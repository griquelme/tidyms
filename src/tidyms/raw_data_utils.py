"""
Functions used to extract information from raw data.

"""

import numpy as np
from .lcms import Chromatogram, LCTrace, MSSpectrum, Roi
from .fileio import MSData
from .utils import find_closest
from . import _constants as c
from . import validation as val
from collections import deque
from copy import deepcopy
from scipy.interpolate import interp1d
from typing import Callable, List, Optional, Tuple, Union


def make_tic(
    ms_data: MSData,
    *,
    kind: str = "tic",
    ms_level: int = 1,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> Chromatogram:
    """
    Creates a total ion chromatogram.

    Parameters
    ----------
    ms_data : MSData
    kind: {"tic", "bpi"}, default="tic"
        `tic` computes the total ion chromatogram. `bpi` computes the base
        peak chromatogram.
    ms_level : int, default=1
        ms level used to build the chromatogram.
    start_time : float, default=0.0
        include scans starting at this acquisition time.
    end_time : float or None, default=None
        Stops when the acquisition time is higher than this value. If None,
        it doesn't filter scans by time.

    Returns
    -------
    chromatograms : lcms.Chromatograms

    """
    if kind == "tic":
        reduce = np.sum
    elif kind == "bpi":
        reduce = np.max
    else:
        msg = "valid modes are tic or bpi"
        raise ValueError(msg)

    n_scan = ms_data.get_n_spectra()
    rt = np.zeros(n_scan)
    tic = np.zeros(n_scan)
    valid_index = list()
    # it is not possible to know a priori how many scans of each level are
    # available in a given file without iterating over it. valid_index holds
    # the index related to the selected level and is used to remove scans
    # from other levels.
    sp_iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level, start_time=start_time, end_time=end_time
    )
    for scan, sp in sp_iterator:
        valid_index.append(scan)
        rt[scan] = sp.time
        if sp.spint.size:
            tic[scan] = reduce(sp.spint)
        else:
            tic[scan] = 0
    tic = tic[valid_index]
    rt = rt[valid_index]
    return Chromatogram(rt, tic, ms_data.separation)


@val.validate_raw_data_utils(val.make_chromatogram_schema)
def make_chromatograms(
    ms_data: MSData,
    mz: np.ndarray,
    *,
    window: Optional[float] = None,
    accumulator: str = "sum",
    fill_missing: bool = True,
    ms_level: int = 1,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[Chromatogram]:
    """
    Computes extracted ion chromatograms using a list of m/z values.

    Parameters
    ----------
    ms_data : MSData
    mz : array
        m/z values used to build the EICs.
    window : positive number or None, default=None
        m/z tolerance used to build the EICs. If ``ms_data.instrument`` is
        ``"qtof"``, the default value is ``0.05``. If ``ms_data.instrument`` is
        ``"orbitrap"`` the default value is ``0.005``.
    accumulator : {"sum", "mean"}, default="sum"
        Mode used to accumulate the values inside the m/z window. ``"sum"``
        computes the total intensity inside the window. ``"mean"`` divides
        the total intensity using the number of points inside the window.
    fill_missing : bool, default=True
        If ``True``, sets the intensity to zero if no signal was found in a
        given scan. If ``False``, missing values are set to NaN.

    ms_level : int, default=1
        ms level used to build the chromatograms.
    start_time : float, default=0.0
        include scans starting at this acquisition time.
    end_time : float or None, default=None
        Stops when the acquisition time is higher than this value.

    Returns
    -------
    chromatograms : List of Chromatograms

    """
    n_sp = ms_data.get_n_spectra()

    # mz_intervals has this shape to be compatible with reduce at
    mz_intervals = np.vstack((mz - window, mz + window)).T.reshape(mz.size * 2)

    eic = np.zeros((mz.size, n_sp))
    if not fill_missing:
        eic[:] = np.nan

    rt = np.zeros(n_sp)
    valid_index = list()
    sp_iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level, start_time=start_time, end_time=end_time
    )
    for scan, sp in sp_iterator:
        valid_index.append(scan)
        rt[scan] = sp.time
        sp_size = sp.mz.size

        # prevents error when working with empty spectra
        if sp_size == 0:
            continue

        # values for each eic in the current scan
        ind_sp = np.searchsorted(sp.mz, mz_intervals)  # slices for each eic
        has_mz = (ind_sp[1::2] - ind_sp[::2]) > 0  # find non-empty slices
        # elements added at the end of mz_sp raise IndexError
        ind_sp[ind_sp >= sp_size] = sp_size - 1
        # this adds the values between two consecutive indices
        tmp_eic = np.where(has_mz, np.add.reduceat(sp.spint, ind_sp)[::2], 0)
        if accumulator == "mean":
            norm = ind_sp[1::2] - ind_sp[::2]
            norm[norm == 0] = 1
            tmp_eic = tmp_eic / norm
        eic[:, scan] = tmp_eic
    valid_index = np.array(valid_index)
    rt = rt[valid_index]
    eic = eic[:, valid_index]

    chromatograms = list()
    for row in eic:
        chromatogram = Chromatogram(rt.copy(), row, mode=ms_data.separation)
        chromatograms.append(chromatogram)
    return chromatograms


@val.validate_raw_data_utils(val.make_roi_schema)
def make_roi(
    ms_data: MSData,
    *,
    tolerance: Optional[float] = None,
    max_missing: Optional[int] = None,
    min_length: Optional[int] = None,
    min_intensity: float = 0.0,
    multiple_match: str = "reduce",
    mz_reduce: Union[str, Callable] = "mean",
    sp_reduce: Union[str, Callable] = "sum",
    targeted_mz: Optional[np.ndarray] = None,
    pad: Optional[int] = None,
    ms_level: int = 1,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    min_snr: float = 10,
    min_distance: Optional[float] = None,
) -> List[Roi]:
    """
    Builds regions of interest (ROI) from raw data.

    ROI are created by connecting values across scans based on the closeness in
    m/z. See the :ref:`user guide <roi-creation>` for a description of the
    algorithm used.

    Parameters
    ----------
    ms_data : MSData
    tolerance : positive number or None, default=None
        m/z tolerance to connect values across scans. If None, the value is
        set based on the value of ``ms_data.instrument``. If ``"qtof"`` is used,
        the tolerance is ``0.01``. If ``"orbitrap"`` is used, the tolerance is
        ``0.005``.
    max_missing : non-negative integer or None, default=None
        maximum number of consecutive missing values in a valid  ROI. If
        ``None``, the value is set to ``1``.
    min_length : positive integer or None, default=None
        The minimum length of a valid ROI, defined as the number of non-NaN
        values in the ROI. If, ``None``, the value is set based on
        ``ms_data.separation``. If ``"uplc"``, the value is set to ``10``. If
        ``"hplc"``, the value is set to ``20``.
    min_intensity : non-negative number , default=0.0
        Minimum intensity in a valid ROI.
    pad: int or None, default=None
        Pad dummy values to the left and right of the ROI. This produces
        better peak picking results when searching low intensity peaks in a
        ROI. Using None set the value to ``2``.
    multiple_match : {"closest", "reduce"}, default="reduce"
        How peaks are matched when there is more than one valid match. If
        ``"closest"`` is used, the closest peak is assigned as a match and
        the others are used to create new ROIs. If ``"reduce"`` is used,
        unique m/z and intensity values are generated using the reduce
        function in `mz_reduce` and `sp_reduce` respectively.
    mz_reduce : "mean" or Callable, default="mean"
        Function used to reduce m/z values. If ``"mean"`` is used, the mean
        value of all valid m/z is used. Any function that accepts numpy
        arrays and return numbers can be used. Used only when
        `multiple_match` is set to ``"reduce"``. See the following
        prototype:

        .. code-block:: python

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

    sp_reduce : {"mean", "sum"} or Callable, default="sum"
        Function used to reduce intensity values. ``"mean"`` computes the
        mean intensity and ``"sum"`` computes the total intensity. Any
        function that accepts numpy arrays and return numbers can be used.
        Only used when `multiple_match` is set to ``"reduce"``. See the
        prototype shown on `mz_reduce`.
    targeted_mz : numpy.ndarray or None, default=None
        A list of m/z values to perform a targeted ROI creation. If this
        value is provided, only ROI with these m/z values will be created.
    ms_level : int, default=1
        ms level used to build the ROI.
    start_time : float, default=0.0
        Use scans starting at this acquisition time.
    end_time : float or None, default=None
        Stops when the acquisition time is higher than this value.
    min_snr : positive number, default=10.0
        Minimum signal-to-noise ratio of the peaks. Used only to convert
        profile data to centroid mode
    min_distance : positive number or None, default=None
        Minimum distance between consecutive peaks. If ``None``, the value
        is set to 0.01 if ``ms_data.instrument`` is ``"qtof"`` or to 0.005 if
        ``ms_data.instrument`` is ``"orbitrap"``. Used only to convert profile
        data to centroid mode.

    Returns
    -------
    roi : list[Roi]
        A list with the detected regions of interest.

    See Also
    --------
    lcms.Roi : Representation of a ROI.
    lcms.LCRoi : ROI used in LC data.

    """

    if targeted_mz is None:
        if min_intensity is None:
            mz_filter = None
        else:
            mz_filter = _make_mz_filter(ms_data, min_intensity, ms_level, start_time, end_time)
        targeted = False
    else:
        mz_filter = np.sort(targeted_mz)
        targeted = True

    if mz_reduce == "mean":
        mz_reduce = np.mean

    if sp_reduce == "sum":
        sp_reduce = np.sum

    rt = np.zeros(ms_data.get_n_spectra())
    processor = _RoiMaker(
        mz_filter,
        max_missing=max_missing,
        min_length=min_length,
        min_intensity=min_intensity,
        tolerance=tolerance,
        multiple_match=multiple_match,
        mz_reduce=mz_reduce,
        sp_reduce=sp_reduce,
        targeted=targeted,
    )

    scans = list()
    sp_iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level, start_time=start_time, end_time=end_time
    )
    for scan, spectrum in sp_iterator:
        rt[scan] = spectrum.time
        scans.append(scan)
        mz, sp = spectrum.find_centroids(min_snr, min_distance)
        processor.feed_spectrum(mz, sp, scan)
        processor.clear_completed_roi()

    # add roi not completed during the last scan
    processor.flag_as_completed()
    processor.clear_completed_roi()
    scans = np.array(scans)
    roi_list = processor.tmp_roi_to_roi(scans, rt, pad, ms_data.separation)

    # TODO: workaround, move code to _RoiMaker
    for k, roi in enumerate(roi_list):
        roi.id = k

    return roi_list


@val.validate_raw_data_utils(val.accumulate_spectra_schema)
def accumulate_spectra(
    ms_data: MSData,
    *,
    start_time: float,
    end_time: float,
    subtract_left_time: Optional[float] = None,
    subtract_right_time: Optional[float] = None,
    ms_level: int = 1,
) -> MSSpectrum:
    """
    accumulates a series of consecutive spectra into a single spectrum.

    Parameters
    ----------
    ms_data : MSData
    start_time: float
        Start accumulating scans at this scan time.
    end_time: float
        Ends accumulation after this scan time.
    subtract_left_time : float or None, default=None
        Scans with acquisition times lower than this value are subtracted from
        the accumulated spectrum. If ``None``, no subtraction is done.
    subtract_right_time : float or None, default=None
        Scans with acquisition times greater than this value are subtracted from
        the accumulated spectrum. If ``None``, no subtraction is done.
    ms_level : int, default=1
        ms level used to build the accumulated spectrum.

    Returns
    -------
    MSSpectrum

    """
    if ms_data.ms_mode == "centroid":
        sp = _accumulate_spectra_centroid(
            ms_data,
            start_time,
            end_time,
            subtract_left_time,
            subtract_right_time,
            ms_level,
        )
    else:  # profile
        sp = _accumulate_spectra_profile(
            ms_data,
            start_time,
            end_time,
            subtract_left_time,
            subtract_right_time,
            ms_level,
        )
    return sp


def _accumulate_spectra_centroid(
    ms_data: MSData,
    start_time: float,
    end_time: float,
    subtract_left_time: float,
    subtract_right_time: float,
    ms_level: int,
) -> MSSpectrum:
    """
    accumulates a series of consecutive spectra into a single spectrum.

    auxiliary function for accumulate_spectra.

    """
    # don't remove any m/z value when detecting rois
    max_missing = ms_data.get_n_spectra()

    roi = make_roi(
        ms_data,
        max_missing=max_missing,
        min_length=1,
        start_time=subtract_left_time,
        end_time=subtract_right_time,
        ms_level=ms_level,
    )

    mz = np.zeros(len(roi))
    spint = mz.copy()

    # set subtract values to negative
    for k, r in enumerate(roi):
        sign = -np.ones(r.time.size)
        start_index, end_index = np.searchsorted(r.time, [start_time, end_time])
        sign[start_index:end_index] = 1
        mz[k] = np.nanmean(r.mz)
        spint[k] = np.nansum(r.spint * sign)

    # remove negative values
    pos_values = spint > 0
    mz = mz[pos_values]
    spint = spint[pos_values]

    # sort values
    sorted_index = np.argsort(mz)
    mz = mz[sorted_index]
    spint = spint[sorted_index]

    sp = MSSpectrum(mz, spint, ms_level=ms_level, instrument=ms_data.instrument)
    return sp


def _accumulate_spectra_profile(
    ms_data: MSData,
    start_time: float,
    end_time: float,
    subtract_left_time: float,
    subtract_right_time: float,
    ms_level: int,
) -> MSSpectrum:
    """
    aux function for accumulate_spectra.

    """
    # The spectra are accumulated in two steps:
    #
    #  1.  iterate through scans to build a grid of m/z values for the
    #      accumulated spectra.
    #  2.  A second iteration is done to interpolate the intensity in each
    #      scan to the m/z grid and generate the accumulated spectrum.
    #
    #  This process is done in two steps to avoid storing the intensity
    #  values from each scan until the grid is built.

    accumulated_mz = None
    # m/z tol. A small value is used to prevent distortions in the results
    tol = 0.00005
    sp_iter = ms_data.get_spectra_iterator(
        ms_level, start_time=subtract_left_time, end_time=subtract_right_time
    )
    # first iteration. Builds a grid of m/z values for the accumulated
    # spectrum. The grid is extended using new m/z values that appear
    # in each new scan
    for scan, sp in sp_iter:
        if accumulated_mz is None:
            accumulated_mz = sp.mz
        ind = find_closest(accumulated_mz, sp.mz)
        no_match = np.abs(accumulated_mz[ind] - sp.mz) > tol
        accumulated_mz = np.sort(np.hstack((accumulated_mz, sp.mz[no_match])))

    accumulated_sp = np.zeros_like(accumulated_mz)
    sp_iter = ms_data.get_spectra_iterator(
        ms_level, start_time=subtract_left_time, end_time=subtract_right_time
    )

    for _, sp in sp_iter:
        interpolator = interp1d(sp.mz, sp.spint, fill_value=0.0)
        if (sp.time < start_time) or (sp.time > end_time):
            sign = -1
        else:
            sign = 1
        accumulated_sp += interpolator(accumulated_mz) * sign

    # set negative values that may result from subtraction to zero
    is_positive_sp = accumulated_sp > 0
    accumulated_mz = accumulated_mz[is_positive_sp]
    accumulated_sp = accumulated_sp[is_positive_sp]

    res = MSSpectrum(
        accumulated_mz,
        accumulated_sp,
        instrument=ms_data.instrument,
        ms_level=ms_level,
        is_centroid=False,
    )
    return res


class _RoiMaker:
    """
    Creates and extends ROIs using spectrum data.

    Auxiliary class to make_roi

    Attributes
    ----------
    tmp_roi_list : _TempRoiList
    valid_roi : List
        Valid ROI.
    mz_filter : sorted array
        m/z values used as a first filter for the values of spectra provided.
        m/z values in spectra with distances larger than `tolerance` are
        ignored.
    max_missing : int
        Maximum number of missing values allowed in a ROI. ROIs with values
        greater than this value are flagged as completed.
    min_length : int
        If the length of a completed ROI is greater than this value, it is
        considered valid and stored in `roi`
    min_intensity : float
        If the maximum intensity in a completed ROI is greater than this value,
        it is considered valid and stored in `roi`.
    tolerance : float
        m/z tolerance used to extend a ROI. If an m/z value in a spectrum is
        closer than this value to the mean m/z of a ROI, then it is used to
        extend it. If the m/z value is not close to any ROI, then if it is
        closer to any value in `mz_seed`, it is used to create a new ROI.
        The remaining values in the spectrum are discarded.
    multiple_match : {"reduce", "closest"}
        Behaviour when more than one m/z value in a spectrum matches a given
        ROI. If `closest` is used, then the closest peak is used to extend the
        ROI and the others are used to create new ROI as described above. If
        `reduce` is used, then a unique m/z and intensity value is computed
        by using the `mz_reduce` and `sp_reduce` functions.
    mz_reduce : Callable
        A function that takes an array of floats and returns a single float
        value (e.g. np.mean)
    sp_reduce : Callable
        Any function that takes an array of floats and returns a single float
        value (e.g. np.sum)
    targeted : bool
        If ``True``, the mean of each ROI is updated after a new element is
        appended. Else, the mean when the ROI was created is used.

    """

    def __init__(
        self,
        mz_filter: Optional[np.ndarray],
        max_missing: int,
        min_length: Optional[int],
        min_intensity: Optional[float],
        tolerance: float,
        multiple_match: str,
        mz_reduce: Optional[Callable],
        sp_reduce: Optional[Callable],
        targeted: bool = False,
    ):
        self.tmp_roi_list = _TempRoiList(update_mean=not targeted)
        self.valid_roi = deque()
        self.mz_filter = mz_filter
        self.max_missing = max_missing
        self.min_intensity = min_intensity
        self.min_length = min_length
        self.tolerance = tolerance
        self.multiple_match = multiple_match
        self.mz_reduce = mz_reduce
        self.sp_reduce = sp_reduce
        self.targeted = targeted

        if targeted:
            self.tmp_roi_list.initialize(mz_filter)

    def feed_spectrum(self, mz: np.ndarray, sp: np.ndarray, scan: int):
        """
        Uses a spectrum to extend and create ROIs.

        Parameters
        ----------
        mz : array
            sorted array of m/z
        sp : array
            Intensity array associated with each m/z value.
        scan : int
            scan number associated with the spectrum.

        """
        if self.mz_filter is not None:
            mz, sp = _filter_invalid_mz(self.mz_filter, mz, sp, self.tolerance)
        if self.tmp_roi_list.roi:
            match_ind, match_mz, match_sp, no_match_mz, no_match_sp = _match_mz(
                self.tmp_roi_list.mz_mean,
                mz,
                sp,
                self.tolerance,
                self.multiple_match,
                self.mz_reduce,
                self.sp_reduce,
            )
            self.tmp_roi_list.extend(match_mz, match_sp, scan, match_ind)
            if not self.targeted:
                self.tmp_roi_list.insert(no_match_mz, no_match_sp, scan)
        else:
            if not self.targeted:
                self.tmp_roi_list.insert(mz, sp, scan)

    def clear_completed_roi(self):
        """
        Flags ROI as completed. Completed valid ROIs are stored  in the `roi`
        attribute. Invalid ROIs are cleared.

        """
        finished_mask = self.tmp_roi_list.missing_count > self.max_missing
        finished_roi_index = np.where(finished_mask)[0]
        if self.min_intensity is not None:
            valid_mask = finished_mask & (self.tmp_roi_list.max_int >= self.min_intensity)
        else:
            valid_mask = finished_mask

        if self.min_length is not None:
            valid_mask &= self.tmp_roi_list.length >= self.min_length

        valid_index = np.where(valid_mask)[0]

        for i in valid_index:
            r = deepcopy(self.tmp_roi_list.roi[i])
            self.valid_roi.append(r)

        self.tmp_roi_list.clear(finished_roi_index)

    def flag_as_completed(self):
        """
        Mark all ROis as completed.

        """
        self.tmp_roi_list.missing_count[:] = self.max_missing + 1

    def tmp_roi_to_roi(
        self, valid_scan: np.ndarray, time: np.ndarray, pad: int, separation: str
    ) -> List[Roi]:
        """
        Converts completed valid _TempRoi objects into Roi.

        Parameters
        ----------
        valid_scan : array
            scan values used to build the ROIs.
        time : array
            acquisition time of each scan.
        pad : int
            Number of dummy values to pad each ROI.
        separation : str
            Separation method of the ms file. Used to create LCRoi objects or
            Roi objects.

        Returns
        -------
        valid_roi: List[Roi]
        """
        valid_roi = list()
        while self.valid_roi:
            tmp = self.valid_roi.popleft()
            tmp.pad(pad, valid_scan)
            roi = tmp.convert_to_roi(time, valid_scan, separation)
            valid_roi.append(roi)
        return valid_roi


class _TempRoiList:
    """
    Container object of Temporary ROI.

    Auxiliary class used in make_roi.

    Attributes
    ----------
    roi : List[_TempRoi]
        List of ROI, sorted by mean m/z value.
    mz_mean : array[float]
        tracks the mean m/z value of each ROI.
    mz_sum : array[float]
        Sum of m/z values in each ROI, used to update the mean.
    max_int : array[float[
        Maximum intensity stored in each ROI.
    missing_count : array[int]
        Number of times that a ROI has not been extended after calling `extend`.
        Each time that a ROI is extended the count is reset to 0.
    length : array[int]
        Number of elements in each ROI.
    update_mean : bool
        If ``True``, the mean of each ROI is updated after a new element is
        appended. Else, the mean when the ROI was created is used.

    """

    def __init__(self, update_mean: bool = True):
        self.roi = list()  # type: List[_TempRoi]
        self.mz_mean = np.array([])  # type: np.ndarray
        self.mz_sum = np.array([])
        self.max_int = np.array([])
        self.missing_count = np.array([], dtype=int)
        self.length = np.array([], dtype=int)
        self.update_mean = update_mean

    def insert(self, mz: np.ndarray, sp: np.ndarray, scan: int):
        """
        Creates new ROI and insert them in the list while keeping the order.

        Parameters
        ----------
        mz : array
            m/z values used to initialize the new ROIs.
        sp:  array
            Intensity values used to initialize the new ROIs.
        scan: int
            scan number used to initialize the new ROIs.

        """
        index = np.searchsorted(self.mz_mean, mz)
        # update roi tracking values
        self.mz_mean = np.insert(self.mz_mean, index, mz)
        self.mz_sum = np.insert(self.mz_sum, index, mz)
        self.max_int = np.insert(self.max_int, index, sp)
        self.missing_count = np.insert(self.missing_count, index, np.zeros_like(index))
        self.length = np.insert(self.length, index, np.ones_like(index))

        # insert new roi
        new_roi = _create_roi_list(mz, sp, scan)
        offset = 0
        for i, roi in zip(index, new_roi):
            self.roi.insert(i + offset, roi)
            offset += 1

    def extend(self, mz: np.ndarray, sp: np.ndarray, scan: int, index: np.ndarray):
        """
        Extends existing ROI.

        Parameters
        ----------
        mz : array
            m/z values used to extend the ROI
        sp : array
            Intensity values used to extend the ROI
        scan : int
            Scan number used to extend the ROI
        index : array
            Indices of the ROI to extend.

        """
        for i, m, s in zip(index, mz, sp):
            self.roi[i].append(m, s, scan)
        self.length[index] += 1
        if self.update_mean:
            self.mz_sum[index] += mz
            self.mz_mean[index] = self.mz_sum[index] / self.length[index]
        self.max_int[index] = np.maximum(self.max_int[index], sp)
        self.missing_count += 1
        self.missing_count[index] = 0

    def clear(self, index: np.ndarray):
        """
        Empties the m/z, intensity and scan values stored in each ROI. The
        mean value of each ROI is kept.

        Parameters
        ----------
        index : array
            Indices of ROI to clear.

        """
        for i in index:
            self.roi[i].clear()

        self.mz_sum[index] = 0
        self.max_int[index] = 0
        self.missing_count[index] = 0
        self.length[index] = 0

    def initialize(self, mz: np.ndarray):
        self.insert(mz, mz, 0)
        self.clear(np.arange(mz.size))


class _TempRoi:
    """
    Stores data from a ROI.

    Auxiliary class used in make_roi

    Attributes
    ----------
    mz : Deque
    spint : Deque
    scan : Deque

    """

    def __init__(self):
        """
        Creates a new empty Temporary ROI

        """
        self.mz = deque()
        self.spint = deque()
        self.scan = deque()

    def append(self, mz: float, spint: float, scan: int):
        """
        Append new m/z, intensity and scan values.

        Parameters
        ----------
        mz : float
        spint : float
        scan : int

        """
        self.mz.append(mz)
        self.spint.append(spint)
        self.scan.append(scan)

    def clear(self):
        """
        Empty the m/z, intensity and scan values stored.

        """
        self.mz = deque()
        self.spint = deque()
        self.scan = deque()

    def pad(self, n: int, valid_scan: np.ndarray):
        """
        Pad the ROI m/z and intensity with NaN. Values are padded only if the
        scan number allows it, that means if there are `n` points to the left or
        the right of the minimum and maximum scan number of the ROI in
        `valid_scans` (see the examples for detail)

        Parameters
        ----------
        n : int
            Number of points to pad to the left and righ
        valid_scan

        Examples
        --------
        >>> roi = _TempRoi()
        >>> roi.append(1, 1, 1)
        >>> valid_scans = np.array([0, 1, 2, 3, 4, 5])
        >>> roi.pad(2, valid_scans)
        >>> roi.scan
        [0, 1, 2, 3]

        """
        first_scan = self.scan[0]
        last_scan = self.scan[-1]
        start, end = np.searchsorted(valid_scan, [first_scan, last_scan + 1])
        left_pad_index = max(0, start - n)
        n_left = start - left_pad_index
        right_pad_index = min(valid_scan.size, end + n)
        n_right = right_pad_index - end

        # left pad
        self.mz.extendleft([np.nan] * n_left)
        self.spint.extendleft([np.nan] * n_left)
        self.scan.extendleft(valid_scan[left_pad_index:start][::-1])

        # right pad
        self.mz.extend([np.nan] * n_right)
        self.spint.extend([np.nan] * n_right)
        self.scan.extend(valid_scan[end:right_pad_index])

    def convert_to_roi(self, rt: np.ndarray, scans: np.ndarray, separation: str) -> Roi:
        """
        Converts a TemporaryRoi into a ROI object

        Parameters
        ----------
        rt: array
            Acquisition times of each scan
        scans : array
            Sorted scan numbers used to build the ROIs.
        separation : mode to pass to ROI creation.

        Returns
        -------
        Roi

        """
        # new arrays that include missing values. These new arrays may include
        # scans from other ms levels that must be removed
        first_scan = self.scan[0]
        last_scan = self.scan[-1]
        size = last_scan + 1 - first_scan
        mz_tmp = np.ones(size) * np.nan
        spint_tmp = mz_tmp.copy()

        # copy values from the ROI to the new arrays
        scan_index = np.array(self.scan) - self.scan[0]
        mz_tmp[scan_index] = self.mz
        spint_tmp[scan_index] = self.spint

        # remove scan numbers from other ms levels (i.e. scan numbers that are
        # not in the scans array)
        start_ind, end_ind = np.searchsorted(scans, [first_scan, last_scan + 1])
        scan_tmp = scans[start_ind:end_ind].copy()
        valid_index = scan_tmp - first_scan
        mz_tmp = mz_tmp[valid_index]
        spint_tmp = spint_tmp[valid_index]
        rt_tmp = rt[scan_tmp].copy()

        # Create ROI objects
        if separation in c.LC_MODES:
            roi = LCTrace(rt_tmp, spint_tmp, mz_tmp, scan_tmp, mode=separation)
        else:
            raise NotImplementedError
            # roi = Roi(spint_tmp, mz_tmp, rt_tmp, scan_tmp, mode=separation)
        return roi


def _make_mz_filter(
    ms_data: MSData,
    min_intensity: float,
    ms_level: int,
    start_time: float,
    end_time: float,
):
    """
    Creates a list of m/z values to initialize ROI for untargeted feature
    detection based on the intensity observed for each m/z value across scans.

    Auxiliary function to make_roi.

    Parameters
    ----------
    ms_data : MSData
    min_intensity : float
        Only include m/z values with intensities higher than this value
    ms_level : int
        Only include scans with this MS level.
    start_time : float, default=0.0
        Ignore scans with acquisition times lower than this value.
    end_time : float or None, default=None
        Ignore scans with acquisition times higher than this value.

    Returns
    -------

    """
    iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level, start_time=start_time, end_time=end_time
    )
    mz_seed = [sp.mz[sp.spint > min_intensity] for _, sp in iterator]
    return np.unique(np.hstack(mz_seed))


def _filter_invalid_mz(
    valid_mz: np.ndarray, mz: np.ndarray, sp: np.ndarray, tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find values in the spectrum that are within tolerance with the m/z values
    in the seed.

    Auxiliary function to _RoiProcessor.extend

    Parameters
    ----------
    valid_mz : np.ndarray
        sorted array of m/z values.
    mz: array
        m/z values to filter
    sp : array
        intensity values associated to each m/z.
    tolerance : float

    Returns
    -------
    mz : np.ndarray
        Filtered m/z values in the spectrum.
    spint : np.ndarray
        Filtered intensity values in the spectrum

    """

    closest_index = find_closest(valid_mz, mz)
    dmz = np.abs(valid_mz[closest_index] - mz)
    match_mask = dmz <= tolerance  # type: np.ndarray
    return mz[match_mask], sp[match_mask]


def _create_roi_list(mz: np.ndarray, sp: np.ndarray, scan: int) -> List[_TempRoi]:
    roi_list = list()
    for m, s in zip(mz, sp):
        roi = _TempRoi()
        roi.append(m, s, scan)
        roi_list.append(roi)
    return roi_list


def _match_mz(
    mz1: np.ndarray,
    mz2: np.ndarray,
    sp2: np.ndarray,
    tolerance: float,
    mode: str,
    mz_reduce: Callable,
    sp_reduce: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    aux function to add method in _RoiProcessor. Find matched values.

    Parameters
    ----------
    mz1: numpy.ndarray
        _RoiProcessor mz_mean
    mz2: numpy.ndarray
        mz values to match
    sp2: numpy.ndarray
        intensity values associated to mz2
    tolerance: float
        tolerance used to match values
    mode: {"closest", "reduce"}
        Behaviour when more than one peak in mz2 matches with a given peak in
        mz1. If mode is `closest`, then the closest peak is assigned as a
        match and the others are assigned to no match. If mode is `merge`, then
        a unique mz and int value is generated using the average of the mz and
        the sum of the intensities.

    Returns
    ------
    match_index: numpy.ndarray
        index when of peaks matching in mz1.
    mz_match: numpy.ndarray
        values of mz2 that matches with mz1
    sp_match: numpy.ndarray
        values of sp2 that matches with mz1
    mz_no_match: numpy.ndarray
    sp_no_match: numpy.ndarray
    """
    closest_index = find_closest(mz1, mz2)
    dmz = np.abs(mz1[closest_index] - mz2)
    match_mask = dmz <= tolerance  # type: np.ndarray
    no_match_mask = ~match_mask
    match_index = closest_index[match_mask]

    # check multiple_matches
    match_unique, match_first, match_count = np.unique(
        match_index, return_counts=True, return_index=True
    )

    # set match values
    match_index = match_unique
    sp_match = sp2[match_mask]
    mz_match = mz2[match_mask]

    # solve multiple matches
    multiple_match_mask = match_count > 1
    multiple_match_first = match_first[multiple_match_mask]
    if match_first.size > 0:
        multiple_match_count = match_count[multiple_match_mask]
        if mode == "reduce":
            for first, count in zip(multiple_match_first, multiple_match_count):
                # mz1 and mz2 are both sorted, the multiple matches are
                # consecutive
                mz_multiple_match = mz_match[first : (first + count)]
                sp_multiple_match = sp_match[first : (first + count)]
                mz_match[first] = mz_reduce(mz_multiple_match)
                sp_match[first] = sp_reduce(sp_multiple_match)
        elif mode == "closest":
            match_index_mz = np.where(match_mask)[0][match_first]
            multiple_match_index_mz = match_index_mz[multiple_match_mask]
            iterator = zip(multiple_match_index_mz, multiple_match_first, multiple_match_count)
            for mz2_index, first, count in iterator:
                closest = np.argmin(dmz[mz2_index : mz2_index + count])
                # flag all multiple matches as no match except the closest one
                no_match_mask[mz2_index : mz2_index + count] = True
                no_match_mask[mz2_index + closest] = False
                mz_match[first] = mz_match[first + closest]
                sp_match[first] = sp_match[first + closest]
        else:
            msg = "mode must be `closest` or `merge`"
            raise ValueError(msg)

    mz_match = mz_match[match_first]
    sp_match = sp_match[match_first]
    mz_no_match = mz2[no_match_mask]
    sp_no_match = sp2[no_match_mask]

    return match_index, mz_match, sp_match, mz_no_match, sp_no_match
