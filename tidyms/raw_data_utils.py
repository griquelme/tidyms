"""
Functions used to extract information from raw data.

"""

import numpy as np
from .lcms import Chromatogram, LCRoi, MSSpectrum, Roi
from .fileio import MSData
from .utils import find_closest
from . import _constants as c
from . import validation as val
from collections import deque, namedtuple
from scipy.interpolate import interp1d
from typing import Callable, List, Optional, Tuple, Union

# remove start and end from params


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
        ms_level=ms_level,
        start_time=start_time,
        end_time=end_time
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
    mz_intervals = (np.vstack((mz - window, mz + window))
                    .T.reshape(mz.size * 2))

    eic = np.zeros((mz.size, n_sp))
    rt = np.zeros(n_sp)
    valid_index = list()
    sp_iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level,
        start_time=start_time,
        end_time=end_time
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
        tmp_eic = np.where(
            has_mz, np.add.reduceat(sp.spint, ind_sp)[::2], 0)
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
    min_distance: Optional[float] = None
) -> List[Roi]:
    """
    Builds regions of interest from raw data.

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
        The minimum length of a valid ROI. If, ``None``, the value is set
        based on ``ms_data.separation``. If ``"uplc"``, the value is set to
        ``10``. If ``"hplc"``, the value is set to ``20``.
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
        sp_seed = ms_data.get_spectrum(0)
        mz_seed, _ = sp_seed.find_centroids(min_snr, min_distance)
        targeted = False
    else:
        mz_seed = targeted_mz
        targeted = True

    rt = np.zeros(ms_data.get_n_spectra())
    processor = _RoiMaker(
        mz_seed,
        max_missing=max_missing,
        min_length=min_length,
        min_intensity=min_intensity,
        tolerance=tolerance,
        multiple_match=multiple_match,
        mz_reduce=mz_reduce,
        sp_reduce=sp_reduce,
        targeted=targeted
    )

    scans = list()  # scan number used in to build ROI
    sp_iterator = ms_data.get_spectra_iterator(
        ms_level=ms_level,
        start_time=start_time,
        end_time=end_time
    )
    for scan, sp in sp_iterator:
        rt[scan] = sp.time
        scans.append(scan)
        sp_mz, sp_spint = sp.find_centroids(min_snr, min_distance)
        processor.extend_roi(sp_mz, sp_spint, scan)
        processor.store_completed_roi()

    # add roi not completed during the last scan
    processor.flag_as_completed()
    processor.store_completed_roi()

    # extend roi, find rt of each roi and convert to Roi objects
    roi = processor.process_completed_roi(scans, rt, pad, ms_data.separation)
    return roi


@val.validate_raw_data_utils(val.accumulate_spectra_schema)
def accumulate_spectra(
    ms_data: MSData,
    *,
    start_time: float,
    end_time: float,
    subtract_left_time: Optional[float] = None,
    subtract_right_time: Optional[float] = None,
    ms_level: int = 1
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
            ms_level
        )
    else:   # profile
        sp = _accumulate_spectra_profile(
            ms_data,
            start_time,
            end_time,
            subtract_left_time,
            subtract_right_time,
            ms_level
        )
    return sp


class _RoiMaker:
    """
    Helper class used by make_roi to create Roi instances from raw data.

    Attributes
    ----------
    mz_mean: numpy.ndarray
        mean value of mz for a given row in mz_array. Used to add new values
        based on a tolerance. Updated after adding a new column
    n_missing: numpy.ndarray
        number of consecutive missing values. Used to detect finished rois
    roi: list[_TemporaryRoi]
    """

    def __init__(self, mz_seed: np.ndarray, max_missing: int = 1,
                 min_length: int = 5, min_intensity: float = 0,
                 tolerance: float = 0.005, multiple_match: str = "closest",
                 mz_reduce: Optional[Callable] = None,
                 sp_reduce: Union[str, Callable] = "sum",
                 targeted: bool = False):
        """

        Parameters
        ----------
        mz_seed: numpy.ndarray
            initial values to build rois
        max_missing: int
            maximum number of missing consecutive values. when a row surpass
            this number the roi is flagged as finished.
        min_length: int
            The minimum length of a finished roi to be considered valid before
            being added to the roi list.
        min_intensity: float
        tolerance: float
            mz tolerance used to connect values.
        multiple_match: {"closest", "reduce"}
            how to match peaks when there is more than one match. If mode is
            `closest`, then the closest peak is assigned as a match and the
            others are assigned to no match. If mode is `reduce`, then a unique
            mz and intensity value is generated using the reduce function in
            `mz_reduce` and `spint_reduce` respectively.
        mz_reduce: callable, optional
            function used to reduce mz values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is ``"reduce"``. See the following prototype:

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

            If None, m/z values are reduced using the mean.

        sp_reduce: str or callable
            function used to reduce spint values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is ``"reduce"``. To use custom functions see the prototype shown on
            `mz_reduce`.
        """
        if multiple_match not in ["closest", "reduce"]:
            msg = "Valid modes are closest or reduce"
            raise ValueError(msg)

        if mz_reduce == "mean":
            self._mz_reduce = np.mean
        else:
            self._mz_reduce = mz_reduce

        if sp_reduce == "mean":
            self._spint_reduce = np.mean
        elif sp_reduce == "sum":
            self._spint_reduce = np.sum
        else:
            self._spint_reduce = sp_reduce

        # temporary roi data
        self.mz_mean = np.unique(mz_seed.copy())
        # roi index maps the values in mz_mean to a temp roi in temp_roi_dict
        self.roi_index = np.arange(mz_seed.size)
        self.n_missing = np.zeros_like(mz_seed, dtype=int)
        self.max_intensity = np.zeros_like(mz_seed)
        self.length = np.zeros_like(mz_seed, dtype=int)
        self.temp_roi_dict = {x: _make_temporary_roi() for x in self.roi_index}
        self.roi = list()

        # parameters used to build roi
        self.min_intensity = min_intensity
        self.max_missing = max_missing
        self.min_length = min_length
        self.tolerance = tolerance
        self.multiple_match = multiple_match
        self.targeted = targeted

    def extend_roi(self, mz: np.ndarray, sp: np.ndarray, scan: int):
        """
        connects mz values with self.mz_mean to extend existing roi.
        Non-matching mz values are used to create new temporary roi.

        """
        self.n_missing += 1

        if mz.size:
            # find matching and non-matching mz values
            match_index, mz_match, sp_match, mz_no_match, sp_no_match = \
                _match_mz(
                    self.mz_mean, mz, sp, self.tolerance,
                    self.multiple_match, self._mz_reduce, self._spint_reduce
                )

            # extend matching roi
            for k, k_mz, k_sp in zip(match_index, mz_match, sp_match):
                k_temp_roi = self.temp_roi_dict[self.roi_index[k]]
                _append_to__roi(k_temp_roi, k_mz, k_sp, scan)

            # update mz_mean and missing values
            updated_mean = (
                (
                    self.mz_mean[match_index] * self.length[match_index]
                    + mz_match
                )
                / (self.length[match_index] + 1)
            )

            self.length[match_index] += 1
            # reset missing count for matching roi
            self.n_missing[match_index] = 0
            self.max_intensity[match_index] = \
                np.maximum(self.max_intensity[match_index], sp_match)

            # if there are non-matching mz values, use them to build new rois.
            # in targeted mode, only roi with specified mz values are built
            if not self.targeted:
                self.mz_mean[match_index] = updated_mean
                self.create_new_roi(mz_no_match, sp_no_match, scan)

    def store_completed_roi(self):
        """
        store completed ROIs. Valid ROI are appended toi roi attribute.
        The validity of the ROI is checked based on roi length and minimum
        intensity.

        """

        # check completed rois
        is_completed = self.n_missing > self.max_missing

        # length and intensity check
        is_valid_roi = ((self.length >= self.min_length) &
                        (self.max_intensity >= self.min_intensity))

        # add valid roi to roi list
        completed_index = np.where(is_completed)[0]
        for ind in completed_index:
            roi_ind = self.roi_index[ind]
            finished_roi = self.temp_roi_dict.pop(roi_ind)
            if is_valid_roi[ind]:
                self.roi.append(finished_roi)

        # remove completed roi
        if self.targeted:
            self.n_missing[is_completed] = 0
            self.length[is_completed] = 0
            self.max_intensity[is_completed] = 0
            max_roi_ind = self.roi_index.max()
            n_completed = is_completed.sum()
            new_indices = np.arange(max_roi_ind + 1,
                                    max_roi_ind + 1 + n_completed)
            self.roi_index[is_completed] = new_indices
            new_tmp_roi = {k: _make_temporary_roi() for k in new_indices}
            self.temp_roi_dict.update(new_tmp_roi)
        else:
            self.mz_mean = self.mz_mean[~is_completed]
            self.n_missing = self.n_missing[~is_completed]
            self.length = self.length[~is_completed]
            self.roi_index = self.roi_index[~is_completed]
            self.max_intensity = self.max_intensity[~is_completed]

    def create_new_roi(self, mz: np.ndarray, sp: np.ndarray, scan: int):
        """
        Creates new temporary roi from non-matching values

        """

        # finds roi index for new temp roi and update metadata
        max_index = self.roi_index.max()
        new_indices = np.arange(mz.size) + max_index + 1
        mz_mean_tmp = np.hstack((self.mz_mean, mz))
        roi_index_tmp = np.hstack((self.roi_index, new_indices))
        n_missing_tmp = np.zeros_like(new_indices, dtype=int)
        n_missing_tmp = np.hstack((self.n_missing, n_missing_tmp))
        length_tmp = np.ones_like(new_indices, dtype=int)
        length_tmp = np.hstack((self.length, length_tmp))
        max_int_tmp = np.zeros_like(new_indices, dtype=float)
        max_int_tmp = np.hstack((self.max_intensity, max_int_tmp))

        # temp roi creation
        for k_index, k_mz, k_sp in zip(new_indices, mz, sp):
            new_roi = _TemporaryRoi([k_mz], [k_sp], [scan])
            self.temp_roi_dict[k_index] = new_roi

        # replace new temp roi metadata
        # roi extension is done using bisection search, all values are sorted
        # using the mz values
        sorted_index = np.argsort(mz_mean_tmp)
        self.mz_mean = mz_mean_tmp[sorted_index]
        self.roi_index = roi_index_tmp[sorted_index]
        self.n_missing = n_missing_tmp[sorted_index]
        self.length = length_tmp[sorted_index]
        self.max_intensity = max_int_tmp[sorted_index]

    def flag_as_completed(self):
        self.n_missing[:] = self.max_missing + 1

    def process_completed_roi(
        self,
        valid_scan: List[int],
        rt: np.ndarray,
        pad: int,
        separation: str
    ) -> List[Roi]:
        """
        Converts valid ROI into ROI objects.
        Parameters
        ----------
        valid_scan : list
            Scan numbers used for ROI creation
        rt : array
            rt values associated to each scan
        pad : int
            Number of dummy values to pad the ROI with
        separation : {"uplc", "hplc"}
            separation value to pass to ROI constructor function.

        Returns
        -------
        List[ROI] : List of completed ROI.

        """
        valid_scan = np.array(valid_scan)
        roi_list = list()
        for r in self.roi:
            # converting to deque makes padding easier
            r = _TemporaryRoi(deque(r.mz), deque(r.sp), deque(r.scan))
            _pad_roi(r, pad, valid_scan)
            r = _build_roi(r, rt, valid_scan, separation)
            roi_list.append(r)
        return roi_list


def _match_mz(mz1: np.ndarray, mz2: np.ndarray, sp2: np.ndarray,
              tolerance: float, mode: str, mz_reduce: Callable,
              sp_reduce: Callable
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray]:
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
    mode: {"closest", "merge"}
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
    match_mask = (dmz <= tolerance)     # type: np.ndarray
    no_match_mask = ~match_mask
    match_index = closest_index[match_mask]

    # check multiple_matches
    unique, first_index, count_index = np.unique(match_index,
                                                 return_counts=True,
                                                 return_index=True)

    # set match values
    match_index = unique
    sp_match = sp2[match_mask][first_index]
    mz_match = mz2[match_mask][first_index]

    # compute matches for duplicates
    multiple_match_mask = count_index > 1
    first_index = first_index[multiple_match_mask]
    if first_index.size > 0:
        first_index_index = np.where(count_index > 1)[0]
        count_index = count_index[multiple_match_mask]
        iterator = zip(first_index_index, first_index, count_index)
        if mode == "closest":
            rm_index = list()   # list of duplicate index to remove
            mz_replace = list()
            spint_replace = list()
            for first_ind, index, count in iterator:
                # check which of the duplicate is closest, the rest are removed
                closest = \
                    np.argmin(dmz[match_mask][index:(index + count)]) + index
                mz_replace.append(mz2[match_mask][closest])
                spint_replace.append(sp2[match_mask][closest])
                remove = np.arange(index, index + count)
                remove = np.setdiff1d(remove, closest)
                rm_index.extend(remove)
            # fix rm_index to full mz2 size
            rm_index = np.where(match_mask)[0][rm_index]
            no_match_mask[rm_index] = True
            mz_match[first_index_index] = mz_replace
            sp_match[first_index_index] = spint_replace
        elif mode == "reduce":
            for first_ind, index, count in iterator:
                # check which of the duplicate is closest
                mz_multiple_match = mz2[match_mask][index:(index + count)]
                sp_multiple_match = sp2[match_mask][index:(index + count)]
                mz_match[first_ind] = mz_reduce(mz_multiple_match)
                sp_match[first_ind] = sp_reduce(sp_multiple_match)
        else:
            msg = "mode must be `closest` or `merge`"
            raise ValueError(msg)

    mz_no_match = mz2[no_match_mask]
    sp_no_match = sp2[no_match_mask]
    return match_index, mz_match, sp_match, mz_no_match, sp_no_match


_TemporaryRoi = namedtuple("TemporaryRoi", ["mz", "sp", "scan"])


def _make_temporary_roi() -> _TemporaryRoi:
    return _TemporaryRoi([], [], [])


def _append_to__roi(roi: _TemporaryRoi, mz: float, sp: float,
                    scan: int):
    roi.mz.append(mz)
    roi.sp.append(sp)
    roi.scan.append(scan)


def _pad_roi(roi: _TemporaryRoi, n: int, valid_scan: np.ndarray):
    first_scan = roi.scan[0]
    last_scan = roi.scan[-1]
    start, end = np.searchsorted(valid_scan, [first_scan, last_scan + 1])
    l_pad_index = max(0, start - n)
    nl = start - l_pad_index
    r_pad_index = min(valid_scan.size, end + n)
    nr = r_pad_index - end

    # fill values
    sp_max = max(roi.sp)
    sp_min = min(roi.sp)
    mz_fill = np.mean(roi.mz)
    sp_threshold = 0.75 * sp_max

    # left pad
    sp_fill_left = sp_max if (roi.sp[0] > sp_threshold) else sp_min
    roi.mz.extendleft([mz_fill] * nl)
    roi.sp.extendleft([sp_fill_left] * nl)
    roi.scan.extendleft(valid_scan[l_pad_index:start][::-1])

    # right pad
    sp_fill_right = sp_max if (roi.sp[-1] > sp_threshold) else sp_min
    roi.mz.extend([mz_fill] * nr)
    roi.sp.extend([sp_fill_right] * nr)
    roi.scan.extend(valid_scan[end:r_pad_index])


def _build_roi(
        roi: _TemporaryRoi, rt: np.ndarray, scans: np.ndarray, separation: str
) -> Roi:
    """
    Converts a TemporaryRoi into a ROI object

    Parameters
    ----------
    roi : TemporaryRoi
    rt: array
        Retention times associated to each scan
    scans : array
        Scans associated used to build the Rois.
    separation : mode to pass to ROI creation.

    Returns
    -------
    Roi

    """
    # build temporal roi arrays, these include scans that must be removed
    # because they are associated with other ms levels.
    first_scan = roi.scan[0]
    last_scan = roi.scan[-1]
    size = last_scan + 1 - first_scan
    mz_tmp = np.ones(size) * np.nan
    spint_tmp = mz_tmp.copy()

    # copy values of the roi to the temporal arrays
    scan_index = np.array(roi.scan) - roi.scan[0]
    mz_tmp[scan_index] = roi.mz
    spint_tmp[scan_index] = roi.sp

    start_ind, end_ind = np.searchsorted(scans,
                                         [first_scan, last_scan + 1])
    scan_tmp = scans[start_ind:end_ind].copy()
    valid_index = scan_tmp - first_scan
    mz_tmp = mz_tmp[valid_index]
    spint_tmp = spint_tmp[valid_index]
    rt_tmp = rt[scan_tmp].copy()

    # temporal sanity check for the roi arrays
    assert rt_tmp.size == mz_tmp.size
    assert rt_tmp.size == spint_tmp.size
    assert rt_tmp.size == scan_tmp.size

    if separation in c.LC_MODES:
        roi = LCRoi(spint_tmp, mz_tmp, rt_tmp, scan_tmp, mode=separation)
    else:
        roi = Roi(spint_tmp, mz_tmp, rt_tmp, scan_tmp, mode=separation)
    return roi


def _accumulate_spectra_centroid(
    ms_data: MSData,
    start_time: float,
    end_time: float,
    subtract_left_time: float,
    subtract_right_time: float,
    ms_level: int
) -> MSSpectrum:
    """
    accumulates a series of consecutive spectra into a single spectrum.

    auxiliary method for accumulate_spectra.

    """
    # don't remove any m/z value when detecting rois
    max_missing = ms_data.get_n_spectra()

    roi = make_roi(
        ms_data,
        max_missing=max_missing,
        min_length=1,
        start_time=subtract_left_time,
        end_time=subtract_right_time,
        ms_level=ms_level
    )

    mz = np.zeros(len(roi))
    spint = mz.copy()

    # set subtract values to negative
    for k, r in enumerate(roi):
        sign = - np.ones(r.time.size)
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
    ms_level: int
) -> MSSpectrum:
    """
    aux method for accumulate_spectra.

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
        ms_level,
        start_time=subtract_left_time,
        end_time=subtract_right_time
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
        ms_level,
        start_time=subtract_left_time,
        end_time=subtract_right_time
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
        is_centroid=False
    )
    return res
