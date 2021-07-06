"""
Functions and objects for working with LC-MS data

Objects
-------
Chromatogram
MSSpectrum
Roi

"""

import bokeh.plotting
import numpy as np
import pyopenms
from collections import namedtuple
from scipy.interpolate import interp1d
from typing import Optional, Iterable, Tuple, Union, List, Callable
from . import peaks
from . import validation
from . import _plot_bokeh
from .utils import find_closest

ms_experiment_type = Union[pyopenms.MSExperiment, pyopenms.OnDiscMSExperiment]


class MSSpectrum:
    """
    Representation of a Mass Spectrum in profile mode. Manages conversion to
    centroids and plotting of data.

    Attributes
    ----------
    mz : array of m/z values
    spint : array of intensity values.
    instrument : str
        MS instrument type. Used to set default values in peak picking.

    """
    def __init__(self, mz: np.ndarray, spint: np.ndarray,
                 instrument: Optional[str] = None):
        """
        Constructor of the MSSpectrum.

        Parameters
        ----------
        mz: array
            m/z values.
        spint: array
            intensity values.

        """
        self.mz = mz
        self.spint = spint
        self.peaks = None

        if instrument is None:
            instrument = "qtof"
        self.instrument = instrument

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        valid_values = ["qtof", "orbitrap"]
        if value in valid_values:
            self._instrument = value
        else:
            msg = "instrument must be one of {}".format(valid_values)
            raise ValueError(msg)

    def find_centroids(self, min_snr: Optional[float] = None,
                       min_distance: Optional[float] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Find centroids in the spectrum.

        Parameters
        ----------
        min_snr : positive number, optional
            Minimum signal to noise ratio of the peaks. Overwrites values
            set by mode. Default value is 10
        min_distance : positive number, optional
            Minimum distance between consecutive peaks. If None, sets the value
            to 0.01 if the `mode` attribute is qtof. If the `mode` is orbitrap,
            sets the value to 0.005

        Returns
        -------
        centroids : array of peak centroids
        area : array of peak area

        """
        params = get_find_centroid_params(self.instrument)
        if min_distance is not None:
            params["min_distance"] = min_distance

        if min_snr is not None:
            params["min_snr"] = min_snr

        centroids, area = peaks.find_centroids(self.mz, self.spint, **params)
        return centroids, area

    def plot(self, draw: bool = True, fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        Plot the spectrum.

        Parameters
        ----------
        draw : bool, optional
            if True run bokeh show function.
        fig_params : dict
            key-value parameters to pass into bokeh figure function.
        line_params : dict
            key-value parameters to pass into bokeh line function.

        Returns
        -------
        bokeh Figure
        """
        return _plot_bokeh.plot_ms_spectrum(self.mz, self.spint, draw=draw,
                                            fig_params=fig_params,
                                            line_params=line_params)


class Chromatogram:
    """
    Representation of a chromatogram. Manages plotting and peak picking.

    Attributes
    ----------
    rt : array
        retention time in each scan.
    spint : array
        intensity in each scan.
    mode : str
        used to set default parameter for peak picking.

    """

    def __init__(self, rt: np.ndarray, spint: np.ndarray,
                 mode: Optional[str] = None):
        """
        Constructor of the Chromatogram.

        Parameters
        ----------
        spint : array of non negative numbers.
            Intensity values of each scan
        rt : array of positive numbers.
            Retention time values.
        mode : {"uplc", "hplc"}, optional
            used to set default parameters in peak picking. If None, `mode` is
            set to uplc.

        """
        if mode is None:
            mode = "uplc"
        self.mode = mode

        self.rt = rt
        self.spint = spint
        self.peaks = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        valid_values = ["uplc", "hplc"]
        if value in valid_values:
            self._mode = value
        else:
            msg = "mode must be one of {}".format(valid_values)
            raise ValueError(msg)

    def find_peaks(self, detect_peaks_params: Optional[dict] = None,
                   descriptors: Optional[dict] = None,
                   filters: Optional[dict] = None) -> List[dict]:
        """
        Find peaks and compute peak descriptors.

        Stores the found peaks in the `peaks` attribute and returns the peaks
        descriptors.

        Parameters
        ----------
        detect_peaks_params : dict, optional
            parameters to pass to :py:func:`tidyms.peaks.detect_peaks`
        descriptors : dict, optional
            descriptors to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
        filters : dict, optional
            filters to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
        Returns
        -------
        params : List[dict]
            List of peak descriptors

        See Also
        --------
        peaks.detect_peaks : peak detection using the CWT algorithm.
        peaks.get_peak_descriptors: computes peak descriptors.
        lcms.get_lc_detect_peaks_params : default value for detect_peak_params
        lcms.get_lc_filter_peaks_params : default value for filters

        """
        if detect_peaks_params is None:
            detect_peaks_params = get_lc_detect_peak_params()

        if descriptors is None:
            descriptors = None

        if filters is None:
            filters = get_lc_filter_peak_params(self.mode)

        peak_list, noise, baseline = peaks.detect_peaks(self.spint,
                                                        **detect_peaks_params)
        peak_list, peak_descriptors = \
            peaks.get_peak_descriptors(self.rt, self.spint, noise, baseline,
                                       peak_list, descriptors=descriptors,
                                       filters=filters)
        self.peaks = peak_list
        return peak_descriptors

    def plot(self, draw: bool = True, fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        Plot the chromatogram.

        Parameters
        ----------
        draw : bool, optional
            if True run bokeh show function.
        fig_params : dict
            key-value parameters to pass into bokeh figure function.
        line_params : dict
            key-value parameters to pass into bokeh line function.

        Returns
        -------
        bokeh Figure

        """
        return _plot_bokeh.plot_chromatogram(self.rt, self.spint, self.peaks,
                                             draw=draw, fig_params=fig_params,
                                             line_params=line_params)


class Roi(Chromatogram):
    """
    mz traces where a chromatographic peak may be found.

    Subclassed from Chromatogram. Used for feature detection in LCMS data.

    Attributes
    ----------
    rt : array
        retention time in each scan.
    spint : array
        intensity in each scan.
    mz : array
        m/z in each scan.
    first_scan : int
        first scan in the raw data where the ROI was detected.

    """
    def __init__(self, spint: np.ndarray, mz: np.ndarray, rt: np.ndarray,
                 first_scan: int, mode: Optional[str] = None):
        super(Roi, self).__init__(rt, spint, mode=mode)
        self.mz = mz
        self.first_scan = first_scan

    def extend(self, rt: np.array, n: int):
        """
        adds a dummy value at the beginning and at the end of the ROI. This
        results in better peak picking results when low intensity peaks are
        cropped.

        """
        max_int = np.nanmax(self.spint)
        mz_fill = np.nanmean(self.mz)
        spint_fill = np.nanmin(self.spint)
        roi_rt = list()
        roi_mz = list()
        roi_spint = list()
        extend_roi_beginning = ((np.isnan(self.spint[0]) or
                                 (self.spint[0] < (0.75 * max_int))))
        extend_roi_end = (((np.isnan(self.spint[-1])) or
                          (self.spint[-1] < (0.75 * max_int))))

        if extend_roi_beginning:
            n_beginning = min(n, self.first_scan)
            roi_rt.append(rt[self.first_scan - n_beginning:self.first_scan])
            roi_mz.append([mz_fill] * n_beginning)
            roi_spint.append([spint_fill] * n_beginning)
            # print(roi_rt)

        roi_rt.append(self.rt)
        roi_spint.append(self.spint)
        roi_mz.append(self.mz)

        if extend_roi_end:
            n_end = min(n, rt.size - self.first_scan - self.rt.size)
            n_roi_end = self.first_scan + self.rt.size
            roi_rt.append(rt[n_roi_end:n_roi_end + n_end])
            roi_mz.append([mz_fill] * n_end)
            roi_spint.append([spint_fill] * n_end)

        self.rt = np.hstack(roi_rt)
        self.spint = np.hstack(roi_spint)
        self.mz = np.hstack(roi_mz)

    def fill_nan(self):
        """
        fill missing intensity values using linear interpolation.

        """

        # if the first or last values are missing, assign an intensity value
        # of zero. This prevents errors in the interpolation and makes peak
        # picking to work better.
        if np.isnan(self.spint[0]):
            self.spint[0] = 0
        if np.isnan(self.spint[-1]):
            self.spint[-1] = 0

        missing = np.isnan(self.spint)
        interpolator = interp1d(self.rt[~missing], self.spint[~missing])
        mz_mean = np.nanmean(self.mz)
        self.mz[missing] = mz_mean
        self.spint[missing] = interpolator(self.rt[missing])

    def get_peaks_mz(self):
        """
        Computes the weighted mean of the m/z for each peak and the m/z
        standard deviation
        Returns
        -------
        mean_mz : array
        mz_std : array

        """
        mz_std = np.zeros(len(self.peaks))
        mz_mean = np.zeros(len(self.peaks))
        for k, peak in enumerate(self.peaks):
            mz_std[k] = self.mz[peak.start:peak.end].std()
            mz_mean[k] = peak.get_loc(self.mz, self.spint)
        return mz_mean, mz_std


def make_chromatograms(ms_experiment: ms_experiment_type, mz: Iterable[float],
                       window: float = 0.005, start: Optional[int] = None,
                       end: Optional[int] = None,
                       accumulator: str = "sum",
                       chromatogram_mode: Optional[str] = None
                       ) -> List[Chromatogram]:
    """
    Computes extracted ion chromatograms for a list of m/z values.

    Parameters
    ----------
    ms_experiment : MSExp or OnDiskMSExp.
    mz : iterable[float]
        mz values used to build the EICs.
    start : int, optional
        first scan to build the chromatograms
    end : int, optional
        last scan to build the chromatograms. The scan with `number` end is not
        included in the chromatograms.
    window : positive number.
               Tolerance to build the EICs.
    accumulator : {"sum", "mean"}
        "mean" divides the intensity in the EIC using the number of points in
        the window.
    chromatogram_mode : {"uplc", "hplc"}, optional
        Mode used to create chromatograms
    Returns
    -------
    chromatograms : List of Chromatograms

    """
    nsp = ms_experiment.getNrSpectra()

    if not isinstance(mz, np.ndarray):
        mz = np.array(mz)

    if start is None:
        start = 0

    if end is None:
        end = nsp

    # validate params
    params = {"start": start, "end": end, "window": window, "mz": mz,
              "accumulator": accumulator}
    validation.validate_make_chromatograms_params(nsp, params)

    # mz_intervals has this shape to be compatible with reduce at
    mz_intervals = (np.vstack((mz - window, mz + window))
                    .T.reshape(mz.size * 2))

    eic = np.zeros((mz.size, end - start))
    rt = np.zeros(end - start)
    for ksp in range(start, end):
        # find rt, mz and intensity values of the current scan
        sp = ms_experiment.getSpectrum(ksp)
        rt[ksp - start] = sp.getRT()
        mz_sp, int_sp = sp.get_peaks()
        ind_sp = np.searchsorted(mz_sp, mz_intervals)

        # check if the slices aren't empty
        has_mz = (ind_sp[1::2] - ind_sp[::2]) > 0
        # elements added at the end of mz_sp raise IndexError
        ind_sp[ind_sp >= int_sp.size] = int_sp.size - 1
        # this adds the values between two consecutive indices
        tmp_eic = np.where(has_mz, np.add.reduceat(int_sp, ind_sp)[::2], 0)
        if accumulator == "mean":
            norm = ind_sp[1::2] - ind_sp[::2]
            norm[norm == 0] = 1
            tmp_eic = tmp_eic / norm
        eic[:, ksp - start] = tmp_eic

    chromatograms = list()
    for row in eic:
        chromatogram = Chromatogram(rt, row, mode=chromatogram_mode)
        chromatograms.append(chromatogram)
    return chromatograms


def make_roi(ms_experiment: ms_experiment_type, tolerance: float,
             max_missing: int, min_length: int, min_intensity: float,
             multiple_match: str, targeted_mz: Optional[np.ndarray] = None,
             start: Optional[int] = None, end: Optional[int] = None,
             mz_reduce: Union[str, Callable] = "mean",
             sp_reduce: Union[str, Callable] = "sum",
             mode: Optional[str] = None
             ) -> List[Roi]:
    """
    Make Region of interest from MS data in centroid mode.
    Used by MSData to as the first step of the centWave algorithm.

    Parameters
    ----------
    ms_experiment: pyopenms.MSExperiment
    max_missing : int
        maximum number of consecutive missing values. when a row surpass this
        number the roi is considered as finished and is added to the roi list if
        it meets the length and intensity criteria.
    min_length : int
        The minimum length of a roi to be considered valid.
    min_intensity : float
        Minimum intensity in a roi to be considered valid.
    tolerance : float
        mz tolerance to connect values across scans
    start : int, optional
        First scan to analyze. If None starts at scan 0
    end : int, optional
        Last scan to analyze. If None, uses the last scan number.
    multiple_match : {"closest", "reduce"}
        How to match peaks when there is more than one match. If mode is
        `closest`, then the closest peak is assigned as a match and the
        others are assigned to no match. If mode is `reduce`, then unique
        mz and intensity values are generated using the reduce function in
        `mz_reduce` and `sp_reduce` respectively.
    mz_reduce : "mean" or Callable
        function used to reduce mz values. Can be a function accepting
        numpy arrays and returning numbers. Only used when `multiple_match`
        is reduce. See the following prototype:

        .. code-block:: python

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

    sp_reduce : {"mean", "sum"} or Callable
        function used to reduce intensity values. Can be a function accepting
        numpy arrays and returning numbers. Only used when `multiple_match`
        is reduce. To use custom functions see the prototype shown on
        `mz_reduce`.
    targeted_mz : numpy.ndarray, optional
        if a list of mz is provided, roi are searched only using this list.

    mode : str, optional
        mode used to create Roi objects.

    Returns
    -------
    roi: list[Roi]

    Notes
    -----
    To create a ROI, m/z values in consecutive scans are connected if they are
    within the tolerance`. If there's more than one possible m/z value to
    connect in the next scan, two different strategies are available, using the
    `multiple_match` parameter: If "closest" is used, then m/z values are
    matched to the closest ones, and the others are used to create new ROI. If
    "reduce" is used, then all values within the tolerance are combined. m/z and
    intensity values are combined using the `mz_reduce`  and `sp_reduce`
    parameters respectively. If no matching value has be found in a scan, a NaN
    is added to the ROI. If no matching values are found in `max_missing`
    consecutive scans the ROI is flagged as finished. In this stage, two
    checks are made before the ROI is considered valid:

    1.  The number of non missing values must be higher than `min_length`.
    2.  The maximum intensity value in the ROI must be higher than
        `min_intensity`.

    If the two conditions are meet, the ROI is added to the list of valid ROI.

    References
    ----------
    .. [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504

    """
    if start is None:
        start = 0

    if end is None:
        end = ms_experiment.getNrSpectra()

    if targeted_mz is None:
        mz_seed, _ = ms_experiment.getSpectrum(start).get_peaks()
        targeted = False
    else:
        mz_seed = targeted_mz
        targeted = True

    size = end - start
    rt = np.zeros(size)
    processor = _RoiMaker(mz_seed, max_missing=max_missing,
                          min_length=min_length,
                          min_intensity=min_intensity, tolerance=tolerance,
                          multiple_match=multiple_match,
                          mz_reduce=mz_reduce, sp_reduce=sp_reduce,
                          mode=mode)
    for k_scan in range(start, end):
        sp = ms_experiment.getSpectrum(k_scan)
        rt[k_scan - start] = sp.getRT()
        mz, spint = sp.get_peaks()
        processor.add(mz, spint, targeted=targeted)
        processor.append_to_roi(rt, targeted=targeted)

    # add roi not completed during last scan
    processor.flag_as_completed()
    processor.append_to_roi(rt)
    return processor.roi


def accumulate_spectra_profile(ms_experiment: ms_experiment_type,
                               start: int, end: int,
                               subtract_left: Optional[int] = None,
                               subtract_right: Optional[int] = None,
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    accumulates a spectra into a single spectrum.

    Parameters
    ----------
    ms_experiment : pyopenms.MSExperiment, pyopenms.OnDiskMSExperiment
    start : int
        start slice for scan accumulation
    end : int
        end slice for scan accumulation.
    subtract_left : int, optional
        Scans between `subtract_left` and `start` are subtracted from the
        accumulated spectrum.
    subtract_right : int, optional
        Scans between `subtract_right` and `end` are subtracted from the
        accumulated spectrum.

    Returns
    -------
    accumulated_mz : array of m/z values
    accumulated_int : array of cumulative intensities.

    """
    if subtract_left is None:
        subtract_left = start

    if subtract_right is None:
        subtract_right = end

    # parameter validation
    params = {"start": start, "end": end, "subtract_left": subtract_left,
              "subtract_right": subtract_right}
    n_sp = ms_experiment.getNrSpectra()
    validation.validate_accumulate_spectra_params(n_sp, params)

    # creates a common mz reference value for the scans
    mz, _ = ms_experiment.getSpectrum(start).get_peaks()
    accumulated_mz = _get_uniform_mz(mz)
    accumulated_sp = np.zeros_like(accumulated_mz)

    # interpolates each scan to the reference. Removes values outside the
    # min and max of the reference.
    for scan in range(subtract_left, subtract_right):
        mz_scan, int_scan = ms_experiment.getSpectrum(scan).get_peaks()
        mz_min, mz_max = mz_scan.min(), mz_scan.max()
        min_ind, max_ind = np.searchsorted(accumulated_mz, [mz_min, mz_max])
        interpolator = interp1d(mz_scan, int_scan, kind="linear")
        tmp_sp = interpolator(accumulated_mz[min_ind:max_ind])
        # accumulate scans
        if (scan < start) or (scan > end):
            accumulated_sp[min_ind:max_ind] -= tmp_sp
        else:
            accumulated_sp[min_ind:max_ind] += tmp_sp

    is_positive_sp = accumulated_sp > 0
    accumulated_mz = accumulated_mz[is_positive_sp]
    accumulated_sp = accumulated_sp[is_positive_sp]
    return accumulated_mz, accumulated_sp


def accumulate_spectra_centroid(ms_experiment: ms_experiment_type,
                                start: int, end: int,
                                subtract_left: Optional[int] = None,
                                subtract_right: Optional[int] = None,
                                tolerance: Optional[float] = None,
                                ):
    """
    accumulates a series of consecutive spectra into a single spectrum.

    Parameters
    ----------
    ms_experiment : pyopenms.MSExperiment, pyopenms.OnDiskMSExperiment
    start : int
        start slice for scan accumulation
    end : int
        end slice for scan accumulation.
    tolerance : float
        m/z tolerance to connect peaks across scans
    subtract_left : int, optional
        Scans between `subtract_left` and `start` are subtracted from the
        accumulated spectrum.
    subtract_right : int, optional
        Scans between `subtract_right` and `end` are subtracted from the
        accumulated spectrum.

    Returns
    -------
    accumulated_mz : array of m/z values
    accumulated_int : array of cumulative intensities.

    """
    n_spectra = ms_experiment.getNrSpectra()

    if subtract_left is None:
        subtract_left = start

    if subtract_right is None:
        subtract_right = end

    max_missing = subtract_right - subtract_left + 1
    params = {"start": start, "end": end, "subtract_left": subtract_left,
              "subtract_right": subtract_right}
    validation.validate_accumulate_spectra_params(n_spectra, params)
    roi = make_roi(ms_experiment, tolerance, max_missing=max_missing,
                   min_length=1, min_intensity=0.0, multiple_match="reduce",
                   start=subtract_left, end=subtract_right, mz_reduce="mean",
                   sp_reduce="sum")
    mask = np.ones(max_missing)
    mask[:start - subtract_left] = -1
    mask[end - subtract_left:] = -1
    mz = np.array([(x.mz * mask).mean() for x in roi])
    spint = np.array([(x.spint * mask).sum() for x in roi])
    mz = mz[spint > 0]
    spint = mz[spint > 0]
    return mz, spint


def get_lc_filter_peak_params(lc_mode: str) -> dict:
    """
    Default filters for peaks detected in LC data.

    Parameters
    ----------
    lc_mode : {"hplc", "uplc"}
        HPLC assumes typical experimental conditions for HPLC experiments:
        longer columns with particle size greater than 3 micron. UPLC is for
        data acquired with short columns with particle size lower than 3 micron.

    Returns
    -------
    filters : dict
        filters to pass to :py:func:`tidyms.peaks.get_peak_descriptors`.

    """
    if lc_mode == "hplc":
        filters = {"width": (10, 90), "snr": (5, None)}
    elif lc_mode == "uplc":
        filters = {"width": (4, 60), "snr": (5, None)}
    else:
        msg = "`mode` must be `hplc` or `uplc`"
        raise ValueError(msg)
    return filters


def get_lc_detect_peak_params() -> dict:
    """
    Default values for performing peak detection on LC data.

    Returns
    -------
    params : dict
        keyword arguments to pass to :py:func:`tidyms.peaks.detect_peaks`
    """
    params = {"smoothing_strength": 1.0}
    return params


def get_roi_params(separation: str = "uplc", instrument: str = "qtof"):
    """
    Creates a dictionary with recommended parameters for the make_roi function
    in different use cases.

    Parameters
    ----------
    separation : {"uplc", "hplc"}
        Mode in which the data was acquired. Used to set minimum length of the
        roi and number of missing values.
    instrument : {"qtof", "orbitrap"}
        Type of MS instrument. Used to set the tolerance.

    Returns
    -------
    roi_parameters : dict
    """
    roi_params = {"min_intensity": 500, "multiple_match": "reduce"}

    if separation == "uplc":
        roi_params.update({"max_missing": 1, "min_length": 10})
    elif separation == "hplc":
        roi_params.update({"max_missing": 1, "min_length": 20})
    else:
        msg = "valid `separation` are uplc and hplc"
        raise ValueError(msg)

    if instrument == "qtof":
        roi_params.update({"tolerance": 0.01})
    elif instrument == "orbitrap":
        roi_params.update({"tolerance": 0.005})
    else:
        msg = "valid `instrument` are qtof and orbitrap"
        raise ValueError(msg)

    roi_params["mode"] = separation

    return roi_params


def get_find_centroid_params(instrument: str):
    """
    Set default parameters to find_centroid method using instrument information.

    Parameters
    ----------
    instrument : {"qtof", "orbitrap"}

    Returns
    -------
    params : dict

    """
    params = {"min_snr": 10}
    if instrument == "qtof":
        md = 0.01
    else:
        # valid values for instrument are qtof or orbitrap
        md = 0.005
    params["min_distance"] = md
    return params


_TempRoi = namedtuple("TempRoi", ["mz", "sp", "scan"])


def _make_empty_temp_roi():
    return _TempRoi(mz=list(), sp=list(), scan=list())


class _RoiMaker:
    """
    Helper class used by make_roi to create Roi instances from raw data.

    Attributes
    ----------
    mz_mean: numpy.ndarray
        mean value of mz for a given row in mz_array. Used to add new values
        based on a tolerance. its updated after adding a new column
    n_missing: numpy.ndarray
        number of consecutive missing values. Used to detect finished rois
    roi: list[ROI]
    """

    def __init__(self, mz_seed: np.ndarray, max_missing: int = 1,
                 min_length: int = 5, min_intensity: float = 0,
                 tolerance: float = 0.005, multiple_match: str = "closest",
                 mz_reduce: Union[str, Callable] = "mean",
                 sp_reduce: Union[str, Callable] = "sum",
                 mode: Optional[str] = None):
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
        mz_reduce: str or callable
            function used to reduce mz values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is reduce. See the following prototype:

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

        sp_reduce: str or callable
            function used to reduce spint values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is reduce. To use custom functions see the prototype shown on
            `mz_reduce`.
        mode: str, optional
            Mode used to create ROI.
        """
        if len(mz_seed.shape) != 1:
            msg = "array must be a vector"
            raise ValueError(msg)

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

        self.mz_mean = mz_seed.copy()
        self.roi_index = np.arange(mz_seed.size)
        self.n_missing = np.zeros_like(mz_seed, dtype=int)
        self.max_intensity = np.zeros_like(mz_seed)
        self.length = np.zeros_like(mz_seed, dtype=int)
        self.index = 0
        self.temp_roi_dict = {x: _make_empty_temp_roi() for x in self.roi_index}
        self.roi = list()
        self.min_intensity = min_intensity
        self.max_missing = max_missing
        self.min_length = min_length
        self.tolerance = tolerance
        self.multiple_match = multiple_match
        self.mode = mode

    def add(self, mz: np.ndarray, sp: np.ndarray, targeted: bool = False):
        """
        Adds new mz and spint values to temporal roi.
        """

        # find matching values
        match_index, mz_match, sp_match, mz_no_match, sp_no_match = \
            _match_mz(self.mz_mean, mz, sp, self.tolerance,
                      self.multiple_match, self._mz_reduce, self._spint_reduce)

        for k, k_mz, k_sp in zip(match_index, mz_match, sp_match):
            k_temp_roi = self.temp_roi_dict[self.roi_index[k]]
            k_temp_roi.mz.append(k_mz)
            k_temp_roi.sp.append(k_sp)
            k_temp_roi.scan.append(self.index)

        # update mz_mean and missing values
        updated_mean = ((self.mz_mean[match_index] * self.length[match_index]
                         + mz_match) / (self.length[match_index] + 1))

        self.length[match_index] += 1
        self.n_missing += 1
        self.n_missing[match_index] = 0
        self.max_intensity[match_index] = \
            np.maximum(self.max_intensity[match_index], sp_match)
        if not targeted:
            self.mz_mean[match_index] = updated_mean
            self.extend(mz_no_match, sp_no_match)
        self.index += 1

    def append_to_roi(self, rt: np.ndarray, targeted: bool = False):
        """
        Remove completed ROI. Valid ROI are appended toi roi attribute.
        """

        # check completed rois
        is_completed = self.n_missing > self.max_missing

        # the most common case are short rois that must be discarded
        is_valid_roi = ((self.length >= self.min_length) &
                        (self.max_intensity >= self.min_intensity))

        # add completed roi
        completed_index = np.where(is_completed)[0]
        for ind in completed_index:
            roi_ind = self.roi_index[ind]
            finished_roi = self.temp_roi_dict.pop(roi_ind)
            if is_valid_roi[ind]:
                roi = tmp_roi_to_roi(finished_roi, rt, mode=self.mode)
                roi.extend(rt, 2)
                self.roi.append(roi)
        if targeted:
            self.n_missing[is_completed] = 0
            self.length[is_completed] = 0
            self.max_intensity[is_completed] = 0
            max_roi_ind = self.roi_index.max()
            n_completed = is_completed.sum()
            new_indices = np.arange(max_roi_ind + 1,
                                    max_roi_ind + 1 + n_completed)
            self.roi_index[is_completed] = new_indices
            new_tmp_roi = {k: _make_empty_temp_roi() for k in new_indices}
            self.temp_roi_dict.update(new_tmp_roi)
        else:
            self.mz_mean = self.mz_mean[~is_completed]
            self.n_missing = self.n_missing[~is_completed]
            self.length = self.length[~is_completed]
            self.roi_index = self.roi_index[~is_completed]
            self.max_intensity = self.max_intensity[~is_completed]

    def extend(self, mz: np.ndarray, sp: np.ndarray):
        """adds new mz values to mz_mean"""
        max_index = self.roi_index.max()
        new_indices = np.arange(mz.size) + max_index + 1
        mz_mean_tmp = np.hstack((self.mz_mean, mz))
        roi_index_tmp = np.hstack((self.roi_index, new_indices))
        sorted_index = np.argsort(mz_mean_tmp)
        n_missing_tmp = np.zeros_like(new_indices, dtype=int)
        n_missing_tmp = np.hstack((self.n_missing, n_missing_tmp))
        length_tmp = np.ones_like(new_indices, dtype=int)
        length_tmp = np.hstack((self.length, length_tmp))
        max_int_tmp = np.zeros_like(new_indices, dtype=float)
        max_int_tmp = np.hstack((self.max_intensity, max_int_tmp))

        for k_index, k_mz, k_sp in zip(new_indices, mz, sp):
            new_roi = _TempRoi(mz=[k_mz], sp=[k_sp], scan=[self.index])
            self.temp_roi_dict[k_index] = new_roi
        self.mz_mean = mz_mean_tmp[sorted_index]
        self.roi_index = roi_index_tmp[sorted_index]
        self.n_missing = n_missing_tmp[sorted_index]
        self.length = length_tmp[sorted_index]
        self.max_intensity = max_int_tmp[sorted_index]

    def flag_as_completed(self):
        self.n_missing[:] = self.max_missing + 1


def _compare_max(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """
    returns the element-wise maximum between old and new

    Parameters
    ----------
    old: numpy.ndarray
    new: numpy.ndarray
        can have nan

    Returns
    -------
    numpy.ndarray
    """
    new[np.isnan(new)] = 0
    return np.maximum(old, new)


def _match_mz(mz1: np.ndarray, mz2: np.ndarray, sp2: np.ndarray,
              tolerance: float, mode: str, mz_reduce: Callable,
              sp_reduce: Callable):
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
        Behaviour when more more than one peak in mz2 matches with a given peak
        in mz1. If mode is `closest`, then the closest peak is assigned as a
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
    match_mask = (dmz <= tolerance)
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


def tmp_roi_to_roi(tmp_roi: _TempRoi, rt: np.ndarray,
                   mode: Optional[str] = None) -> Roi:
    first_scan = tmp_roi.scan[0]
    last_scan = tmp_roi.scan[-1]
    size = last_scan + 1 - first_scan
    mz_tmp = np.ones(size) * np.nan
    spint_tmp = mz_tmp.copy()
    tmp_index = np.array(tmp_roi.scan) - tmp_roi.scan[0]
    rt_tmp = rt[first_scan:(last_scan + 1)].copy()
    mz_tmp[tmp_index] = tmp_roi.mz
    spint_tmp[tmp_index] = tmp_roi.sp
    roi = Roi(spint_tmp, mz_tmp, rt_tmp, first_scan, mode=mode)
    return roi


def _get_uniform_mz(mz: np.ndarray):
    """returns a new uniformly sampled m/z array."""
    mz_min = mz.min()
    mz_max = mz.max()
    mz_res = np.diff(mz).min()
    uniform_mz = np.arange(mz_min, mz_max, mz_res)
    return uniform_mz
