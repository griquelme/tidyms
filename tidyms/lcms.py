"""
Functions and objects for working with LC-MS data read from pyopenms.

Objects
-------
Chromatogram
MSSpectrum
Roi

Functions
---------
make_chromatograms
make_roi
accumulate_spectra_profile
accumulate_spectra_centroid
get_lc_filter_peak_params
get_roi_params
get_find_centroid_params

"""

import bokeh.plotting
import numpy as np
from collections import namedtuple, deque
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, Union, List, Callable
from scipy.interpolate import interp1d
from . import peaks
from . import _plot_bokeh
from .utils import find_closest


class MSSpectrum:
    """
    Representation of a Mass Spectrum. Manages conversion to
    centroid and plotting of data.

    Attributes
    ----------
    mz : array
        m/z data
    spint : array
        Intensity data
    time : float or None
        Time at which the spectrum was acquired
    ms_level : int
        MS level of the scan
    polarity : int or None
        Polarity used to acquire the data.
    instrument : {"qtof", "orbitrap"}, default="qtof"
        MS instrument type. Used to set default values in methods.
    is_centroid : bool
        True if the data is in centroid mode.

    """

    def __init__(
        self,
        mz: np.ndarray,
        spint: np.ndarray,
        time: Optional[float] = None,
        ms_level: int = 1,
        polarity: Optional[int] = None,
        instrument: str = "qtof",
        is_centroid: bool = True
    ):
        self.mz = mz
        self.spint = spint
        self.time = time
        self.ms_level = ms_level
        self.polarity = polarity
        self.instrument = instrument
        self.is_centroid = is_centroid

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        valid_values = ["qtof", "orbitrap"]
        if value in valid_values:
            self._instrument = value
        else:
            msg = "instrument must be one of `qtof` or `orbitrap`. Got {}"
            raise ValueError(msg.format(value))

    def find_centroids(
        self,
        min_snr: float = 10.0,
        min_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Find centroids in the spectrum.

        Parameters
        ----------
        min_snr : positive number, default=10.0
            Minimum signal-to-noise ratio of the peaks.
        min_distance : positive number or None, default=None
            Minimum distance between consecutive peaks. If ``None``, the value
            is set to 0.01 if ``self.instrument`` is ``"qtof"`` or to 0.005 if
            ``self.instrument`` is ``"orbitrap"``.

        Raises
        ------
        ValueError: if the spectrum is in centroid mode

        Returns
        -------
        centroid : array
            m/z centroids. If ``self.is_centroid`` is ``True``, returns
            ``self.mz``.
        area : array
            peak area. If ``self.is_centroid`` is ``True``, returns
            ``self.spint``.

        """
        if self.is_centroid:
            centroid, area = self.mz, self.spint
        else:
            params = get_find_centroid_params(self.instrument)
            if min_distance is not None:
                params["min_distance"] = min_distance

            if min_snr is not None:
                params["min_snr"] = min_snr

            centroid, area = peaks.find_centroids(self.mz, self.spint, **params)
        return centroid, area

    def plot(
        self,
        fig_params: Optional[dict] = None,
        line_params: Optional[dict] = None,
        show: bool = True
    ) -> bokeh.plotting.Figure:     # pragma: no cover
        """
        Plot the spectrum using Bokeh.

        Parameters
        ----------
        fig_params : dict or None, default=None
            key-value parameters to pass to ``bokeh.plotting.figure``.
        line_params : dict, or None, default=None
            key-value parameters to pass to ``bokeh.plotting.Figure.line``.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.
        Returns
        -------
        bokeh.plotting.Figure

        """
        default_fig_params = _plot_bokeh.get_spectrum_figure_params()
        if fig_params:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params
        else:
            fig_params = default_fig_params
        fig = bokeh.plotting.Figure(**fig_params)

        if self.is_centroid:
            plotter = _plot_bokeh.add_stems
        else:
            plotter = _plot_bokeh.add_line
        plotter(fig, self.mz, self.spint, line_params=line_params)
        _plot_bokeh.set_ms_spectrum_axis_params(fig)
        if show:
            bokeh.plotting.show(fig)
        return fig


class Chromatogram:
    """
    Representation of a chromatogram. Manages plotting and peak detection.

    Attributes
    ----------
    rt : array
        Retention time data.
    spint : array
        Intensity data.
    mode : {"uplc", "hplc"}, default="uplc"
        Analytical platform used for separation. Sets default values for peak
        detection.

    """

    def __init__(self, rt: np.ndarray, spint: np.ndarray, mode: str = "uplc"):
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
            msg = "mode must be one of {}. Got {}"
            raise ValueError(msg.format(valid_values, value))

    def find_peaks(
        self,
        smoothing_strength: Optional[float] = 1.0,
        descriptors: Optional[dict] = None,
        filters: Optional[dict] = None,
        find_peaks_params: Optional[dict] = None,
        return_signal_estimators: bool = False
    ) -> List[dict]:
        """
        Find peaks and compute peak descriptors.

        Stores the detected peaks in the `peaks` attribute and returns the peaks
        descriptors.

        Parameters
        ----------
        descriptors : dict or None, default=None
            descriptors to pass to :py:func:`tidyms.peaks.get_peak_descriptors`.
        find_peaks_params : dict or None, default=None
            parameters to pass to :py:func:`tidyms.peaks.detect_peaks`
        smoothing_strength: positive number or None, default=1.0
            Width of a gaussian window used to smooth the signal. If ``None``,
            no smoothing is applied.
        filters : dict or None, default=None
            filters to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
        return_signal_estimators : bool, default=False
            If True, returns a dictionary with the noise, baseline and the
            smoothed signal

        Returns
        -------
        params : List[dict]
            List of peak descriptors
        estimators : dict
            a dictionary with the noise, baseline and smoothed signal used
            inside the function. Only if `return_signal_estimators` is ``True``.

        Notes
        -----
        Peak detection is done in five steps:

        1. Estimate the noise level.
        2. Apply a gaussian smoothing to the chromatogram.
        3. Estimate the baseline.
        4. Detect peaks in the chromatogram.
        5. Compute peak descriptors and filter peaks.

        See Also
        --------
        peaks.estimate_noise : noise estimation of 1D signals
        peaks.estimate_baseline : baseline estimation of 1D signals
        peaks.detect_peaks : peak detection of 1D signals.
        peaks.get_peak_descriptors: computes peak descriptors.
        lcms.get_lc_filter_peaks_params : default value for filters

        """
        if find_peaks_params is None:
            find_peaks_params = dict()

        if filters is None:
            filters = get_lc_filter_peak_params(self.mode)

        noise = peaks.estimate_noise(self.spint)

        if smoothing_strength is None:
            x = self.spint
        else:
            x = gaussian_filter1d(self.spint, smoothing_strength)

        baseline = peaks.estimate_baseline(x, noise)
        peak_list = peaks.detect_peaks(x, noise, baseline, **find_peaks_params)

        peak_list, peak_descriptors = peaks.get_peak_descriptors(
            self.rt,
            self.spint,
            noise,
            baseline,
            peak_list,
            descriptors=descriptors,
            filters=filters
        )
        self.peaks = peak_list

        if return_signal_estimators:
            estimators = {"smoothed": x, "noise": noise, "baseline": baseline}
            res = peak_descriptors, estimators
        else:
            res = peak_descriptors
        return res

    def plot(
        self,
        fig_params: Optional[dict] = None,
        line_params: Optional[dict] = None,
        fill_params: Optional[dict] = None,
        palette: Optional[str] = None,
        show: bool = True
    ) -> bokeh.plotting.Figure:     # pragma: no cover
        """
        Plot the chromatogram.

        Parameters
        ----------
        fig_params : dict or None, default=None
            key-value parameters to pass to ``bokeh.plotting.figure``.
        line_params : dict or None, default=None
            key-value parameters to pass to ``bokeh.plotting.Figure.line``.
            Used to draw the chromatogram line.
        fill_params : dict or None, default=None
            key-value parameters to pass to ``bokeh.plotting.Figure.varea``.
            Used to fill the area under the peaks.
        palette : str or None, default=None
            Color palette used to fill the area under the peaks.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh.plotting.Figure

        """
        default_fig_params = _plot_bokeh.get_chromatogram_figure_params()
        if fig_params:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params
        else:
            fig_params = default_fig_params
        fig = bokeh.plotting.Figure(**fig_params)

        _plot_bokeh.add_line(fig, self.rt, self.spint, line_params=line_params)
        if self.peaks:
            _plot_bokeh.fill_peaks(
                fig,
                self.rt,
                self.spint,
                self.peaks,
                palette,
                varea_params=fill_params
            )
        _plot_bokeh.set_chromatogram_axis_params(fig)
        if show:
            bokeh.plotting.show(fig)
        return fig


class Roi(Chromatogram):
    """
    m/z traces where a chromatographic peak may be found. m/z information
    is stored besides time and intensity information.

    Subclassed from Chromatogram. Used for feature detection in LCMS data.

    Attributes
    ----------
    rt : array
        retention time in each scan.
    spint : array
        intensity in each scan.
    mz : array
        m/z in each scan.
    scan : array
        scan numbers where the ROI is defined.
    mode : {"uplc", "hplc"}
        Analytical platform used separation. Sets default values for peak
        detection.

    """
    def __init__(self, spint: np.ndarray, mz: np.ndarray, rt: np.ndarray,
                 scan: np.ndarray, mode: str = "uplc"):
        super(Roi, self).__init__(rt, spint, mode=mode)
        self.mz = mz
        self.scan = scan

    def fill_nan(self, fill_value: Optional[float] = None):
        """
        Fill missing intensity values using linear interpolation.

        Parameters
        ----------
        fill_value : float or None
            Missing intensity values are replaced with this value. If None,
            values are filled using linear interpolation.

        """

        # if the first or last values are missing, assign an intensity value
        # of zero. This prevents errors in the interpolation and makes peak
        # picking work better.

        if np.isnan(self.spint[0]):
            self.spint[0] = 0
        if np.isnan(self.spint[-1]):
            self.spint[-1] = 0

        missing = np.isnan(self.spint)
        mz_mean = np.nanmean(self.mz)
        if fill_value is None:
            interpolator = interp1d(self.rt[~missing], self.spint[~missing])
            self.mz[missing] = mz_mean
            self.spint[missing] = interpolator(self.rt[missing])
        else:
            self.mz[missing] = mz_mean
            self.spint[missing] = fill_value

    def get_peaks_mz(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the weighted mean of the m/z for each peak and the m/z
        standard deviation.

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


def get_find_centroid_params(instrument: str) -> dict:
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
    else:   # orbitrap
        md = 0.005
    params["min_distance"] = md
    return params


class RoiMaker:
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
    # deque extendleft from right to left
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

    roi = Roi(spint_tmp, mz_tmp, rt_tmp, scan_tmp, mode=separation)
    return roi
