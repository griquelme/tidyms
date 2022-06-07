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
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from . import peaks
from . import _plot_bokeh
from . import _constants as c


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
        instrument: str = c.QTOF,
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
        valid_values = c.MS_INSTRUMENTS
        if value in valid_values:
            self._instrument = value
        else:
            msg = "{} is not a valid instrument. Valid values are: {}."
            raise ValueError(msg.format(value, c.MS_INSTRUMENTS))

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


class Roi:
    """
        m/z traces extracted from raw data. m/z information is stored besides
        time and intensity information.

        Attributes
        ----------
        time : array
            time in each scan.
        spint : array
            intensity in each scan.
        mz : array
            m/z in each scan.
        scan : array
            scan numbers where the ROI is defined.
        mode : {"uplc", "hplc"}
            Analytical platform used separation. Sets default values for peak
            detection.
        features : OptionalList[Feature]]

        """
    def __init__(self, spint: np.ndarray, mz: np.ndarray, time: np.ndarray,
                 scan: np.ndarray, mode: str):
        self.mode = mode
        self.time = time
        self.spint = spint
        self.mz = mz
        self.scan = scan
        self.features = None    # Type: Optional[List[Feature]]

    def plot(
        self,
        figure: Optional[bokeh.plotting.Figure] = None,
        show: bool = True
    ) -> bokeh.plotting.Figure:     # pragma: no cover
        raise NotImplementedError

    def extract_features(self, **kwargs) -> List["Feature"]:
        raise NotImplementedError

    def describe_features(
        self,
        custom_descriptors: Optional[dict] = None,
        filters: Optional[dict] = None
    ) -> List[Dict[str, float]]:
        """
        Computes descriptors for the features detected in the ROI.

        Parameters
        ----------
        custom_descriptors : dict or None, default=None
            A dictionary of strings to callables, used to estimate custom
            descriptors of a feature. The function must have the following
            signature:

            .. code-block:: python

                "estimator_func(roi: Roi, feature: Feature) -> float"

        filters : dict or None, default=None
            A dictionary of descriptor names to a tuple of minimum and maximum
            acceptable values. To use only minimum/maximum values, use None
            (e.g. (None, max_value) in the case of using only maximum). Features
            with descriptors outside those ranges are removed. Filters for
            custom descriptors can also be used.

        Returns
        -------
        features : List[Feature]
            filtered list of features.
        descriptors: List[Dict[str, float]]
            Descriptors for each feature.

        """

        if custom_descriptors is None:
            custom_descriptors = dict()

        if filters is None:
            filters = self.get_default_filters()
        _fill_filter_boundaries(filters)

        valid_features = list()
        descriptor_list = list()      # Type: List[Dict[str, float]]
        for f in self.features:
            f_descriptors = f.get_descriptors(self)
            for descriptor, func in custom_descriptors.items():
                f_descriptors[descriptor] = func(self, f)

            if _has_all_valid_descriptors(f_descriptors, filters):
                valid_features.append(f)
                descriptor_list.append(f_descriptors)
        self.features = valid_features
        return descriptor_list

    def get_default_filters(self) -> Dict[str, float]:
        raise NotImplementedError

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
        if fill_value is None:
            interpolator = interp1d(self.time[~missing], self.spint[~missing])
            self.spint[missing] = interpolator(self.time[missing])
        else:
            self.spint[missing] = fill_value

        if isinstance(self.mz, np.ndarray):
            mz_mean = np.nanmean(self.mz)
            self.mz[missing] = mz_mean


class LCRoi(Roi):
    """
        m/z traces where chromatographic peaks may be found. m/z information
        is stored besides time and intensity information.

        Subclassed from Roi. Used for feature detection in LCMS data.

        Attributes
        ----------
        time : array
            time in each scan.
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

    def __init__(self, spint: np.ndarray, mz: np.ndarray, time: np.ndarray,
                 scan: np.ndarray, mode: str = c.UPLC):
        super(LCRoi, self).__init__(spint, mz, time, scan, mode)
        self.baseline = None
        self.noise = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in c.LC_MODES:
            self._mode = value
        else:
            msg = "{} is not a valid mode. Valid values are: {}."
            raise ValueError(msg.format(value, c.LC_MODES))

    def extract_features(
        self,
        smoothing_strength: Optional[float] = 1.0,
        store_smoothed: bool = False,
        **kwargs
    ) -> List["Peak"]:
        """
        Detect chromatographic peaks.

        Peaks are stored in the `features` attribute.

        Parameters
        ----------
        smoothing_strength : float or None, default=1.0
            Scale of a Gaussian function used to smooth the signal. If None,
            no smoothing is applied.
        store_smoothed : bool, default=True
            If True, replaces the original data with the smoothed version.
        **kwargs :
            Parameters to pass to :py:func:`tidyms.peaks.detect_peaks`.

        Notes
        -----
        Peak detection is done in five steps:

        1. Estimate the noise level.
        2. Apply a gaussian smoothing to the chromatogram.
        3. Estimate the baseline.
        4. Detect peaks in the chromatogram.

        A complete description can be found :ref:`here <feature-extraction>`.

        See Also
        --------
        tidyms.peaks.estimate_noise : noise estimation of 1D signals
        tidyms.peaks.estimate_baseline : baseline estimation of 1D signals
        tidyms.peaks.detect_peaks : peak detection of 1D signals.

        """
        self.fill_nan(0.0)
        noise = peaks.estimate_noise(self.spint)

        if smoothing_strength is None:
            x = self.spint
        else:
            x = gaussian_filter1d(self.spint, smoothing_strength)

        if store_smoothed:
            self.spint = x

        baseline = peaks.estimate_baseline(self.spint, noise)
        start, apex, end = peaks.detect_peaks(
            self.spint, noise, baseline, **kwargs
        )
        features = [Peak(s, a, e) for s, a, e in zip(start, apex, end)]

        self.features = features
        self.baseline = baseline
        self.noise = noise
        return features

    def plot(
        self,
        figure: Optional[bokeh.plotting.Figure] = None,
        show: bool = True
    ) -> bokeh.plotting.Figure:     # pragma: no cover
        """
        Plot the ROI.

        Parameters
        ----------
        figure : bokeh.plotting.Figure or None, default=None
            Figure to add the plot. If None, a new figure is created.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh.plotting.Figure

        """
        if figure is None:
            fig_params = _plot_bokeh.get_chromatogram_figure_params()
            figure = bokeh.plotting.Figure(**fig_params)

        _plot_bokeh.add_line(figure, self.time, self.spint)
        if self.features:
            palette = _plot_bokeh.get_palette()
            palette_cycler = _plot_bokeh.palette_cycler(palette)
            for f, color in zip(self.features, palette_cycler):
                _plot_bokeh.fill_area(
                    figure,
                    self.time,
                    self.spint,
                    f.start,
                    f.end,
                    color,
                )
        _plot_bokeh.set_chromatogram_axis_params(figure)
        if show:
            bokeh.plotting.show(figure)
        return figure

    def get_default_filters(self) -> dict:
        """
        Default filters for peaks detected in LC data.
        """
        if self.mode == c.HPLC:
            filters = {"width": (10, 90), "snr": (5, None)}
        else:   # mode = "uplc"
            filters = {"width": (4, 60), "snr": (5, None)}
        return filters


class Chromatogram(LCRoi):
    """
    Representation of a chromatogram. Manages plotting and peak detection.

    Subclassed from LCRoi.

    Attributes
    ----------
    time : array
        Retention time data.
    spint : array
        Intensity data.
    mode : {"uplc", "hplc"}, default="uplc"
        Analytical platform used for separation. Sets default values for peak
        detection.

    """
    def __init__(self, time: np.ndarray, spint: np.ndarray, mode: str = c.UPLC):
        super(Chromatogram, self).__init__(spint, None, time, None, mode)


class InvalidPeakException(ValueError):
    """
    Exception raised when invalid indices are used in the construction of
    Peak objects.

    """
    pass


class Feature:
    """
    Abstract representation of a feature.

    Attributes
    ----------
    start: int
        index where the peak begins. Must be smaller than `apex`
    end: int
        index where the peak ends. Start and end used as slices defines the
        peak region.

    """

    def __init__(self, start: int, end: int):

        try:
            assert start < end
        except AssertionError:
            msg = "start must be lower than end"
            raise ValueError(msg)

        self.start = start
        self.end = end

    def __repr__(self):
        str_repr = "{}(start={}, end={})"
        name = self.__class__.__name__
        str_repr = str_repr.format(name, self.start, self.end)
        return str_repr

    def plot(
        self,
        roi: Roi,
        figure: bokeh.plotting.Figure,
        color: str,
        **varea_params
    ):
        _plot_bokeh.fill_area(
            figure, roi.time, roi.spint, self.start, self.end, color,
            **varea_params
        )

    def get_descriptors(self, roi: Roi):
        raise NotImplementedError


class Peak(Feature):
    """
    Representation of a chromatographic peak. Computes peak descriptors.

    Attributes
    ----------
    start: int
        index where the peak begins. Must be smaller than `apex`
    apex: int
        index where the apex of the peak is located. Must be smaller than `end`
    end: int
        index where the peak ends. Start and end used as slices defines the
        peak region.

    """

    def __init__(self, start: int, apex: int, end: int):
        super(Peak, self).__init__(start, end)
        try:
            assert self.start < apex
            assert apex < self.end
        except AssertionError:
            msg = "start must be lower than loc and loc must be lower than end"
            raise InvalidPeakException(msg)
        self.apex = apex

    def __repr__(self):
        str_repr = "{}(start={}, apex={}, end={})"
        name = self.__class__.__name__
        str_repr = str_repr.format(name, self.start, self.apex, self.end)
        return str_repr

    def get_rt(self, roi: LCRoi) -> float:
        """
        Finds the peak location in the ROI rt, using spint as weights.

        Parameters
        ----------
        roi: Roi
            ROI where the peak was detected

        Returns
        -------
        loc : float

        """
        weights = roi.spint[self.start:self.end]
        weights[weights < 0] = 0
        loc = np.abs(np.average(roi.time[self.start:self.end], weights=weights))
        return loc

    def get_height(self, roi: LCRoi) -> float:
        """
        Computes the height of the peak, defined as the difference between the
        value of intensity in the ROI and the baseline at the peak apex.

        Parameters
        ----------
        roi : LCRoi
            ROI where the peak was detected

        Returns
        -------
        height : non-negative number. If the baseline estimation is greater
        than y, the height is set to zero.

        """
        height = roi.spint[self.apex] - roi.baseline[self.apex]
        return max(0.0, height)

    def get_area(self, roi: LCRoi) -> float:
        """
        Computes the area in the region defined by the peak.

        If the baseline area is greater than the peak area, the area is set
        to zero.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        area : positive number.

        """
        baseline_corrected = (roi.spint[self.start:self.end] -
                              roi.baseline[self.start:self.end])
        area = trapz(baseline_corrected, roi.time[self.start:self.end])
        return max(0.0, area)

    def get_width(self, roi: LCRoi) -> float:
        """
        Computes the peak width, defined as the region where the 95 % of the
        total peak area is distributed.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        width : positive number.

        """
        height = (
                roi.spint[self.start:self.end] -
                roi.baseline[self.start:self.end]
        )
        area = cumtrapz(height, roi.time[self.start:self.end])
        if area[-1] > 0:
            relative_area = area / area[-1]
            percentile = [0.025, 0.975]
            start, end = self.start + np.searchsorted(relative_area, percentile)
            width = roi.time[end] - roi.time[start]
        else:
            width = 0.0
        return max(0.0, width)

    def get_extension(self, roi: LCRoi) -> float:
        """
        Computes the peak extension, defined as the length of the peak region.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        extension : positive number

        """
        return roi.time[self.end] - roi.time[self.start]

    def get_snr(self, roi: LCRoi) -> float:
        """
        Computes the peak signal-to-noise ratio, defined as the quotient
        between the peak height and the noise level at the apex.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        snr : float

        """

        peak_noise = roi.noise[self.apex]
        if np.isclose(peak_noise, 0):
            snr = np.inf
        else:
            snr = self.get_height(roi) / peak_noise
        return snr

    def get_mean_mz(self, roi: LCRoi) -> float:
        """
        Computes the weighted average m/z of the peak.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        mz_mean : float

        """
        if roi.mz is None:
            mz_mean = None
        else:
            weights = roi.spint[self.start:self.end]
            weights[weights < 0] = 0
            mz_mean = np.average(roi.mz[self.start:self.end], weights=weights)
            mz_mean = max(0.0, mz_mean)
        return mz_mean

    def get_mz_std(self, roi: LCRoi) -> float:
        """
        Computes the standard deviation of the m/z in the peak

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        mz_std : float

        """
        if roi.mz is None:
            mz_std = None
        else:
            mz_std = roi.mz[self.start:self.end].std()
        return mz_std

    def get_descriptors(self, roi: LCRoi) -> Dict[str, float]:
        """
        Computes peak height, area, location, width and SNR.

        Parameters
        ----------
        roi : Roi
            ROI where the peak was detected

        Returns
        -------
        descriptors: dict
            A mapping of descriptor names to descriptor values.
        """
        descriptors = {
            c.HEIGHT: self.get_height(roi),
            c.AREA: self.get_area(roi),
            c.RT: self.get_rt(roi),
            c.WIDTH: self.get_width(roi),
            c.SNR: self.get_snr(roi),
            c.MZ: self.get_mean_mz(roi),
            c.MZ_STD: self.get_mz_std(roi)
        }
        return descriptors


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
    if instrument == c.QTOF:
        md = 0.01
    else:   # orbitrap
        md = 0.005
    params["min_distance"] = md
    return params


def _fill_filter_boundaries(filter_dict: Dict[str, Tuple]):
    """
    Replaces None in the filter boundaries to perform comparisons.

    aux function of get_peak_descriptors
    """
    for k in filter_dict:
        lb, ub = filter_dict[k]
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf
        filter_dict[k] = (lb, ub)


def _has_all_valid_descriptors(peak_descriptors: Dict[str, float],
                               filters: Dict[str, Tuple[float, float]]) -> bool:
    """
    Check that the descriptors of a peak are in a valid range.

    aux function of get_peak_descriptors.

    Parameters
    ----------
    peak_descriptors : dict
        mapping of descriptor names to descriptor values.
    filters : dict
        Dictionary from descriptors names to minimum and maximum acceptable
        values.

    Returns
    -------
    is_valid : bool
        True if all descriptors are inside the valid ranges.

    """
    res = True
    for descriptor, (lb, ub) in filters.items():
        d = peak_descriptors[descriptor]
        is_valid = (d >= lb) and (d <= ub)
        if not is_valid:
            res = False
            break
    return res
