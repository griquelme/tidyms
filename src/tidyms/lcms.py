"""
Functions and objects for working with LC-MS data.

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

import bisect
import bokeh.plotting
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import Any, cast, Optional, Sequence, Tuple, Type, TypeVar, Union
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from . import peaks
from . import _plot_bokeh
from . import _constants as c
from .utils import array1d_to_str, str_to_array1d


# Replace with Self when code is updated to Python 3.11
AnyRoi = TypeVar("AnyRoi", bound="Roi")
AnyFeature = TypeVar("AnyFeature", bound="Feature")


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
        is_centroid: bool = True,
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
        self, min_snr: float = 10.0, min_distance: Optional[float] = None
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
        show: bool = True,
    ) -> bokeh.plotting.figure:  # pragma: no cover
        """
        Plot the spectrum using Bokeh.

        Parameters
        ----------
        fig_params : dict or None, default=None
            key-value parameters to pass to ``bokeh.plotting.figure``.
        line_params : dict, or None, default=None
            key-value parameters to pass to ``bokeh.plotting.figure.line``.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh.plotting.figure

        """
        default_fig_params = _plot_bokeh.get_spectrum_figure_params()
        if fig_params:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params
        else:
            fig_params = default_fig_params
        fig = bokeh.plotting.figure(**fig_params)

        if self.is_centroid:
            plotter = _plot_bokeh.add_stems
        else:
            plotter = _plot_bokeh.add_line
        plotter(fig, self.mz, self.spint, line_params=line_params)
        _plot_bokeh.set_ms_spectrum_axis_params(fig)
        if show:
            bokeh.plotting.show(fig)
        return fig


class Roi(ABC):

    """
    Regions of interest extracted from raw MS data.

    """

    def __init__(self, index: int):
        self.index = index
        self.features: Optional[list[Feature]] = None

    def plot(self) -> bokeh.plotting.figure:
        ...

    @abstractmethod
    def extract_features(self, **kwargs) -> list["Feature"]:
        ...

    def get_default_filters(self) -> dict[str, float]:
        raise NotImplementedError

    @classmethod
    def from_string(cls: Type[AnyRoi], s: str) -> AnyRoi:
        """Loads a ROI from a JSON string."""

        d = cls._deserialize(s)
        features = d.pop(c.ROI_FEATURE_LIST)
        roi = cls(**d)
        ft_class = cls._get_feature_type()
        if features is not None:
            roi.features = [ft_class.from_str(x, roi) for x in features]
        return roi

    @staticmethod
    @abstractmethod
    def _deserialize(s: str) -> dict:
        """
        Converts JSON str into a dictionary used to create a ROI instance.

        This method needs to be overwritten in case that the attributes needs
        to be further deserialized before instantiating a ROI.

        See MZTrace as an example.
        """
        ...

    @abstractmethod
    def to_string(self) -> str:
        """Serializes a ROI into a string."""
        ...

    @staticmethod
    @abstractmethod
    def _get_feature_type() -> Type["Feature"]:
        """Feature Class stored under features attribute."""
        ...


class MZTrace(Roi):
    """
    ROI Implementation using MZ traces.

    MZ traces are 1D traces containing time, intensity and m/z associated with
    each scan.

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

    def __init__(
        self,
        time: np.ndarray[Any, np.dtype[np.floating]],
        spint: np.ndarray[Any, np.dtype[np.floating]],
        mz: np.ndarray[Any, np.dtype[np.floating]],
        scan: np.ndarray[Any, np.dtype[np.integer]],
        index: int = 0,
        mode: Optional[str] = None,
        noise: Optional[np.ndarray] = None,
        baseline: Optional[np.ndarray] = None,
    ):
        super().__init__(index)
        self.mode = mode
        self.time = time
        self.spint = spint
        self.mz = mz
        self.scan = scan
        self.features: Optional[list[Feature]] = None
        self.baseline = baseline
        self.noise = noise

    @staticmethod
    def _deserialize(s: str) -> dict[str, Any]:
        d = json.loads(s)
        d[c.SPINT] = str_to_array1d(d[c.SPINT])
        d[c.MZ] = None if d[c.MZ] is None else str_to_array1d(d[c.MZ])
        d[c.TIME] = str_to_array1d(d[c.TIME])
        d[c.SCAN] = None if d[c.SCAN] is None else str_to_array1d(d[c.SCAN])
        d[c.NOISE] = None if d[c.NOISE] is None else str_to_array1d(d[c.NOISE])
        d[c.BASELINE] = None if d[c.BASELINE] is None else str_to_array1d(d[c.BASELINE])
        return d

    def to_string(self) -> str:
        """
        Serializes the LCRoi into a JSON str.

        Returns
        -------
        str

        """
        d = dict()
        d["index"] = self.index
        d[c.MODE] = self.mode
        d[c.TIME] = array1d_to_str(self.time)
        d[c.SPINT] = array1d_to_str(self.spint)
        d[c.SCAN] = None if self.scan is None else array1d_to_str(self.scan)
        d[c.MZ] = None if self.mz is None else array1d_to_str(self.mz)
        d[c.BASELINE] = None if self.baseline is None else array1d_to_str(self.baseline)
        d[c.NOISE] = None if self.noise is None else array1d_to_str(self.noise)
        if self.features is None:
            d[c.ROI_FEATURE_LIST] = None
        else:
            d[c.ROI_FEATURE_LIST] = [x.to_str() for x in self.features]

        d_json = json.dumps(d)
        return d_json

    def fill_nan(self, **kwargs):
        """
        Fill missing values in the trace.

        Missing m/z values are filled using the mean m/z of the ROI. Missing intensity
        values are filled using linear interpolation. Missing values on the boundaries
        are filled by extrapolation. Negative values are set to 0.

        Parameters
        ----------
        kwargs:
            Parameters to pass to :func:`scipy.interpolate.interp1d`

        """

        # if the first or last values are missing, assign an intensity value
        # of zero. This prevents errors in the interpolation and makes peak
        # picking work better.

        missing = np.isnan(self.spint)
        if missing.any():
            interpolator = interp1d(
                self.time[~missing], self.spint[~missing], assume_sorted=True, **kwargs
            )
            sp_max = np.nanmax(self.spint)
            sp_min = np.nanmin(self.spint)
            self.spint[missing] = interpolator(self.time[missing])
            # bound extrapolated values to max and min observed values
            self.spint = np.maximum(self.spint, sp_min)
            self.spint = np.minimum(self.spint, sp_max)
        if isinstance(self.mz, np.ndarray):
            self.mz[missing] = np.nanmean(self.mz)


class LCTrace(MZTrace):
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

    features: list["Peak"]

    def __init__(
        self,
        time: np.ndarray[Any, np.dtype[np.floating]],
        spint: np.ndarray[Any, np.dtype[np.floating]],
        mz: np.ndarray[Any, np.dtype[np.floating]],
        scan: np.ndarray[Any, np.dtype[np.integer]],
        index: int = 0,
        mode: Optional[str] = c.UPLC,
        noise: Optional[np.ndarray] = None,
        baseline: Optional[np.ndarray] = None,
    ):
        super().__init__(time, spint, mz, scan, index, mode, noise, baseline)

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
        **kwargs,
    ) -> list["Peak"]:
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
        self.fill_nan(fill_value="extrapolate")
        noise = peaks.estimate_noise(self.spint)
        if smoothing_strength is None:
            x = self.spint
        else:
            x = cast(
                np.ndarray[Any, np.dtype[np.floating]],
                gaussian_filter1d(self.spint, smoothing_strength),
            )

        if store_smoothed:
            self.spint = x

        baseline = peaks.estimate_baseline(x, noise)
        start, apex, end = peaks.detect_peaks(x, noise, baseline, **kwargs)
        n_peaks = start.size
        features = [
            Peak(s, a, e, self, i) for s, a, e, i in zip(start, apex, end, range(n_peaks))
        ]

        self.features = features
        self.baseline = baseline
        self.noise = noise
        return features

    def plot(
        self, figure: Optional[bokeh.plotting.figure] = None, show: bool = True
    ) -> bokeh.plotting.figure:  # pragma: no cover
        """
        Plot the ROI.

        Parameters
        ----------
        figure : bokeh.plotting.figure or None, default=None
            Figure to add the plot. If None, a new figure is created.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh.plotting.figure

        """
        if figure is None:
            fig_params = _plot_bokeh.get_chromatogram_figure_params()
            figure = bokeh.plotting.figure(**fig_params)

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

    @staticmethod
    def _get_feature_type():
        return Peak


class Chromatogram(LCTrace):
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

    mz: Optional[np.ndarray]
    scan: Optional[np.ndarray]

    def __init__(
        self, time: np.ndarray, spint: np.ndarray, index: int = 0, mode: str = c.UPLC
    ):
        super(Chromatogram, self).__init__(time, spint, spint, spint, index, mode)
        self.mz = None
        self.scan = None


class InvalidPeakException(ValueError):
    """
    Exception raised when invalid indices are used in the construction of
    Peak objects.

    """

    pass


class Feature(ABC):
    """
    Abstract representation of a feature.

    Attributes
    ----------
    roi: Roi
    annotation: Optional[Annotation]
    index: int

    """

    def __init__(self, roi: Roi, index: int = 0):
        self.roi = roi
        self._mz = None
        self._area = None
        self._height = None
        self.annotation = Annotation(-1, -1, -1, -1)
        self.index = index

    @property
    def mz(self) -> float:
        if self._mz is None:
            self._mz = self.get_mz()
        return self._mz

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.get_area()
        return self._area

    @property
    def height(self) -> float:
        if self._height is None:
            self._height = self.get_height()
        return self._height

    def __lt__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz < other
        elif isinstance(other, Feature):
            return self.mz < other.mz

    def __le__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz <= other
        elif isinstance(other, Feature):
            return self.mz <= other.mz

    def __gt__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz > other
        elif isinstance(other, Feature):
            return self.mz > other.mz

    def __ge__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz >= other
        elif isinstance(other, Feature):
            return self.mz >= other.mz

    @abstractmethod
    def get_mz(self) -> float:
        ...

    @abstractmethod
    def get_area(self) -> float:
        ...

    @abstractmethod
    def get_height(self) -> float:
        ...

    @abstractmethod
    def describe(self) -> dict[str, float]:
        ...

    @abstractmethod
    def compare(self, other: "Feature") -> float:
        ...

    @abstractmethod
    def to_str(self) -> str:
        ...

    @classmethod
    def from_str(cls, s: str, roi: Roi) -> "Feature":
        d = cls._deserialize(s)
        return cls(roi=roi, **d)

    @staticmethod
    @abstractmethod
    def _deserialize(s: str) -> dict:
        ...

    @staticmethod
    @abstractmethod
    def compute_isotopic_envelope(
        feature: Sequence["Feature"],
    ) -> Tuple[list[float], list[float]]:
        ...


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
    roi: LCTrace
        ROI associated with the Peak.
    index: int
        Unique index for features detected a ROI.

    """

    roi: LCTrace

    def __init__(self, start: int, apex: int, end: int, roi: LCTrace, index: int = 0):
        try:
            assert start < end
            assert start < apex
            assert apex < end
        except AssertionError:
            msg = "start must be lower than loc and loc must be lower than end"
            raise InvalidPeakException(msg)
        super().__init__(roi, index)
        self.start = int(start)
        self.end = int(end)
        self.apex = int(apex)

    def to_str(self) -> str:
        d = {
            c.START: self.start,
            c.APEX: self.apex,
            c.END: self.end,
            "index": self.index,
        }
        s = json.dumps(d)
        return s

    @staticmethod
    def _deserialize(s: str) -> dict:
        return json.loads(s)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(start={self.start}, apex={self.apex}, end={self.end})"

    def plot(self, figure: bokeh.plotting.figure, color: str, **varea_params):
        _plot_bokeh.fill_area(
            figure,
            self.roi.time,
            self.roi.spint,
            self.start,
            self.end,
            color,
            **varea_params,
        )

    def get_rt_start(self) -> float:
        """
        Computes the start of the peak, in time units

        Returns
        -------
        float

        """
        return self.roi.time[self.start]

    def get_rt_end(self) -> float:
        """
        Computes the end of the peak, in time units

        Returns
        -------
        float

        """
        return self.roi.time[self.end - 1]

    def get_rt(self) -> float:
        """
        Finds the peak location in the ROI rt, using spint as weights.

        Returns
        -------
        rt : float

        """
        weights = self.roi.spint[self.start : self.end]
        if self.roi.baseline is not None:
            weights = weights - self.roi.baseline[self.start : self.end]
        weights = np.maximum(weights, 0)
        loc = np.abs(np.average(self.roi.time[self.start : self.end], weights=weights))
        return loc

    def get_height(self) -> float:
        """
        Computes the height of the peak, defined as the difference between the
        value of intensity in the ROI and the baseline at the peak apex.

        Returns
        -------
        height : non-negative number. If the baseline estimation is greater
        than y, the height is set to zero.

        """
        height = self.roi.spint[self.apex]
        if self.roi.baseline is not None:
            height -= self.roi.baseline[self.apex]
        return max(0.0, height)

    def get_area(self) -> float:
        """
        Computes the area in the region defined by the peak.

        If the baseline area is greater than the peak area, the area is set
        to zero.

        Returns
        -------
        area : positive number.

        """
        peak_extension = self.roi.spint[self.start : self.end]
        if self.roi.baseline is not None:
            peak_extension = peak_extension - self.roi.baseline[self.start : self.end]

        area = trapz(peak_extension, self.roi.time[self.start : self.end])
        return max(0.0, area)

    def get_width(self) -> float:
        """
        Computes the peak width, defined as the region where the 95 % of the
        total peak area is distributed.

        Returns
        -------
        width : positive number.

        """
        height = (
            self.roi.spint[self.start : self.end] - self.roi.baseline[self.start : self.end]
        )
        area = cumtrapz(height, self.roi.time[self.start : self.end])
        if area[-1] > 0:
            relative_area = area / area[-1]
            percentile = [0.025, 0.975]
            start, end = self.start + np.searchsorted(relative_area, percentile)
            width = float(self.roi.time[end] - self.roi.time[start])
        else:
            width = 0.0
        return max(0.0, width)

    def get_extension(self) -> float:
        """
        Computes the peak extension, defined as the length of the peak region.

        Returns
        -------
        extension : positive number

        """
        return self.roi.time[self.end] - self.roi.time[self.start]

    def get_snr(self) -> float:
        """
        Computes the peak signal-to-noise ratio, defined as the quotient
        between the peak height and the noise level at the apex.

        Returns
        -------
        snr : float

        """

        peak_noise = self.roi.noise[self.apex]
        if np.isclose(peak_noise, 0):
            snr = np.inf
        else:
            snr = self.get_height() / peak_noise
        return snr

    def get_mz(self) -> float:
        """
        Computes the weighted average m/z of the peak.

        Returns
        -------
        mz_mean : float

        """
        if self.roi.mz is None:
            msg = "mz not specified for ROI."
            raise ValueError(msg)
        else:
            weights = self.roi.spint[self.start : self.end]
            weights[weights < 0.0] = 0
            mz_mean = np.average(self.roi.mz[self.start : self.end], weights=weights)
            mz_mean = max(0.0, mz_mean.item())
        return mz_mean

    def get_mz_std(self) -> Optional[float]:
        """
        Computes the standard deviation of the m/z in the peak

        Returns
        -------
        mz_std : float

        """
        if self.roi.mz is None:
            mz_std = None
        else:
            mz_std = self.roi.mz[self.start : self.end].std()
        return mz_std

    def describe(self) -> dict[str, float]:
        """
        Computes peak height, area, location, width and SNR.

        Returns
        -------
        descriptors: dict
            A mapping of descriptor names to descriptor values.
        """
        descriptors = {
            c.HEIGHT: self.get_height(),
            c.AREA: self.get_area(),
            c.RT: self.get_rt(),
            c.WIDTH: self.get_width(),
            c.SNR: self.get_snr(),
            c.MZ: self.get_mz(),
            c.MZ_STD: self.get_mz_std(),
            c.RT_START: self.get_rt_start(),
            c.RT_END: self.get_rt_end(),
        }
        return descriptors

    def compare(self, other: "Peak") -> float:
        return _compare_features_lc(self, other)

    @staticmethod
    def compute_isotopic_envelope(features: list["Peak"]) -> Tuple[list[float], list[float]]:
        """
        Computes a m/z and relative abundance for a list of features.

        """
        scan_start = 0
        scan_end = 10000000000  # dummy value
        for ft in features:
            scan_start = max(scan_start, ft.roi.scan[ft.start])
            scan_end = min(scan_end, ft.roi.scan[ft.end - 1])

        p = list()
        mz = list()
        if scan_start < scan_end:
            for ft in features:
                start = bisect.bisect(ft.roi.scan, scan_start)
                end = bisect.bisect(ft.roi.scan, scan_end)
                apex = (start + end) // 2  # dummy value
                tmp_peak = Peak(start, apex, end, ft.roi)
                p_area = trapz(tmp_peak.roi.spint[start:end], tmp_peak.roi.time[start:end])
                p.append(p_area)
                mz.append(tmp_peak.mz)
        total_area = sum(p)
        p = [x / total_area for x in p]
        return mz, p


@dataclass
class Annotation:
    """
    Contains annotation information of features.

    If an annotation is not available, ``-1`` is used.

    Attributes
    ----------
    label : int
        Correspondence label of features.
    isotopologue_label : int
        Groups features from the same isotopic envelope.
    isotopologue_index : int
        Position of the feature in an isotopic envelope.
    charge : int
        Charge state.

    """

    label: int
    isotopologue_label: int
    isotopologue_index: int
    charge: int


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
    params = {"min_snr": 10.0}
    if instrument == c.QTOF:
        md = 0.01
    else:  # orbitrap
        md = 0.005
    params["min_distance"] = md
    return params


def _compare_features_lc(ft1: Peak, ft2: Peak) -> float:
    """
    Feature similarity function used in LC-MS data.
    """
    start1 = ft1.roi.scan[ft1.start]
    start2 = ft2.roi.scan[ft2.start]
    if start1 > start2:
        ft1, ft2 = ft2, ft1
    overlap_ratio = _overlap_ratio(ft1, ft2)
    min_overlap = 0.5
    if overlap_ratio > min_overlap:
        os1, oe1, os2, oe2 = _get_overlap_index(ft1, ft2)
        norm1 = np.linalg.norm(ft1.roi.spint[ft1.start : ft1.end])
        norm2 = np.linalg.norm(ft2.roi.spint[ft2.start : ft2.end])
        x1 = ft1.roi.spint[os1:oe1] / norm1
        x2 = ft2.roi.spint[os2:oe2] / norm2
        similarity = np.dot(x1, x2)
    else:
        similarity = 0.0
    return similarity


def _overlap_ratio(ft1: Peak, ft2: Peak) -> float:
    """
    Computes the overlap ratio, defined as the quotient between the overlap
    region and the extension of the longest feature.

    `ft1` must start before `ft2`

    Parameters
    ----------
    roi1 : LCRoi
    ft1 : Peak
    roi2 : LCRoi
    ft2 : Peak

    Returns
    -------
    overlap_ratio : float

    """
    start2 = ft2.roi.scan[ft2.start]
    end1 = ft1.roi.scan[ft1.end - 1]
    end2 = ft2.roi.scan[ft2.end - 1]
    # start1 <= start2. end1 > start2 is a sufficient condition for overlap
    if end1 > start2:
        # the overlap ratio is the quotient between the length overlapped region
        # and the extension of the shortest feature.
        if end1 <= end2:
            start2_index_in1 = bisect.bisect_left(ft1.roi.scan, start2)
            overlap_length = ft1.end - start2_index_in1
        else:
            overlap_length = ft2.end - ft2.start
        min_length = min(ft1.end - ft1.start, ft2.end - ft2.start)
        res = overlap_length / min_length
    else:
        res = 0.0
    return res


def _get_overlap_index(ft1: Peak, ft2: Peak) -> Tuple[int, int, int, int]:
    """
    Computes the overlap indices for ft1 and ft2.

    `ft1` must start before `ft2`

    Parameters
    ----------
    ft1 : Peak
    ft2 : Peak

    Returns
    -------
    overlap_start1 : int
    overlap_end1 : int
    overlap_start2 : int
    overlap_end2 : int

    """
    end1 = ft1.roi.scan[ft1.end - 1]
    end2 = ft2.roi.scan[ft2.end - 1]
    start2 = ft2.roi.scan[ft2.start]
    if end1 >= end2:
        overlap_start1 = bisect.bisect_left(ft1.roi.scan, start2)
        overlap_end1 = bisect.bisect(ft1.roi.scan, end2)
        overlap_start2 = ft2.start
        overlap_end2 = ft2.end
    else:
        overlap_start1 = bisect.bisect_left(ft1.roi.scan, start2)
        overlap_end1 = ft1.end
        overlap_start2 = ft2.start
        overlap_end2 = bisect.bisect(ft2.roi.scan, end1)
    return overlap_start1, overlap_end1, overlap_start2, overlap_end2
