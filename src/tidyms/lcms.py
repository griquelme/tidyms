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

from __future__ import annotations

import bisect
import bokeh.plotting
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Any, cast, Optional, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz

from .core.models import Annotation, Feature, Roi
from . import peaks
from . import _plot_bokeh
from .core import constants as c
from .core.spectrum import MSSpectrum
from .utils import array_to_json_str, json_str_to_array


class MZTrace(Roi):
    """
    ROI Implementation using MZ traces.

    MZ traces are 1D traces containing time, intensity and m/z associated with
    each scan.

    Attributes
    ----------
    time : array[float]
        time in each scan. All values are assumed to be non-negative.
    spint : array[float]
        intensity in each scan. All values are assumed to be non-negative.
    mz : array[float]
        m/z in each scan. All values are assumed to be non-negative.
    scan : array[int]
        scan numbers where the ROI is defined. All values are assumed to be
        non-negative.
    features : OptionalList[Feature]]

    """

    def __init__(
        self,
        time: np.ndarray[Any, np.dtype[np.floating]],
        spint: np.ndarray[Any, np.dtype[np.floating]],
        mz: np.ndarray[Any, np.dtype[np.floating]],
        scan: np.ndarray[Any, np.dtype[np.integer]],
        noise: Optional[np.ndarray] = None,
        baseline: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.time = time
        self.spint = spint
        self.mz = mz
        self.scan = scan
        self.features: Optional[list[Feature]] = None
        self.fill_nan()

        if noise is None:
            noise = peaks.estimate_noise(self.spint)
        self.noise = noise

        if baseline is None:
            baseline = peaks.estimate_baseline(self.spint, self.noise)
        self.baseline = baseline

    @staticmethod
    def _deserialize(s: str) -> dict[str, Any]:
        d = json.loads(s)
        d[c.SPINT] = json_str_to_array(d[c.SPINT])
        d[c.MZ] = None if d[c.MZ] is None else json_str_to_array(d[c.MZ])
        d[c.TIME] = json_str_to_array(d[c.TIME])
        d[c.SCAN] = None if d[c.SCAN] is None else json_str_to_array(d[c.SCAN])
        d[c.NOISE] = None if d[c.NOISE] is None else json_str_to_array(d[c.NOISE])
        d[c.BASELINE] = (
            None if d[c.BASELINE] is None else json_str_to_array(d[c.BASELINE])
        )
        return d

    def to_str(self) -> str:
        """
        Serialize the LCRoi into a JSON str.

        Returns
        -------
        str

        """
        d = dict()
        d[c.TIME] = array_to_json_str(self.time)
        d[c.SPINT] = array_to_json_str(self.spint)
        d[c.SCAN] = None if self.scan is None else array_to_json_str(self.scan)
        d[c.MZ] = None if self.mz is None else array_to_json_str(self.mz)
        d[c.BASELINE] = (
            None if self.baseline is None else array_to_json_str(self.baseline)
        )
        d[c.NOISE] = None if self.noise is None else array_to_json_str(self.noise)
        d_json = json.dumps(d)
        return d_json

    def fill_nan(self):
        """
        Fill missing values in the trace.

        Missing m/z values are filled using the mean m/z of the ROI. Missing intensity
        values are filled using linear interpolation. Missing values on the boundaries
        are filled by extrapolation. Negative values are set to 0.

        """
        missing = np.isnan(self.spint)
        if missing.any():
            interpolator = interp1d(
                self.time[~missing],
                self.spint[~missing],
                assume_sorted=True,
                fill_value="extrapolate",
            )
            sp_max = np.nanmax(self.spint)
            sp_min = np.nanmin(self.spint)
            self.spint[missing] = interpolator(self.time[missing])
            # bound extrapolated values to max and min observed values
            self.spint = np.maximum(self.spint, sp_min)
            self.spint = np.minimum(self.spint, sp_max)
        if isinstance(self.mz, np.ndarray):
            self.mz[missing] = np.nanmean(self.mz)

    def smooth(self, smoothing_strength: float):
        """
        Smooths the intensity of the trace using a gaussian kernel.

        Parameters
        ----------
        smoothing_strength : float
            Standard deviation of the gaussian kernel.
        """
        self.spint = gaussian_filter1d(self.spint, smoothing_strength)


class LCTrace(MZTrace):
    """
    m/z traces where chromatographic peaks may be found.

    m/z information is stored besides time and intensity information.

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

    """

    features: Optional[list[Peak]]

    def extract_features(self, **kwargs):
        """
        Detect chromatographic peaks.

        Peaks are stored in the `features` attribute.

        Parameters
        ----------
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
        if self.noise is None:
            self.noise = peaks.estimate_noise(self.spint)

        if self.baseline is None:
            self.baseline = peaks.estimate_baseline(self.spint, self.noise)

        start, apex, end = peaks.detect_peaks(
            self.spint, self.noise, self.baseline, **kwargs
        )
        self.features = [Peak(s, a, e, self) for s, a, e in zip(start, apex, end)]

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

    def __eq__(self, other: "LCTrace") -> bool:
        is_eq = (
            np.array_equal(self.mz, other.mz)
            and np.array_equal(self.time, other.time)
            and np.array_equal(self.spint, other.spint)
            and np.array_equal(self.scan, other.scan)
        )
        return is_eq


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

    """

    def __init__(self, time: np.ndarray, spint: np.ndarray):
        super(Chromatogram, self).__init__(time, spint, spint, spint)


class InvalidPeakException(ValueError):
    """Exception raised when invalid indices are passed to the Peak constructor."""

    pass


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

    def __init__(
        self,
        start: int,
        apex: int,
        end: int,
        roi: LCTrace,
        annotation: Optional[Annotation] = None,
    ):
        try:
            assert start < end
            assert start < apex
            assert apex < end
        except AssertionError:
            msg = "start must be lower than loc and loc must be lower than end"
            raise InvalidPeakException(msg)
        super().__init__(roi, annotation)
        self.start = int(start)
        self.end = int(end)
        self.apex = int(apex)

    def to_str(self) -> str:
        """Serialize the peak into a string representation."""
        d = {c.START: self.start, c.APEX: self.apex, c.END: self.end}
        s = json.dumps(d)
        return s

    @staticmethod
    def _deserialize(s: str) -> dict:
        return json.loads(s)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(start={self.start}, apex={self.apex}, end={self.end})"

    def plot(self, figure: bokeh.plotting.figure, color: str, **varea_params):
        """Plot the LC trace."""
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
        Compute the start of the peak, in time units.

        Returns
        -------
        float

        """
        return self.roi.time[self.start]

    def get_rt_end(self) -> float:
        """
        Compute the end of the peak, in time units.

        Returns
        -------
        float

        """
        return self.roi.time[self.end - 1]

    def get_rt(self) -> float:
        """
        Find the peak location in the ROI rt, using spint as weights.

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
        Compute the height of the peak.

        The height is defined as the difference between the value of intensity
        in the ROI and the baseline at the peak apex.

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
        Compute the peak area.

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
        Compute the peak width.

        The peak width is defined as the region where the 95 % of the total peak
        area is distributed.

        Returns
        -------
        width : positive number.

        """
        height = (
            self.roi.spint[self.start : self.end]
            - self.roi.baseline[self.start : self.end]
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
        Compute the peak extension.

        The peak extension is defined as the length of the peak region.

        Returns
        -------
        extension : positive number

        """
        return self.roi.time[self.end - 1] - self.roi.time[self.start]

    def get_snr(self) -> float:
        """
        Compute the peak signal-to-noise ratio (SNR).

        The SNR is defined as the quotient between the peak height and the noise
        level at the apex.

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
        Compute the weighted average m/z of the peak.

        Returns
        -------
        mz_mean : float

        """
        if self.roi.mz is None:
            msg = "mz not specified for ROI."
            raise ValueError(msg)
        else:
            weights = cast(np.ndarray, self.roi.spint[self.start : self.end])
            weights[weights < 0.0] = 0
            mz_mean = np.average(self.roi.mz[self.start : self.end], weights=weights)
            mz_mean = max(0.0, mz_mean.item())
        return mz_mean

    def get_mz_std(self) -> Optional[float]:
        """
        Compute the standard deviation of the peak m/z.

        Returns
        -------
        float

        """
        if self.roi.mz is None:
            mz_std = None
        else:
            mz_std = self.roi.mz[self.start : self.end].std()
        return mz_std

    def compare(self, other: Peak) -> float:
        """
        Compute the similarity between a pair of peaks.

        The similarity is defined as the cosine distance between the overlapping
        region of two peaks.

        """
        return _compare_features_lc(self, other)

    @staticmethod
    def compute_isotopic_envelope(
        features: list[Peak],
    ) -> Tuple[list[float], list[float]]:
        """
        Compute the isotopic envelope (m/z and abundance) of a list of peaks.

        Parameters
        ----------
        features : list[Peak]

        Returns
        -------
        mz : np.ndarray
            The mean m/z of each peak.
        abundance : np.ndarray
            The abundance of each peak (normalized to 1).

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
                p_area = trapz(
                    tmp_peak.roi.spint[start:end], tmp_peak.roi.time[start:end]
                )
                p.append(p_area)
                mz.append(tmp_peak.mz)
        total_area = sum(p)
        p = [x / total_area for x in p]
        return mz, p


def _compare_features_lc(ft1: Peak, ft2: Peak) -> float:
    """Feature similarity function used in LC-MS data."""
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
        x1 = cast(np.ndarray, ft1.roi.spint[os1:oe1]) / norm1
        x2 = cast(np.ndarray, ft2.roi.spint[os2:oe2]) / norm2
        similarity = np.dot(x1, x2)
    else:
        similarity = 0.0
    return similarity


def _overlap_ratio(ft1: Peak, ft2: Peak) -> float:
    """
    Compute the overlap ratio, between a pair of peaks.

    The overlap ratio is the quotient between the overlap region and the
    maximum value of the extension.

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
    Compute the overlap indices for ft1 and ft2.

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
