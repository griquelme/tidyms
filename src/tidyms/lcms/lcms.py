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
from math import nan

import numpy as np
import pydantic
from pydantic.functional_validators import model_validator
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self

from ..core.models import AnnotableFeature, IsotopicEnvelope, Roi
from ..core.numpy import FloatingArray, IntArray


class MZTrace(Roi):
    """
    ROI implementation using m/z traces.

    An m/z trace is a 1D trace containing m/z, time and intensity information
    across scans.

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

    mz: FloatingArray
    spint: FloatingArray
    time: FloatingArray
    scan: IntArray
    noise: FloatingArray | None = None
    baseline: FloatingArray | None = None

    @model_validator(mode="after")
    def check_noise_all_non_negative(self) -> Self:
        """Validate noise."""
        if self.noise is not None:
            msg = "All noise array elements must be non-negative."
            assert np.all(self.noise >= 0.0), msg
        return self

    @model_validator(mode="after")
    def check_baseline_lower_or_equal_than_intensity(self) -> Self:
        """Validate baseline."""
        if self.baseline is not None:
            msg = "All baseline elements must be lower than their corresponding trace intensity."
            assert np.all(self.spint >= self.baseline), msg
        return self

    def get_slice_height(self, start: int, end: int) -> FloatingArray:
        """
        Compute the trace height in a slice.

        Parameters
        ----------
        start : int
            slice start index.
        end : int
            slice end index.

        """
        height = self.spint[start:end]
        if self.baseline is not None:
            height = height - self.baseline[start:end]
        return np.maximum(height, 0.0)

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
                fill_value="extrapolate", # type: ignore
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

    def __eq__(self, other: Self) -> bool:
        return (
            np.array_equal(self.mz, other.mz)
            and np.array_equal(self.time, other.time)
            and np.array_equal(self.spint, other.spint)
            and np.array_equal(self.scan, other.scan)
            and self.id == other.id
        )


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

    pass

class InvalidPeakException(ValueError):
    """Exception raised when invalid indices are passed to the Peak constructor."""

    pass


class Peak(AnnotableFeature[MZTrace]):
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
    mz: float
        Peak m/z, defined as the weighted average of the trace m/z in the peak
        region. The trace height is used as weights.
    rt : float
        Peak retention time, defined as the weighted average of the trace time
        in the peak region. The trace height is used as weights.
    rt_start : float
        Peak start time.
    rt_end : float
        Peak end time.
    height: float
        Peak height, defined as the difference between the peak intensity
        and the peak baseline at the apex.
    area: float
        Peak area.
    snr: float
        The peak signal-to-noise ratio, defined as the quotient between the
        peak height and the noise level at the apex. If the noise level is not
        available, the SNR is set to ``nan``.
    mz_std: float
        The peak m/z standard deviation.
    width: float
        The peak width, defined as the time window comprised by the 2.5 % and
        97.5 % of the cumulative peak area.
    extension : float
        The peak extension, defined as the length of the peak region.

    """

    start: pydantic.NonNegativeInt
    apex: pydantic.PositiveInt
    end: pydantic.PositiveInt

    area: float = nan
    extension: float = nan
    height: float = nan
    mz_std: float = nan
    rt: float = nan
    rt_start: float = nan
    rt_end: float = nan
    snr: float = nan
    width: float = nan

    @model_validator(mode="after")
    def check_peak_definition(self) -> Self:
        """
        Perform peak sanity check.

        Check that start, apex and end indices are strictly sorted. Check that
        peak end index is lower than trace length.

        """
        msg = "start must be lower than loc and loc must be lower than end"
        assert (self.start < self.apex < self.end), msg

        msg = "peak end must be lower than m/z trace length."
        assert (self.end < self.roi.scan.size), msg

        return self

    def _set_rt_start(self) -> None:
        self.rt_start = self.roi.time[self.start].item()

    def _set_rt_end(self) -> None:
        self.rt_end = self.roi.time[self.end - 1].item()

    def _set_rt(self) -> None:
        try:
            weights = self.roi.get_slice_height(self.start, self.end)
            self.rt = np.average(self.roi.time[self.start : self.end], weights=weights).item()
        except ZeroDivisionError:
            self.rt = nan

    def _set_height(self) -> None:
        height = self.roi.spint[self.apex].item()
        if self.roi.baseline is not None:
            height = height - self.roi.baseline[self.apex].item()
        self.height = height

    def _set_area(self) -> None:
        peak_extension = self.roi.get_slice_height(self.start, self.end)
        self.area = trapz(peak_extension, self.roi.time[self.start : self.end]).item()

    def _set_width(self) -> None:
        """
        Compute the peak width.

        The peak width is defined as the region where the 95 % of the total peak
        area is distributed.

        Returns
        -------
        width : positive number.

        """
        height = self.roi.get_slice_height(self.start, self.end)

        area = cumtrapz(height, self.roi.time[self.start : self.end])
        if area[-1] > 0:
            relative_area = area / area[-1]
            percentile = [0.025, 0.975]
            start, end = self.start + np.searchsorted(relative_area, percentile)
            width = float(self.roi.time[end] - self.roi.time[start])
        else:
            width = nan
        self.width = width

    def _set_extension(self) -> None:
        self.extension = self.roi.time[self.end - 1] - self.roi.time[self.start]

    def _set_snr(self) -> None:
        if self.roi.noise is None or np.allclose(self.roi.noise[self.apex], 0.0):
            self.snr = nan
        else:
            self.snr = self.get("height") / self.roi.noise[self.apex]


    def _set_mz(self) -> None:
        weights = self.roi.get_slice_height(self.start, self.end)
        try:
            mz_mean = np.average(self.roi.mz[self.start : self.end], weights=weights)
            self.mz = max(0.0, mz_mean.item())
        except ZeroDivisionError:
            self.mz = nan

    def _set_mz_std(self) -> None:
        self.mz_std = self.roi.mz[self.start : self.end].std()

    def compare(self, other: Self) -> float:
        """
        Compute the similarity between a pair of peaks.

        The similarity is defined as the cosine distance between the overlapping
        region of two peaks.

        """
        return _compare_features_lc(self, other)

    @staticmethod
    def compute_isotopic_envelope(
        features: list[Peak],
    ) -> IsotopicEnvelope:
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
        # TODO: refactor this.
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
                tmp_peak = Peak(start=start, apex=apex, end=end, roi=ft.roi)
                p_area = trapz(
                    tmp_peak.roi.spint[start:end], tmp_peak.roi.time[start:end]
                )
                p.append(p_area)
                mz.append(ft.mz)
        total_area = sum(p)
        p = [x / total_area for x in p]
        return IsotopicEnvelope(mz=mz, p=p)


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
        x1 = ft1.roi.spint[os1:oe1] / norm1
        x2 = ft2.roi.spint[os2:oe2] / norm2
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


def _get_overlap_index(ft1: Peak, ft2: Peak) -> tuple[int, int, int, int]:
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
