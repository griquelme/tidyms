"""
functions and objects used to detect peaks.
"""

import numpy as np
from scipy.signal import _peak_finding
from scipy.signal import find_peaks
from scipy.signal.wavelets import ricker, cwt
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional
from .utils import _find_closest_sorted

# TODO : restructure module.
# TODO : PeakLocation should only store: loc, start, end, scale and signal,
#   obtained from CWT estimation.
# TODO : add find_peaks_cwt function that find peaks using only a list of
#   widths, and ridgeline tolerance, ridgeline min_length and ridgeline
#   n_missing. This function should return a list of PeakLocation.
# TODO : add estimate_peak_params function that takes a list of PeakLocation,
#   the signal used to detect the peaks, a dictionary of estimators and a
#   dictionary of filter params. This function should estimate parameters from
#   the peaks and filter them. The estimator dictionary should have as keys
#   parameters to estimate of the peaks (area, intensity, width) or estimations
#   for the whole signal (noise, baseline). The values of the dictionary
#   should be strings for predefined estimation functions or callables for
#   custom estimators. Using these estimators peaks can also be filtered using
#   the filter dictionary: available filters should be: snr, width,
#   baseline_ratio.


class PeakLocation:
    """
    Holds peak information for peak picking with the CWT algorithm.

    Attributes
    ----------
    loc: int
        index where a the apex of a peak was found.
    start: int
    scale: float
        width value where the peak optimally was found.
    end: int
    snr: float
        Signal to noise ratio estimation.
    baseline: float
        baseline estimation.
    """

    def __init__(self, loc, scale, start, end, snr, baseline):
        self.loc = loc
        self.scale = scale
        self.start = start
        self.end = end
        self.snr = snr
        self.baseline = baseline

    def rescale(self, old_scale, new_scale) -> 'PeakLocation':
        """
        create a new PeakLocation object using values from the new scale.

        Parameters
        ----------
        old_scale: numpy.ndarray
        new_scale: numpy.ndarray

        Returns
        -------
        PeakLocation
        """
        old_points = old_scale[[self.start, self.loc, self.end]]
        start, loc, end = _find_closest_sorted(new_scale, old_points)
        return PeakLocation(loc, self.scale, start, end, self.snr,
                            self.baseline)

    def get_peak_params(self, y: np.ndarray, x: Optional[np.ndarray] = None,
                        subtract_bl: bool = True,
                        center_estimation: str = "weighted") -> dict:
        """
        compute peak parameters based on x and y.

        Parameters
        ----------
        y: numpy.ndarray
        x: numpy.ndarray, optional
            if None computes parameters using index values.
        subtract_bl: bool
            If True subtract baseline to peak intensity and area
        center_estimation: {"weighted", "apex"}
            Only used when x is provided. if "weighted", the location of the
            peak is computed as the weighted mean of x in the extension of the
            peak, using y as weights. If "apex", the location is simply the
            location obtained after peak picking.

        Returns
        -------
        dict
        """

        if x is None:
            width = self.end - self.start
            area = trapz(y[self.start:(self.end + 1)])
            location = self.loc
        else:
            width = x[self.end] - x[self.start]
            area = trapz(y[self.start:(self.end + 1)],
                         x[self.start:(self.end + 1)])
            if center_estimation == "weighted":
                # weighted mean
                y_sum = y[self.start:self.end].sum()
                location = (x[self.start:self.end]
                            * y[self.start:self.end] / y_sum).sum()
            elif center_estimation == "apex":
                location = x[self.loc]
            else:
                msg = "center_estimation must be `weighted` or `apex`"
                raise ValueError(msg)

        intensity = y[self.loc]

        if subtract_bl:
            area -= width * self.baseline

        peak_params = {"location": location, "intensity": intensity,
                       "width": width, "area": area}
        return peak_params


def make_widths(x: np.ndarray, max_width: float) -> np.ndarray:
    """
    Create an array of widths to use in CWT.

    Parameters
    ----------
    x: numpy.ndarray
        vector of x axis. It's assumed that x is sorted.
    max_width: float
    Returns
    -------
    widths: numpy.ndarray
    """
    min_x_distance = np.diff(x).min()
    n = int((max_width - min_x_distance) / min_x_distance)
    first_half = np.linspace(min_x_distance, 10 * min_x_distance, 40)
    second_half = np.linspace(11 * min_x_distance, max_width, n - 10)
    widths = np.hstack((first_half, second_half))

    return widths


def process_ridge_lines(y: np.ndarray,
                        cwt_array: np.ndarray,
                        ridge_lines: List[Tuple[np.ndarray, np.ndarray]],
                        min_width: int,
                        max_width: int,
                        min_length=None,
                        min_snr: float = 3,
                        min_bl_ratio: float = 2) -> List[PeakLocation]:
    """
    Filter ridge lines and estimate peak parameters.

    Parameters
    ----------
    y: np.ndarray
        intensity vector used estimate baseline and noise
    cwt_array: numpy.ndarray
        CWT using y and widths, obtained using scipy.signal.wavelets.cwt
    ridge_lines: list[(numpy.ndarray, numpy.ndarray)]
        Ridge lines from cwt_array computed using
        scipy.signal._peak_finding._identify_ridge_lines.
    min_width: int
    max_width: int
    min_length: int
        minimum length of a ridge line
    min_snr: float
    min_bl_ratio: float
        minimum ratio between the intensity and the baseline

    Returns
    -------

    """
    # TODO: add BL estimator
    if min_length is None:
        min_length = np.ceil(cwt_array.shape[0] / 8)

    peaks = list()

    for row_ind, col_ind in ridge_lines:
        rl_peaks = list()   # peaks in the current ridge line

        # min_length check
        if len(row_ind) < min_length:
            continue

        # snr check and width check for all scale max
        line = cwt_array[row_ind, col_ind]
        max_index = find_peaks(line)[0]

        for ind in max_index:
            scale_index, peak_index = row_ind[ind], col_ind[ind]
            extension = _find_peak_extension(cwt_array, scale_index, peak_index)

            peak_width = extension[1] - extension[0]
            width_check = ((peak_width >= min_width)
                           and (peak_width <= max_width))
            baseline, snr = snr_calculation(y, peak_index, extension)
            snr_check = snr >= min_snr
            bl_ratio_check = y[peak_index] > min_bl_ratio * baseline
            if snr_check and width_check and bl_ratio_check:
                temp_peak = PeakLocation(peak_index, scale_index,
                                         extension[0], extension[1],
                                         snr, baseline)
                rl_peaks.append(temp_peak)

        if len(rl_peaks) > 0:
            best_peak = max(rl_peaks,
                            key=lambda x: cwt_array[x.scale, x.loc])
            peaks.append(best_peak)
    return peaks


def snr_calculation(y: np.ndarray,
                    peak_index: int,
                    extension: Tuple[int, int]):

    # baseline and noise estimation.  region at the left and right of the peak
    # are used to estimate baseline.
    # BL is the mean of the 25 % lower values of each region.
    # Noise is the std of the same region
    # The BL is finally chosen as the smaller value of left and right regions.
    # Noise is the higher of both regions.

    # prevents to use negative indices or indices greater than size of y
    width = (extension[1] - extension[0])
    left = max(0, extension[0] - width)
    right = min(y.size, extension[1] + width)

    left_25_lower = np.sort(y[left:extension[0]])
    # left_25_lower = left_25_lower[:(left_25_lower.size // 4)]
    lperc_5 = int(5 * left_25_lower.size / 100)
    lperc_95 = int(95 * left_25_lower.size / 100)
    left_25_lower = left_25_lower[lperc_5:lperc_95]
    if left_25_lower.size > 0:
        left_bl = left_25_lower.mean()
        left_noise = left_25_lower.std()
    else:
        left_bl, left_noise = np.inf, np.inf
    right_25_lower = np.sort(y[extension[1]:right])
    # right_25_lower = right_25_lower[:(right_25_lower.size // 4)]
    rperc_5 = int(5 * right_25_lower.size / 100)
    rperc_95 = int(95 * right_25_lower.size / 100)
    right_25_lower = right_25_lower[rperc_5:rperc_95]
    if right_25_lower.size > 0:
        right_bl = right_25_lower.mean()
        right_noise = right_25_lower.std()
    else:
        right_bl, right_noise = np.inf, np.inf

    baseline = min(left_bl, right_bl)
    noise = min(left_noise, right_noise)

    if noise:
        snr = (y[peak_index] - baseline) / noise
    else:
        snr = 0
    return baseline, snr


def _find_peak_extension(cwt_array: np.ndarray, scale_index: int,
                         peak_index: int) -> Tuple[int, int]:
    """
    Find peak extensions in CWT algorithm. Helper function of
    process_ridge_lines.

    Parameters
    ----------
    cwt_array: numpy.ndarray
    scale_index: int
    peak_index: int

    Returns
    -------
    extension: Tuple[int, int]
    """
    min_index = find_peaks(-cwt_array[scale_index, :])[0]
    # 0 and 1 are included in case there's no minimum
    min_index = np.hstack((0, min_index, cwt_array.shape[1] - 1))
    ext = np.searchsorted(min_index, peak_index)
    extension = min_index[ext - 1], min_index[ext]
    return extension


def resample_data(x: np.ndarray,
                  y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data to uniform sampling. Helper function to cwt peak_picking

    Parameters
    ----------
    x: np.ndarray
    y: np.ndarray

    Returns
    -------
    xr: np.ndarray
        X resampled
    yr: np.ndarray
        y resampled
    """
    min_dist = np.diff(x).min()
    x_max = x.max()
    x_min = x.min()
    n_interp = int((x_max - x_min) / min_dist) + 1
    xr = np.linspace(x_min, x_max, n_interp)
    interpolator = interp1d(x, y)
    yr = interpolator(xr)
    return xr, yr


def convert_to_original_scale(x_orig: np.ndarray, x_rescaled: np.ndarray,
                              peak: PeakLocation) -> PeakLocation:

    start, loc, end = _find_closest_sorted(x_orig,
                                           x_rescaled[[peak.start, peak.loc,
                                                      peak.end]])
    return PeakLocation(loc, peak.scale, start, end, peak.snr, peak.baseline)


def pick_cwt(x: np.ndarray, y: np.ndarray, widths: np.ndarray, snr: float = 3,
             bl_ratio: float = 2, min_width: Optional[float] = 5,
             max_width: Optional[float] = 60,
             max_distance: Optional[int] = None,
             min_length: Optional[int] = None,
             gap_thresh: int = 1) -> List[PeakLocation]:

    # Convert to uniform sampling
    xu, yu = resample_data(x, y)    # x uniform, y uniform

    # convert min_width and max_width to number of points
    min_width = int(min_width / (xu[1] - xu[0])) + 1
    max_width = int(max_width / (xu[1] - xu[0])) + 1

    # widths = make_widths(xu, max_width=max_width)
    widths = widths / (xu[1] - xu[0])

    # Setting max_distance
    if max_distance is None:
        max_distance = np.ones_like(widths) * 3
    else:
        max_distance = int(max_distance / (xu[1] - xu[0])) + 1
        max_distance = np.ones_like(widths) * max_distance

    w = cwt(yu, ricker, widths)
    ridge_lines = _peak_finding._identify_ridge_lines(w, max_distance,
                                                      gap_thresh)
    peaks = process_ridge_lines(yu, w, ridge_lines, min_width, max_width,
                                min_length=min_length, min_snr=snr,
                                min_bl_ratio=bl_ratio)

    # convert back from uniform sampling to original x scale
    peaks = [p.rescale(xu, x) for p in peaks]
    # sort peaks based on loc

    peaks.sort(key=lambda x: x.loc)
    # correct overlap between consecutive peaks:

    remove = list()
    for k in range(len(peaks) - 1):
        left, right = peaks[k], peaks[k + 1]
        if left.end > right.start:
            right.start = left.end
        if right.start > right.end:
            remove.append(k+1)

    for ind in reversed(remove):
        del peaks[ind]

    for p in peaks:
        if p.start > p.end:
            print(p.end - p.start)


    return peaks
