"""
Utilities to work with peaks
"""

import numpy as np
from scipy.signal import _peak_finding
from scipy.signal import find_peaks
from scipy.signal.wavelets import ricker, cwt
from scipy.integrate import trapz
from scipy.interpolate import interp1d
# from scipy.optimize import curve_fit
from typing import Tuple, List, Optional
from .utils import find_closest_sorted


class PeakLocation:
    """
    Holds peak information

    Attributes
    ----------
    loc: int
    start: int
    scale: float
    end: int
    snr: float
    baseline: float
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
        rescale:
        Parameters
        ----------
        old_scale: numpy.ndarray
        new_scale: numpy.ndarray

        Returns
        -------
        PeakLocation
        """
        old_points = old_scale[[self.start, self.loc, self.end]]
        start, loc, end = find_closest_sorted(new_scale, old_points)
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
        left_bl, left_noise = 0, 0
    right_25_lower = np.sort(y[extension[1]:right])
    # right_25_lower = right_25_lower[:(right_25_lower.size // 4)]
    rperc_5 = int(5 * right_25_lower.size / 100)
    rperc_95 = int(95 * right_25_lower.size / 100)
    right_25_lower = right_25_lower[rperc_5:rperc_95]
    if right_25_lower.size > 0:
        right_bl = right_25_lower.mean()
        right_noise = right_25_lower.std()
    else:
        right_bl = 0
        right_noise = 0

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

    start, loc, end = find_closest_sorted(x_orig,
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


# def get_peak_params2(cwt_array: np.ndarray, spint: np.ndarray, max_width: int,
#                      peak_loc: int, extension: Tuple[int, int], mode: str,
#                      min_snr: Optional[float] = 3,
#                      min_sblr: Optional[float] = 0.25,
#                      noise_perc: Optional[float] = 0.25) -> Optional[dict]:
#     """
#     Estimate peaks parameters.
#
#     Parameters
#     ----------
#     cwt_array: numpy.ndarray
#     spint: numpy.ndarray
#     max_width: int
#     peak_loc: int
#     extension: Tuple[int, int]
#     mode: {"centwave", "cwt"}
#         Estimate parameters according to centwave algorithm [1] or cwt peak
#         picking [2].
#     min_snr: float
#         Minimum SNR of peaks. SNR definition varies according to the method
#         selected.
#     min_sblr: float
#         Minimum signal to baseline ratio. Only used in centwave algorithm.
#
#     Returns
#     -------
#     peak_params: dict
#     """
#     if mode == "centwave":
#         start = max(peak_loc - max_width, 0)
#         end = min(peak_loc + max_width, spint.size)
#         # noise percentile intensity to estimate bl and noise
#         tmp_int = np.sort(spint[start:end])
#         tmp_int = tmp_int[:int(noise_perc * tmp_int.size)]
#         bl = tmp_int.mean()
#         noise = tmp_int.std()
#         ft_max = spint[peak_loc] - bl
#         snr = ft_max / noise
#         sblr = spint[peak_loc] / bl     # signal to baseline ratio
#
#         if (snr >= min_snr) and (sblr >= min_sblr):
#             area = trapz(spint[extension[0]:extension[1]])
#
#
# def get_peak_params(spint, peak_params, min_snr=3, blr=0.25):
#     results = list()
#     for peak in peak_params:
#         ext = 30
#         start = max(peak[1] - ext, 0)
#         end = min(peak[2] + ext, spint.size)
#         tmp_int = np.sort(spint[start:end])
#         # TODO: check negative index
#         tmp_int = tmp_int[:int(0.25 * tmp_int.size)]
#         bl = tmp_int.mean()
#         noise = tmp_int.std()
#         ft_int = spint[peak[0]] - bl
#         print(peak[0])
#         print(ft_int)
#         print(bl)
#         print(ft_int / noise)
#         print("-----")
#         if ((ft_int / noise) >= min_snr) and ((bl / spint[peak[0]]) <= blr):
#             results.append((peak[0], peak[1], peak[2], ft_int, ft_int / noise,
#                             bl, peak[3]))
#     return results


# def peak_cwt(y: np.ndarray, x: Optional[np.ndarray] = None,
#              snr: Optional[float: 3,], min_width: float = 3,
#              max_width: float = 60, min_roi_length: Optional[int] = None,
#              min_window_size: Optional[int] = None):
#
#     if min


# def pick(x, y, fwhm=None, height=None, asymmetry=False,
#          tailing=False, integrate=False, integrate_height=0.95):
#
#     peak_params = dict()
#     peaks, _ = find_peaks(y, height, prominence=500)
#     if fwhm:
#         fwhm_index = get_peak_widths(y, peaks, 0.5)
#         peaks_fwhm = x[fwhm_index[1, :]] - x[fwhm_index[0, :]]
#         # fwhm filter
#         fwhm_filter = (peaks_fwhm >= fwhm[0]) & (peaks_fwhm <= fwhm[1])
#         fwhm_index = fwhm_index[:, fwhm_filter]
#         peaks = peaks[fwhm_filter]
#         peak_params["fwhm left"] = x[fwhm_index[0, :]]
#         peak_params["fwhm right"] = x[fwhm_index[1, :]]
#         peak_params["fwhm"] = peaks_fwhm[fwhm_filter]
#         peak_params["fwhm overlap"] = find_overlap(fwhm_index)
#     if asymmetry:
#         peak_params["asymmetry"] = analyse_peak_shape(x, y, peaks, "asymmetry")
#     if tailing:
#         peak_params["tailing"] = analyse_peak_shape(x, y, peaks, "tailing")
#     if integrate:
#         lims = get_peak_widths(y, peaks, integrate_height)
#         area = np.zeros(lims.shape[1])
#         for k in range(area.size):
#             y_lims = y[lims[0, k]:(lims[1, k] + 1)]
#             x_lims = x[lims[0, k]:(lims[1, k] + 1)]
#             area[k] = trapz(y_lims, x_lims)
#         peak_params["area"] = area
#         peak_params["area left"] = x[lims[0, :]]
#         peak_params["area right"] = x[lims[1, :]]
#         peak_params["area overlap"] = find_overlap(lims)
#     peak_params["index"] = peaks
#     peak_params["loc"] = x[peaks]
#     peak_params["height"] = y[peaks]
#     return peak_params
#
#
# def make_empty_peaks():
#     d = dict()
#     params = ["index", "fwhm left", "fwhm right", "fwhm", "fwhm overlap",
#               "asymmetry", "tailing", "area", "area left", "area right",
#               "area overlap", "loc", "height"]
#     for param in params:
#         d[param] = np.array([])
#     return d
#
#
# def get_peak_widths(y, peaks, rel_height):
#
#     w = peak_widths(y, peaks, rel_height)
#     left_index = np.round(w[2]).astype(int)
#     right_index = np.round(w[3]).astype(int)
#     return np.vstack((left_index, right_index))
#
#
# def analyse_peak_shape(x, y, peaks, mode):
#     """
#     computes peak asymmetry or peak tailing factor.
#
#     Parameters
#     ----------
#     x : np.array
#     y : np.array
#     peaks : peaks index
#     mode : {"asymmetry", "tailing"}
#
#     Returns
#     -------
#     factor : np.array
#     """
#
#     rel_height = {"asymmetry": 0.9, "tailing": 0.95}
#     rel_height = rel_height[mode]
#
#     w = get_peak_widths(y, peaks, rel_height)
#     left_width = x[peaks] - x[w[0, :]]
#     right_width = x[w[1, :]] - x[peaks]
#     left_width[left_width == 0] = np.nan
#     if mode == "asymmetry":
#         factor = right_width / left_width
#     elif mode == "tailing":
#         factor = (left_width + right_width) / (2 * left_width)
#     else:
#         raise ValueError("mode should be `asymmetry` or `tailing`")
#     return factor
#
#
# def find_overlap(intervals):
#     reshaped = intervals.T.reshape(intervals.size)
#     has_overlap = np.where(np.diff(reshaped)[1::2] < 0)[0]
#     has_overlap = np.hstack((has_overlap, has_overlap + 1))
#     overlap = np.zeros(intervals.shape[1], dtype=bool)
#     overlap[has_overlap] = True
#     return overlap
#
#
# def fit_gaussian(x, y):
#     """
#     Fit multiple gaussians.
#
#     Parameters
#     ----------
#     x: np.array
#     y: np.array
#
#     Returns
#     -------
#     result
#     """
#     # initial parameters guess
#     peak_list = pick(x, y, fwhm=[0, 1], asymmetry=True)
#     mu = peak_list["loc"]
#     # correct peaks fwhm in cases of overlap
#     fwhm = guess_fwhm(peak_list["fwhm"], peak_list["fwhm overlap"])
#     sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
#     amp = peak_list["height"]
#     guess = np.vstack((mu, sigma, amp)).T.flatten()
#     popt, pcov = curve_fit(utils.gaussian_mixture, x, y, p0=guess)
#     return popt, pcov
#
#
# def overlap_groups(overlap):
#     """
#     Group contiguous overlapped peaks.
#
#     Parameters
#     ----------
#     overlap: list[bool].
#         returned from pick.
#
#     Returns
#     -------
#     groups: list[int].
#         List of peaks with overlap.
#     """
#     groups = list()
#     group = list()
#     for i, has_overlap in enumerate(overlap):
#         group.append(i)
#         if not has_overlap:
#             groups.append(group)
#             group = list()
#     if group:
#         groups.append(group)
#     return groups
#
#
# def guess_fwhm(fwhm, overlap):
#     groups = overlap_groups(overlap)
#     guess = np.zeros_like(fwhm)
#     for group in groups:
#         guess[group] = fwhm[group].mean()
#     return guess
