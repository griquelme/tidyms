"""
functions and objects used to detect peaks.
"""

import numpy as np
from scipy.signal import _peak_finding
from scipy.signal import find_peaks
from scipy.signal import argrelmax
from scipy.signal.wavelets import ricker, cwt
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.stats import median_absolute_deviation as mad
from typing import Tuple, List, Optional, Union, Callable, Dict
from .utils import _find_closest_sorted
_estimator_type = Dict[str, Callable]


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

    """

    def __init__(self, loc, scale, start, end):
        self.loc = loc
        self.scale = scale
        self.start = start
        self.end = end

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
        return PeakLocation(loc, self.scale, start, end)

    def get_loc(self, x: np.ndarray, y: np.ndarray, baseline: np.ndarray):
        """
        Estimate the location of the peak apex using a weighted average of x.
        """
        weights = y[self.start:self.end] - baseline[self.start:self.end]
        weights[weights < 0] = 0
        loc = np.average(x[self.start:self.end], weights=weights)
        return loc

    def get_extension(self, x: np.ndarray):
        width = x[self.end] - x[self.start]
        return width

    def get_area(self, x: np.ndarray, y: np.ndarray,
                 baseline: Optional[np.ndarray] = None):
        """
        Estimates the peak area.
        """
        if baseline is None:
            baseline_area = 0
        else:
            baseline_area = trapz(baseline[self.start:(self.end + 1)],
                                  x[self.start:(self.end + 1)])
        total_area = trapz(y[self.start:(self.end + 1)],
                           x[self.start:(self.end + 1)])
        peak_area = max(0, total_area - baseline_area)
        return peak_area

    def __repr__(self):
        str_repr = "PeakLocation(loc={}, start={}, end={})"
        str_repr = str_repr.format(self.loc, self.start, self.end)
        return str_repr


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
    # min_index = find_peaks(-cwt_array[scale_index, :])[0]
    min_index = argrelmax(-cwt_array[scale_index, :])[0]
    # 0 and 1 are included in case there's no minimum
    min_index = np.hstack((0, min_index, cwt_array.shape[1] - 1))
    ext = np.searchsorted(min_index, peak_index)
    extension = min_index[ext - 1], min_index[ext]
    return extension


def _is_uniform_sampled(x: np.ndarray) -> bool:
    """
    check if the distance between points is constant
    """
    dx = np.diff(x)
    is_uniform = (dx == dx[0]).all()
    return is_uniform


def _resample_data(x: np.ndarray, y: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data to uniform sampling. Helper function to detect_peaks

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

    if _is_uniform_sampled(x):
        xu, yu = x, y
    else:
        min_dist = np.diff(x).min()
        x_max = x.max()
        x_min = x.min()
        n_interp = int((x_max - x_min) / min_dist) + 1
        xu = np.linspace(x_min, x_max, n_interp)
        interpolator = interp1d(x, y)
        yu = interpolator(xu)
    return xu, yu


def _process_ridge_lines(cwt_array: np.ndarray, y_peaks: np.ndarray,
                         ridge_lines: List[Tuple[np.ndarray, np.ndarray]],
                         min_length, max_distance) -> List[PeakLocation]:
    """
    find peaks in each ridge line. A peak is a local maximum in a ridge line
    If more than one maximum is found in a ridge line, the maximum with found
    at the lowest scale is selected.

    Parameters
    ----------
    cwt_array: numpy.ndarray
        CWT using y and widths, obtained using scipy.signal.wavelets.cwt
    ridge_lines: list[(numpy.ndarray, numpy.ndarray)]
        Ridge lines from cwt_array computed using
        scipy.signal._peak_finding._identify_ridge_lines.
    min_length: int
        minimum length of a ridge line

    Returns
    -------
    list[PeakLocation]
    """

    if min_length is None:
        min_length = np.ceil(cwt_array.shape[0] / 8)

    peaks = list()

    for k, (row_ind, col_ind) in enumerate(ridge_lines):

        # sometimes the function to detect ridge lines repeats rows values.
        # TODO : implement a custom function for ridge line detection.
        row_ind, unique_ind = np.unique(row_ind, return_index=True)
        col_ind = col_ind[unique_ind]

        # min_length check
        if len(row_ind) < min_length:
            continue

        # snr check and width check for all scale max
        line = cwt_array[row_ind, col_ind]
        # max_index = find_peaks(line)[0]
        max_index = argrelmax(line)[0]
        if max_index.size == 0:
            max_index = np.array([line.size - 1])

        for ind in max_index:
            scale_index, peak_index = row_ind[ind], col_ind[ind]
            start, end = _find_peak_extension(cwt_array, scale_index,
                                              peak_index)
            temp_peak = PeakLocation(peak_index, scale_index, start, end)
            temp_peak = _validate_peak(temp_peak, y_peaks, col_ind,
                                       max_distance[0])
            if temp_peak is not None:
                peaks.append(temp_peak)
                break
    return peaks


def baseline_noise_estimation(y: np.ndarray) -> Tuple[np.ndarray, float]:
    # Noise estimation
    # ----------------
    # if y = s + b + e
    # where s is the peak signal, b is a baseline and e is an error term.
    # some assumptions:
    # 1. s is symmetric. This ensures that the cumulative sum of the
    #     difference is of the peak is ~ 0.
    # 2. e ~ N(0, sigma) iid.
    # 3. The derivative of b, |db/dx| is small. in particular , for two co
    # consecutive points |b[n + 1] - b[n]| << sigma
    #
    # From this we can say that for two consecutive points the following
    # approximation is valid:
    #
    #  dy[n] = y[n + 1] - y[n] ~ s[n + 1] - s[n] + e
    #
    # If there's no peak signal, then:
    #
    # dy[n] ~= e ~ N(0, sqrt(2) * sigma)
    #
    # (The sqrt(2) term comes from adding two independent normal random
    # variables with std = sigma.
    # To remove zones with peaks we use an iterative approach, where we remove
    # the higher 90th percentile of the signal. The noise is computed as the std
    # of the remaining values from dy. The MAD is used as a robust estimator of
    # std. Using this noise value, we find baseline points and using these
    # points we compute a new noise value using the dy values. If the difference
    # is greater than 50 percent, the procedure is repeated, but now using
    # the higher 80th percentile of the signal...
    #
    # Baseline estimation
    # -------------------
    # The points where dy is smaller than three times the noise are considered
    # as baseline. The baseline is then interpolated in the peak zones.

    quantiles = np.linspace(0.1, 0.9, 9)[::-1]
    dy = np.diff(y)
    dy_abs = np.abs(dy)
    noise_found = False
    noise = 0

    for q in quantiles:

        # initial noise estimation
        threshold = np.quantile(y, q)
        noise = mad(dy[y[1:] < threshold]) / np.sqrt(2)

        # prevent noise equal to zero or nan
        if np.isnan(noise) or np.isclose(noise, 0):
            noise = np.finfo(np.float64).eps

        # detect baseline points
        baseline_mask = (dy_abs <= (3 * noise)) & (y[1:] < threshold)
        baseline_index = np.where(baseline_mask)[0] + 1

        # compare the noise value obtained with the index selected as baseline
        new_noise = mad(dy[baseline_index - 1]) / np.sqrt(2)
        dnoise = np.abs(new_noise - noise) / noise

        # checks the difference with the new noise value
        if dnoise <= 0.5:
            noise = new_noise
            baseline_mask = (dy_abs <= (3 * noise)) & (y[1:] < threshold)
            baseline_index = np.where(baseline_mask)[0] + 1
            if baseline_index.size:
                baseline_index = _remove_non_consecutive_index(baseline_index)
                noise_found = True
                break

    # fallback to the noise value using q = 0.25 if there was no convergence
    if (not noise_found) or (baseline_index.size == 0):
        threshold = np.quantile(y, 0.5)
        noise = mad(dy[y[1:] < threshold]) / np.sqrt(2)
        baseline_index = np.where(dy_abs <= (3 * noise))[0] + 1
        # if baseline is still empty, return a constant baseline
        if baseline_index.size == 0:
            noise = max(np.finfo(np.float64).eps, mad(y[y < threshold]))
            baseline = np.ones_like(y) * y.min()
            return baseline, noise

    # append first and last elements if they are not part of the baseline
    # this is a necessary step before interpolation.
    baseline_x, baseline_y = _get_baseline_points(y, baseline_index)

    # interpolate baseline to have the same size as y
    interpolator = interp1d(baseline_x, baseline_y)
    baseline = interpolator(np.arange(y.size))
    return baseline, noise


def _get_baseline_points(y, ind):
    """
    adds first and last points to the baseline.

    """
    if ind[0] > 0:
        start_x = 0
        start_y = y[ind[0]]
    else:
        start_x = np.array([], dtype=int)
        start_y = np.array([])

    if ind[-1] < (y.size - 1):
        end_x = y.size - 1
        end_y = y[ind[-1]]
    else:
        end_x = np.array([], dtype=int)
        end_y = np.array([])
    baseline_x = np.hstack([start_x, ind, end_x])
    baseline_y = np.hstack([start_y, y[ind], end_y])
    return baseline_x, baseline_y


def _remove_non_consecutive_index(x):
    """
    Remove non consecutive values from x. Ignores first and last elements.

    Parameters
    ----------
    x: sorted 1-D array of integers.

    Returns
    -------
    x_consecutive: array
    """
    prev_diff = (x - np.roll(x, 1)) == 1
    next_diff = (np.roll(x, -1) - x) == 1
    is_consecutive = prev_diff | next_diff
    is_consecutive[[0, -1]] = True
    return x[is_consecutive]


def _estimate_params(x: np.ndarray, y: np.ndarray, widths: np.ndarray,
                     w: np.ndarray, peaks: List[PeakLocation], snr: float,
                     min_width: float, max_width: float,
                     estimators: Union[str, _estimator_type],
                     baseline: Optional[np.ndarray] = None,
                     noise: Optional[np.ndarray] = None,
                     append_empty_params: bool = False):

    filtered_peaks = list()
    peak_params = list()

    if noise is None:
        noise = np.nan

    if baseline is None:
        baseline = np.zeros_like(y)

    for peak in peaks:
        if estimators == "default":
            peak_width = peak.get_extension(x)
            peak_height = max(0, y[peak.loc] - baseline[peak.loc])
            peak_snr = abs(peak_height) / noise
            # if peak_snr > snr:
            #     print("snr:", peak_snr)
            #     print("height:", peak_height)
            #     print("width:", peak_width)
            #     print("loc: ", x[peak.loc])
        elif estimators == "cwt":
            peak_width = widths[peak.scale] * (x[1] - x[0])
            peak_height = y[peak.loc]
            peak_snr = abs(w[peak.scale, peak.loc] / w[0, peak.loc])
        else:
            peak_width = estimators["width"](x, y, peak, baseline)
            peak_height = estimators["height"](x, y, peak, baseline)
            peak_snr = abs(peak_height) / noise

        is_valid_peak = ((peak_snr >= snr) and (peak_width >= min_width) and
                         (peak_width <= max_width))
        # if the peak is valid, then the area is computed
        if is_valid_peak:
            if estimators == "default":
                peak_area = peak.get_area(x, y, baseline)
                peak_loc = peak.get_loc(x, y, baseline)
            elif estimators == "cwt":
                peak_area = w[peak.scale, peak.loc]
                peak_loc = x[peak.loc]
            else:
                peak_area = estimators["area"](x, y, peak, baseline)
                peak_loc = estimators["loc"](x, y, peak, baseline)
            peak_param = {"area": peak_area, "loc": peak_loc,
                          "height": peak_height, "width": peak_width}
            filtered_peaks.append(peak)
            peak_params.append(peak_param)
        else:
            if append_empty_params:
                empty_dict = dict()
                peak_params.append(empty_dict)
    return filtered_peaks, peak_params


def detect_peaks(x: np.ndarray, y: np.ndarray, widths: np.ndarray,
                 min_length: int = 5, max_distance: int = 2,
                 gap_threshold: int = 1, snr: float = 3, min_width: float = 5,
                 max_width: float = 60,
                 estimators: Union[str, _estimator_type] = "default"):
    r"""
    Find peaks in a 1D signal.

    Peaks are detected using a modified version of the algorithm described in
    [1].

    Parameters
    ----------
    x : sorted array
    y : array of intensities
    widths : array
        Array of widths, in x units. Used as scales to build the wavelet
        array.
    min_length : int
        Minimum number of points in a ridge line.
    max_distance : float
        Maximum x distance between consecutive points in a ridge line, in x
        units.
    gap_threshold : int
        Maximum number of consecutive missing peaks in a ridge line.
    snr : positive number
        Signal-to-noise- ratio used to filter peaks. Defined as follows:

        .. math::

            SNR = \frac{peak height - baseline}{noise}

    min_width : positive number
        Minimum width of the peaks
    max_width : positive number
        Maximum width of the peaks
    estimators : str or dict
        How to estimate baseline, noise, peak height, peak width, peak area and
        peak location. If `estimators` is 'cwt', parameters are computed as
        described in [1]. Check the Notes to see how estimations in 'default'
        mode are computed or how custom estimators can be used.

    Returns
    -------
    peaks : List of PeakLocation
    params : dict of peak parameters

    Notes
    -----
    Peaks are detected using the CWT algorithm described in [DP06]. The optimum
    scale where each peak is detected is the local maximum at the lowest scale
    in the ridge line. If no local maximum was found, the scale with the maximum
    coefficient is chosen. After finding a peak, the extension of the peak
    is found by finding the nearest local minimum at both sides of the peak,
    using the wavelet coefficients with the best scale. A peak is represented
    then by three indices specifying the peak location, peak start and peak end.
    These three values, together with baseline and noise estimations are used
    to estimate peak parameters. If the mode used is 'default`, the peak
    parameters are defined as follows:

        baseline :
            A baseline is built using y values where no peak was detected. These
            values are interpolated to build the baseline.
        noise :
            The noise is computed as the standard deviation of the values used
            to build the baseline. To obtain a robust estimation, the median
            absolute deviation of the baseline is used.
        height :
            The height of a peak is computed as the difference between the
            y value baseline value at the peak location
        snr :
            The quotient between the height of the peak and the noise.
        area :
            Area of the peak obtained by integration between the start and
            the end of the peak. The area of the baseline is subtracted.
        width :
            The peak width is computed as the peak extension, that is, the
            difference between the end and the start of the peak.

    After computing these parameters, peaks are filtered based on SNR and peak
    width. Peak overlap between the filtered peaks is analyzed then. Two
    peaks are overlapping if there is superposition in their peak extensions.
    Overlapping peaks are flagged, their extension corrected and  corrected peak
    parameters are computed again.

    Custom estimators can be used for noise, baseline, peak height, peak
    location, peak width and peak area:

    .. code-block:: python

            estimators = {"baseline": baseline_func, "noise": noise_func,
                          "height": height_func,  "loc": loc_func,
                          "width": width_func, "area": area_func}

            # x and y are the same array used in the function
            # peaks is a list of PeakLocation instances
            # peak is a single PeakLocation instance

            # baseline must have the same size as x and y
            baseline = baseline_func(x, y, peaks)
            # noise is a positive number
            noise = noise_func(x, y, peaks)
            # peak_parameters are all positive numbers
            # (area and height can be zero)
            height = height_func(x, y, peak, baseline)
            area = area_func(x, y, peak, baseline)
            width = width_func(x, y, peak, baseline)
            loc = loc_func(x, y, peak, baseline)

    References
    ----------

    .. [DP06] Pan Du, Warren A. Kibbe, Simon M. Lin, Improved peak detection in
        mass spectrum by incorporating continuous wavelet transform-based
        pattern matching, Bioinformatics, Volume 22, Issue 17, 1 September 2006,
        Pages 2059–2065, https://doi.org/10.1093/bioinformatics/btl355

    """

    # Convert to uniform sampling
    xu, yu = _resample_data(x, y)

    # convert parameters to number of points
    widths, max_distance = \
        _convert_to_points(xu, widths, max_distance)

    # detect peaks in the ridge lines
    w = cwt(yu, ricker, widths)
    ridge_lines = \
        _peak_finding._identify_ridge_lines(w, max_distance, gap_threshold)
    # y_peaks are the local maxima of y and are used to validate peaks
    # y_peaks = find_peaks(yu)[0]
    y_peaks = argrelmax(yu, order=2)[0]
    peaks = _process_ridge_lines(w, y_peaks, ridge_lines, min_length,
                                 max_distance)

    # baseline and noise estimation
    if estimators == "default":
        baseline, noise = baseline_noise_estimation(yu)
    elif estimators == "cwt":
        baseline, noise = None, None
    else:
        baseline = estimators["baseline"](xu, yu, peaks)
        noise = estimators["noise"](xu, yu, peaks)

    # peak filtering and parameter estimation
    peaks, params = \
        _estimate_params(xu, yu, widths, w, peaks, snr, min_width, max_width,
                         estimators, baseline=baseline, noise=noise)

    # sort peaks based on location
    sorted_index = sorted(range(len(peaks)), key=lambda s: peaks[s].loc)
    peaks = [peaks[k] for k in sorted_index]
    params = [params[k] for k in sorted_index]

    # find and correct overlap between consecutive peaks:
    overlap_index = list()
    rm_index = list()
    for k in range(len(peaks) - 1):
        left, right = peaks[k], peaks[k + 1]
        is_same_peak = right.loc == left.loc
        merge = (right.loc - left.loc) <= max_distance[0]
        has_overlap = left.end > right.start
        if is_same_peak:
            rm_index.append(k + (left.scale < right.scale))
        elif merge:
            rm_index.append(k)
            right.start = left.start
            right.loc = (left.loc + right.loc) // 2
        elif has_overlap:
            _fix_peak_extension(left, right, yu)
            overlap_index.extend([k, k + 1])
        # remove invalid peaks after the extension was fixed
        if yu[left.loc] < max(yu[left.start], yu[left.end]):
            rm_index.append(k)

    overlap_peaks = [peaks[x] for x in overlap_index]

    # if there are peaks with overlap, then compute again peak parameters after
    # correction
    if overlap_index:
        _, overlap_params = \
            _estimate_params(xu, yu, widths, w, overlap_peaks, snr, min_width,
                             max_width, estimators, baseline=baseline,
                             noise=noise, append_empty_params=True)
        # replace corrected values in params:
        for k, param in zip(overlap_index, overlap_params):
            if len(param):
                params[k] = param
            else:
                rm_index.append(k)

    # remove invalid peaks and back scale peaks
    peaks = [p.rescale(xu, x) for (k, p) in enumerate(peaks)
             if k not in rm_index]
    params = [p for k, p in enumerate(params) if (len(p) and k not in rm_index)]

    return peaks, params


def _fix_peak_extension(left: PeakLocation, right: PeakLocation,
                        y: np.ndarray):
    """
    Correct the peak extension and location when there's peak overlap.
    aux function for _detect_peaks
    """

    # There are four possible cases of overlap
    # 1 - left.loc < right.start and left.end < right.loc
    # 2 - left.loc < right.start and left.end > right.loc
    # 3 - left.loc > right.start and left.end < right.loc
    # 4 - left.loc > right.start and left.end > right.loc
    # in the case 1, a middle point is computed as the boundary between peaks.
    # In the case 2 and 3, one of the peaks is assumed to have the correct
    # boundary and the value of the other peak is corrected.
    # The case 4 only can occur if one of the peaks is invalid. This is because
    # two local maximum cannot be consecutive without a minimum between them.

    # values used for the case 4

    if left.loc < right.start:
        if left.end < right.loc:
            # TODO : maybe is a better alternative to use the min
            #   between left.loc and right.loc
            middle = (left.end + right.start) // 2
            left.end = middle
            right.start = middle
        else:
            left.end = right.start
    else:
        if left.end < right.loc:
            right.start = left.end
        else:
            # one of the peaks is invalid and must be removed
            # the least intense peak is assumed to be invalid
            if y[left.loc] > y[right.loc]:
                # dummy values to remove the peak
                right.start = right.loc - 1
                right.end = right.loc + 1
            else:
                left.start = left.loc - 1
                left.end = left.loc + 1


def _convert_to_points(xu: np.ndarray, widths: np.ndarray,
                       max_distance: float):
    """
    Convert the parameters for detect_peaks function from x units to point
    units.
    """
    dxu = xu[1] - xu[0]
    widths = widths / dxu
    if widths[0] > 1:
        widths = np.hstack([1, widths])
    max_distance = np.rint(max_distance / dxu)
    max_distance = np.ones_like(widths, dtype=int) * max_distance
    return widths, max_distance


def _validate_peak(peak: PeakLocation, y_peaks: np.ndarray, rl_col: np.ndarray,
                   max_distance: int):
    """
    Check if there is a local maximum in y close to the found peak.
    Auxiliary function to _process_ridgelines.
    """
    start, end = np.searchsorted(y_peaks, [peak.start, peak.end])
    ext_peaks = y_peaks[start:end]
    # check if there's at least a local maximum in the defined peak extension
    if ext_peaks.size:
        # check is close in any scale
        crop_rl_col = rl_col[:(peak.scale + 1)]
        closest_scale = _find_closest_sorted(ext_peaks, crop_rl_col)
        peak_dist = np.abs(ext_peaks[closest_scale] - crop_rl_col)
        min_dist = peak_dist.min()
        if min_dist <= max_distance:
            new_loc = crop_rl_col[np.argmin(peak_dist)]
            peak.loc = new_loc
        else:
            peak = None
    else:
        peak = None
    return peak


def find_centroids(x: np.ndarray, y: np.ndarray, snr: float,
                   min_distance: float):

    r"""
    Convert peaks to centroid mode.


    Parameters
    ----------
    x : sorted array
    y : array
    snr : positive number
        Signal to noise ratio
    min_distance : positive number
        Minimum distance between consecutive peaks.

    Returns
    -------
    x_centroid : the centroid of each peak.
    y_area : area of each peak.
    peak_index : index of each centroid in the original x array.

    Notes
    -----
    Each peak is found a a local maximum in the data. To remove peaks, the
    SNR of the peak is computed as follows:

    .. math::

        SNR = \frac{peak intensity - baseline}{noise}

    Peak boundaries are
    defined as the closest local minimum to the peak. The area of each peak
    is the area between the boundaries, after subtracting the baseline.

    See Also
    --------
    baseline_noise_estimation

    """

    # baseline and noise estimation
    baseline, noise = baseline_noise_estimation(y)
    peak_index = argrelmax(y, order=2)[0]

    # filter peaks by baseline and noise
    yb = y - baseline
    yb[yb < 0] = 0
    peak_index = peak_index[(yb[peak_index] / noise) > snr]

    # find peak boundaries: peak boundaries are the
    # baseline points closest at each side of a peak
    # bl_index = np.where(yb <= 0)[0]
    y_min = find_peaks(-y)[0]
    boundaries = np.searchsorted(y_min, peak_index)
    boundaries[boundaries < 1] = 1
    boundaries[boundaries >= y_min.size] = y_min.size - 1
    start = y_min[boundaries - 1]
    end = y_min[boundaries] + 1

    # merge close peaks
    merge_index = np.where(np.diff(x[peak_index]) < min_distance)[0]
    peak_index[merge_index] = (peak_index[merge_index] +
                               peak_index[merge_index + 1]) // 2
    end[merge_index] = end[merge_index + 1]
    rm_mask = np.ones(shape=peak_index.size, dtype=bool)
    rm_mask[merge_index + 1] = False
    peak_index = peak_index[rm_mask]
    start = start[rm_mask]
    end = end[rm_mask]

    # find peak area and centroid for each peak
    x_centroid = np.zeros(peak_index.size)
    y_area = np.zeros(peak_index.size)
    for k, ks, ke, in zip(range(boundaries.size), start, end):
        y_area[k] = trapz(yb[ks:ke], x[ks:ke])
        try:
            cent = np.average(x[ks:ke], weights=yb[ks:ke])
        except ZeroDivisionError:
            cent = 0
        x_centroid[k] = cent

    # if the distance between two peaks is smaller than min_dist
    # conserve only the peak with the greatest area
    dx_centroid = np.diff(x_centroid)
    dy_area = np.diff(y_area)
    rm_index = np.where(dx_centroid < min_distance)[0]
    rm_index += dy_area[rm_index] < 0
    rm_mask = np.ones(shape=x_centroid.size, dtype=bool)
    rm_mask[rm_index] = False
    rm_mask[x_centroid <= 0] = False    # remove values with zero weight
    x_centroid = x_centroid[rm_mask]
    y_area = y_area[rm_mask]
    peak_index = peak_index[rm_mask]

    return x_centroid, y_area, peak_index
