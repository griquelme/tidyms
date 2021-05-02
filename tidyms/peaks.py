"""
functions and objects used to detect peaks.

Objects
-------

- PeakLocation : Stores peak location and extension. Computes peak parameters.

Functions
---------
- estimate_noise(x) : Estimates noise level in a 1D signal
- estimate_baseline(x, noise) : Estimates the baseline in a 1D signal
- detect_peaks(x, y) : Detects peaks in a 1D signal
- detect_peaks_cwt(x, y) : Detects peaks in a 1D signal using the CWT algorithm
- find_centroids(x, y) : Computes the centroid and area of peaks in a 1D signal.

"""

import numpy as np
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.signal import _peak_finding
from scipy.signal import find_peaks
from scipy.signal import argrelmax
from scipy.signal.wavelets import ricker, cwt
from scipy.signal.windows import gaussian
from scipy.special import erfc
from scipy.stats import median_absolute_deviation as mad
from typing import Tuple, List, Optional, Union, Callable, Dict
from .utils import _find_closest_sorted
_estimator_type = Dict[str, Callable]


class PeakLocation:
    """
    Holds peak information for peak picking and methods for peak parameter
    estimation.

    Attributes
    ----------
    loc: int
        index where the apex of the peak is located.
    start: int
        index where the peak begins.
    end: int
        index where the peak ends
    scale: int, optional
        scale at which the peak was found in the CWT algorithm

    """

    def __init__(self, loc: int, start: int, end: int,
                 scale: Optional[int] = None):
        self.loc = loc
        self.start = start
        self.end = end
        self.scale = scale

    def __repr__(self):
        str_repr = "PeakLocation(loc={}, start={}, end={})"
        str_repr = str_repr.format(self.loc, self.start, self.end)
        return str_repr

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
        return PeakLocation(loc, start, end, self.scale)

    def get_loc(self, x: np.array, y: np.array, baseline: np.array):
        weights = (y[self.start:(self.end + 1)] -
                   baseline[self.start:(self.end + 1)])
        weights[weights < 0] = 0
        loc = np.abs(np.average(x[self.start:(self.end + 1)],
                                weights=weights))
        return loc

    def get_height(self, y: np.array, baseline: np.array):
        height = y[self.loc] - baseline[self.loc]
        return max(0.0, height)

    def get_area(self, x: np.array, y: np.array, baseline: np.array):
        baseline_corrected = (y[self.start:(self.end + 1)] -
                              baseline[self.start:(self.end + 1)])
        area = trapz(baseline_corrected, x[self.start:(self.end + 1)])
        return max(0.0, area)

    def get_width(self, x, y, baseline):
        height = (y[self.start:(self.end + 1)] -
                  baseline[self.start:(self.end + 1)])
        area = cumtrapz(height, x[self.start:(self.end + 1)])
        relative_area = area / area[-1]
        start, end = self.start + np.searchsorted(relative_area, [0.025, 0.975])
        width = x[end] - x[start]
        return max(0.0, width)

    def get_extension(self, x: np.ndarray):
        width = x[self.end] - x[self.start]
        return width

    def get_prominence(self, y: np.array, baseline: np.array):
        height = self.get_height(y, baseline)
        if height == 0.0:
            prominence = 0.0
        else:
            prominence = min((y[self.loc] - y[self.start]) / height,
                             (y[self.loc] - y[self.end]) / height)
            prominence = max(0, prominence)
        return prominence

    def get_snr(self, y: np.array, noise: np.array, baseline: np.array):
        peak_noise = noise[self.loc]
        if np.isclose(peak_noise, 0):
            snr = np.inf
        else:
            snr = self.get_height(y, baseline) / peak_noise
        return snr

    def get_noise_probability(self, y: np.array, noise: np.array):
        left_noise = (noise[self.start:self.loc + 1] ** 2).sum()
        right_noise = (noise[self.loc:self.end + 1] ** 2).sum()
        left = (y[self.loc] - y[self.start]) / np.sqrt(left_noise * 2)
        right = (y[self.loc] - y[self.end]) / np.sqrt(right_noise * 2)
        probability = erfc((left, right)).max()
        return probability

    def get_params(self, x: np.array, y: np.array, noise: np.array,
                   baseline: np.array):
        params = {"height": self.get_height(y, baseline),
                  "area": self.get_area(x, y, baseline),
                  "loc": self.get_loc(x, y, baseline),
                  "width": self.get_width(x, y, baseline),
                  "snr": self.get_snr(y, noise, baseline)}
        return params


# noise and baseline estimation


def _estimate_local_noise(x: np.ndarray, robust: bool = True):
    r"""
    Estimates noise in a signal. Assumes that the noise is gaussian iid.

    Parameters
    ----------
    x : array
    robust : bool
        If True, estimates the noise using the median absolute deviation. Else
        uses the standard deviation.

    Notes
    -----
    The noise is estimated using the standard deviation of the second finite
    difference of x. If we assume that the signal has three additive components:
    signal, baseline and noise:

    .. math::

        x[n] = s[n] + b[n] + e[n]

    The peak signal, s is a  positive signal that is mostly close to zero except
    where a peak is present. The baseline b is a slow changing positive
    function, and the noise term e, are observations from a normal
    distribution with a mean of zero and standard deviation :math:`\sigma`.
    The second difference of the signal is:

    .. math::
        d2x[n] = (x[n] - x[n-1]) - (x[n-1] - x[n-2]) \\
        d2x[n] =  (s[n] - s[n-1]) - (s[n-1] - s[n-2]) +
            (b[n] - b[n-1]) - (b[n-1] - b[n-2]) +
            (e[n] - e[n-1]) - (e[n-1] - e[n-2])

    We want to consider only the contribution from the noise. Under the
    assumption that the baseline is a slow changing function, we can ignore its
    contribution. If we compute the absolute value of the second difference,
    the highest values are going to have a significant contribution from
    the signal. Using this fact we remove values above the 90th percentile
    of the absolute value of the second difference. The remaining values can
    be described approximately by:

    .. math::

        d2x[n] \approx (e[n] - e[n-1]) - (e[n-1] - e[n-2]

    We can describe then d2x as a normal distribution with mean zero and
    standard deviation :math:`2 \sigma`

    """
    d2x = np.diff(x, n=2)
    sorted_index = np.argsort(np.abs(d2x))
    d2x = d2x[sorted_index]
    # the noise mean should be at least three times smaller than its standard
    # deviation. d2x with high absolute values are removed until this
    # condition is meet. start at 90th percentile and decrease it in each
    # iteration
    n_deviations = 3    # dummy value to initialize the loop
    percentile_counter = 9
    noise_std = 0
    while (n_deviations > 2) and (percentile_counter > 2):
        percentile = percentile_counter * x.size // 10
        if robust:
            noise_std = mad(d2x[:percentile])
            noise_mean = np.median(d2x[:percentile])
        else:
            noise_std = d2x[:percentile].std()
            noise_mean = d2x[:percentile].mean()
        n_deviations = abs(noise_mean / noise_std)
        percentile_counter -= 1
    noise = noise_std / 2
    return noise


def estimate_noise(x: np.ndarray, min_slice_size: int = 200,
                   n_slices: int = 5, robust: bool = True):
    """
    Estimates the noise in a signal.

    Splits x into several slices and estimates the noise assuming that the
    noise is gaussian iid in each slice.

    Parameters
    ----------
    x : array
    min_slice_size : int, optional
        Minimum size of a slice. If the size of x is smaller than this value,
        the noise is estimated using the whole array. The default value is 100.
    n_slices: int, optional
        Number of slices to create. The default value is 5
    robust : bool
        If True, estimates the noise using the median absolute deviation. Else
        uses the standard deviation.
    Returns
    -------
    noise: array with the same size as x

    """
    noise = np.zeros_like(x)
    slice_size = x.size // n_slices
    # if slice_size < min_slice_size:
    #     slice_size = min_slice_size
    start = 0
    while start < x.size:
        end = min(start + slice_size, x.size)
        if (x.size - end) < (min_slice_size // 2):
            # prevent short slices at the end of x
            end = x.size
        slice_noise = _estimate_local_noise(x[start:end], robust=robust)
        noise[start:end] = slice_noise
        start = end
    return noise


def estimate_baseline(x, noise, return_index=False):
    """
    Estimate the baseline of a signal.

    Parameters
    ----------
    x : array
    noise : array obtained with _estimate noise
    return_index : bool
        If True returns the index in x flagged as baseline

    Returns
    -------
    baseline : array

    Notes
    -----
    The baseline is estimated by computing how likely is that the increment
    observed in an interval of the signal can be attributed to the noise. See
    here for more details.

    """
    # find points that only have contribution from the baseline
    baseline_index = _find_baseline_points(x, noise)

    # interpolate baseline points to match x size
    baseline = x[baseline_index]
    interpolator = interp1d(baseline_index, baseline)
    baseline = interpolator(np.arange(x.size))

    # prevents that interpolated points have higher values than x.
    baseline = np.minimum(baseline, x)

    if return_index:
        return baseline, baseline_index
    else:
        return baseline


def _find_baseline_points(x, noise):
    # x is smoothed to reduce the number of intervals in noisy signals
    smoothed = _smooth(x, 1)
    # find local maxima and minima
    local_min = find_peaks(smoothed)[0]
    local_max = find_peaks(-smoothed)[0]
    ext = np.hstack([0, local_max, local_min, x.size - 1])
    ext = np.sort(ext)

    # estimate the probability of an interval to be only noise
    noise_probability = _estimate_noise_probability(noise, smoothed, ext)
    noise_threshold = 0.05
    # creates a vector with indices where baseline was found
    baseline_index = _build_baseline_index(x, noise, noise_probability,
                                           noise_threshold, ext)
    return baseline_index


def _smooth(x, strength):
    g = gaussian(x.size, strength)
    g /= g.sum()
    xs = np.convolve(x, g, "same")
    return xs


def _estimate_noise_probability(noise, smoothed, ext):
    # compute the difference at the start and end of each monotonic interval
    ext_reshape = np.vstack([ext,
                             np.roll(ext, -1)]).T.reshape(ext.size * 2)[:-2]
    delta = (smoothed[np.roll(ext, -1)] - smoothed[ext])[:-1]
    # noise level in each interval
    delta_noise = np.sqrt(np.add.reduceat(noise ** 2, ext_reshape)[::2])
    noise_probability = erfc(np.abs(delta) / (delta_noise * np.sqrt(2)))
    return noise_probability


def _build_baseline_index(x, noise, noise_probability, noise_threshold, ext):
    is_signal = noise_probability < noise_threshold
    is_signal = is_signal | np.roll(is_signal, 1) | np.roll(is_signal, -1)
    baseline_index = list()

    for k in range(ext.size - 1):
        if not is_signal[k]:
            baseline_index.append(np.arange(ext[k], ext[k + 1] + 1))

    if len(baseline_index) == 0:
        baseline_index = np.array([0, x.size - 1], dtype=int)
    else:
        stack = list()
        if baseline_index[0][0] != 0:
            stack.append([0])
        stack.extend(baseline_index)
        if baseline_index[-1][-1] != x.size - 1:
            stack.append([x.size - 1])
        baseline_index = np.hstack(stack)

    # add local minimums such that (x - b) / noise  <= 3
    x_min = find_peaks(-x)[0]
    if x_min.size > 0:
        closest_ind = _find_closest_sorted(baseline_index, x_min)
        closest_ind = baseline_index[closest_ind]
        add_min = (x[x_min] - x[closest_ind]) < noise[x_min]
        add_min = x_min[add_min]
        baseline_index = np.hstack((baseline_index, add_min))
        baseline_index = np.unique(baseline_index)
    return baseline_index


# peak picking algorithm

def detect_peaks(x: np.ndarray, y: np.ndarray, noise: Optional[np.array] = None,
                 baseline: Optional[np.array] = None,
                 peak_probability: float = 0.975,
                 min_prominence: float = 0.1, min_distance: float = 3,
                 smoothing_strength: Optional[float] = None,
                 estimators: Optional[dict] = None,
                 filters: Optional[dict] = None):
    r"""
    Finds peaks in a 1D signal.

    Parameters
    ----------
    x : sorted array
    y : array with the same size as x
    noise: array with the same size as x, optional
        Noise estimation of y. If None, noise is estimated using the default
        method.
    baseline: array with the same size as x, optional
        Baseline estimation of y. If None, the baseline is estimated using the
        default method.
    peak_probability : number between 0 and 1.
        The probability of  observing similar intensity values from a peak
        from noise. Higher values reduce the number of peaks detected.
    min_prominence : number between 0 and 1
        Minimum signal prominence, expressed as a fraction of the peak height.
        Peaks with values under this threshold are removed or merged if they
        are overlapping with another peak.
    min_distance : positive number
        Minimum distance between consecutive peaks, in `x` units. Peaks closer
        than this distance are merged.
    smoothing_strength : positive number, optional
        Width of a gaussian used to smooth the signal using convolution. If
        None, no smoothing is applied.
    estimators : dict, optional
        A dictionary of strings to callables, used to estimate custom parameters
        on a peak. See notes for a detailed explanation.
    filters : dict, optional
        A dictionary of parameter names to a tuple of minimum and maximum
        acceptable values. Peaks with parameters outside this range are removed.
        By default it can filter snr, height, area and width, but it can also
        filter custom parameters added by `estimators`.

    Returns
    -------
    peaks : List[PeakLocation]
    params : dict

    Notes
    -----
    The algorithm for peak finding is as follows:

    1.  first, the noise level is estimated for each point in the signal.
    2.  Using the noise estimation, a baseline level is estimated for the
        signal. This allow us to classify a point in the signal as `baseline`
        or as `peak`.
    3.  Local maxima are found in the signal and are conserved only if they
        are in a region classified as peak.
    4.  For each local maximum the SNR is computed using the following equation:

        .. math::

            SNR = \frac{y - b}{n}

        where y is the local maximum intensity, b is the baseline level at the
        local maximum and n is the noise level at the local maximum. Peaks with
        SNR values lower than `min_snr` are removed.
    5.  Local maxima closer than `min_distance` are merged. The most intense
        value is conserved.
    6.  After this step, PeakLocation objects are built. Each object holds
        information about the peak start, apex and end. For each local maximum
        the start and end of the peaks are found as the closest baseline point
    7.  Peak overlap is detected by evaluating  overlapping start-end segments
        between peaks. Overlap is solved by finding the minimum value
        between overlapping peaks and assigning it as the new boundary.
    8.  Once the overlap is solved, the prominence of each peak is analyzed.
        The prominence of the peak is defined as:

        .. math::

            prominence = \min{\frac{y_{apex} - y_{start}}{y_{apex} - b_{apex}},
                \frac{y_{apex} - y_{end}}{y_{apex} - b_{apex}}

        If the prominence of a peak is lower than `min_prominence` then the peak
        is removed. The prominence is useful to set a level to split or merge
        overlapping peaks.
    9.  After this step, peak parameters are computed. By default, four
        parameters are computed:

        height : difference between the maximum and the baseline level
        area : peak area after subtracting the baseline.
        loc  : peak location, using the weighted average of the peak.
        width : peak width, as the region of the peak that hold the 95 % of the
            total area. This is a robust way to estimate the peak width for
            asymmetric peaks.

        Custom estimators can be added using the estimators parameter.
        Estimators must be a dictionary that maps a string with the estimator
        name to the estimator function. The function must have the following
        signature:

        .. code-block:: python

            "estimator_func(x, y, noise, baseline, peak) -> float"

        Where `x`, `y`, `noise` and `baseline` are 1D numpy arrays with the same
        size and peak is a PeakLocation object

    See Also
    --------
    estimate_noise
    estimate_baseline

    """
    if noise is None:
        noise = estimate_noise(y)

    if smoothing_strength is not None:
        y = _smooth(y, smoothing_strength)

    if baseline is None:
        baseline, baseline_index = estimate_baseline(y, noise,
                                                     return_index=True)
    else:
        baseline_index = np.where(np.abs(y - baseline) < noise)[0]

    peaks = find_peaks(y)[0]
    peaks = np.setdiff1d(peaks, baseline_index, assume_unique=True)
    # snr = (y[peaks] - baseline[peaks]) / noise[peaks]
    # snr filter
    # peaks = peaks[snr >= min_snr]

    # find peak extension: the peak extension is a pair of indices that
    # correspond to the beginning and end of a peak. A first extension is
    # obtained by finding the closest baseline point
    start, end = _find_peak_extension(peaks, baseline_index)

    # remove close peaks using the min_distance parameter. If two peaks
    # are close then the least intense is removed
    peaks, start, end = _merge_min_distance(x, y, peaks, start, end,
                                            min_distance)

    peaks = [PeakLocation(p, s, e) for p, s, e in zip(peaks, start, end)]
    # filter peaks using SNR and prominence. Fix overlap between peaks
    peaks = _filter_invalid_peaks(y, noise, baseline, peaks, peak_probability,
                                  min_prominence)
    params = _estimate_peak_parameters(x, y, baseline, noise, peaks, estimators)
    if filters is not None:
        fill_filter_boundaries(filters)
        peaks, params = _filter_peaks(peaks, params, filters)

    return peaks, params


def _find_peak_extension(peaks: np.array, baseline_index: np.array):
    ext_index = np.searchsorted(baseline_index, peaks)
    start = baseline_index[ext_index - 1]
    end = baseline_index[ext_index]
    return start, end


def _merge_min_distance(x, y, peaks, left, right, min_distance):
    rm_index = np.where(np.diff(x[peaks]) <= min_distance)[0]
    most_intense = y[peaks[rm_index]] > y[peaks[rm_index + 1]]
    rm_mask = np.ones(peaks.size, dtype=bool)
    rm_mask[rm_index + most_intense] = False
    peaks = peaks[rm_mask]
    left = left[rm_mask]
    right = right[rm_mask]
    return peaks, left, right


def _fix_overlap(y: np.array, left_peak: PeakLocation,
                 right_peak: PeakLocation):

    if left_peak.end >= right_peak.start:
        peak_boundary = (np.argmin(y[left_peak.loc:right_peak.loc]) +
                         left_peak.loc)
        left_peak.end = peak_boundary
        right_peak.start = peak_boundary


def _check_is_valid_peak(y: np.array, noise: np.array, baseline: np.array,
                         peak: PeakLocation, min_snr: float,
                         min_prominence: float):
    try:
        is_valid_snr = peak.get_snr(y, noise, baseline) >= min_snr
        prominence = peak.get_prominence(y, baseline)
        height = peak.get_height(y, baseline)
        is_valid_prominence = prominence >= min_prominence
        # check prominence in snr units
        prominence_snr = prominence * height / noise[peak.loc]
        is_valid_prominence_snr = prominence_snr >= min_snr
        is_valid_peak = (is_valid_prominence and is_valid_snr and
                         is_valid_prominence_snr)
    except IndexError:
        is_valid_peak = False
    return is_valid_peak


def _filter_invalid_peaks(y: np.array, noise: np.array, baseline: np.array,
                          peaks: List[PeakLocation], peak_probability: float,
                          min_prominence: float) -> List[PeakLocation]:
    # maximum probability of a peak being considered noise
    p_thresh = 1 - peak_probability
    # a dummy peak appended to analyze all the peaks inside the loop
    peaks.append(PeakLocation(y.size + 1, y.size + 1, y.size + 1))
    new_peak_list = list()
    # initialization of variables used in the loop
    last_valid_peak = None
    left_peak = peaks[0]
    is_last_valid_overlap = False

    # peaks are analyzed from left to right using the following rules:
    #   1.  Overlap between two consecutive peaks is evaluated. If there is
    #       overlap, the boundary between the two is adjusted using the
    #       minimum value between the two peaks
    #   2.  Peak SNR and prominence is evaluated for each peak. If the peak
    #       pass this check, it is appended to the definitive peak list.
    #   3.  If a peak is flagged as invalid, then three cases are possible:
    #       the peak is not overlapping with any other peak, the peak is
    #       overlapping with the next peak or the peak is overlapping with the
    #       previous peak. In the first case, the peak is removed. In the second
    #       case, the peak is merged with the peak to the right (even if there
    #       is overlap with the next peak). In the third case, the peak is
    #       merged with the previous peak.

    for right_peak in peaks[1:]:

        # check and fix overlap with the next peak
        is_overlap = left_peak.end > right_peak.start
        _fix_overlap(y, left_peak, right_peak)

        # check snr and prominence
        # is_valid_peak = _check_is_valid_peak(y, noise, baseline, left_peak,
        #                                      min_snr, min_prominence)
        # if is_valid_peak:
        #     if is_last_valid_overlap:
        #         _fix_overlap(y, last_valid_peak, left_peak)
        #     last_valid_peak = left_peak
        #     is_last_valid_overlap = is_overlap
        #     new_peak_list.append(left_peak)
        # elif is_last_valid_overlap:
        #     _merge_peaks(last_valid_peak, left_peak)
        #     is_last_valid_overlap = is_overlap
        # elif is_overlap:
        #     _merge_peaks(right_peak, left_peak)
        #     is_last_valid_overlap = False
        # left_peak = right_peak
        is_valid_proba = left_peak.get_noise_probability(y, noise) <= p_thresh
        is_valid_prom = left_peak.get_prominence(y, baseline) >= min_prominence
        # sanity check
        is_peak = ((y[left_peak.loc] > y[left_peak.start]) and
                   (y[left_peak.loc] > y[left_peak.end]))
        is_valid_peak = is_valid_prom and is_valid_proba and is_peak
        if is_valid_peak:
            if is_last_valid_overlap:
                _fix_overlap(y, last_valid_peak, left_peak)
            last_valid_peak = left_peak
            is_last_valid_overlap = is_overlap
            new_peak_list.append(left_peak)
        elif is_overlap:
            _merge_peaks(right_peak, left_peak)
            # is_last_valid_overlap = False
        elif is_last_valid_overlap:
            _merge_peaks(last_valid_peak, left_peak)
            is_last_valid_overlap = False
        left_peak = right_peak

    return new_peak_list


def _merge_peaks(p1: PeakLocation, p2: PeakLocation):
    p1.start = min(p1.start, p2.start)
    p1.end = max(p1.end, p2.end)


def _filter_peaks(peak_list: List[PeakLocation], param_list: List[dict],
                  filters: dict):
    filtered_param_list = list()
    filtered_peak_list = list()
    for peak, param in zip(peak_list, param_list):
        is_valid_peak = True
        for p, values in filters.items():
            is_valid_p = _is_valid_range(param[p], *values)
            is_valid_peak = is_valid_peak and is_valid_p
        if is_valid_peak:
            filtered_peak_list.append(peak)
            filtered_param_list.append(param)
    return filtered_peak_list, filtered_param_list


def _is_valid_range(value: float, min_valid: float, max_valid: float) -> bool:
    return (value >= min_valid) and (value <= max_valid)


def _estimate_peak_parameters(x, y, baseline, noise, peaks, estimators):
    params = list()

    if estimators is None:
        estimators = dict()

    for p in peaks:
        p_params = p.get_params(x, y, noise, baseline)
        for param, estimator in estimators.items():
            p_params[param] = estimator(x, y, noise, baseline, p)
        params.append(p_params)
    return params


def fill_filter_boundaries(filter_dict):
    for k in filter_dict:
        lb, ub = filter_dict[k]
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf
        filter_dict[k] = (lb, ub)


# CWT peak picking


def _find_peak_extension_cwt(cwt_array: np.ndarray, scale_index: int,
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
            start, end = _find_peak_extension_cwt(cwt_array, scale_index,
                                                  peak_index)
            temp_peak = PeakLocation(peak_index, start, end, scale_index)
            temp_peak = _validate_peak_cwt(temp_peak, y_peaks, col_ind,
                                           max_distance[0])
            if temp_peak is not None:
                peaks.append(temp_peak)
                break
    return peaks


def _estimate_params_cwt(x: np.ndarray, y: np.ndarray, widths: np.ndarray,
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


def detect_peaks_cwt(x: np.ndarray, y: np.ndarray, widths: np.ndarray,
                     min_length: int = 5, max_distance: int = 2,
                     gap_threshold: int = 1, snr: float = 3,
                     min_width: float = 5, max_width: float = 60,
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
        noise = estimate_noise(yu)
        baseline = estimate_baseline(yu, noise)
        noise = noise.mean()
    elif estimators == "cwt":
        baseline, noise = None, None
    else:
        baseline = estimators["baseline"](xu, yu, peaks)
        noise = estimators["noise"](xu, yu, peaks)

    # peak filtering and parameter estimation
    peaks, params = \
        _estimate_params_cwt(xu, yu, widths, w, peaks, snr, min_width,
                             max_width, estimators, baseline=baseline,
                             noise=noise)

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
            _fix_peak_extension_cwt(left, right, yu)
            overlap_index.extend([k, k + 1])
        # remove invalid peaks after the extension was fixed
        if yu[left.loc] < max(yu[left.start], yu[left.end]):
            rm_index.append(k)

    overlap_peaks = [peaks[x] for x in overlap_index]

    # if there are peaks with overlap, then compute again peak parameters after
    # correction
    if overlap_index:
        _, overlap_params = \
            _estimate_params_cwt(xu, yu, widths, w, overlap_peaks, snr,
                                 min_width, max_width, estimators,
                                 baseline=baseline, noise=noise,
                                 append_empty_params=True)
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


def _fix_peak_extension_cwt(left: PeakLocation, right: PeakLocation,
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


def _validate_peak_cwt(peak: PeakLocation, y_peaks: np.ndarray,
                       rl_col: np.ndarray, max_distance: int):
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


def find_centroids(x: np.ndarray, y: np.ndarray, min_snr: float,
                   min_distance: float):

    r"""
    Convert peaks to centroid mode.

    Parameters
    ----------
    x : sorted array
    y : array
    min_snr : positive number
        Signal to noise ratio
    min_distance : positive number
        Minimum distance between consecutive peaks.

    Returns
    -------
    centroid : sorted array, centroid of the peaks.
    area : array, area of the peaks.
    index : array of int, position of the peaks in y.

    Notes
    -----
    Uses detect_peaks to build a peak list.

    See Also
    --------
    detect_peaks

    """
    peaks, params = detect_peaks(x, y, min_distance=min_distance,
                                 filters={"snr": (min_snr, None)})
    index = np.array([x.loc for x in peaks])
    centroid, area = np.array([[x["loc"], x["area"]] for x in params]).T
    return centroid, area, index
