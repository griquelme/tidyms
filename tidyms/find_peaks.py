import numpy as np
from .utils import gauss
from scipy.signal import find_peaks
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from scipy.special import erfc
from typing import List, Optional
from scipy.stats import median_absolute_deviation as mad


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

    """

    def __init__(self, loc: int, start: int, end: int):
        self.loc = loc
        self.start = start
        self.end = end

    def __repr__(self):
        str_repr = "PeakLocation(loc={}, start={}, end={})"
        str_repr = str_repr.format(self.loc, self.start, self.end)
        return str_repr

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

    def get_params(self, x, y, baseline):
        params = {"height": self.get_height(y, baseline),
                  "area": self.get_area(x, y, baseline),
                  "loc": self.get_loc(x, y, baseline),
                  "width": self.get_width(x, y, baseline)}
        return params


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
    # second difference filter using a rational transfer function
    d2x = lfilter([1, -2, 1], 1, x)[2:]
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
    if slice_size < min_slice_size:
        slice_size = min_slice_size
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


def estimate_baseline(x, noise):
    """
    Estimate the baseline of a signal.

    Parameters
    ----------
    x : array
    noise : array obtained with _estimate noise

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

    return baseline


def _find_baseline_points(x, noise):
    # x is smoothed to reduce the number of intervals in noisy signals
    smoothed = _smooth(x)
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
                                           noise_threshold, ext, local_min)
    return baseline_index


def _smooth(x):
    half_x = x.size // 2 + x.size % 2
    smoothed = np.convolve(x, gauss(np.arange(x.size), half_x, 1, 1))
    smoothed = smoothed[half_x:half_x + x.size] / np.sqrt(2)
    return smoothed


def _estimate_noise_probability(noise, smoothed, ext):
    # compute the difference at the start and end of each monotonic interval
    ext_reshape = np.vstack([ext,
                             np.roll(ext, -1)]).T.reshape(ext.size * 2)[:-2]
    delta = (smoothed[np.roll(ext, -1)] - smoothed[ext])[:-1]
    # noise level in each interval
    delta_noise = np.sqrt(np.add.reduceat(noise ** 2, ext_reshape)[::2])
    noise_probability = erfc(np.abs(delta) / delta_noise)
    return noise_probability


def _build_baseline_index(x, noise, noise_probability, noise_threshold,
                          ext, local_min):
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
    baseline_index = np.hstack((baseline_index,
                                local_min[
                                    x[local_min] < (3 * noise[local_min])]))
    baseline_index = np.sort(baseline_index)
    return baseline_index


def detect_peaks(x: np.ndarray, y: np.ndarray, min_snr: float = 10,
                 min_prominence: float = 0.1, min_distance: float = 3,
                 min_width: Optional[float] = None,
                 max_width: Optional[float] = None,
                 estimators: Optional[dict] = None,
                 noise_estimator_params: Optional[dict] = None):
    r"""
    Finds peaks in a 1D signal.

    Parameters
    ----------
    x : array
    y : array
    min_snr : positive number
        Minimum signal to noise ratio
    min_prominence : number between 0 and 1
        Minimum signal prominence, expressed as a fraction of the peak height.
    min_distance : positive number
        Minimum distance between consecutive peaks, in `x` units
    min_width : positive number, optional
        Minimum width of a peak, in x units.
    max_width : positive number, optional
        Maximum width of a peak, in x units.
    estimators : dict, optional
        A dictionary of strings to callables, used to estimate custom parameters
        on a peak. See notes for a detailed explanation.
    noise_estimator_params : dict, optional
        key, value parameters to pass to the noise estimator

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
    if min_width is None:
        min_width = 0

    if max_width is None:
        max_width = x.max()

    if noise_estimator_params is None:
        noise_estimator_params = dict()

    peaks = find_peaks(y)[0]

    # baseline and noise estimation
    noise = estimate_noise(y, **noise_estimator_params)
    baseline = estimate_baseline(y, noise)
    snr = (y - baseline) / noise

    # snr filter
    peaks = peaks[snr[peaks] >= min_snr]

    # find peak extension: the peak extension is a pair of indices that
    # correspond to the beginning and end of a peak. A first extension is
    # obtained by finding the closest baseline point
    baseline_index = np.where(np.isclose(y, baseline))[0]
    ext_index = np.searchsorted(baseline_index, peaks)
    left = baseline_index[ext_index - 1]
    right = baseline_index[ext_index]

    # peak remove close peaks using the min_distance parameter. If two peaks
    # are close then the least intense is removed
    peaks, left, right = _merge_min_distance(x, y, peaks, left, right,
                                             min_distance)

    peaks = [PeakLocation(l, s, e) for l, s, e in zip(peaks, left, right)]
    peaks = _filter_prominence(y, noise, baseline, peaks, min_prominence,
                               min_snr)
    params = _estimate_peak_parameters(x, y, baseline, noise, peaks, estimators)
    peaks, params = _filter_width(peaks, params, min_width, max_width)

    return peaks, params


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


def _filter_prominence(y: np.array, noise: np.array, baseline: np.array,
                       peaks: List[PeakLocation], min_prominence: float,
                       min_snr: float):

    # peaks are analyzed from left to right
    # a dummy peak location object is added at the end to analyze all the peaks
    # inside the loop
    peaks.append(PeakLocation(y.size - 1, y.size - 1, y.size - 1))
    new_peak_list = list()
    last_valid_peak = None
    left_peak = peaks[0]
    is_last_valid_overlap = False
    for right_peak in peaks[1:]:
        is_overlap = left_peak.end > right_peak.start
        _fix_overlap(y, left_peak, right_peak)
        # check if the prominence is greater than min_prominence,
        # in SNR units, if there's no overlap, the SNR of the peak is used
        # instead
        peak_height = left_peak.get_height(y, baseline)
        left_prom = y[left_peak.loc] - y[left_peak.start]
        right_prom = y[left_peak.loc] - y[left_peak.end]
        is_valid_snr = min(left_prom / noise[left_peak.start],
                           right_prom / noise[left_peak.end]) > min_snr
        is_valid_prom = ((min(left_prom, right_prom) / peak_height) >
                         min_prominence)
        is_valid_peak = is_valid_prom and is_valid_snr

        if is_valid_peak:
            if is_last_valid_overlap:
                _fix_overlap(y, last_valid_peak, left_peak)
            last_valid_peak = left_peak
            is_last_valid_overlap = is_overlap
            new_peak_list.append(left_peak)
        elif is_last_valid_overlap:
            _merge_peaks(last_valid_peak, left_peak)
            # fix_overlap2(y, last_valid_peak, right_peak)
        elif is_overlap:
            _merge_peaks(right_peak, left_peak)
            is_last_valid_overlap = False
        left_peak = right_peak
    return new_peak_list


def _merge_peaks(p1: PeakLocation, p2: PeakLocation):
    p1.start = min(p1.start, p2.start)
    p1.end = max(p1.end, p2.end)


def _filter_width(peaks: List[PeakLocation], params: List[dict],
                  min_width: float, max_width: float):
    new_params = list()
    new_peaks = list()
    for k in range(len(peaks)):
        if _is_valid_width(params[k]["width"], min_width, max_width):
            new_peaks.append(peaks[k])
            new_params.append(params[k])
    return new_peaks, new_params


def _is_valid_width(w: float, min_width: float, max_width: float) -> bool:
    return (w >= min_width) and (w <= max_width)


def _estimate_peak_parameters(x, y, baseline, noise, peaks, estimators):
    params = list()

    if estimators is None:
        estimators = dict()

    for p in peaks:
        p_params = p.get_params(x, y, baseline)
        for param, estimator in estimators.items():
            p_params[param] = estimator(x, y, noise, baseline, p)
        params.append(p_params)
    return params
