import numpy as np
from . peaks import PeakLocation
from scipy.signal import find_peaks
from scipy.signal import lfilter
from scipy.stats import median_absolute_deviation as mad
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from typing import List, Optional, Tuple


def _estimate_local_noise(x: np.ndarray):
    r"""
    Estimates noise in a signal. Assumes that the noise is gaussian iid.

    Parameters
    ----------
    x : array

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
    standard deviation :math:`sqrt{6} \sigma`

    """
    # second difference filter using a rational transfer function
    d2x = lfilter([1, -2, 1], 1, x)[2:]
    sorted_index = np.argsort(np.abs(d2x))
    percentile90 = x.size * 9 // 10
    d2x = d2x[sorted_index[:percentile90]]
    # the MAD reduces the effect of outliers with signal contribution.
    noise = mad(d2x) / np.sqrt(6)
    return noise


def _estimate_noise(x: np.ndarray, min_slice_size: int = 100,
                    n_slices: int = 5):
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
        slice_noise = _estimate_local_noise(x[start:end])
        noise[start:end] = slice_noise
        start = end
    return noise


def _estimate_baseline(x, noise):
    """
    Estimate the baseline of a signal.

    Parameters
    ----------
    x : array
    noise : array obtained with _estimate noise

    Returns
    -------
    baseline : array

    """
    # Points in x are classified into signal points or baseline points. Baseline
    # points are then interpolated to build the baseline array.

    # # local maxima and minima in x
    local_min, local_max = _find_local_extrema(x)

    # using local_max and local_min, regions where the x increases and decreases
    # uniformly are defined. The idea behind this if that if the increment and
    # decrement is greater than the expected contribution by the noise, then
    # that region is not considered as baseline.
    # to estimate the noise threshold value, the noise estimation is used and
    # the threshold is estimated as noise * sqrt(length), assuming gaussian iid
    # noise.
    baseline_index = _find_baseline_points(x, noise, local_min, local_max)
    # interpolate baseline points to match x size
    baseline = x[baseline_index]
    interpolator = interp1d(baseline_index, baseline)
    baseline = interpolator(np.arange(x.size))
    baseline = np.minimum(baseline, x)
    return baseline


def _find_local_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima in a 1D signal. Auxiliary function to
    _estimate_baseline.

    Parameters
    ----------
    x : array

    Returns
    -------
    local_min : array
    local_max : array

    """
    # local maxima and minima in x
    local_max = find_peaks(x)[0]
    local_min = find_peaks(-x)[0]

    if local_min.size <= 1:
        local_min = np.array([0, x.size - 1])
        local_max = np.array([np.argmax(x)])
    else:
        # every local maximum must be surrounded by a local minimum
        if local_max[0] < local_min[0]:
            local_max = local_max[1:]
        if local_max[-1] > local_min[-1]:
            local_max = local_max[:-1]
    return local_min, local_max


def _find_baseline_points(x: np.ndarray, noise: np.ndarray,
                          local_min: np.ndarray, local_max: np.ndarray):
    local_max_index = np.searchsorted(local_min, local_max)
    local_max_index[local_max_index >= local_min.size] = local_min.size - 1
    # each region of increment goes from local_min to lmax and each region of
    # decrement goes from lmax to local_min
    increment_start = local_min[local_max_index - 1]
    increment_end = local_max
    decrement_start = local_max
    decrement_end = local_min[local_max_index]
    increment_length = increment_end - increment_start + 1
    decrement_length = decrement_end - decrement_start + 1
    increment_threshold = noise[local_max] * np.sqrt(increment_length)
    decrement_threshold = noise[local_max] * np.sqrt(decrement_length)

    # increment and decrement cumulative sum, relative to the minimum value
    # this value is compared against the thresholds to check if the region
    # can be considered baseline or not
    # x_cumsum = x.cumsum()
    # increment_cumsum = (x_cumsum[increment_end] - x_cumsum[increment_start]
    #                     - x[increment_start] * (increment_length - 1))
    # decrement_cumsum = (x_cumsum[decrement_end] - x_cumsum[decrement_start]
    #                     - x[decrement_end] * (decrement_length - 1))
    #
    # intervals below the threshold are set as baseline points
    # valid_increment = increment_cumsum < increment_threshold
    # valid_decrement = decrement_cumsum < decrement_threshold
    # baseline_mask = valid_increment & valid_decrement
    lsnr = (x[increment_end] - x[increment_start]) / (
                noise[increment_end] * np.sqrt(increment_length))
    rsnr = (x[decrement_start] - x[decrement_end]) / (
                noise[decrement_start] * np.sqrt(decrement_length))
    snr = np.maximum(lsnr, rsnr)
    baseline_mask = snr < 1

    baseline_start = increment_start[baseline_mask]
    baseline_end = decrement_end[baseline_mask]

    # this builds an array with the index of each baseline point
    baseline_index = list()
    for s, e in zip(baseline_start, baseline_end):
        baseline_index.append(np.arange(s, e))
    if len(baseline_index):
        baseline_index = np.hstack(baseline_index)
    else:
        baseline_index = np.array([], dtype=int)

    # remove values from the baseline that are above the 90th percentile in y
    # mostly a sanity check...
    threshold = np.quantile(x, 0.90)
    rm_ind = np.where(x >= threshold)[0]
    baseline_index = np.setdiff1d(baseline_index, rm_ind)

    # add first and last point to the baseline
    if baseline_index.size == 0:
        baseline_index = np.array([0, x.size - 1])
    stack = [baseline_index]
    if baseline_index[0] != 0:
        stack = [0] + stack
    if baseline_index[-1] != (x.size - 1):
        stack = stack + [x.size - 1]
    baseline_index = np.hstack(stack)
    return baseline_index


def detect_peaks(x: np.ndarray, y: np.ndarray, min_snr: float = 10,
                 min_prominence: float = 10, min_distance: float = 3,
                 min_width: Optional[float] = None,
                 max_width: Optional[float] = None, estimators=None):
    r"""
    Finds peaks in a 1D signal.

    Parameters
    ----------
    x : array
    y : array
    min_snr : positive number
        Minimum signal to noise ratio
    min_prominence : positive number
        Minimum signal prominence, expressed as multiples of the noise
    min_distance : positive number
        Minimum distance between consecutive peaks, in `x` units
    min_width : positive number, optional
        Minimum width of a peak, in x units.
    max_width : positive number, optional
        Maximum width of a peak, in x units.
    estimators : dict, optional
        A dictionary of strings to callables, used to estimate custom parameters
        on a peak. See notes for a detailed explanation.

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

            prominence = \min{\frac{y_{apex} - y_{start}}{\sigma},
                \frac{y_{apex} - y_{end}}{\sigma}}

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
    _estimate_noise
    _estimate_baseline

    """
    if min_width is None:
        min_width = 0

    if max_width is None:
        max_width = x.max()

    # find peaks
    peaks = find_peaks(y)[0]

    # baseline and noise estimation
    noise = _estimate_noise(y)
    baseline = _estimate_baseline(y, noise)
    snr = (y - baseline) / noise

    # snr filter
    peaks = peaks[snr[peaks] >= min_snr]

    # find peak extension: the peak extension is a pair of indices that
    # correspond to the beginning and end of a peak. A first extension is
    # obtained by finding the closest baseline point
    baseline_index = np.where(np.isclose(y, baseline))[0]
    ext_index = np.searchsorted(baseline_index, peaks)
    left = baseline_index[ext_index - 1]
    # right = baseline_index[ext_index + 1]
    right = baseline_index[ext_index]

    # peak remove close peaks using the min_distance parameter. If two peaks
    # are close then the least intense is removed
    peaks, left, right = _merge_min_distance(x, y, peaks, left, right,
                                             min_distance)

    # fix overlapping peaks: when there are overlapping peaks, they share the
    # same extension. The left and right extension are fixed by selecting a
    # local minima between the two peaks
    left, right = fix_overlap(x, y, peaks, left, right)

    # peak prominence filter: The left (right) prominence of a peak is defined
    # as the difference in intensity between the intensity at the apex and the
    # intensity at the left (right) extension of the peak. Overlapping Peaks
    # with prominence under the prominence threshold are merged and non
    # overlapping peaks are removed.
    peaks = _filter_prominence(y, noise, peaks, left, right, min_prominence)
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


def fix_overlap(x, y, peaks, left, right):
    has_overlap = np.diff(right) == 0
    local_min = find_peaks(-y)[0]
    overlap_index = np.where(has_overlap)[0]
    left_candidates = np.searchsorted(local_min, peaks)
    for ind in overlap_index:
        start, end = left_candidates[ind:ind + 2]
        min_min = np.argmin(y[local_min[start:end]])
        right[ind] = local_min[start + min_min]
        left[ind + 1] = right[ind]
    return left, right


def get_overlap_groups(left, right):
    overlap_index = np.where(np.roll(right, 1) != left)[0]
    overlap_index = np.hstack((overlap_index, left.size))
    return overlap_index


def _merge_peaks(p1: PeakLocation, p2: PeakLocation, y: np.ndarray):
    loc = max((p1.loc, p2.loc), key=lambda x: y[x])
    p1.loc = loc
    p1.end = p2.end


def _get_left_prominence(y: np.ndarray, peak: PeakLocation):
    return y[peak.loc] - y[peak.start]


def _get_right_prominence(y: np.ndarray, peak: PeakLocation):
    return y[peak.loc] - y[peak.end]


def _filter_prominence(y, noise, peaks, left, right, min_prominence):
    """
    Merge peaks when the middle peak has invalid prominences from both sides
    """
    # TODO: this function is a clusterfuck, but I couldn't make a simpler
    #  version...
    # Peaks are merged according to their prominence:
    # 1. If a peak has a left and right prominence greater than the threshold,
    #    its considered valid and added to the merged list.
    # 2. peaks that aren't valid, and have overlap with a valid peak are merged
    # 3. If a peak can be merged to both a peak to the left and a peak to the
    #    right, the peak is merged in the direction where its prominence is
    #    lower.
    merged_peak_list = list()
    prominence_threshold = noise * min_prominence
    # here peaks are grouped by overlap
    overlap_index = get_overlap_groups(left, right)
    for k in range(overlap_index.size - 1):
        # for each group of overlapped peaks, the prominence is evaluated on
        # each peak
        last_valid = None
        last_invalid = None
        for peak_index in range(overlap_index[k], overlap_index[k + 1]):
            current_peak = PeakLocation(peaks[peak_index], None,
                                        left[peak_index], right[peak_index])
            if last_invalid is not None:
                _merge_peaks(last_invalid, current_peak, y)
                current_peak, last_invalid = last_invalid, None

            left_prominence = _get_left_prominence(y, current_peak)
            right_prominence = _get_right_prominence(y, current_peak)
            prominence = min(left_prominence, right_prominence)
            is_valid = prominence > prominence_threshold[current_peak.loc]
            if is_valid:
                merged_peak_list.append(current_peak)
                last_valid = current_peak
            elif (last_valid is not None) and (
                    left_prominence < right_prominence):
                _merge_peaks(last_valid, current_peak, y)
            else:
                last_invalid = current_peak
    return merged_peak_list


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


def _estimate_height(x, y, baseline, noise, peak):
    return abs(y[peak.loc] - baseline[peak.loc])


def _estimate_area(x, y, baseline, noise, peak):
    baseline_corrected = y[peak.start:(peak.end + 1)] - baseline[peak.start:(
                peak.end + 1)]
    area = trapz(baseline_corrected, x[peak.start:(peak.end + 1)])
    return abs(area)


def _estimate_loc(x, y, baseline, noise, peak):
    weights = y[peak.start:(peak.end + 1)] - baseline[peak.start:(peak.end + 1)]
    loc = np.abs(np.average(x[peak.start:(peak.end + 1)], weights=weights))
    return loc


def _estimate_width(x, y, baseline, noise, peak):
    bsignal = y[peak.start:(peak.end + 1)] - baseline[peak.start:(peak.end + 1)]
    area = cumtrapz(bsignal, x[peak.start:(peak.end + 1)])
    rarea = area / area[-1]
    start, end = peak.start + np.searchsorted(rarea, [0.025, 0.975])
    return x[end] - x[start]


def _estimate_peak_parameters(x, y, baseline, noise, peaks, estimators):
    base_estimators = {"height": _estimate_height,
                       "area": _estimate_area,
                       "loc": _estimate_loc,
                       "width": _estimate_width}
    peak_params = list()
    if estimators is None:
        estimators = dict()
    base_estimators.update(estimators)
    for p in peaks:
        p_params = dict()
        for param, param_estimator in base_estimators.items():
            p_params[param] = param_estimator(x, y, baseline, noise, p)
        peak_params.append(p_params)
    return peak_params
