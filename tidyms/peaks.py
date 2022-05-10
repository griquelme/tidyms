"""
functions and objects used to detect peaks.

Objects
-------

- Peak : Stores peak location and extension. Computes peak parameters.

Functions
---------
- estimate_noise(x) : Estimates noise level in a 1D signal
- estimate_baseline(x, noise) : Estimates the baseline in a 1D signal
- detect_peaks(x, y) : Detects peaks in a 1D signal
- find_centroids(x, y) : Computes the centroid and area of peaks in a 1D signal.

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.special import erfc
from scipy.stats import median_abs_deviation as mad
from typing import List, Optional, Tuple


def detect_peaks(
        x: np.ndarray,
        noise: np.ndarray,
        baseline: np.ndarray,
        find_peaks_params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Finds peaks in a 1D signal.

    Parameters
    ----------
    x : array
        Signal with peaks.
    noise : array with the same size as x
        Noise level of x.
    baseline : array with the same size as x
        Baseline estimation of x
    find_peaks_params : dict or None, default=None
        parameters to pass to :py:func:`scipy.signal.find_peaks`.

    Returns
    -------
    start : array
        Indices where peaks start
    apex : array
        Indices where the peak maximum was found
    end : array
        Indices where peaks end

    Notes
    -----
    The algorithm for peak finding is as follows:

    1.  Peaks are detected using :py:func:`scipy.signal.find_peaks`. Peaks
        with a prominence lower than three times the noise or in regions
        classified as baseline  are removed.
    2.  Points from :math:`x` are considered baseline is the following
        condition is meet:

        .. math::
            |x[k] - b[k]| < e[k]

        where :math:`b` is the baseline and :math:`e` is the noise. If a
        detected peak is classified as baseline is removed.
    3.  The extension of each peak is found by finding the closest baseline
        point to its left and right.
    4.  If there are overlapping peaks (i.e. overlapping peak extensions),
        the extension is fixed by defining a boundary between the peaks as
        the minimum value between the two peaks.

    See Also
    --------
    estimate_noise : estimates noise in a 1D signal
    estimate_baseline : estimates the baseline in a 1D signal
    Peak : stores peak start, apex and end indices
    get_peak_descriptors : computes descriptors on a list of Peak objects

    """

    baseline_index = np.where((x - baseline) < noise)[0]
    if baseline_index.size == 0:
        baseline_index = np.array([0, x.size - 1], dtype=int)

    prominence = 3 * noise

    if find_peaks_params is None:
        find_peaks_params = {"prominence": prominence, "distance": 3}
    else:
        find_peaks_params["prominence"] = prominence

    peaks = find_peaks(x, **find_peaks_params)[0]
    # remove peaks close to baseline level
    peaks = np.setdiff1d(peaks, baseline_index, assume_unique=True)

    start, end = _find_peak_extension(peaks, baseline_index)
    start, end = _fix_peak_overlap(x, start, peaks, end)
    start, peaks, end = _normalize_peaks(x, start, peaks, end)
    return start, peaks, end


def estimate_noise(
        x: np.ndarray,
        min_slice_size: int = 200,
        n_slices: int = 5,
        robust: bool = True
) -> np.ndarray:
    """
    Estimates the noise level in a signal.

    Splits x into several slices and estimates the noise assuming that the
    noise is gaussian iid in each slice. See [ADD LINK] for a detailed
    description of how the method works

    Parameters
    ----------
    x : 1D array
    min_slice_size : int, default=200
        Minimum size of a slice. If the size of x is smaller than this value,
        the noise is estimated using the whole array.
    n_slices: int, default=5
        Number of slices to create. The size of each slice must be greater than
        `min_slice_size`.
    robust : bool, default=True
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
        if (x.size - end) < min_slice_size:
            # prevent short slices at the end of x
            end = x.size
        slice_noise = _estimate_local_noise(x[start:end], robust=robust)
        noise[start:end] = slice_noise
        start = end
    return noise


def estimate_baseline(
        x: np.ndarray,
        noise: np.ndarray,
        min_proba: float = 0.05
) -> np.ndarray:
    """
    Computes the baseline of a 1D signal.

    The baseline is estimated by classifying each point in the signal as either
    signal or baseline. The baseline is obtained by interpolation of baseline
    points. See [ADD LINK] for a detailed explanation of how the method works.

    Parameters
    ----------
    x : non-empty 1D array
    noise : array
        Noise estimation obtained with ``estimate_noise``
    min_proba : number between 0 and 1, default=0.05

    Returns
    -------
    baseline : array with the same size as x

    """
    # find points that only have contribution from the baseline
    baseline_index = _find_baseline_points(x, noise, min_proba)

    # interpolate baseline points to match x size
    baseline = x[baseline_index]
    interpolator = interp1d(baseline_index, baseline)
    baseline = interpolator(np.arange(x.size))

    # prevents that interpolated points have higher values than x.
    baseline = np.minimum(baseline, x)
    return baseline


def find_centroids(mz: np.ndarray, spint: np.ndarray, min_snr: float,
                   min_distance: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the centroid of a mass spectrum in profile mode.

    Parameters
    ----------
    mz : array
    spint : array
    min_snr : positive number
        Minimum signal-to-noise ratio
    min_distance : positive number
        Minimum m/z distance between consecutive centroids

    Returns
    -------
    centroid_mz : array
        centroid m/z of peaks
    centroid_int : array
        area of peaks

    """
    noise = estimate_noise(spint)
    baseline = estimate_baseline(spint, noise)
    baseline_index = np.where((spint - baseline) < noise)[0]
    prominence = 3 * noise
    find_peaks_params = {"prominence": prominence, "distance": 3}

    peaks = find_peaks(spint, **find_peaks_params)[0]
    # remove peaks close to baseline level
    peaks = np.setdiff1d(peaks, baseline_index, assume_unique=True)
    peaks = peaks[((spint[peaks] - baseline[peaks]) / noise[peaks]) > min_snr]

    start, end = _find_peak_extension(peaks, baseline_index)
    start, end = _fix_peak_overlap(spint, start, peaks, end)
    start, peaks, end = _normalize_peaks(spint, start, peaks, end)

    # peak centroid and total intensity computation
    # if m[0], ...,  m[n] is the m/z array and i[0], ..., i[n] is the
    # intensity array, for a peak with start and indices k and l respectively,
    # the total intensity A is A = \sum_{j=k}^{l} i[j] and the centroid C,
    # computed as the weighted mean of the m/z is
    # C = \sum_{j=k}^{l} m[j] * i[j] / A
    # If we define the cumulative intensity I_{k} = \sum_{j=0}^{k} i[j]
    # It is easy to see that A = I[l - 1] - I[k - 1]. The same can be done
    # for the centroids defining the weights W[k] = \sum_{j=0}^{k} m[j] * i[j]
    # C = (W[l - 1] - W[k - 1]) / A
    if start.size:
        cumulative_spint = np.cumsum(spint)
        weights = np.cumsum(mz * spint)
        start_cumulative_spint = cumulative_spint[start - 1]
        if start[0] == 0:
            # prevents using the last value from cumulative_spint
            start_cumulative_spint[0] = 0
        total_spint = cumulative_spint[end - 1] - start_cumulative_spint

        start_weight = weights[start - 1]
        if start[0] == 0:
            start_weight[0] = 0
        centroid = (weights[end] - start_weight) / total_spint
    else:
        centroid = np.array([])
        total_spint = np.array([])

    if centroid.size:
        _merge_close_peaks(centroid, total_spint, min_distance)
    return centroid, total_spint


def _find_peak_extension(peaks: np.array, baseline_index: np.array
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the closest baseline points to the left and right of each peak.

    aux function of detect_peaks

    Parameters
    ----------
    peaks : array of indices
    baseline_index : array of indices

    Returns
    -------
    start : array
    end : array

    """
    ext_index = np.searchsorted(baseline_index, peaks)
    ext_index[ext_index >= baseline_index.size] = baseline_index.size - 1
    start = baseline_index[ext_index - 1]
    end = baseline_index[ext_index] + 1
    return start, end


def _fix_peak_overlap(y: np.ndarray, start: np.ndarray, peaks: np.ndarray,
                      end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix boundaries of overlapping peaks.

    aux function of detect_peaks

    Parameters
    ----------
    y : array
    start : array
    peaks : array
    end : array

    Returns
    -------
    start : array
    end : array

    """
    local_min = find_peaks(-y)[0]
    # find overlapping peaks indices
    overlap_mask = end > np.roll(start, -1)
    if overlap_mask.size:
        overlap_mask[-1] = False
    overlap_index = np.where(overlap_mask)[0]
    for k in overlap_index:
        # search local min in the region defined by the overlapping peaks
        ks, ke = np.searchsorted(local_min, [peaks[k], peaks[k + 1]])
        k_min = np.argmin(y[local_min[ks:ke]])
        boundary = local_min[ks + k_min]
        end[k] = boundary + 1
        start[k + 1] = boundary
    return start, end


def _normalize_peaks(y: np.ndarray, start: np.ndarray, peaks: np.ndarray,
                     end: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sanity check of peaks.

    Finds the local maximum in the region defined between start and end for each
    peak. If no maximum is found, the peak is removed. Also, corrects the peak
    index using this value that can be slightly shifted if smoothing was
    applied to the signal.

    aux function of detect_peaks.

    Parameters
    ----------
    y : array
    start : array of indices
    peaks : array of indices
    end : array of indices

    Returns
    -------
    start : array of indices
    fixed_peaks : array of indices
    end : array of indices

    """
    local_max = find_peaks(y)[0]
    start_index = np.searchsorted(local_max, start)
    end_index = np.searchsorted(local_max, end, side="right")
    fixed_peaks = np.zeros_like(peaks)
    valid_peaks = np.zeros(peaks.size, dtype=bool)
    for k in range(peaks.size):
        # search local min in the region defined by the overlapping peaks
        local_max_slice = y[local_max[start_index[k]:end_index[k]]]
        if local_max_slice.size > 0:
            k_max = np.argmax(local_max_slice)
            new_peak = local_max[start_index[k] + k_max]
            valid_peaks[k] = (start[k] < new_peak) and (new_peak < end[k])
            fixed_peaks[k] = new_peak
    start = start[valid_peaks]
    end = end[valid_peaks]
    fixed_peaks = fixed_peaks[valid_peaks]
    return start, fixed_peaks, end


def _estimate_local_noise(x: np.ndarray, robust: bool = True) -> float:
    r"""
    Estimates noise in a 1D signal. Assumes that the noise is gaussian iid.

    aux function of estimate_noise

    Parameters
    ----------
    x : 1D array
        The size of x must be at least 4. If the size is smaller, the function
        will return 0.
    robust : bool
        If True, estimates the noise using the median absolute deviation. Else
        uses the standard deviation.

    Returns
    -------
    noise : non-negative number.

    """
    d2x = np.diff(x, n=2)
    sorted_index = np.argsort(np.abs(d2x))
    d2x = d2x[sorted_index]
    # if d2x follows a normal distribution ~ N(0, 2*sigma), its sample mean
    # has a normal distribution ~ N(0,  2 * sigma / sqrt(n - 2)) where n is the
    # size of d2x.
    # d2x with high absolute values are removed until this the mean of d2x is
    # lower than its standard deviation.
    # start at 90th percentile and decrease it in each iteration.
    # The loop stops at the 20th percentile even if this condition is not meet
    n_deviations = 3    # dummy values to initialize the loop
    percentile_counter = 9  # start at 90th percentile
    noise_std = 0

    if robust:
        mean_func = np.median
        std_func = lambda x: mad(x, scale="normal")
    else:
        mean_func = np.mean
        std_func = np.std

    if d2x.size >= 2:   # we need at least 2 points to compute the std
        while (n_deviations > 1.0) and (percentile_counter > 2):
            percentile_index = percentile_counter * d2x.size // 10
            # the minimum number of elements required to compute the MAD
            if percentile_index <= 2:
                break
            noise_std = std_func(d2x[:percentile_index])
            noise_mean = mean_func(d2x[:percentile_index])

            # if all the values in d2x are equal, noise_std is equal to zero
            if noise_std > 0:
                n_deviations = abs(noise_mean / noise_std)
            else:
                break
            percentile_counter -= 1
        noise = noise_std / 2
        if np.isclose(noise, 0):
            noise = d2x.std()
    else:
        # in the case where d2x < 2 we cannot compute the std, the noise is
        # defined as 0
        noise = 0.0
    return noise


def _find_baseline_points(x: np.ndarray, noise: np.ndarray,
                          min_proba: float) -> np.ndarray:
    """
    Finds points flagged as baseline.

    Aux function of estimate_baseline.

    Parameters
    ----------
    x : 1D array
    noise : 1D array
    min_proba : number between 0 and 1

    Returns
    -------
    baseline_index : array of indices.

    """
    extrema = _find_local_extrema(x)
    # check how likely is that the difference observed in each min-max slice
    # can be attributed to noise.
    noise_proba = _estimate_noise_probability(noise, x, extrema)
    # creates a vector with indices where baseline was found
    baseline_index = _build_baseline_index(x, noise_proba, min_proba, extrema)
    return baseline_index


def _find_local_extrema(x: np.ndarray) -> np.ndarray:
    """
    Finds all local minima and maxima in an 1D array.

    aux function of _find_baseline_points.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray : sorted array of local extrema.

    """
    local_max = find_peaks(x)[0]
    local_min = find_peaks(-x)[0]
    if local_max.size:
        # include first and last indices
        extrema = np.hstack([0, local_min, local_max, x.size - 1])
    else:
        extrema = np.array([], dtype=int)
    return np.unique(extrema)


def _estimate_noise_probability(noise: np.ndarray, x: np.ndarray,
                                extrema: np.ndarray) -> np.ndarray:
    """
    Computes the probability that the variation observed in each slice is
    due to noise only.

    Aux function of _find_baseline_points.

    Parameters
    ----------
    noise : 1D array
    x : 1D array
    extrema : array of indices.

    Returns
    -------
    noise_probability : array

    """

    if extrema.size:
        noise_slice = _get_noise_slice_sum_std(noise, extrema[:-1])
        # The difference between maximum and minimum in each slice
        # delta = np.abs(x[np.roll(extrema, -1)] - x[extrema])[:-1]
        x_sum = _get_signal_sum(x, extrema)
        # here we are computing p(abs(sum noise) > delta) assuming a normal
        # distribution
        noise_probability = erfc(x_sum / (noise_slice * np.sqrt(2)))
    else:
        # prevents raising an exception when no extrema had been found
        noise_probability = np.array([])
    return noise_probability


def _get_noise_slice_sum_std(noise: np.ndarray,
                             extrema: np.ndarray) -> np.ndarray:
    """
    Computes the standard deviation of the sum of slices between local maxima.

    aux function of _estimate_noise_probability.

    Parameters
    ----------
    noise : array with the noise level in the signal
    extrema : array with sorted indices of local max/min.

    Returns
    -------
    array

    """
    # the values in noise are an estimation of the standard deviation of the
    # noise. If the noise is iid, the std of the sum is the sum of variances.

    # reshape the extrema indices to compute the sum of elements between
    # consecutive slices, i.e: the sum between of y between extrema[0] and
    # extrema[1], extrema[1] and extrema[2]...
    # The last element is not used
    reduce_ind = (np.vstack([extrema, np.roll(extrema + 1, -1)])
                    .T.reshape(extrema.size * 2)[:-1])
    return np.sqrt(np.add.reduceat(noise ** 2, reduce_ind)[::2])


def _get_signal_sum(x, extrema):
    # we want to compute the sum of x between a, b, where a and b two
    # consecutive local maxima indices. if X is the cumulative sum of x, then
    # the sum between a and b is X[b] - X[a - 1]. The sum must also be relative
    # to the minimum between a and b. Because we are comparing intervals
    # between loca extrema, the intervals are monotonic, and we just need to
    # subtract the min(x[a], x[b]) multiplied by the length of the interval.
    # this code achieves this in a vectorized way
    cum_x = np.cumsum(x)
    ext_shift = np.roll(extrema, -1)
    n_times = ext_shift - extrema + 1
    start_cum_int = cum_x[extrema - 1]
    # first value set to 0 to avoid errors associated with roll
    start_cum_int[0] = 0
    x_sum = cum_x[ext_shift] - start_cum_int
    x_min = np.minimum(x[ext_shift], x[extrema])
    res = x_sum - n_times * x_min
    res = res[:-1]
    return res


def _build_baseline_index(x: np.ndarray, noise_probability: np.ndarray,
                          min_p: float, extrema: np.ndarray):
    """
    builds an array with indices of points flagged as baseline.

    aux function of _find_baseline_points

    Returns
    -------
    array

    """
    # define regions of signal based on noise probability
    is_signal = noise_probability < min_p
    # extend regions of signals to right and left
    is_signal = is_signal | np.roll(is_signal, 1) | np.roll(is_signal, -1)
    baseline_index = list()

    for k in range(extrema.size - 1):
        if not is_signal[k]:
            slice_indices = np.arange(extrema[k], extrema[k + 1] + 1)
            baseline_index.append(slice_indices)

    baseline_index = _include_first_and_last_index(x, baseline_index)
    return baseline_index


def _include_first_and_last_index(x: np.ndarray,
                                  baseline_index: List[np.ndarray]
                                  ) -> np.ndarray:
    """
    adds first and last indices of x to the baseline indices.

    aux function of _build_baseline_index

    Parameters
    ----------
    x : array
    baseline_index : array of indices

    Returns
    -------
    baseline_index : array of indices

    """
    if len(baseline_index):
        stack = list()
        # include first and last indices
        if baseline_index[0][0] != 0:
            stack.append([0])
        stack.extend(baseline_index)
        if baseline_index[-1][-1] != x.size - 1:
            stack.append([x.size - 1])
        baseline_index = np.hstack(stack)
    else:
        baseline_index = np.array([0, x.size - 1], dtype=int)
    return baseline_index


def _merge_close_peaks(mz: np.ndarray, spint: np.ndarray,
                       min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    dmz = np.diff(mz)
    is_close_mask = (dmz < min_distance) & (np.roll(dmz, -1) > min_distance)
    is_close_mask[-1] = (mz[-1] - mz[-2]) < min_distance    # boundary case
    close_index = np.where(is_close_mask)[0]
    while close_index.size > 0:
        # merge close centroids
        new_spint = spint[close_index] + spint[close_index + 1]
        new_mz = (mz[close_index] * spint[close_index] +
                  mz[close_index + 1] * spint[close_index + 1])
        new_mz /= new_spint
        spint[close_index] = new_spint
        mz[close_index] = new_mz
        # remove merged centroids
        mz = np.delete(mz, close_index + 1)
        spint = np.delete(spint, close_index + 1)

        # repeat until no close peaks are detected
        dmz = np.diff(mz)
        is_close_mask = (dmz < min_distance) & (np.roll(dmz, -1) > min_distance)
        is_close_mask[-1] = (mz[-1] - mz[-2]) < min_distance
        close_index = np.where(is_close_mask)[0]
    return mz, spint
