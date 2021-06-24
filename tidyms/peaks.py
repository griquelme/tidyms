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
- find_centroids(x, y) : Computes the centroid and area of peaks in a 1D signal.

"""

import numpy as np
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import erfc
from scipy.stats import median_abs_deviation as mad
from typing import Callable, Dict, List, Optional, Tuple


class PeakLocation:
    """
    Representation of a peak. Computes peak descriptors.

    Attributes
    ----------
    apex: int
        index where the apex of the peak is located. Must be smaller than `end`
    start: int
        index where the peak begins. Must be smaller than `loc`
    end: int
        index where the peak ends. Start and end used as slices defines the
        peak region.

    """

    def __init__(self, start: int, apex: int, end: int):

        try:
            assert start < apex
            assert apex < end
        except AssertionError:
            msg = "start must be lower than loc and loc must be lower than end"
            raise InvalidPeaKException(msg)

        self.apex = apex
        self.start = start
        self.end = end

    def __repr__(self):
        str_repr = "PeakLocation(start={}, loc={}, end={})"
        str_repr = str_repr.format(self.start, self.apex, self.end)
        return str_repr

    def get_loc(self, x: np.ndarray, y: np.ndarray):
        """
        Finds the peak location as the weighted average of x using y as weights.

        Parameters
        ----------
        x : sorted array
        y : array with the same size as x

        Returns
        -------
        loc : float

        """
        weights = y[self.start:self.end]
        weights[weights < 0] = 0
        loc = np.abs(np.average(x[self.start:self.end], weights=weights))
        return loc

    def get_height(self, y: np.ndarray, baseline: np.ndarray):
        """
        Computes the height of the peak, defined as the difference between the
        value of y and the baseline at the peak apex.

        Parameters
        ----------
        y : array
        baseline : array with the same size as y

        Returns
        -------
        height : non-negative number. If the baseline estimation is greater
        than y, the height is set to zero.

        """
        height = y[self.apex] - baseline[self.apex]
        return max(0.0, height)

    def get_area(self, x: np.ndarray, y: np.ndarray, baseline: np.ndarray):
        """
        Computes the area in the region defined by the peak.

        Parameters
        ----------
        x : sorted array
        y : array with the same size as y
        baseline : array with the same size as y

        Returns
        -------
        area : positive number. If the baseline area is greater than the peak
            area, the area is set to zero.

        """
        baseline_corrected = (y[self.start:self.end] -
                              baseline[self.start:self.end])
        area = trapz(baseline_corrected, x[self.start:self.end])
        return max(0.0, area)

    def get_width(self, x: np.ndarray, y, baseline):
        """
        Computes the peak width, defined as the region where the 95 % of the
        total peak area is distributed.

        Parameters
        ----------
        x : sorted array
        y : array with the same size as y
        baseline : array with the same size as y

        Returns
        -------
        width : positive number.

        """
        height = (y[self.start:self.end] -
                  baseline[self.start:self.end])
        area = cumtrapz(height, x[self.start:self.end])
        if area[-1] > 0:
            relative_area = area / area[-1]
            percentile = [0.025, 0.975]
            start, end = self.start + np.searchsorted(relative_area, percentile)
            width = x[end] - x[start]
        else:
            width = 0.0
        return max(0.0, width)

    def get_extension(self, x: np.ndarray):
        """
        Computes the peak extension, defined as the length of the peak region.

        Parameters
        ----------
        x: sorted array

        Returns
        -------
        extension : positive number

        """
        return x[self.end] - x[self.start]

    def get_snr(self, y: np.array, noise: np.array, baseline: np.array):
        """
        Computes the peak signal-to-noise ratio, defined as the quotient
        between the peak height and the noise level at the apex.

        Parameters
        ----------
        y : array
        noise : array with the same size as y
        baseline : array with the same size as y

        Returns
        -------
        snr : float

        """

        peak_noise = noise[self.apex]
        if np.isclose(peak_noise, 0):
            snr = np.inf
        else:
            snr = self.get_height(y, baseline) / peak_noise
        return snr

    def get_descriptors(self, x: np.array, y: np.array, noise: np.array,
                        baseline: np.array):
        """
        Computes peak height, area, location, width and SNR.

        Parameters
        ----------
        x : sorted array
        y : array with the same size as x
        noise : array with the same size as x
        baseline : array with the same size as x

        Returns
        -------
        descriptors: a dictionary from descriptor names to values.
        """
        descriptors = {"height": self.get_height(y, baseline),
                       "area": self.get_area(x, y, baseline),
                       "loc": self.get_loc(x, y),
                       "width": self.get_width(x, y, baseline),
                       "snr": self.get_snr(y, noise, baseline)}
        return descriptors


def detect_peaks(x: np.ndarray, smoothing_strength: Optional[float] = 1.0,
                 find_peaks_params: Optional[dict] = None
                 ) -> Tuple[List[PeakLocation], np.ndarray, np.ndarray]:
    r"""
    Finds peaks in a 1D signal.

    Parameters
    ----------
    x : array
    smoothing_strength: positive number, optional
        Width of a gaussian window used to smooth the signal. If None, no
        smoothing is applied.
    find_peaks_params : dict, optional
        parameters to pass to :py:function:`scipy.signal.find_peaks`.

    Returns
    -------
    peaks : List[PeakLocation]
    noise : array
    baseline : array

    Notes
    -----
    The algorithm for peak finding is as follows:

    1.  Noise is estimated for the signal.
    2.  Using the noise, each point in the signal is classified as either
        baseline or signal.
    3.  Peaks are detected using :py:function:`scipy.signal.find_peaks`. Peaks
        with a prominence lower than three times the noise or in regions
        classified as baseline  are removed.
    4.  For each peak its extension is found by finding the closest baseline
        point to the left and right.
    5.  If there are overlapping peaks (i.e. overlapping peak extensions),
        the extension is fixed by defining a boundary between the peaks as
        the minimum value between the two peaks.

    See Also
    --------
    estimate_noise
    estimate_baseline
    PeakLocation
    get_peak_descriptors

    """
    noise = estimate_noise(x)

    if smoothing_strength is not None:
        # xs = smooth(x, gaussian, smoothing_strength)
        xs = gaussian_filter1d(x, smoothing_strength)
    else:
        xs = x

    baseline, baseline_index = estimate_baseline(xs, noise, return_index=True)
    prominence = 3 * noise

    if find_peaks_params is None:
        find_peaks_params = {"prominence": prominence, "distance": 3}
    else:
        find_peaks_params["prominence"] = prominence

    peaks = find_peaks(xs, **find_peaks_params)[0]
    peaks = np.setdiff1d(peaks, baseline_index, assume_unique=True)

    start, end = _find_peak_extension(peaks, baseline_index)
    start, end = _fix_peak_overlap(x, start, peaks, end)
    start, peaks, end = _normalize_peaks(x, start, peaks, end)
    peaks = [PeakLocation(s, p, e) for s, p, e in zip(start, peaks, end)]
    return peaks, noise, baseline


def get_peak_descriptors(x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                         baseline: np.ndarray, peaks: List[PeakLocation],
                         descriptors: Optional[Dict[str, Callable]] = None,
                         filters: Optional[Dict[str, Tuple]] = None
                         ) -> Tuple[List[PeakLocation], List[Dict[str, float]]]:
    """
    Computes peak descriptors for a list of peaks.

    Parameters
    ----------
    x : sorted array
    y : array with the same size as x
    noise: array with the same size as x. Noise estimation of y.
    baseline: array with the same size as x. Baseline estimation of y.
    peaks : List of peaks obtained with the detect_peaks function
    descriptors : dict, optional
        A dictionary of strings to callables, used to estimate custom parameters
        on a peak. The function must have the following signature:

        .. code-block:: python

            "estimator_func(x, y, noise, baseline, peak) -> float"

    filters : dict, optional
        A dictionary of descriptor names to a tuple of minimum and maximum
        acceptable values. To use only minimum/maximum values, use None
        (e.g. (None, max_value) in the case of using only maximum).Peaks with
        descriptors outside these ranges are removed. Filters for custom
        descriptors can be used also.

    Returns
    -------
    peaks : List[PeakLocation]
    descriptors: List[dict]
        By default, the location, height, area, width and SNR of each peak are
        computed.

    See Also
    --------
    detect_peaks
    estimate_baseline
    estimate_noise

    """

    if descriptors is None:
        descriptors = dict()

    if filters is None:
        filters = dict()
    _fill_filter_boundaries(filters)

    valid_peaks = list()
    descriptor_list = list()
    for p in peaks:
        p_descriptors = p.get_descriptors(x, y, noise, baseline)
        for descriptor, func in descriptors.items():
            p_descriptors[descriptor] = func(x, y, noise, baseline, p)

        if _has_all_valid_descriptors(p_descriptors, filters):
            valid_peaks.append(p)
            descriptor_list.append(p_descriptors)
    return valid_peaks, descriptor_list


def estimate_noise(x: np.ndarray, min_slice_size: int = 200,
                   n_slices: int = 5, robust: bool = True) -> np.ndarray:
    """
    Estimates the noise in a signal.

    Splits x into several slices and estimates the noise assuming that the
    noise is gaussian iid in each slice. See [ADD LINK] for a detailed
    description of how the method works

    Parameters
    ----------
    x : 1D array
    min_slice_size : int, optional
        Minimum size of a slice. If the size of x is smaller than this value,
        the noise is estimated using the whole array.
    n_slices: int, optional
        Number of slices to create. The size of each slice must be greater than
        `min_slice_size`.
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


def estimate_baseline(x: np.ndarray, noise: np.ndarray, min_proba: float = 0.05,
                      return_index=False):
    """
    Computes the baseline of a 1D signal.

    The baseline is estimated by classifying each point in the signal as either
    signal or baseline. The baseline is obtained by interpolation of baseline
    points. See [ADD LINK] for a detailed explanation of how the method works.

    Parameters
    ----------
    x : non empty 1D array
    noise : array obtained with estimate noise
    min_proba : number between 0 and 1
    return_index : bool
        If True returns the indices of elements flagged as baseline

    Returns
    -------
    baseline : array
    baseline_index = array, only returned if return_index is True

    """
    # find points that only have contribution from the baseline
    baseline_index = _find_baseline_points(x, noise, min_proba)

    # interpolate baseline points to match x size
    baseline = x[baseline_index]
    interpolator = interp1d(baseline_index, baseline)
    baseline = interpolator(np.arange(x.size))

    # prevents that interpolated points have higher values than x.
    baseline = np.minimum(baseline, x)

    if return_index:
        baseline_index = np.where(np.isclose(x, baseline))[0]
        return baseline, baseline_index
    else:
        return baseline


class InvalidPeaKException(ValueError):
    """
    Exception raised when invalid indices are used in the construction of
    PeakLocation objects.

    """
    pass


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


def _fill_filter_boundaries(filter_dict: Dict[str, Tuple]):
    """
    Replaces None in the filter boundaries to perform comparisons.

    aux function of get_peak_descriptors
    """
    for k in filter_dict:
        lb, ub = filter_dict[k]
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf
        filter_dict[k] = (lb, ub)


def _has_all_valid_descriptors(peak_descriptors: Dict[str, float],
                               filters: Dict[str, Tuple[float, float]]) -> bool:
    """
    Check that the descriptors of a peak are in a valid range.

    aux function of get_peak_descriptors.

    Parameters
    ----------
    peak_descriptors : dict
        Dictionary from descriptors names to values.
    filters : dict
        Dictionary from descriptors names to minimum and maximum acceptable
        values.

    Returns
    -------
    is_valid : bool
        True if all descriptors are inside the valid ranges.

    """
    res = True
    for descriptor, (lb, ub) in filters.items():
        d = peak_descriptors[descriptor]
        is_valid = (d >= lb) and (d <= ub)
        if not is_valid:
            res = False
            break
    return res


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
    noise : non negative number.

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
    while (n_deviations > 1.0) and (percentile_counter > 2):
        percentile_index = percentile_counter * d2x.size // 10
        # the minimum number of elements required to compute the MAD
        if percentile_index <= 2:
            break
        # dev_threshold = 2 / np.sqrt(percentile - 2)

        if robust:
            noise_std = mad(d2x[:percentile_index], scale="normal")
            noise_mean = np.median(d2x[:percentile_index])
        else:
            noise_std = d2x[:percentile_index].std()
            noise_mean = d2x[:percentile_index].mean()

        # if all the values in d2x are equal, noise_std is equal to zero
        if noise_std > 0:
            n_deviations = abs(noise_mean / noise_std)
        else:
            break
        percentile_counter -= 1
    noise = noise_std / 2
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
        delta = np.abs(x[np.roll(extrema, -1)] - x[extrema])[:-1]
        # here we are computing p(abs(sum noise) > delta) assuming a normal
        # distribution
        noise_probability = erfc(delta / (noise_slice * np.sqrt(2)))
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
