import tidyms as ms
import numpy as np
import pytest
from scipy.signal.windows import gaussian
from scipy.special import erfc
from scipy.ndimage import gaussian_filter1d
# from itertools import product

# random seed
SEED = 1234


# noise estimation tests

@pytest.fixture
def noise():
    sigma = 1.0
    np.random.seed(SEED)
    return np.random.normal(size=500, scale=sigma), sigma


def test_estimate_local_noise_empty_signal():
    x = np.array([])
    noise = ms.peaks._estimate_local_noise(x)
    assert np.isclose(noise, 0.0)


@pytest.mark.parametrize("x", [np.array([1]), np.array([1, 2])])
def test_estimate_local_noise_signal_length_lower_than_two(x):
    noise = ms.peaks._estimate_local_noise(x)
    assert np.isclose(noise, 0.0)


def test_estimate_local_noise(noise):
    # check that the noise estimation is close to the std of a normal
    # distribution
    x, sigma = noise
    noise_estimation = ms.peaks._estimate_local_noise(x)
    # noise should be close to sigma, check with a 20 % tolerance
    assert (sigma < 1.2 * noise_estimation)


def test_estimate_local_noise_non_robust(noise):
    x, sigma = noise
    noise_estimation = ms.peaks._estimate_local_noise(x, robust=False)
    # noise should be close to sigma, check with a 20 % tolerance
    assert (sigma < 1.2 * noise_estimation)


def test_estimate_noise_empty_array():
    x = np.array([])
    noise = ms.peaks.estimate_noise(x)
    assert noise.size == 0.0


@pytest.mark.parametrize("x", [np.array([1]), np.array([1, 3]),
                               np.array([1, 4, 6])])
def test_estimate_noise_signal_length_lower_than_two(x):
    noise_estimation = ms.peaks.estimate_noise(x)
    assert np.allclose(noise_estimation, 0.0)


def test_estimate_noise_check_size(noise):
    noise, sigma = noise
    noise_estimation = ms.peaks.estimate_noise(noise, n_slices=2)
    assert noise.size == noise_estimation.size


def test_estimate_noise_n_slices(noise):
    noise, sigma = noise
    noise_estimation = ms.peaks.estimate_noise(noise, n_slices=2)
    size = noise.size
    half = size // 2
    # check that the noise estimation was done for 2 slices
    assert np.allclose(noise_estimation[:half], noise_estimation[0])
    assert np.allclose(noise_estimation[half:], noise_estimation[half])
    # check that the estimation on each slice is different
    assert noise_estimation[0] != noise_estimation[half]


def test_estimate_noise_min_slice_size(noise):
    noise, sigma = noise
    n_slices = 5
    min_slice_size = 150
    noise_estimation = ms.peaks.estimate_noise(noise, n_slices=n_slices,
                                               min_slice_size=min_slice_size)
    # noise has a size of 500, the slice is going to be 100 < 150
    # check that 150 is used instead.
    slice_boundaries = [0, 150, 300, 500]   # the last slice is extended to 200
    # to prevent the creation of a slice of size 50
    for k in range(len(slice_boundaries) - 1):
        start = slice_boundaries[k]
        end = slice_boundaries[k + 1]
        assert np.allclose(noise_estimation[start:end], noise_estimation[start])


# Test baseline estimation

def test_find_local_extrema():
    x = np.arange(10)
    # reflect and merge the concatenate x. local extrema should be 0, 9, 19
    x = np.hstack((x, x[::-1]))
    test_output = ms.peaks._find_local_extrema(x)
    expected_output = [0, 9, 19]
    assert np.array_equal(test_output, expected_output)


def test_find_local_extrema_no_local_maximum():
    x = np.arange(10)
    test_output = ms.peaks._find_local_extrema(x)
    expected_output = np.array([])
    assert np.array_equal(test_output, expected_output)


test_noise_sum_params = [[np.array([0, 1]), np.sqrt([25, 25])],
                         [np.array([0]), np.sqrt([34])]]


@pytest.mark.parametrize("index,expected", test_noise_sum_params)
def test_get_noise_sum_slice_std(index, expected):
    index = np.array(index)
    expected = np.array(expected)
    x = np.array([3, 4, 2, 2, 1])
    test_output = ms.peaks._get_noise_slice_sum_std(x, index)
    assert np.allclose(test_output, expected)


def test_estimate_noise_probability():
    noise = np.ones(7)
    x = np.array([0, 0.1, 0.4, 2, 1.25, 1.1, 1.0])
    extrema = np.array([0, 3, 6])
    # two slices of size 4 and 2 respectively, the expected output should
    # be erfc(1/sqrt(2) and erfc(1)
    expected_output = erfc([2.5 * np.sqrt(1 / 2) / 2,
                            1.35 * np.sqrt(1 / 2) / 2])
    test_output = ms.peaks._estimate_noise_probability(noise, x, extrema)
    assert np.allclose(expected_output, test_output)


def test_build_baseline_index():
    x = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0])
    extrema = np.array([0, 2, 4, 6, 8, 10, 12])
    noise_probability = np.array([0, 0.25, 0.25, 0.25, 0, 0])
    min_proba = 0.05
    expected = np.array([0, 4, 5, 6, 12])
    test = ms.peaks._build_baseline_index(x, noise_probability, min_proba,
                                          extrema)
    assert np.array_equal(expected, test)


def test_estimate_baseline():
    # a simple test, a noise array is built using a noise level greater
    # than the noise level in the signal. All points should be classified as
    # baseline
    n = 100
    x = np.random.normal(size=n, scale=1)
    noise = np.ones(n) * 5
    baseline = ms.peaks.estimate_baseline(x, noise)
    expected_baseline_index = np.arange(n)
    test_baseline_index = np.where(np.abs(x - baseline) < noise)[0]
    assert np.array_equal(expected_baseline_index, test_baseline_index)


@pytest.fixture
def single_peak(noise):
    noise, sigma = noise
    x = gaussian(noise.size, 2) * 20
    return x


@pytest.fixture
def two_non_overlapping_peaks(noise):
    noise, sigma = noise
    x = np.arange(noise.size)
    params = np.array([[100, 2, 50], [150, 2, 25]])
    y = ms.utils.gaussian_mixture(x, params).sum(axis=0)
    return y, params


def test_detect_peaks_one_peak(single_peak, noise):
    noise, sigma = noise
    x = single_peak + noise
    noise_estimation = ms.peaks.estimate_noise(x)
    # smooth x to reduce the number of detected peaks
    x = gaussian_filter1d(x, 1.0)
    baseline_estimation = ms.peaks.estimate_baseline(x, noise)
    peaks = ms.peaks.detect_peaks(x, noise_estimation, baseline_estimation)
    assert len(peaks[0]) == 1


def test_detect_peaks_two_non_overlapping_peaks(two_non_overlapping_peaks,
                                                noise):
    noise, sigma = noise
    x, _ = two_non_overlapping_peaks
    x = x + noise
    noise_estimation = ms.peaks.estimate_noise(x)
    # smooth x to reduce the number of detected peaks
    x = gaussian_filter1d(x, 1.0)
    baseline_estimation = ms.peaks.estimate_baseline(x, noise)
    peaks = ms.peaks.detect_peaks(x, noise_estimation, baseline_estimation)
    assert len(peaks[0]) == 2


@pytest.fixture
def two_overlapping_peaks(noise):
    noise, sigma = noise
    x = np.arange(noise.size)
    params = np.array([[100, 2, 50], [108, 2, 25]])
    y = ms.utils.gaussian_mixture(x, params).sum(axis=0)
    return y, params


def test_detect_peaks_two_overlapping_peaks(two_overlapping_peaks, noise):
    noise, sigma = noise
    x, _ = two_overlapping_peaks
    x = x + noise
    noise_estimation = ms.peaks.estimate_noise(x)
    # smooth x to reduce the number of detected peaks
    x = gaussian_filter1d(x, 1.0)
    baseline_estimation = ms.peaks.estimate_baseline(x, noise)
    peaks = ms.peaks.detect_peaks(x, noise_estimation, baseline_estimation)
    start, apex, end = peaks
    # only two peaks are detected
    assert len(start) == 2
    # check the boundary of the overlapping peaks
    assert end[0] == (start[1] + 1)
