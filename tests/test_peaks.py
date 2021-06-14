import tidyms as ms
import numpy as np
import pytest
from scipy.signal.windows import gaussian
from scipy.special import erfc
# from itertools import product

# random seed
SEED = 1234
np.random.seed(SEED)


# noise estimation tests

@pytest.fixture
def noise():
    sigma = 1.0
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
    # noise should be close to sigma, check with a 5 % tolerance
    assert (sigma < 1.05 * noise_estimation)


def test_estimate_local_noise_non_robust(noise):
    x, sigma = noise
    noise_estimation = ms.peaks._estimate_local_noise(x, robust=False)
    # noise should be close to sigma, check with a 5 % tolerance
    assert (sigma < 1.05 * noise_estimation)


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

def test_smooth():
    sigma = 2
    x1 = gaussian(200, sigma)
    # compare manual convolution with smooth function
    expected_output = np.convolve(x1, x1 / x1.sum(), "same")
    test_output = ms.peaks.smooth(x1, gaussian, sigma)
    assert np.allclose(expected_output, test_output)


def test_find_local_extrema():
    x = np.arange(10)
    # reflect and merge the concatenate x. local extrema should be 0, 9, 19
    x = np.hstack((x, x[::-1]))
    test_output = ms.peaks._find_local_extrema(x)
    expected_output = [0, 9, 19]
    assert np.array_equal(test_output, expected_output)


def test_estimate_noise_probability():
    noise = np.ones(6)
    x = np.array([0, 0.1, 0.4, 2, 0.75, 0.0])
    extrema = np.array([0, 3, 5])
    # two slices of size 4 and 2 respectively, the expected output should
    # be erfc(1/sqrt(2) and erfc(1)
    expected_output = erfc([np.sqrt(1 / 2), 1.0])
    test_output = ms.peaks._estimate_noise_probability(noise, x, extrema)
    assert np.allclose(expected_output, test_output)
