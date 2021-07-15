import tidyms as ms
import numpy as np
import pytest
from scipy.signal.windows import gaussian
from scipy.special import erfc
from scipy.integrate import trapz
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
    assert len(peaks) == 1


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
    assert len(peaks) == 2


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
    # only two peaks are detected
    assert len(peaks) == 2
    # check the boundary of the overlapping peaks
    assert peaks[0].end == (peaks[1].start + 1)


# test PeakLocation

def test_peak_location_init():
    # test peak construction
    ms.peaks.Peak(0, 10, 20)
    assert True


def test_peak_location_end_lower_than_loc():
    with pytest.raises(ms.peaks.InvalidPeakException):
        ms.peaks.Peak(0, 10, 10)


def test_peak_location_loc_lower_than_start():
    with pytest.raises(ms.peaks.InvalidPeakException):
        ms.peaks.Peak(10, 9, 20)


@pytest.fixture
def x_y_peak():
    n = 200
    x = np.arange(n)
    # it is not necessary that the signal is an actual peak and make tests
    # easier
    y = np.ones(n)
    apex = n // 2
    peak = ms.peaks.Peak(apex - 10, apex, apex + 10)
    return x, y, peak


def test_peak_loc(x_y_peak):
    # check that the location of the peak is close to the estimation
    x, y, peak = x_y_peak
    test_loc = peak.get_loc(x, y)
    expected_loc = (x[peak.start] + x[peak.end - 1]) / 2
    assert np.isclose(test_loc, expected_loc)


def test_peak_height(x_y_peak):
    # check that the height of the peak is close to the estimation
    x, y, peak = x_y_peak
    baseline = np.zeros_like(x)
    test_height = peak.get_height(y, baseline)
    expected_height = y[peak.apex]
    assert test_height == expected_height


def test_peak_area(x_y_peak):
    # check that the area of the peak is close to the estimation
    x, y, peak = x_y_peak
    baseline = np.zeros_like(x)
    test_area = peak.get_area(x, y, baseline)
    expected_area = trapz(y[peak.start:peak.end], x[peak.start:peak.end])
    assert test_area == expected_area


def test_peak_width(x_y_peak):
    # check that the area of the peak is close to the estimation
    x, y, peak = x_y_peak
    baseline = np.zeros_like(x)
    test_width = peak.get_width(x, y, baseline)
    width_bound = x[peak.end] - x[peak.start]
    assert test_width <= width_bound


def test_peak_width_bad_width():
    # test that the width is zero when the peak is badly shaped
    peak = ms.peaks.Peak(10, 20, 30)
    y = np.zeros(100)
    x = np.arange(100)
    baseline = y
    test_width = peak.get_width(x, y, baseline)
    assert np.isclose(test_width, 0.0)


def test_peak_extension(x_y_peak):
    x, y, peak = x_y_peak
    test_extension = peak.get_extension(x)
    expected_extension = x[peak.end] - x[peak.start]
    assert expected_extension == test_extension


def test_peak_snr(x_y_peak):
    x, y, peak = x_y_peak
    noise = np.ones_like(x)
    baseline = np.zeros_like(x)
    test_snr = peak.get_snr(y, noise, baseline)
    expected_snr = 1.0
    assert np.isclose(test_snr, expected_snr)


def test_peak_snr_zero_noise(x_y_peak):
    x, y, peak = x_y_peak
    noise = np.zeros_like(x)
    baseline = np.zeros_like(x)
    test_snr = peak.get_snr(y, noise, baseline)
    expected_snr = np.inf
    assert np.isclose(test_snr, expected_snr)


# test peak descriptors

def test_fill_filter_boundaries_fill_upper_bound():
    filters = {"loc": (50, None), "snr": (5, 10)}
    ms.peaks._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][1], np.inf)


def test_fill_filter_boundaries_fill_lower_bound():
    filters = {"loc": (None, 50), "snr": (5, 10)}
    ms.peaks._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][0], -np.inf)


def test_has_all_valid_descriptors():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (3, 10)}
    assert ms.peaks._has_all_valid_descriptors(descriptors, filters)


def test_has_all_valid_descriptors_descriptors_outside_valid_ranges():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (10, 20)}
    assert not ms.peaks._has_all_valid_descriptors(descriptors, filters)


def test_get_descriptors(x_y_peak):
    x, y, peak = x_y_peak
    peaks = [peak]
    noise = np.zeros_like(x)
    baseline = np.zeros_like(x)
    ms.peaks.get_peak_descriptors(x, y, noise, baseline, peaks)
    assert True


def test_get_descriptors_custom_descriptors(x_y_peak):
    x, y, peak = x_y_peak
    peaks = [peak]
    noise = np.zeros_like(x)
    baseline = np.zeros_like(x)

    def return_one(x, y, noise, baseline, peak):
        return 1

    custom_descriptor = {"custom": return_one}

    _, descriptors = \
        ms.peaks.get_peak_descriptors(x, y, noise, baseline, peaks,
                                      descriptors=custom_descriptor)
    assert descriptors[0]["custom"] == 1
