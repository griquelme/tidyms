import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from scipy.special import erfc
from tidyms import utils
from tidyms.algorithms import peaks

SEED = 1234

SIGMA = 1.0

@pytest.fixture(scope="module")
def noise():
    np.random.seed(SEED)
    return np.random.normal(size=500, scale=SIGMA)


class TestEstimateLocalNoise:
    def test_empty_signal_returns_zero(self):
        x = np.array([])
        noise = peaks._estimate_local_noise(x)
        assert np.isclose(noise, 0.0)

    @pytest.mark.parametrize("x", [np.array([1]), np.array([1, 2])])
    def test_signal_length_lower_than_two(self, x):
        noise = peaks._estimate_local_noise(x)
        assert np.isclose(noise, 0.0)

    @pytest.mark.parametrize("robust", [False, True])
    def test_noise_estimation_from_gaussian_noise_is_close_to_std(self, noise, robust):
        # check that the noise estimation is close to the std of a normal distribution
        actual = peaks._estimate_local_noise(noise, robust=robust)
        # noise should be close to sigma, check with a 20 % tolerance
        assert abs(actual - SIGMA) / SIGMA < 0.2


class TestEstimateNoise:
    def empty_array_returns_empty_array(self):
        x = np.array([])
        noise = peaks.estimate_noise(x)
        assert noise.size == 0.0

    @pytest.mark.parametrize("x", [np.array([1]), np.array([1, 3]), np.array([1, 4, 6])])
    def test_signal_length_lower_than_two(self, x):
        noise_estimation = peaks.estimate_noise(x)
        assert np.allclose(noise_estimation, 0.0)

    def test_noise_estimation_size(self, noise):
        n_slices = 2
        actual = peaks.estimate_noise(noise, n_slices=n_slices)
        assert np.all(actual >= 0.0)
        assert np.unique(actual).size == n_slices
        assert noise.size == actual.size

    def test_estimate_noise_number_of_unique_values_in_estimation(self, noise):
        n_slices = 2
        actual = peaks.estimate_noise(noise, n_slices=n_slices)
        assert np.all(actual >= 0.0)
        assert np.unique(actual).size == n_slices


    def test_min_slice_size_has_higher_priority_than_n_slices_to_set_slice_size(self, noise):
        n_slices = 5
        min_slice_size = 150
        noise_estimation = peaks.estimate_noise(noise, n_slices=n_slices, min_slice_size=min_slice_size)
        # noise has a size of 500, the slice is going to be 100 < 150 check that
        # 150 is used instead. There should be three unique values
        actual_n_slices = np.unique(noise_estimation).size
        assert actual_n_slices == 3


class TestFindLocalExtrema:
    def test_first_middle_and_last_indices_are_extrema_in_triangular_pulse(self):
        x = np.arange(10)
        triangular_pulse = np.hstack((x, x[::-1]))
        actual = peaks._find_local_extrema(triangular_pulse)
        expected = [0, 9, 19]
        assert np.array_equal(actual, expected)


    def test_monotone_signal_has_no_local_extrema(self):
        monotone_signal = np.arange(10)
        actual = peaks._find_local_extrema(monotone_signal)
        expected = np.array([])
        assert np.array_equal(actual, expected)


class TestBaseline:
    test_noise_sum_params = [[np.array([0, 1]), np.sqrt([25, 25])], [np.array([0]), np.sqrt([34])]]

    @pytest.mark.parametrize("index,expected", test_noise_sum_params)
    def test_get_noise_sum_slice_std(self, index, expected):
        index = np.array(index)
        expected = np.array(expected)
        x = np.array([3, 4, 2, 2, 1])
        test_output = peaks._get_noise_slice_sum_std(x, index)
        assert np.allclose(test_output, expected)


    def test_estimate_noise_probability(self):
        noise = np.ones(7)
        x = np.array([0, 0.1, 0.4, 2, 1.25, 1.1, 1.0])
        extrema = np.array([0, 3, 6])
        # two slices of size 4 and 2 respectively, the expected output should
        # be erfc(1/sqrt(2) and erfc(1)
        expected_output = erfc([2.5 * np.sqrt(1 / 2) / 2, 1.35 * np.sqrt(1 / 2) / 2])
        test_output = peaks._estimate_noise_probability(noise, x, extrema)
        assert np.allclose(expected_output, test_output)


    def test_build_baseline_index(self):
        x = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0])
        extrema = np.array([0, 2, 4, 6, 8, 10, 12])
        noise_probability = np.array([0, 0.25, 0.25, 0.25, 0, 0])
        min_proba = 0.05
        expected = np.array([0, 4, 5, 6, 12])
        test = peaks._build_baseline_index(x, noise_probability, min_proba, extrema)
        assert np.array_equal(expected, test)

    def test_estimate_baseline(self):
        # a simple test, a noise array is built using a noise level greater
        # than the noise level in the signal. All points should be classified as
        # baseline
        n = 100
        x = np.random.normal(size=n, scale=1)
        noise = np.ones(n) * 5
        baseline = peaks.estimate_baseline(x, noise)
        expected_baseline_index = np.arange(n)
        test_baseline_index = np.where(np.abs(x - baseline) < noise)[0]
        assert np.array_equal(expected_baseline_index, test_baseline_index)


class TestDetectPeaks:

    @pytest.fixture(scope="class")
    def single_peak(self, noise):
        x = gaussian(noise.size, 2) * 20
        return x

    @pytest.fixture(scope="class")
    def two_non_overlapping_peaks(self, noise):
        x = np.arange(noise.size)
        params = np.array([[100, 2, 50], [150, 2, 25]])
        return utils.gaussian_mixture(x, params).sum(axis=0)

    @pytest.fixture
    def two_overlapping_peaks(self, noise):
        x = np.arange(noise.size)
        params = np.array([[100, 2, 50], [108, 2, 25]])
        return utils.gaussian_mixture(x, params).sum(axis=0)

    def test_signal_with_one_peak_detect_single_peak(self, single_peak, noise):
        x = single_peak + noise
        noise_estimation = peaks.estimate_noise(x)

        # smooth x to reduce the number of detected peaks
        x = gaussian_filter1d(x, 1.0)

        baseline_estimation = peaks.estimate_baseline(x, noise)
        peak_list = peaks.detect_peaks(x, noise_estimation, baseline_estimation)

        assert len(peak_list[0]) == 1

    def test_detect_peaks_two_non_overlapping_peaks(self, two_non_overlapping_peaks, noise):
        x = two_non_overlapping_peaks + noise
        noise_estimation = peaks.estimate_noise(x)
        # smooth x to reduce the number of detected peaks
        x = gaussian_filter1d(x, 1.0)
        baseline_estimation = peaks.estimate_baseline(x, noise)
        peak_list = peaks.detect_peaks(x, noise_estimation, baseline_estimation)
        assert len(peak_list[0]) == 2

    def test_detect_peaks_two_overlapping_peaks(self, two_overlapping_peaks, noise):
        x = two_overlapping_peaks + noise
        noise_estimation = peaks.estimate_noise(x)
        # smooth x to reduce the number of detected peaks
        x = gaussian_filter1d(x, 1.0)

        baseline_estimation = peaks.estimate_baseline(x, noise)

        peak_list = peaks.detect_peaks(x, noise_estimation, baseline_estimation)
        start, apex, end = peak_list

        expected_n_peaks = 2
        assert len(start) == expected_n_peaks

        # check the boundary of the overlapping peaks
        assert end[0] == (start[1] + 1)
