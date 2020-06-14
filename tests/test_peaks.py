import ms_feature_validation as mfv
import numpy as np
import pytest
from itertools import product

# test peak picking functions using typical values expected in chromatography

@pytest.fixture
def peak_data():
    # the same data is generated always
    np.random.seed(1234)
    x = np.arange(100)
    noise = np.random.normal(size=x.size, scale=0.25)
    # the scale used for peak picking
    baseline = np.ones_like(x) * 4
    widths = [np.linspace(0.25, 5, 20), np.linspace(6, 20, 8),
              np.linspace(25, 60, 8)]
    widths = np.hstack(widths)
    return x, widths, baseline, noise


def test_one_peak_several_widths(peak_data):
    x, widths, baseline, noise = peak_data
    peak_loc = 50
    peak_height = 10
    peak_widths = np.linspace(2, 5, 10)
    for w in peak_widths:
        y = mfv.utils.gauss(x, peak_loc, w, peak_height)
        y += baseline + noise
        peaks, params = mfv.peaks.detect_peaks(x, y, widths, max_width=60,
                                               snr=5)
        # the number of peaks should be 1
        # the error in the peak location should be smaller than the peak width
        assert abs(params[0]["loc"] - peak_loc) <= (w + 1)


def test_one_peak_several_snr(peak_data):
    x, widths, baseline, _ = peak_data
    np.random.seed(1234)
    peak_loc = 50
    peak_height = 10
    peak_width = 3
    # snr used in peak pick is 10, max noise std should be 1
    noise_list = np.linspace(0.1, 0.9, 10)
    for n in noise_list:
        y = mfv.utils.gauss(x, peak_loc, peak_width, peak_height)
        noise = np.random.normal(size=x.size, scale=n)
        y += baseline + noise
        peaks, params = mfv.peaks.detect_peaks(x, y, widths, max_width=60,
                                               snr=5)
        if len(peaks) == 0:
            print(n)
        assert len(peaks) == 1
        assert len(peaks) == 1
        assert abs(params[0]["loc"] - peak_loc) <= (peak_width + 1)


def test_two_overlapping_peaks(peak_data):
    x, widths, baseline, noise = peak_data
    np.random.seed(1234)
    peak_loc = 50
    peak_height = 20
    peak_widths = [2, 3, 4]
    peak_dist = np.arange(2, 10)
    peak_ratio = np.logspace(-0.5, 1, 20)
    for peak_width, dist, ratio in product(peak_widths, peak_dist, peak_ratio):
        gm_params = [[peak_loc, peak_width, peak_height],
                     [peak_loc + dist, peak_width, peak_height * ratio]]
        gm_params = np.array(gm_params)
        y = mfv.utils.gaussian_mixture(x, gm_params).sum(axis=0)
        y += baseline + noise
        peaks, params = mfv.peaks.detect_peaks(x, y, widths, max_width=100,
                                               snr=5, max_distance=1)
        if len(peaks) == 0:
            print(peak_width, dist, ratio)
        assert len(peaks) <= 2
        assert (((params[0]["loc"] - peak_loc) <= peak_width + 1) or
                ((-params[0]["loc"] - peak_loc - dist) <= peak_width + 1))
