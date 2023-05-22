from tidyms import lcms
from tidyms import utils
import tidyms as ms
import numpy as np
import pytest
from scipy.integrate import trapz
from typing import Tuple
from math import isclose

# # Test Chromatogram object


@pytest.fixture
def chromatogram_data():
    rt = np.arange(200)
    spint = utils.gauss(rt, 50, 2, 100)
    spint += np.random.normal(size=rt.size, scale=1.0)
    return rt, spint


def test_chromatogram_creation(chromatogram_data):
    # test building a chromatogram with default mode
    rt, spint = chromatogram_data
    lcms.Chromatogram(rt, spint)


def test_chromatogram_find_peaks(chromatogram_data):
    rt, spint = chromatogram_data
    chromatogram = lcms.Chromatogram(rt, spint)
    chromatogram.smooth(1.0)
    chromatogram.extract_features()
    # chromatogram.describe_features()
    assert len(chromatogram.features) == 1


# Test MSSpectrum


@pytest.fixture
def centroid_mzml():
    mz = np.linspace(100, 110, 1000)
    spint = utils.gauss(mz, 105, 0.005, 100)
    spint += +np.random.normal(size=mz.size, scale=1.0)
    return mz, spint


def test_ms_spectrum_creation(centroid_mzml):
    sp = lcms.MSSpectrum(*centroid_mzml)
    assert sp.instrument == "qtof"


def test_ms_spectrum_creation_with_instrument(centroid_mzml):
    instrument = "orbitrap"
    sp = lcms.MSSpectrum(*centroid_mzml, instrument=instrument)
    assert sp.instrument == instrument


def test_ms_spectrum_creation_invalid_instrument(centroid_mzml):
    with pytest.raises(ValueError):
        instrument = "invalid-mode"
        lcms.MSSpectrum(*centroid_mzml, instrument=instrument)


def test_find_centroids_qtof(centroid_mzml):
    sp = lcms.MSSpectrum(*centroid_mzml)
    # the algorithm is tested on test_peaks.py
    sp.find_centroids()
    assert True


# Test ROI


@pytest.fixture
def roi_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rt = np.arange(200)
    spint = utils.gauss(rt, 50, 2, 100)
    mz = np.random.normal(loc=150.0, scale=0.001, size=spint.size)
    # add some nan values
    nan_index = [0, 50, 100, 199]
    spint[nan_index] = np.nan
    mz[nan_index] = np.nan
    return rt, mz, spint


@pytest.fixture
def lc_roi_with_peak() -> tuple[lcms.LCTrace, lcms.Peak]:
    n = 200
    # it is not necessary that the signal is an actual peak and make tests easier
    x = np.arange(n)
    y = np.ones_like(x).astype(float)
    noise = np.zeros_like(y)
    lc_roi = ms.lcms.LCTrace(x.astype(float), y, y, x, noise, noise)

    apex = n // 2
    peak = ms.lcms.Peak(apex - 10, apex, apex + 10, lc_roi)
    return lc_roi, peak


def test_fill_nan(roi_data):
    rt, mz, spint = roi_data
    roi = lcms.LCTrace(rt, spint, mz, rt)
    roi.fill_nan(fill_value="extrapolate")
    has_nan = np.any(np.isnan(roi.mz) & np.isnan(roi.spint))
    assert not has_nan


def test_peak_location_init(lc_roi_with_peak):
    # test peak construction
    roi, _ = lc_roi_with_peak
    ms.lcms.Peak(0, 10, 20, roi)
    assert True


def test_peak_location_end_lower_than_loc(lc_roi_with_peak):
    roi, _ = lc_roi_with_peak
    with pytest.raises(ms.lcms.InvalidPeakException):
        ms.lcms.Peak(0, 10, 10, roi)


def test_peak_location_loc_lower_than_start(lc_roi_with_peak):
    roi, _ = lc_roi_with_peak
    with pytest.raises(ms.lcms.InvalidPeakException):
        ms.lcms.Peak(10, 9, 20, roi)


def test_peak_rt(lc_roi_with_peak):
    # check that the location of the peak is close to the estimation
    lc_roi, peak = lc_roi_with_peak
    test_rt = peak.get_rt()
    expected_rt = (lc_roi.time[peak.start] + lc_roi.time[peak.end - 1]) / 2
    assert np.isclose(test_rt, expected_rt)


def test_Peak_get_rt_start(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    test_rt_start = peak.get_rt_start()
    expected_rt_start = lc_roi.time[peak.start]
    assert np.isclose(test_rt_start, expected_rt_start)


def test_Peak_get_rt_end(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    test_rt_end = peak.get_rt_end()
    expected_rt_end = lc_roi.time[peak.end - 1]
    assert np.isclose(test_rt_end, expected_rt_end)


def test_peak_height(lc_roi_with_peak):
    # check that the height of the peak is close to the estimation
    lc_roi, peak = lc_roi_with_peak
    baseline = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = baseline
    test_height = peak.get_height()
    expected_height = lc_roi.spint[peak.apex]
    assert test_height == expected_height


def test_peak_area(lc_roi_with_peak):
    # check that the area of the peak is close to the estimation
    lc_roi, peak = lc_roi_with_peak
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_area = peak.get_area()
    y = lc_roi.spint[peak.start : peak.end]
    x = lc_roi.time[peak.start : peak.end]
    expected_area = trapz(y, x)
    assert test_area == expected_area


def test_peak_width(lc_roi_with_peak):
    # check that the area of the peak is close to the estimation
    lc_roi, peak = lc_roi_with_peak
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_width = peak.get_width()
    width_bound = lc_roi.time[peak.end] - lc_roi.time[peak.start]
    assert test_width <= width_bound


def test_peak_width_bad_width():
    # test that the width is zero when the peak is badly shaped
    y = np.zeros(100)
    x = np.arange(100)
    lc_roi = ms.lcms.LCTrace(y, y, x.astype(float), x)
    peak = ms.lcms.Peak(10, 20, 30, lc_roi)
    lc_roi.baseline = y
    test_width = peak.get_width()
    assert np.isclose(test_width, 0.0)


def test_peak_extension(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    test_extension = peak.get_extension()
    expected_extension = lc_roi.time[peak.end - 1] - lc_roi.time[peak.start]
    assert expected_extension == test_extension


def test_peak_snr(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    lc_roi.noise = np.ones_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_snr = peak.get_snr()
    expected_snr = 1.0
    assert np.isclose(test_snr, expected_snr)


def test_peak_snr_zero_noise(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    lc_roi.noise = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_snr = peak.get_snr()
    expected_snr = np.inf
    assert np.isclose(test_snr, expected_snr)


def test_Feature_describe(lc_roi_with_peak):
    lc_roi, peak = lc_roi_with_peak
    lc_roi.noise = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    peak.describe()
    assert True


# Test ROI serialization


def test_LCRoi_serialization_no_noise_no_baseline_no_features(lc_roi_with_peak):
    roi, _ = lc_roi_with_peak
    roi_str = roi.to_string()
    roi_from_str = lcms.LCTrace.from_string(roi_str)
    assert np.array_equal(roi.time, roi_from_str.time)
    assert np.array_equal(roi.spint, roi_from_str.spint, equal_nan=True)
    assert np.array_equal(roi.mz, roi_from_str.mz, equal_nan=True)
    assert np.array_equal(roi.scan, roi_from_str.scan)


def test_LCRoi_serialization(lc_roi_with_peak):
    roi, _ = lc_roi_with_peak
    roi.extract_features()
    roi_str = roi.to_string()
    roi_from_str = lcms.LCTrace.from_string(roi_str)
    assert np.array_equal(roi.time, roi_from_str.time)
    assert np.array_equal(roi.spint, roi_from_str.spint, equal_nan=True)
    assert np.array_equal(roi.mz, roi_from_str.mz, equal_nan=True)
    assert np.array_equal(roi.scan, roi_from_str.scan)
    assert np.array_equal(roi.noise, roi_from_str.noise)
    assert np.array_equal(roi.baseline, roi_from_str.baseline)
    for expected, test in zip(roi.features, roi_from_str.features):
        assert expected.start == test.start
        assert expected.apex == test.apex
        assert expected.end == test.end


def test__overlap_ratio_overlapping_peaks():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    time_roi1 = scans_roi1.astype(float)
    scans_roi2 = scans[25:55]
    time_roi2 = scans_roi2.astype(float)
    roi1 = lcms.LCTrace(time_roi1, time_roi1, time_roi1, scans_roi1)
    roi2 = lcms.LCTrace(time_roi2, time_roi2, time_roi2, scans_roi2)
    ft1 = lcms.Peak(5, 10, 15, roi1)
    ft2 = lcms.Peak(5, 10, 15, roi2)
    test_result = lcms._overlap_ratio(ft1, ft2)
    expected_result = 0.5
    assert isclose(expected_result, test_result)


def test__overlap_ratio_non_overlapping_peaks():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = lcms.Peak(5, 10, 15, roi1)
    ft2 = lcms.Peak(15, 16, 20, roi2)
    test_result = lcms._overlap_ratio(ft1, ft2)
    expected_result = 0.0
    assert isclose(expected_result, test_result)


def test__overlap_ratio_perfect_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = lcms.Peak(10, 15, 20, roi1)
    ft2 = lcms.Peak(0, 5, 10, roi2)
    test_result = lcms._overlap_ratio(ft1, ft2)
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__get_overlap_index_partial_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[25:55]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = lcms.Peak(5, 10, 15, roi1)
    ft2 = lcms.Peak(5, 10, 15, roi2)
    test_result = lcms._get_overlap_index(ft1, ft2)
    expected_result = 10, 15, 5, 10
    assert test_result == expected_result


def test__get_overlap_index_perfect_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[25:55]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = lcms.Peak(5, 10, 15, roi1)
    ft2 = lcms.Peak(0, 5, 10, roi2)
    test_result = lcms._get_overlap_index(ft1, ft2)
    expected_result = 5, 15, 0, 10
    assert test_result == expected_result


def test__overlap_ratio_ft2_contained_in_ft1():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = lcms.Peak(10, 15, 20, roi1)
    ft2 = lcms.Peak(2, 5, 8, roi2)
    test_result = lcms._overlap_ratio(ft1, ft2)
    # if ft2 is contained in ft1, the overlap ratio is 1.0
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_same_features():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = lcms.Peak(10, 15, 20, roi1)
    test_result = ft1.compare(ft1)
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_non_overlapping_features():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = lcms.Peak(10, 15, 20, roi1)
    scans_roi2 = scans[50:70]
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft2 = lcms.Peak(10, 15, 20, roi2)
    test_result = ft1.compare(ft2)
    expected_result = 0.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_non_overlapping_features_ft1_starts_after_ft2():
    scans = np.arange(100)
    scans_roi1 = scans[50:70]
    roi1 = lcms.LCTrace(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = lcms.Peak(10, 15, 20, roi1)
    scans_roi2 = scans[20:40]
    roi2 = lcms.LCTrace(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft2 = lcms.Peak(10, 15, 20, roi2)
    test_result = ft1.compare(ft2)
    expected_result = 0.0
    assert isclose(expected_result, test_result)
