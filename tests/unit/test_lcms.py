from tidyms import lcms
from tidyms import utils
import tidyms as ms
import numpy as np
import pytest
from scipy.integrate import trapz

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
    chromatogram = lcms.Chromatogram(rt, spint)
    assert chromatogram.mode == "uplc"


def test_chromatogram_creation_with_mode(chromatogram_data):
    rt, spint = chromatogram_data
    chromatogram = lcms.Chromatogram(rt, spint, mode="hplc")
    assert chromatogram.mode == "hplc"


def test_chromatogram_creation_invalid_mode(chromatogram_data):
    rt, spint = chromatogram_data
    with pytest.raises(ValueError):
        lcms.Chromatogram(rt, spint, mode="invalid-mode")


def test_chromatogram_find_peaks(chromatogram_data):
    chromatogram = lcms.Chromatogram(*chromatogram_data)
    chromatogram.extract_features()
    chromatogram.describe_features()
    assert len(chromatogram.features) == 1


# Test MSSpectrum


@pytest.fixture
def centroid_mzml():
    mz = np.linspace(100, 110, 1000)
    spint = utils.gauss(mz, 105, 0.005, 100)
    spint += + np.random.normal(size=mz.size, scale=1.0)
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
def roi_data():
    rt = np.arange(200)
    spint = utils.gauss(rt, 50, 2, 100)
    mz = np.random.normal(loc=150.0, scale=0.001, size=spint.size)
    # add some nan values
    nan_index = [0, 50, 100, 199]
    spint[nan_index] = np.nan
    mz[nan_index] = np.nan
    mode = "uplc"
    return rt, mz, spint, mode


def test_roi_creation(roi_data):
    rt, mz, spint, mode = roi_data
    lcms.Roi(spint, mz, rt, rt, mode)
    assert True


def test_fill_nan(roi_data):
    rt, mz, spint, mode = roi_data
    roi = lcms.Roi(spint, mz, rt, rt, mode)
    roi.fill_nan()
    has_nan = np.any(np.isnan(roi.mz) & np.isnan(roi.spint))
    assert not has_nan


def test_peak_location_init():
    # test peak construction
    ms.lcms.Peak(0, 10, 20)
    assert True


def test_peak_location_end_lower_than_loc():
    with pytest.raises(ms.lcms.InvalidPeakException):
        ms.lcms.Peak(0, 10, 10)


def test_peak_location_loc_lower_than_start():
    with pytest.raises(ms.lcms.InvalidPeakException):
        ms.lcms.Peak(10, 9, 20)


@pytest.fixture
def lc_roi_peak():
    n = 200
    # it is not necessary that the signal is an actual peak and make tests
    # easier
    x = np.arange(n)
    y = np.ones_like(x)
    lc_roi = ms.lcms.LCRoi(y, y, x, x)
    apex = n // 2
    peak = ms.lcms.Peak(apex - 10, apex, apex + 10)
    return lc_roi, peak


def test_peak_rt(lc_roi_peak):
    # check that the location of the peak is close to the estimation
    lc_roi, peak = lc_roi_peak
    test_rt = peak.get_rt(lc_roi)
    expected_rt = (lc_roi.time[peak.start] + lc_roi.time[peak.end - 1]) / 2
    assert np.isclose(test_rt, expected_rt)


def test_Peak_get_rt_start(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    test_rt_start = peak.get_rt_start(lc_roi)
    expected_rt_start = lc_roi.time[peak.start]
    assert np.isclose(test_rt_start, expected_rt_start)


def test_Peak_get_rt_end(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    test_rt_end = peak.get_rt_end(lc_roi)
    expected_rt_end = lc_roi.time[peak.end - 1]
    assert np.isclose(test_rt_end, expected_rt_end)


def test_peak_height(lc_roi_peak):
    # check that the height of the peak is close to the estimation
    lc_roi, peak = lc_roi_peak
    baseline = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = baseline
    test_height = peak.get_height(lc_roi)
    expected_height = lc_roi.spint[peak.apex]
    assert test_height == expected_height


def test_peak_area(lc_roi_peak):
    # check that the area of the peak is close to the estimation
    lc_roi, peak = lc_roi_peak
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_area = peak.get_area(lc_roi)
    y = lc_roi.spint[peak.start:peak.end]
    x = lc_roi.time[peak.start:peak.end]
    expected_area = trapz(y, x)
    assert test_area == expected_area


def test_peak_width(lc_roi_peak):
    # check that the area of the peak is close to the estimation
    lc_roi, peak = lc_roi_peak
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_width = peak.get_width(lc_roi)
    width_bound = lc_roi.time[peak.end] - lc_roi.time[peak.start]
    assert test_width <= width_bound


def test_peak_width_bad_width():
    # test that the width is zero when the peak is badly shaped
    peak = ms.lcms.Peak(10, 20, 30)
    y = np.zeros(100)
    x = np.arange(100)
    lc_roi = ms.lcms.LCRoi(y, y, x, x)
    lc_roi.baseline = y
    test_width = peak.get_width(lc_roi)
    assert np.isclose(test_width, 0.0)


def test_peak_extension(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    test_extension = peak.get_extension(lc_roi)
    expected_extension = lc_roi.time[peak.end] - lc_roi.time[peak.start]
    assert expected_extension == test_extension


def test_peak_snr(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    lc_roi.noise = np.ones_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_snr = peak.get_snr(lc_roi)
    expected_snr = 1.0
    assert np.isclose(test_snr, expected_snr)


def test_peak_snr_zero_noise(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    lc_roi.noise = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    test_snr = peak.get_snr(lc_roi)
    expected_snr = np.inf
    assert np.isclose(test_snr, expected_snr)


# test peak descriptors

def test_fill_filter_boundaries_fill_upper_bound():
    filters = {"loc": (50, None), "snr": (5, 10)}
    ms.lcms._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][1], np.inf)


def test_fill_filter_boundaries_fill_lower_bound():
    filters = {"loc": (None, 50), "snr": (5, 10)}
    ms.lcms._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][0], -np.inf)


def test_has_all_valid_descriptors():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (3, 10)}
    assert ms.lcms._has_all_valid_descriptors(descriptors, filters)


def test_has_all_valid_descriptors_descriptors_outside_valid_ranges():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (10, 20)}
    assert not ms.lcms._has_all_valid_descriptors(descriptors, filters)


def test_get_descriptors(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    lc_roi.noise = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)
    peak.get_descriptors(lc_roi)
    assert True


def test_get_descriptors_custom_descriptors(lc_roi_peak):
    lc_roi, peak = lc_roi_peak
    lc_roi.features = [peak]
    lc_roi.noise = np.zeros_like(lc_roi.spint)
    lc_roi.baseline = np.zeros_like(lc_roi.spint)

    def return_one(roi, peak):
        return 1

    custom_descriptor = {"custom": return_one}
    descriptors = lc_roi.describe_features(custom_descriptors=custom_descriptor)
    assert descriptors[0]["custom"] == 1


# Test ROI serialization

@pytest.fixture
def lc_roi(roi_data):
    rt, mz, spint, mode = roi_data
    return lcms.LCRoi(
        spint,
        mz,
        rt,
        rt,
        mode=mode
    )


def test_LCRoi_serialization_no_noise_no_baseline_no_features(lc_roi):
    roi_str = lc_roi.to_json()
    roi_from_str = lcms.LCRoi.from_json(roi_str)
    assert np.array_equal(lc_roi.time, roi_from_str.time)
    assert np.array_equal(lc_roi.spint, roi_from_str.spint, equal_nan=True)
    assert np.array_equal(lc_roi.mz, roi_from_str.mz, equal_nan=True)
    assert np.array_equal(lc_roi.scan, roi_from_str.scan)


def test_LCRoi_serialization(lc_roi):
    lc_roi.extract_features()
    roi_str = lc_roi.to_json()
    roi_from_str = lcms.LCRoi.from_json(roi_str)
    assert np.array_equal(lc_roi.time, roi_from_str.time)
    assert np.array_equal(lc_roi.spint, roi_from_str.spint, equal_nan=True)
    assert np.array_equal(lc_roi.mz, roi_from_str.mz, equal_nan=True)
    assert np.array_equal(lc_roi.scan, roi_from_str.scan)
    assert np.array_equal(lc_roi.noise, roi_from_str.noise)
    assert np.array_equal(lc_roi.baseline, roi_from_str.baseline)
    for expected, test in zip(lc_roi.features, roi_from_str.features):
        assert expected.start == test.start
        assert expected.apex == test.apex
        assert expected.end == test.end
