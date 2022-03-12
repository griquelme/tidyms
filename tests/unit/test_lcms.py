from tidyms import lcms
from tidyms import utils
import numpy as np
import pytest
from itertools import product

# # Test Chromatogram object
#
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
    chromatogram.find_peaks()
    assert len(chromatogram.peaks) == 1


# Test MSSPectrum


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

    return rt, mz, spint


def test_roi_creation(roi_data):
    rt, mz, spint = roi_data
    lcms.Roi(spint, mz, rt, rt)
    assert True


def test_fill_nan(roi_data):
    rt, mz, spint = roi_data
    roi = lcms.Roi(spint, mz, rt, rt)
    roi.fill_nan()
    has_nan = np.any(np.isnan(roi.mz) & np.isnan(roi.spint))
    assert not has_nan


# roi making tests


def test_match_mz_no_multiple_matches():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([40, 51, 78, 91, 99, 130, 150])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([0, 2, 4], dtype=int)
    mz2_match_index = np.array([1, 4, 6], dtype=int)
    mz2_no_match_index = np.array([0, 2, 3, 5], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
        lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)

def test_match_mz_no_matches():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([40, 53, 78, 91, 97, 130, 154])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([], dtype=int)
    mz2_match_index = np.array([], dtype=int)
    mz2_no_match_index = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
        lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_all_match():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([51, 77, 99, 126, 150])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
    mz2_no_match_index = np.array([], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
        lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_multiple_matches_mode_closest():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    # in closest mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 4, 6, 7], dtype=int)
    mz2_no_match_index = np.array([1, 2, 3, 5, 8], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
        lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_multiple_matches_mode_reduce():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150], dtype=float)
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151], dtype=float)
    sp2 = np.array([100] * mz2.size, dtype=float)
    # expected values for match/no match indices
    # in closest mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 1, 3, 4, 5, 6, 7, 8], dtype=int)
    mz2_no_match_index = np.array([2], dtype=int)
    expected_mz2_match = [50.0, 100.0, 126.0, 150.5]
    expected_sp2_match = [200, 300, 100, 200]
    mode = "reduce"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
        lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.sum)
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.allclose(mz2_match, expected_mz2_match)
    assert np.allclose(sp2_match, expected_sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_invalid_mode():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    # in closest mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 4, 6, 7], dtype=int)
    mz2_no_match_index = np.array([1, 2, 3, 5, 8], dtype=int)
    mode = "invalid-mode"
    with pytest.raises(ValueError):
        test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = \
            lcms._match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)


# test default parameter functions

def test_get_lc_filter_params_uplc():
    lcms.get_lc_filter_peak_params("uplc")
    assert True


def test_get_lc_filter_params_hplc():
    lcms.get_lc_filter_peak_params("hplc")
    assert True


def test_get_lc_filter_params_invalid_mode():
    with pytest.raises(ValueError):
        lcms.get_lc_filter_peak_params("invalid-mode")
