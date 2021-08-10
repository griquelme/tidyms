from tidyms import lcms
from tidyms import utils
import numpy as np
import pytest
from itertools import product

mz_list = [200, 250, 300, 420, 450]


@pytest.fixture
def simulated_experiment():
    mz = np.array(mz_list)
    rt = np.linspace(0, 100, 100)

    # simulated features params
    mz_params = np.array([mz_list,
                          [3, 10, 5, 31, 22]])
    mz_params = mz_params.T
    rt_params = np.array([[30, 40, 60, 80, 80],
                          [1, 2, 2, 3, 3],
                          [1, 1, 1, 1, 1]])
    rt_params = rt_params.T

    noise_level = 0.1
    sim_exp = utils.SimulatedExperiment(mz, rt, mz_params, rt_params,
                                        noise=noise_level, mode="centroid")
    return sim_exp


# parameters of make_chromatograms are tested in the test_validation module

def test_make_chromatograms(simulated_experiment):
    # test that the chromatograms generated are valid

    # create chromatograms
    n_sp = simulated_experiment.getNrSpectra()
    n_mz = simulated_experiment.mz_params.shape[0]
    rt = np.zeros(n_sp)
    chromatogram = np.zeros((n_mz, n_sp))
    for scan in range(n_sp):
        sp = simulated_experiment.getSpectrum(scan)
        rt[scan] = sp.getRT()
        _, spint = sp.get_peaks()
        chromatogram[:, scan] = spint

    expected_chromatograms = [lcms.Chromatogram(rt, x) for x in chromatogram]
    test_chromatograms = lcms.make_chromatograms(simulated_experiment, mz_list)
    assert len(test_chromatograms) == len(expected_chromatograms)
    for ec, tc in zip(expected_chromatograms, test_chromatograms):
        assert np.array_equal(ec.rt, tc.rt)
        assert np.array_equal(ec.spint, tc.spint)


def test_make_chromatograms_accumulator_mean(simulated_experiment):
    lcms.make_chromatograms(simulated_experiment, mz_list, accumulator="mean")
    assert True


def test_make_chromatograms_start(simulated_experiment):
    n_sp = simulated_experiment.getNrSpectra()
    start = 10
    chromatogram_length = n_sp - start
    chromatograms = lcms.make_chromatograms(simulated_experiment, mz_list,
                                            start=start)
    for c in chromatograms:
        assert c.rt.size == chromatogram_length
        assert c.rt[0] == simulated_experiment.getSpectrum(start).getRT()


def test_make_chromatograms_end(simulated_experiment):
    end = 90
    chromatograms = lcms.make_chromatograms(simulated_experiment, mz_list,
                                            end=end)
    for c in chromatograms:
        assert c.rt.size == end
        assert c.rt[-1] == simulated_experiment.getSpectrum(end - 1).getRT()


def test_make_chromatograms_outside_range_mz(simulated_experiment):
    # the total intensity of the chromatogram should be equal to zero
    chromatograms = lcms.make_chromatograms(simulated_experiment, [550])
    assert np.isclose(chromatograms[0].spint.sum(), 0)


# def test_accumulate_spectra(simulated_experiment):
#     lcms.accumulate_spectra_profile(simulated_experiment, start=10, end=20)
#     assert True
#
#
# def test_accumulate_spectra_subtract(simulated_experiment):
#     lcms.accumulate_spectra_profile(simulated_experiment, start=10, end=20,
#                                     subtract_left=5, subtract_right=25)
#     assert True
#
#
# def test_get_roi_params():
#     func_params = [("uplc", "qtof"), ("uplc", "orbitrap"), ("hplc", "qtof"),
#               ("hplc", "orbitrap")]
#     n_sp = 100  # dummy value for the validator
#     for separation, instrument in func_params:
#         params = lcms.get_roi_params(separation, instrument)
#         validation.validate_make_roi_params(n_sp, params)
#     assert True
#
#
# def test_get_roi_params_bad_separation():
#     with pytest.raises(ValueError):
#         lcms.get_roi_params("bad-value", "qtof")
#
#
# def test_get_roi_params_bad_instrument():
#     with pytest.raises(ValueError):
#         lcms.get_roi_params("uplc", "bad-value")
#
#
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
def ms_data():
    mz = np.linspace(100, 110, 1000)
    spint = utils.gauss(mz, 105, 0.005, 100)
    spint += + np.random.normal(size=mz.size, scale=1.0)
    return mz, spint


def test_ms_spectrum_creation(ms_data):
    sp = lcms.MSSpectrum(*ms_data)
    assert sp.instrument == "qtof"


def test_ms_spectrum_creation_with_instrument(ms_data):
    instrument = "orbitrap"
    sp = lcms.MSSpectrum(*ms_data, instrument=instrument)
    assert sp.instrument == instrument


def test_ms_spectrum_creation_invalid_instrument(ms_data):
    with pytest.raises(ValueError):
        instrument = "invalid-mode"
        lcms.MSSpectrum(*ms_data, instrument=instrument)


def test_find_centroids_qtof(ms_data):
    sp = lcms.MSSpectrum(*ms_data)
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


def test_make_roi(simulated_experiment):
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce")
    assert len(roi_list) == simulated_experiment.mz_params.shape[0]


def test_make_roi_targeted_mz(simulated_experiment):
    # the first three m/z values generated by simulated experiment are used
    targeted_mz = simulated_experiment.mz_params[:, 0][:3]
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", targeted_mz=targeted_mz)
    assert len(roi_list) == targeted_mz.size


def test_make_roi_min_intensity(simulated_experiment):
    min_intensity = 15
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0,
                             min_intensity=min_intensity,
                             multiple_match="reduce")
    # only two roi should have intensities greater than 15
    assert len(roi_list) == 2


def test_make_roi_start(simulated_experiment):
    start = 10
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", start=start)
    n_sp = simulated_experiment.getNrSpectra()
    for r in roi_list:
        assert r.mz.size == (n_sp - start)


def test_make_roi_end(simulated_experiment):
    end = 10
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", end=end)
    n_sp = simulated_experiment.getNrSpectra()
    for r in roi_list:
        assert r.mz.size == end

def test_make_roi_multiple_match_closest(simulated_experiment):
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="closest")
    assert len(roi_list) == simulated_experiment.mz_params.shape[0]


def test_make_roi_multiple_match_reduce_merge(simulated_experiment):
    # set a tolerance such that two mz values are merged
    # test is done in targeted mode to force a multiple match by removing
    # one of the mz values
    targeted_mz = simulated_experiment.mz_params[:, 0]
    targeted_mz = np.delete(targeted_mz, 3)
    tolerance = 31
    roi_list = lcms.make_roi(simulated_experiment, tolerance=tolerance,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", targeted_mz=targeted_mz)
    assert len(roi_list) == (simulated_experiment.mz_params.shape[0] - 1)


def test_make_roi_multiple_match_reduce_custom_mz_reduce(simulated_experiment):
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", mz_reduce=np.median)
    assert len(roi_list) == simulated_experiment.mz_params.shape[0]


def test_make_roi_multiple_match_reduce_custom_sp_reduce(simulated_experiment):
    sp_reduce = lambda x: 1
    roi_list = lcms.make_roi(simulated_experiment, tolerance=0.005,
                             max_missing=0, min_length=0, min_intensity=0,
                             multiple_match="reduce", sp_reduce=sp_reduce)
    assert len(roi_list) == simulated_experiment.mz_params.shape[0]


def test_make_roi_invalid_multiple_match(simulated_experiment):
    with pytest.raises(ValueError):
        lcms.make_roi(simulated_experiment, tolerance=0.005,  max_missing=0,
                      min_length=0, min_intensity=0,
                      multiple_match="invalid-value")


# test accumulate spectra

def test_accumulate_spectra_centroid(simulated_experiment):
    n_sp = simulated_experiment.getNrSpectra()
    sp = lcms.accumulate_spectra_centroid(simulated_experiment, 0, n_sp - 1,
                                          tolerance=0.005)
    assert sp.mz.size == simulated_experiment.mz_params.shape[0]


def test_accumulate_spectra_centroid_subtract_left(simulated_experiment):
    sp = lcms.accumulate_spectra_centroid(simulated_experiment, 70, 90,
                                          subtract_left=20, tolerance=0.005)
    # only two peaks at rt 80 should be present
    assert sp.mz.size == 2


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


@pytest.mark.parametrize("separation,instrument",
                         list(product(["hplc", "uplc"], ["qtof", "orbitrap"])))
def test_get_roi_params(separation, instrument):
    lcms.get_roi_params(separation, instrument)
    assert True


def test_get_roi_params_bad_separation():
    with pytest.raises(ValueError):
        lcms.get_roi_params("invalid-separation", "qtof")


def test_get_roi_params_bad_ms_mode():
    with pytest.raises(ValueError):
        lcms.get_roi_params("uplc", "invalid-ms-mode")
