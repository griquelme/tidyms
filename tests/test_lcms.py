from tidyms import lcms
from tidyms import validation
from tidyms import utils
import numpy as np
import pytest

mz_list = [200, 250, 300, 420, 450]

@pytest.fixture
def simulated_experiment():
    mz_res = 0.001
    min_mz = 100
    max_mz = 500
    n_mz = int((max_mz - min_mz) / mz_res)
    mz = np.linspace(min_mz, max_mz, n_mz)

    rt = np.linspace(0, 100, 100)

    # simulated features params
    mz_params = np.array([mz_list,
                          [0.02, 0.02, 0.02, 0.02, 0.02],
                          [3, 10, 5, 31, 22]])
    mz_params = mz_params.T
    rt_params = np.array([[30, 40, 60, 80, 80],
                          [1, 2, 2, 3, 3],
                          [3, 10, 5, 31, 22]])
    rt_params = rt_params.T

    noise_level = 1
    simexp = utils.SimulatedExperiment(mz, rt, mz_params, rt_params,
                                       noise=noise_level, mode="profile")
    return simexp


# parameters of make_chromatograms are tested in the test_validation module

def test_make_chromatograms(simulated_experiment):
    lcms.make_chromatograms(simulated_experiment, mz_list)
    assert True


def test_make_chromatograms_accumulator_mean(simulated_experiment):
    lcms.make_chromatograms(simulated_experiment, mz_list, accumulator="mean")
    assert True


def test_make_chromatograms_start(simulated_experiment):
    n_sp = simulated_experiment.getNrSpectra()
    start = 10
    chromatogram_length = n_sp - start
    rt, chromatograms = lcms.make_chromatograms(simulated_experiment, mz_list,
                                                start=start)
    assert rt.size == chromatogram_length
    assert rt[0] == simulated_experiment.getSpectrum(start).getRT()


def test_make_chromatograms_end(simulated_experiment):
    n_sp = simulated_experiment.getNrSpectra()
    end = 90
    rt, chromatograms = lcms.make_chromatograms(simulated_experiment, mz_list,
                                                end=end)
    assert rt.size == end
    assert rt[-1] == simulated_experiment.getSpectrum(end - 1).getRT()


def test_make_chromatograms_outside_range_mz(simulated_experiment):
    # the total intensity of the chromatogram should be equal to zero
    _, chromatograms = lcms.make_chromatograms(simulated_experiment, [550])
    assert np.isclose(chromatograms.sum(), 0)


def test_accumulate_spectra(simulated_experiment):
    lcms.accumulate_spectra_profile(simulated_experiment, start=10, end=20)
    assert True


def test_accumulate_spectra_subtract(simulated_experiment):
    lcms.accumulate_spectra_profile(simulated_experiment, start=10, end=20,
                                    subtract_left=5, subtract_right=25)
    assert True


def test_make_widths_lc():
    lcms.make_widths_lc(mode="hplc")
    lcms.make_widths_lc(mode="uplc")
    with pytest.raises(ValueError):
        lcms.make_widths_lc(mode="invalid_mode")
    assert True


def test_make_widths_ms():
    lcms.make_widths_ms(mode="qtof")
    lcms.make_widths_ms(mode="orbitrap")
    with pytest.raises(ValueError):
        lcms.make_widths_ms(mode="invalid_mode")
    assert True


def test_get_lc_cwt_params():
    params = lcms.get_lc_detect_peak_params("uplc", "cwt")
    validation.validate_cwt_peak_picking_params(params)
    params = lcms.get_lc_detect_peak_params("hplc", "cwt")
    validation.validate_cwt_peak_picking_params(params)
    with pytest.raises(ValueError):
        params = lcms.get_lc_detect_peak_params("invalid_mode", "cwt")


def test_get_lc_peak_params_max():
    params = lcms.get_lc_detect_peak_params("uplc", "max")
    validation.validate_max_peak_picking_params(params)
    params = lcms.get_lc_detect_peak_params("hplc", "max")
    validation.validate_max_peak_picking_params(params)
    with pytest.raises(ValueError):
        params = lcms.get_lc_detect_peak_params("invalid_mode", "max")


def test_get_ms_cwt_params():
    params = lcms.get_ms_cwt_params("qtof")
    validation.validate_cwt_peak_picking_params(params)
    params = lcms.get_ms_cwt_params("orbitrap")
    validation.validate_cwt_peak_picking_params(params)
    with pytest.raises(ValueError):
        params = lcms.get_ms_cwt_params("invalid_mode")


def test_get_roi_params():
    func_params = [("uplc", "qtof"), ("uplc", "orbitrap"), ("hplc", "qtof"),
              ("hplc", "orbitrap")]
    n_sp = 100  # dummy value for the validator
    for separation, instrument in func_params:
        params = lcms.get_roi_params(separation, instrument)
        validation.validate_make_roi_params(n_sp, params)
    assert True


def test_get_roi_params_bad_separation():
    with pytest.raises(ValueError):
        lcms.get_roi_params("bad-value", "qtof")


def test_get_roi_params_bad_instrument():
    with pytest.raises(ValueError):
        lcms.get_roi_params("uplc", "bad-value")


# Test Chromtatogram object

@pytest.fixture
def chromatogram_data():
    rt = np.arange(200)
    spint = utils.gauss(rt, 50, 2, 100)
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
        chromatogram = lcms.Chromatogram(rt, spint, mode="invalid-mode")


def test_chromatogram_find_peaks(chromatogram_data):
    chromatogram = lcms.Chromatogram(*chromatogram_data)
    chromatogram.find_peaks()
    assert len(chromatogram.peaks) == 1


def test_chromatogram_find_peaks_custom_params(chromatogram_data):
    chromatogram = lcms.Chromatogram(*chromatogram_data)
    # a min_width greater than the width of the peak is used
    # the peak list should be empty
    peak_params = {"min_width": 120}
    chromatogram.find_peaks(params=peak_params, method="cwt")
    assert len(chromatogram.peaks) == 0


# Test MSSPectrum

@pytest.fixture
def ms_data():
    mz = np.linspace(100, 150, 10000)
    spint = utils.gauss(mz, 125, 0.005, 100)
    return mz, spint


def test_ms_spectrum_creation(ms_data):
    sp = lcms.MSSpectrum(*ms_data)
    assert  sp.mode == "qtof"


def test_ms_spectrum_creation_with_mode(ms_data):
    mode = "orbitrap"
    sp = lcms.MSSpectrum(*ms_data, instrument=mode)
    assert sp.mode == mode


def test_ms_spectrum_creation_invalid_mode(ms_data):
    with pytest.raises(ValueError):
        mode = "invalid-mode"
        sp = lcms.MSSpectrum(*ms_data, instrument=mode)


def test_find_centroids_qtof(ms_data):
    sp = lcms.MSSpectrum(*ms_data)
    # the algorithm is tested on test_peaks.py
    sp.find_centroids()
    assert True


def test_find_centroids_non_default_parameters(ms_data):
    sp = lcms.MSSpectrum(*ms_data, instrument="orbitrap")
    sp.find_centroids(min_snr=10, min_distance=0.003)
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
    lcms.Roi(spint, mz, rt, 50)
    assert True


def test_fill_nan(roi_data):
    rt, mz, spint = roi_data
    roi = lcms.Roi(spint, mz, rt, 50)
    roi.fill_nan()
    has_nan = np.any(np.isnan(roi.mz) & np.isnan(roi.spint))
    assert not has_nan
