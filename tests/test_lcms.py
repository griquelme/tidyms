from tidyms import lcms
from tidyms import validation
from tidyms.utils import SimulatedExperiment
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
    simexp = SimulatedExperiment(mz, rt, mz_params, rt_params,
                                 noise=noise_level)
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
    lcms.accumulate_spectra(simulated_experiment, start=10, end=20)
    assert True


def test_accumulate_spectra_subtract(simulated_experiment):
    lcms.accumulate_spectra(simulated_experiment, start=10, end=20,
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
    params = lcms.get_lc_cwt_params("uplc")
    validation.validate_cwt_peak_picking_params(params)
    params = lcms.get_lc_cwt_params("hplc")
    validation.validate_cwt_peak_picking_params(params)
    with pytest.raises(ValueError):
        params = lcms.get_lc_cwt_params("invalid_mode")


def test_get_ms_cwt_params():
    params = lcms.get_ms_cwt_params("qtof")
    validation.validate_cwt_peak_picking_params(params)
    params = lcms.get_ms_cwt_params("orbitrap")
    validation.validate_cwt_peak_picking_params(params)
    with pytest.raises(ValueError):
        params = lcms.get_ms_cwt_params("invalid_mode")
