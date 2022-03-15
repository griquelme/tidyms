"""
Test lcms and fileio functionality with real data.

"""

import tidyms as ms
import numpy as np
import pytest
import os


@pytest.fixture
def ms_data_centroid() -> ms.MSData:
    tidyms_path = ms.fileio.get_tidyms_path()
    filename = "centroid-data-zlib-indexed-compressed.mzML"
    data_path = os.path.join(tidyms_path, "test-raw-data", filename)
    return ms.MSData(data_path)


def test_ms_data_invalid_ms_mode_setter(ms_data_centroid):
    with pytest.raises(ValueError):
        ms_data_centroid.ms_mode = "invalid-mode"


def test_ms_data_invalid_instrument_setter(ms_data_centroid):
    with pytest.raises(ValueError):
        ms_data_centroid.instrument = "invalid-instrument"


def test_ms_data_invalid_separation_setter(ms_data_centroid):
    with pytest.raises(ValueError):
        ms_data_centroid.separation = "invalid-separation"


def test_make_chromatogram_ms_level_1(ms_data_centroid):
    mz = np.array([205.098, 524.37, 188.07])   # some m/z observed in the data
    chromatograms = ms_data_centroid.make_chromatograms(mz)
    rt = ms_data_centroid.get_rt(ms_level=1)
    for c in chromatograms:
        assert np.array_equal(rt, c.rt)
        assert c.rt.size == c.spint.size


def test_ms_data_get_spectrum(ms_data_centroid):
    ms_data_centroid.get_spectrum(0)
    assert True


def test_make_tic_ms_level_1(ms_data_centroid):
    tic = ms_data_centroid.make_tic(ms_level=1)
    rt = ms_data_centroid.get_rt(ms_level=1)
    assert np.array_equal(rt, tic.rt)
    assert tic.rt.size == tic.spint.size


def test_make_chromatogram_ms_level_2(ms_data_centroid):
    mz = np.array([205.098, 524.37, 188.07])   # some m/z observed in the data
    chromatograms = ms_data_centroid.make_chromatograms(mz, ms_level=2)
    rt = ms_data_centroid.get_rt(ms_level=2)
    for c in chromatograms:
        assert np.array_equal(rt, c.rt)
        assert c.rt.size == c.spint.size


def test_make_roi(ms_data_centroid):
    roi_list = ms_data_centroid.make_roi()
    for r in roi_list:
        # The three arrays must have the same size
        assert r.rt.size == r.spint.size
        assert r.rt.size == r.scan.size


def test_accumulate_spectra(ms_data_centroid):
    sp = ms_data_centroid.accumulate_spectra(20, 30)
    assert sp.mz.size == sp.spint.size
