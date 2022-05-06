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
    chromatograms = ms.make_chromatograms(ms_data_centroid, mz)
    rt = list()
    for _, sp in ms_data_centroid.get_spectra_iterator(ms_level=1):
        rt.append(sp.time)
    rt = np.array(rt)
    for c in chromatograms:
        assert np.array_equal(rt, c.time)
        assert c.time.size == c.spint.size


def test_ms_data_get_spectrum(ms_data_centroid):
    ms_data_centroid.get_spectrum(0)
    assert True


def test_make_tic_ms_level_1(ms_data_centroid):
    tic = ms.make_tic(ms_data_centroid, ms_level=1)
    rt = list()
    for _, sp in ms_data_centroid.get_spectra_iterator(ms_level=1):
        rt.append(sp.time)
    rt = np.array(rt)
    assert np.array_equal(rt, tic.time)
    assert tic.time.size == tic.spint.size


def test_make_chromatogram_ms_level_2(ms_data_centroid):
    mz = np.array([205.098, 524.37, 188.07])   # some m/z observed in the data
    ms_level = 2
    chromatograms = ms.make_chromatograms(
        ms_data_centroid, mz, ms_level=ms_level)
    rt = list()
    for _, sp in ms_data_centroid.get_spectra_iterator(ms_level=ms_level):
        rt.append(sp.time)
    rt = np.array(rt)
    for c in chromatograms:
        assert np.array_equal(rt, c.time)
        assert c.time.size == c.spint.size


def test_make_roi(ms_data_centroid):
    roi_list = ms.make_roi(ms_data_centroid)
    for r in roi_list:
        # The three arrays must have the same size
        assert r.time.size == r.spint.size
        assert r.time.size == r.scan.size


def test_accumulate_spectra(ms_data_centroid):
    sp = ms.accumulate_spectra(ms_data_centroid, start_time=20, end_time=30)
    assert sp.mz.size == sp.spint.size
