from tidyms import fileio
from tidyms.lcms import Chromatogram
from tidyms.utils import get_cache_path
import os
import pytest
import numpy as np


mz_list = np.array([200, 250, 300, 420, 450])


def test_read_mzmine():
    dataset_name = "test-mzmine"
    cache_path = get_cache_path()
    data_path = os.path.join(cache_path, dataset_name)
    data_matrix_path = os.path.join(data_path, "data.csv")
    sample_metadata_path = os.path.join(data_path, "sample.csv")
    try:
        fileio.read_mzmine(data_matrix_path, sample_metadata_path)
    except FileNotFoundError:
        fileio._download_dataset(dataset_name)
        fileio.read_mzmine(data_matrix_path, sample_metadata_path)
    assert True


def test_read_progenesis():
    # progenesis data is contained in one file
    dataset_name = "test-progenesis"
    cache_path = get_cache_path()
    data_path = os.path.join(cache_path, dataset_name)
    data_matrix_path = os.path.join(data_path, "data.csv")
    try:
        fileio.read_progenesis(data_matrix_path)
    except FileNotFoundError:
        fileio._download_dataset(dataset_name)
        fileio.read_progenesis(data_matrix_path)
    assert True


def test_read_xcms():
    dataset_name = "test-xcms"
    cache_path = get_cache_path()
    data_path = os.path.join(cache_path, dataset_name)
    data_matrix_path = os.path.join(data_path, "data.csv")
    sample_metadata_path = os.path.join(data_path, "sample.csv")
    feature_metadata_path = os.path.join(data_path, "feature.csv")
    try:
        fileio.read_xcms(data_matrix_path, feature_metadata_path,
                         sample_metadata_path)
    except FileNotFoundError:
        fileio._download_dataset(dataset_name)
        fileio.read_xcms(data_matrix_path, feature_metadata_path,
                         sample_metadata_path)
    assert True


@pytest.fixture
def centroid_mzml():
    cache_path = get_cache_path()
    dataset_name = "test-raw-data"
    filename = "centroid-data-zlib-indexed-compressed.mzML"
    data_path = os.path.join(cache_path, dataset_name, filename)
    ms_data = fileio.MSData(data_path, ms_mode="profile")
    return ms_data


@pytest.fixture
def profile_mzml():
    cache_path = get_cache_path()
    filename = "profile-data-zlib-indexed-compressed.mzML"
    data_path = os.path.join(cache_path, "test-raw-data", filename)
    ms_data = fileio.MSData(data_path, ms_mode="profile")
    return ms_data


def test_read_compressed_indexed_mzml(centroid_mzml):
    n_spectra = centroid_mzml.get_n_spectra()
    n_chromatogram = centroid_mzml.get_n_chromatograms()

    # test spectra
    for k in range(n_spectra):
        centroid_mzml.get_spectrum(k)

    # test chromatogram
    for k in range(n_chromatogram):
        centroid_mzml.get_chromatogram(k)

    assert True


def test_read_uncompressed_indexed_mzml():
    cache_path = get_cache_path()
    filename = "centroid-data-indexed-uncompressed.mzML"
    data_path = os.path.join(cache_path, "test-raw-data", filename)
    ms_data = fileio.MSData(data_path)
    n_spectra = ms_data.get_n_spectra()
    n_chromatogram = ms_data.get_n_chromatograms()

    # test spectra
    for k in range(n_spectra):
        ms_data.get_spectrum(k)

    # test chromatogram
    for k in range(n_chromatogram):
        ms_data.get_n_chromatograms()

    assert True


def test_read_compressed_no_index_mzml():
    cache_path = get_cache_path()
    filename = "centroid-data-zlib-no-index-compressed.mzML"
    data_path = os.path.join(cache_path, "test-raw-data", filename)
    ms_data = fileio.MSData(data_path)
    n_spectra = ms_data.get_n_spectra()
    n_chromatogram = ms_data.get_n_chromatograms()

    # test spectra
    for k in range(n_spectra):
        ms_data.get_spectrum(k)

    # test chromatogram
    for k in range(n_chromatogram):
        ms_data.get_n_chromatograms()

    assert True


def test_get_spectra_iterator_start(centroid_mzml):
    start = 9
    sp_iterator = centroid_mzml.get_spectra_iterator(start=start)
    for scan, sp in sp_iterator:
        assert scan >= start


def test_get_spectra_iterator_end(centroid_mzml):
    expected_end = 20
    print(centroid_mzml.path)
    sp_iterator = centroid_mzml.get_spectra_iterator(end=expected_end)
    for scan, sp in sp_iterator:
        assert scan < expected_end


def test_get_spectra_iterator_ms_level(centroid_mzml):
    expected_ms_level = 2
    sp_iterator = centroid_mzml.get_spectra_iterator(ms_level=expected_ms_level)
    for scan, sp in sp_iterator:
        assert sp.ms_level == expected_ms_level


def test_get_spectra_iterator_start_time(centroid_mzml):
    start_time = 10
    sp_iterator = centroid_mzml.get_spectra_iterator(start_time=start_time)
    for scan, sp in sp_iterator:
        assert sp.time >= start_time


def test_get_spectra_iterator_end_time(centroid_mzml):
    end_time = 20
    sp_iterator = centroid_mzml.get_spectra_iterator(end_time=end_time)
    for scan, sp in sp_iterator:
        assert sp.time < end_time


@pytest.fixture
def sim_ms_data():
    mz = np.array(mz_list)
    rt = np.linspace(0, 100, 100)

    # simulated features params
    mz_params = np.array([mz_list, [3, 10, 5, 31, 22]])
    mz_params = mz_params.T
    rt_params = np.array(
        [[30, 40, 60, 80, 80], [1, 2, 2, 3, 3], [1, 1, 1, 1, 1]]
    )
    rt_params = rt_params.T

    noise_level = 0.1
    sim_exp = fileio.SimulatedMSData(
        mz, rt, mz_params, rt_params, noise=noise_level
    )
    return sim_exp


def test_make_roi(sim_ms_data):
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1
    )
    assert len(roi_list) == sim_ms_data.mz_params.shape[0]


def test_make_roi_targeted_mz(sim_ms_data):
    # the first three m/z values generated by simulated experiment are used
    targeted_mz = sim_ms_data.mz_params[:, 0][:3]
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        min_intensity=0,
        targeted_mz=targeted_mz)
    assert len(roi_list) == targeted_mz.size


def test_make_roi_min_intensity(sim_ms_data):
    min_intensity = 15
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        min_intensity=min_intensity,
    )
    # only two roi should have intensities greater than 15
    assert len(roi_list) == 2


def test_make_roi_start(sim_ms_data):
    start = 10
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        start=start
    )
    n_sp = sim_ms_data.get_n_spectra()
    for r in roi_list:
        assert r.mz.size == (n_sp - start)


def test_make_roi_end(sim_ms_data):
    end = 10
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        end=end)
    for r in roi_list:
        assert r.mz.size == end


def test_make_roi_multiple_match_closest(sim_ms_data):
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        multiple_match="closest")
    assert len(roi_list) == sim_ms_data.mz_params.shape[0]


def test_make_roi_multiple_match_reduce_merge(sim_ms_data):
    # set a tolerance such that two mz values are merged
    # test is done in targeted mode to force a multiple match by removing
    # one of the mz values
    targeted_mz = sim_ms_data.mz_params[:, 0]
    targeted_mz = np.delete(targeted_mz, 3)
    tolerance = 31
    roi_list = sim_ms_data.make_roi(
        tolerance=tolerance,
        max_missing=0,
        min_length=1,
        targeted_mz=targeted_mz)
    assert len(roi_list) == (sim_ms_data.mz_params.shape[0] - 1)


def test_make_roi_multiple_match_reduce_custom_mz_reduce(sim_ms_data):
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        mz_reduce=np.median
    )
    assert len(roi_list) == sim_ms_data.mz_params.shape[0]


def test_make_roi_multiple_match_reduce_custom_sp_reduce(sim_ms_data):
    roi_list = sim_ms_data.make_roi(
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        sp_reduce=lambda x: 1
    )
    assert len(roi_list) == sim_ms_data.mz_params.shape[0]


def test_make_roi_invalid_multiple_match(sim_ms_data):
    with pytest.raises(ValueError):
        sim_ms_data.make_roi(
            tolerance=0.005,
            max_missing=0,
            min_length=0,
            multiple_match="invalid-value"
        )


# # test accumulate spectra

def test_accumulate_spectra_centroid(sim_ms_data):
    n_sp = sim_ms_data.get_n_spectra()
    sp = sim_ms_data.accumulate_spectra(0, n_sp - 1)
    assert sp.mz.size == sim_ms_data.mz_params.shape[0]


def test_accumulate_spectra_centroid_subtract_left(sim_ms_data):
    sp = sim_ms_data.accumulate_spectra(70, 90, subtract_left=20)
    # only two peaks at rt 80 should be present
    assert sp.mz.size == 2


# test make_chromatogram

def test_make_chromatograms(sim_ms_data):
    # test that the chromatograms generated are valid

    # create chromatograms
    n_sp = sim_ms_data.get_n_spectra()
    n_mz = sim_ms_data.mz_params.shape[0]
    rt = np.zeros(n_sp)
    chromatogram = np.zeros((n_mz, n_sp))
    for scan, sp in sim_ms_data.get_spectra_iterator():
        sp = sim_ms_data.get_spectrum(scan)
        rt[scan] = sp.time
        chromatogram[:, scan] = sp.spint

    expected_chromatograms = [Chromatogram(rt, x) for x in chromatogram]
    test_chromatograms = sim_ms_data.make_chromatograms(mz_list)
    assert len(test_chromatograms) == len(expected_chromatograms)
    for ec, tc in zip(expected_chromatograms, test_chromatograms):
        assert np.array_equal(ec.rt, tc.rt)
        assert np.array_equal(ec.spint, tc.spint)


def test_make_chromatograms_accumulator_mean(sim_ms_data):
    sim_ms_data.make_chromatograms(mz_list, accumulator="mean")
    assert True


def test_make_tic(sim_ms_data):
    sim_ms_data.make_tic(kind="tic")
    assert True


def test_make_tic_bpi(sim_ms_data):
    sim_ms_data.make_tic(kind="bpi")
    assert True


def test_accumulate_spectra_profile(profile_mzml):
    sp = profile_mzml.accumulate_spectra(start=5, end=10,)
    assert sp.mz.size == sp.spint.size


def test_load_dataset():
    for d in fileio.list_available_datasets():
        fileio.load_dataset(d)


def test_load_dataset_invalid_dataset():
    with pytest.raises(ValueError):
        fileio.load_dataset("invalid-dataset")