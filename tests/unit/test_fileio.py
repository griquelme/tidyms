from tidyms import fileio
from tidyms.utils import get_tidyms_path
import os
import pytest


def test_read_mzmine():
    dataset_name = "test-mzmine"
    cache_path = get_tidyms_path()
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
    cache_path = get_tidyms_path()
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
    cache_path = get_tidyms_path()
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
    cache_path = get_tidyms_path()
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
    cache_path = get_tidyms_path()
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


def test_centroids(profile_mzml):
    sp = profile_mzml.get_spectrum(0)
    centroids = sp.find_centroids()
    assert True


def test_load_dataset():
    for d in fileio.list_available_datasets():
        fileio.load_dataset(d)


def test_load_dataset_invalid_dataset():
    with pytest.raises(ValueError):
        fileio.load_dataset("invalid-dataset")
