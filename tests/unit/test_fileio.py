from tidyms import fileio
import os


def test_read_mzmine():
    dataset_name = "test-mzmine"
    cache_path = fileio._get_cache_path()
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
    cache_path = fileio._get_cache_path()
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
    cache_path = fileio._get_cache_path()
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
