from tidyms import fileio
import os

data_path = os.path.join("tests", "data")


def test_read_mzmine():
    data_matrix_path = os.path.join(data_path, "mzmine-data-matrix.csv")
    sample_metadata_path = os.path.join(data_path, "mzmine-sample-metadata.csv")
    fileio.read_mzmine(data_matrix_path, sample_metadata_path)
    assert True


def test_read_progenesis():
    data_matrix_path = os.path.join(data_path, "progenesis-data-matrix.csv")
    fileio.read_progenesis(data_matrix_path)
    assert True


def test_read_xcms():
    data_matrix_path = os.path.join(data_path, "xcms-data-matrix.tsv")
    feature_metadata = os.path.join(data_path, "xcms-feature-metadata.tsv")
    sample_metadata = os.path.join(data_path, "xcms-sample-metadata.tsv")
    fileio.read_xcms(data_matrix_path, feature_metadata, sample_metadata)
    assert True
