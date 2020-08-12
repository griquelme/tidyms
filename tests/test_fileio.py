from tidyms import fileio


def test_read_mzmine():
    data_matrix, _, sample_metadata = \
        fileio._load_csv_files("mzmine", test_files=True)
    fileio.read_mzmine(data_matrix, sample_metadata)
    assert True


def test_read_progenesis():
    # progenesis data is contained in one file
    data_matrix, _, _ = fileio._load_csv_files("progenesis", test_files=True)
    fileio.read_progenesis(data_matrix)
    assert True


def test_read_xcms():
    data_matrix, feature_metadata, sample_metadata = \
        fileio._load_csv_files("xcms", test_files=True)
    fileio.read_xcms(data_matrix, feature_metadata, sample_metadata)
    assert True
