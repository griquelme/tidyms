import tidyms as ms
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def assay(tmpdir) -> ms.LegacyAssay:
    tidyms_path = ms.fileio.get_tidyms_path()
    data_path = Path(tidyms_path).joinpath("test-nist-raw-data")
    assay_path = Path(tmpdir).joinpath("test-assay")
    return ms.LegacyAssay(assay_path, data_path)


@pytest.fixture
def detect_features_params() -> dict:
    # a list of known m/z values to reduce computing time
    mz_list = np.array([144.081, 146.060, 195.086, 189.0734, 205.0967, 188.071])
    return {
        "tolerance": 0.015,
        "min_intensity": 5000,
        "targeted_mz": mz_list,
        "smoothing_strength": 1.0,
    }


def test_detect_features(assay, detect_features_params):
    assay.detect_features(**detect_features_params)
    assert True


def test_extract_features(assay, detect_features_params):
    assay.detect_features(**detect_features_params)
    assay.extract_features()
    assert True


def test_describe_features(assay, detect_features_params):
    assay.detect_features(**detect_features_params)
    assay.extract_features()
    assay.describe_features()
    assert True


def test_build_feature_table(assay, detect_features_params):
    assay.detect_features(**detect_features_params)
    assay.extract_features()
    assay.describe_features()
    assay.build_feature_table()
    assert True


# def test_match_features(assay, detect_features_params):
#     assay.detect_features(**detect_features_params)
#     assay.extract_features()
#     assay.describe_features()
#     assay.build_feature_table()
#     assay.match_features()
#     assert True


# def test_build_data_matrix(assay, detect_features_params):
#     assay.detect_features(**detect_features_params)
#     assay.extract_features()
#     assay.describe_features()
#     assay.build_feature_table()
#     assay.match_features()
#     assay.make_data_matrix()
#     assert True
