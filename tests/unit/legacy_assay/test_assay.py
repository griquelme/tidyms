from tidyms.base import legacy_assay
from tidyms.lcms import LCTrace, Peak
from tidyms import _constants as c
import pytest
from pathlib import Path
import os
import numpy as np
import pandas as pd


def fill_dummy_data_dir(path, sample_names):
    """
    Creates n files called Sample1.mzML to SampleN.mzML inside the path.

    Parameters
    ----------
    path : Path
    sample_names: list

    """
    for sample in sample_names:
        path.joinpath(sample).touch()


def create_sample_names(n: int, start: int = 1):
    template = "Sample{}.mzML"
    return [template.format(k) for k in range(start, n + start)]


def create_assay_dir(
    path, n, start=1, data_dir_name: str = "data-dir", assay_dir_name: str = "assay-dir"
):
    tmpdir = Path(str(path))
    assay_path = tmpdir.joinpath(assay_dir_name)
    data_path = tmpdir.joinpath(data_dir_name)
    # assay_path.mkdir()
    data_path.mkdir(exist_ok=True)
    sample_names = create_sample_names(n, start=start)
    fill_dummy_data_dir(data_path, sample_names)
    return assay_path, data_path


def create_dummy_assay_manager(assay_path, data_path, metadata) -> legacy_assay._AssayManager:
    return legacy_assay._AssayManager(
        assay_path=assay_path,
        data_path=data_path,
        sample_metadata=metadata,
        ms_mode=c.CENTROID,
        instrument=c.QTOF,
        separation=c.UPLC,
        data_import_mode=c.SIMULATED,
    )


def create_dummy_data_frame(n: int, classes: list, start: int = 1):
    sample_names = create_sample_names(n, start=start)
    sample_names = [x.split(".")[0] for x in sample_names]
    df = pd.DataFrame(data=sample_names, columns=[c.SAMPLE])
    df[c.CLASS] = np.random.choice(classes, n)
    return df


def test_assay_metadata_creation(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    assert True


# get_path_list


def test_get_path_list_single_file_str(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.mzML"
    path = Path(file_path)
    path.touch()
    path_list = legacy_assay._get_path_list(file_path)
    assert len(path_list) == 1


def test_get_path_list_single_file_invalid_extension(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.bad-ext"
    path = Path(file_path)
    path.touch()
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(file_path)


def test_get_path_list_single_file_file_not_exists(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.mzML"
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(file_path)


def test_get_path_list_list_str(tmpdir):
    file_name_template = str(tmpdir) + os.path.sep + "Sample{}.mzML"
    file_list = [file_name_template.format(x) for x in range(1, 10)]
    for file in file_list:
        path = Path(file)
        path.touch()
    path_list = legacy_assay._get_path_list(file_list)
    assert len(path_list) == len(file_list)


def test_get_path_list_list_str_bad_extension(tmpdir):
    file_name_template = str(tmpdir) + os.path.sep + "Sample{}.bad-ext"
    file_list = [file_name_template.format(x) for x in range(1, 10)]
    for file in file_list:
        path = Path(file)
        path.touch()
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(file_list)


def test_get_path_list_list_str_not_exists(tmpdir):
    file_name_template = str(tmpdir) + os.path.sep + "Sample{}.bad-ext"
    file_list = [file_name_template.format(x) for x in range(1, 10)]
    for k, file in enumerate(file_list):
        # skip a file
        if k != 3:
            path = Path(file)
            path.touch()
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(file_list)


def test_get_path_path(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.mzML"
    path = Path(file_path)
    path.touch()
    path_list = legacy_assay._get_path_list(path)
    assert len(path_list) == 1


def test_get_path_path_invalid_extension(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.bad-ext"
    path = Path(file_path)
    path.touch()
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(path)


def test_get_path_path_not_exist(tmpdir):
    file_path = str(tmpdir) + os.path.sep + "sample.mzML"
    path = Path(file_path)
    with pytest.raises(ValueError):
        legacy_assay._get_path_list(path)


def test_assay_manager_invalid_sample_metadata_missing_samples(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata = sample_metadata.drop(index=[0, 1, 2])
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_invalid_sample_metadata_extra_samples(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata2 = sample_metadata.copy()
    sample_metadata2[c.SAMPLE] = sample_metadata2[c.SAMPLE] + "asdf"
    sample_metadata = pd.concat([sample_metadata, sample_metadata2])
    sample_metadata = sample_metadata.reset_index()
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_without_sample_column(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata = sample_metadata.drop(columns=[c.SAMPLE])
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_without_class_column(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata = sample_metadata.drop(columns=[c.CLASS])
    metadata = create_dummy_assay_manager(assay_path, data_path, sample_metadata)
    assert (metadata.sample_metadata[c.CLASS] == 0).all()


def test_assay_manager_sample_metadata_without_order_column(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata = sample_metadata.drop(columns=[c.CLASS])
    metadata = create_dummy_assay_manager(assay_path, data_path, sample_metadata)
    assert (metadata.sample_metadata[c.ORDER] == np.arange(n) + 1).all()


def test_assay_manager_sample_metadata_with_order_column(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata[c.ORDER] = np.arange(n) + 1
    create_dummy_assay_manager(assay_path, data_path, sample_metadata)
    assert True


def test_assay_manager_sample_metadata_invalid_order_type(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata[c.ORDER] = np.arange(n) + 1
    sample_metadata[c.ORDER] = sample_metadata[c.ORDER].astype(str)
    with pytest.raises(TypeError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_invalid_order_non_pos_values(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata[c.ORDER] = np.arange(n) + 1
    sample_metadata.at[2, c.ORDER] = -1
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_invalid_order_repeated_values(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    sample_metadata[c.ORDER] = np.arange(n) + 1
    sample_metadata.at[2, c.ORDER] = 1
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_with_batch_column(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    n1 = n // 2
    n2 = n - n1
    sample_metadata[c.BATCH] = [1] * n1 + [2] * n2
    create_dummy_assay_manager(assay_path, data_path, sample_metadata)
    assert True


def test_assay_manager_sample_metadata_invalid_batch_type(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    n1 = n // 2
    n2 = n - n1
    sample_metadata[c.BATCH] = [0] * n1 + [2] * n2
    sample_metadata[c.BATCH] = sample_metadata[c.BATCH].astype(str)
    with pytest.raises(TypeError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_sample_metadata_invalid_batch_values(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    n1 = n // 2
    n2 = n - n1
    sample_metadata[c.BATCH] = [0] * n1 + [2] * n2
    with pytest.raises(ValueError):
        create_dummy_assay_manager(assay_path, data_path, sample_metadata)


def test_assay_manager_create_assay_dir(tmpdir):
    n = 20
    classes = ["a", "b", "c"]
    assay_path, data_path = create_assay_dir(tmpdir, n)
    sample_metadata = create_dummy_data_frame(n, classes)
    metadata = create_dummy_assay_manager(assay_path, data_path, sample_metadata)
    metadata.create_assay_dir()

    # check dir
    roi_path = metadata.assay_path.joinpath("roi")
    assert roi_path.is_dir()
    ft_path = metadata.assay_path.joinpath("feature")
    assert ft_path.is_dir()


def test_assay_manager_check_previous_step_first_step(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    step = c.DETECT_FEATURES
    metadata.check_step(step)
    assert True


def test_assay_manager_check_previous_middle_step(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    previous = c.DETECT_FEATURES
    metadata.flag_completed(previous)
    step = c.EXTRACT_FEATURES
    metadata.check_step(step)
    assert True


def test_assay_manager_check_invalid_order(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    step = c.EXTRACT_FEATURES
    with pytest.raises(legacy_assay.PreprocessingOrderError):
        metadata.check_step(step)


def test_assay_manager_get_sample_path_using_sample_name(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    name = metadata.get_sample_names()[0]
    metadata.get_sample_path(name)


def test_assay_manager_get_sample_path_invalid_sample_name(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    with pytest.raises(ValueError):
        metadata.get_sample_path("invalid-sample-name")


def test_assay_manager_get_roi_dir_path_sample_name(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    name = metadata.get_sample_names()[0]
    metadata.get_roi_path(name)


def test_assay_manager_get_roi_dir_path_invalid_sample_name(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    with pytest.raises(ValueError):
        metadata.get_roi_path("invalid-sample-name")


def test_assay_manager_get_feature_path_sample_name(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    metadata.create_assay_dir()
    sample_name = metadata.get_sample_names()[0]
    ft_filename = "{}.pickle".format(sample_name)
    expected_path = metadata.assay_path.joinpath("feature", ft_filename)
    ft_path = metadata.get_feature_path(sample_name)
    assert ft_path == expected_path


def test_assay_manager_add_samples(tmpdir):
    n1 = 20
    n2 = 20
    assay_path, data_path = create_assay_dir(tmpdir, n1)
    metadata = create_dummy_assay_manager(assay_path, data_path, None)
    metadata.create_assay_dir()
    _, new_data_path = create_assay_dir(
        tmpdir, n2, start=n1 + 1, data_dir_name="new-data-path"
    )
    metadata.add_samples(new_data_path, None)

    assert metadata.sample_metadata.shape[0] == n1 + n2

    # check new sample names
    for sample_path in new_data_path.glob("*"):
        filename = sample_path.stem
        assert filename in metadata.sample_metadata.index


# Test AssayTemplate

# create a subclass with dummy methods


class DummyAssay(legacy_assay.LegacyAssay):
    n_roi = 5
    roi_length = 20
    n_ft = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, data_import_mode=c.SIMULATED)

    def get_ms_data(self, sample: str):
        return sample


def detect_features_dummy(ms_data: str, **kwargs):
    results = list()
    for k in range(DummyAssay.n_roi):
        scan = np.arange(DummyAssay.roi_length)
        time = scan.astype(float)
        roi = LCTrace(time, time, time, scan)
        results.append(roi)
    return results


def extract_features_dummy(roi, **kwargs):
    roi.noise = np.zeros_like(roi.spint) + 1e-8
    roi.baseline = np.zeros_like(roi.spint) + 1e-8
    roi.features = [Peak(0, 5, 10, roi) for _ in range(DummyAssay.n_ft)]


def test_assay_creation(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    DummyAssay(assay_path, data_path)
    assert True


def test_assay_creation_invalid_separation(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    with pytest.raises(ValueError):
        DummyAssay(assay_path, data_path, separation="invalid-value")


def test_assay_creation_invalid_instrument(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    with pytest.raises(ValueError):
        DummyAssay(assay_path, data_path, instrument="invalid-instrument")


def test_assay_from_existing_assay(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay_path = test_assay.manager.assay_path
    DummyAssay(test_assay_path, data_path)
    assert True


def test_assay_creation_invalid_dir(tmpdir):
    path = Path(str(tmpdir))
    path = path.joinpath("bad-dir.tidyms-assay")
    path.mkdir()
    with pytest.raises(ValueError):
        DummyAssay(path, path)


def test_assay_detect_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy, verbose=False)
    assert True


def test_assay_detect_features_verbose(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy, verbose=True)
    assert True


def test_assay_detect_features_multiple_times(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.detect_features(strategy=detect_features_dummy)
    assert True


def test_assay_detect_features_multiple_times_with_different_params(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.detect_features(strategy=detect_features_dummy, dummy_param=2)
    assert True


def test_assay_load_roi(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    sample_name = test_assay.manager.get_sample_names()[0]
    r = test_assay.load_roi(sample_name, 0)
    assert r.spint.size == DummyAssay.roi_length


def test_assay_load_roi_invalid_roi_index(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    create_dummy_assay_manager(assay_path, data_path, None)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    with pytest.raises(IndexError):
        sample_name = test_assay.manager.get_sample_names()[0]
        test_assay.load_roi(sample_name, 1000)


def test_assay_load_roi_list(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    sample_list = test_assay.manager.get_sample_names()
    for sample in sample_list:
        roi_list = test_assay.load_roi_list(sample)
        assert len(roi_list) == DummyAssay.n_roi
        for r in roi_list:
            assert r.time.size == DummyAssay.roi_length


def test_assay_extract_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features(verbose=False)
    assert True


def test_assay_extract_features_verbose(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features(verbose=True)
    assert True


def test_assay_extract_features_saved_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features(strategy=extract_features_dummy)
    sample_list = test_assay.manager.get_sample_names()
    for sample in sample_list:
        roi_list = test_assay.load_roi_list(sample)
        for r in roi_list:
            assert len(r.features) == DummyAssay.n_ft


def test_assay_extract_features_before_detect_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    with pytest.raises(legacy_assay.PreprocessingOrderError):
        test_assay.extract_features()


def test_assay_describe_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features()
    test_assay.describe_features(verbose=False)
    assert True


def test_assay_describe_features_verbose(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features()
    test_assay.describe_features(verbose=True)
    assert True


def test_assay_load_features(tmpdir):
    assay_path, data_path = create_assay_dir(tmpdir, 20)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features(strategy=extract_features_dummy)
    filters = {c.WIDTH: (0, None), c.SNR: (0, None)}
    test_assay.describe_features(filters=filters)
    sample_list = test_assay.manager.get_sample_names()
    for sample in sample_list:
        features = test_assay.load_features(sample)
        n_features = DummyAssay.n_ft * DummyAssay.n_roi
        # 12 descriptors = area, width, height, snr, mz, mz_std, rt, rt start,
        # rt end, roi index, ft index, extension
        n_descriptors = 12
        expected_shape = (n_features, n_descriptors)
        assert features.shape == expected_shape
    assert True


def test_assay_build_feature_table(tmpdir):
    n_samples = 20
    assay_path, data_path = create_assay_dir(tmpdir, n_samples)
    test_assay = DummyAssay(assay_path, data_path)
    test_assay.detect_features(strategy=detect_features_dummy)
    test_assay.extract_features(strategy=extract_features_dummy)
    filters = {c.WIDTH: (0, None), c.SNR: (0, None)}
    test_assay.describe_features(filters=filters)
    test_assay.build_feature_table()
    n_features = DummyAssay.n_ft * DummyAssay.n_roi * n_samples
    n_descriptors = 14
    expected_shape = (n_features, n_descriptors)
    assert test_assay.feature_table.shape == expected_shape


# test peak descriptors


def test_fill_filter_boundaries_fill_upper_bound():
    filters = {"loc": (50, None), "snr": (5, 10)}
    legacy_assay._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][1], np.inf)


def test_fill_filter_boundaries_fill_lower_bound():
    filters = {"loc": (None, 50), "snr": (5, 10)}
    legacy_assay._fill_filter_boundaries(filters)
    assert np.isclose(filters["loc"][0], -np.inf)


def test_has_all_valid_descriptors():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (3, 10)}
    assert legacy_assay._all_valid_descriptors(descriptors, filters)


def test_has_all_valid_descriptors_descriptors_outside_valid_ranges():
    descriptors = {"loc": 50, "height": 10, "snr": 5}
    filters = {"snr": (10, 20)}
    assert not legacy_assay._all_valid_descriptors(descriptors, filters)
