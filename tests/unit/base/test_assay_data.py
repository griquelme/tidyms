import numpy as np
from tidyms.base.assay_data import AssayData, Sample, SampleData
from tidyms.lcms import LCTrace, Peak, Annotation
from pathlib import Path
import pytest
from typing import cast


def create_dummy_sample(path: Path, suffix: int, group: str = "") -> Sample:
    file = path / f"sample-{suffix}.mzML"
    sample = Sample(path=str(file), id=file.stem, group=group)
    return sample


def create_dummy_lc_trace():
    time = np.linspace(0, 20, 50)
    spint = np.abs(np.random.normal(size=time.size))
    mz = spint
    scans = np.arange(time.size)
    noise = np.zeros_like(mz)
    baseline = np.zeros_like(mz)
    return LCTrace(time, spint, mz, scans, noise=noise, baseline=baseline)


def add_dummy_peaks(lc_trace_list: list[LCTrace], n: int):
    label_counter = 0
    for lc_trace in lc_trace_list:
        features = list()
        for _ in range(n):
            annotation = Annotation(label=label_counter)
            ft = Peak(0, 1, 2, lc_trace, annotation)
            features.append(ft)
            label_counter += 1
        lc_trace.features = features


def test_create_AssayData():
    AssayData("", LCTrace, Peak)


def test_AssayData_add_samples_add_empty_list():
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples([])


def test_AssayData_get_samples_from_empty_db():
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples([])
    sample_list = assay_data.get_samples()
    assert len(sample_list) == 0


def test_AssayData_add_samples_add_single_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(sample)


def test_AssayData_get_samples_single_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    expected_sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(expected_sample)
    test_sample = assay_data.get_samples()
    assert expected_sample == test_sample


def test_AssayData_add_samples_multiple_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(sample_list)


def test_AssayData_get_samples_multiple_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    expected_sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(expected_sample_list)
    test_sample_list = assay_data.get_samples()
    assert test_sample_list == expected_sample_list


def test_AssayData_add_samples_add_existing_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    expected_sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(expected_sample)
    with pytest.raises(ValueError):
        assay_data.add_samples(expected_sample)


def test_AssayData_delete_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(sample_list)

    # check samples before delete
    expected_before_delete = assay_data.get_samples()
    assert sample_list == expected_before_delete

    # delete last sample and check again
    rm_samples = [str("sample-19")]
    assay_data.delete_samples(rm_samples)
    sample_list.pop()
    test_sample_list = assay_data.get_samples()
    assert test_sample_list == sample_list


def test_AssayData_flag_processed(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    expected = samples[:3]
    pipeline = "detect_features"
    assay_data.flag_processed(expected, pipeline)
    actual = assay_data.get_processed_samples(pipeline)
    assert actual == expected


def test_AssayData_flag_unprocessed(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    expected = samples[:3]
    pipeline = "detect_features"

    # flag samples and check
    assay_data.flag_processed(expected, pipeline)
    actual = assay_data.get_processed_samples(pipeline)
    assert actual == expected

    # remove flag
    assay_data.flag_unprocessed(pipeline)
    actual = assay_data.get_processed_samples(pipeline)
    assert len(actual) == 0


def test_AssayData_set_processing_parameters():
    assay_data = AssayData("", LCTrace, Peak)
    step = "detect_features"
    pipeline = "pipeline"
    parameters = {"a": 1, "b": 2.0, "c": "a"}
    assay_data.set_processing_parameters(step, pipeline, parameters)


def test_AssayData_get_processing_parameters():
    assay_data = AssayData("", LCTrace, Peak)
    step = "detect_features"
    pipeline = "pipeline"
    expected = {"a": 1, "b": 2.0, "c": "a"}
    assay_data.set_processing_parameters(step, pipeline, expected)
    actual = assay_data.get_processing_parameters(step)
    assert expected == actual


def test_AssayData_get_processing_parameters_invalid_step():
    assay_data = AssayData("", LCTrace, Peak)
    step = "invalid step"
    with pytest.raises(ValueError):
        assay_data.get_processing_parameters(step)


def test_AssayData_get_pipeline_parameters():
    expected = {
        "step1": {"param1": 20.0, "param2": "value-1"},
        "step2": {"param3": [1.0, 2.0], "param4": True},
    }
    name = "pipeline1"
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_pipeline_parameters(name, expected)
    actual = assay_data.get_pipeline_parameters(name)
    assert actual == expected


def test_AssayData_update_pipeline_parameters():
    expected = {
        "step1": {"param1": 20.0, "param2": "value-1"},
        "step2": {"param3": [1.0, 2.0], "param4": True},
    }
    name = "pipeline1"
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_pipeline_parameters(name, expected)
    expected["step2"]["param4"] = False

    # update only params of step 1
    new_params = expected.copy()
    new_params.pop("step1")
    assay_data.update_pipeline_parameters(name, new_params)
    actual = assay_data.get_pipeline_parameters(name)
    assert expected == actual


def test_AssayData_add_roi_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for x in range(20)]
    assay_data.add_roi_list(roi_list, sample)


def test_AssayData_get_roi_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    expected_roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(expected_roi_list, sample)
    test_roi_list = assay_data.get_roi_list(sample)
    assert expected_roi_list == test_roi_list


def test_AssayData_add_roi_list_empty_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = list()
    assay_data.add_roi_list(roi_list, sample)


def test_AssayData_get_roi_list_empty_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    expected_roi_list = list()
    assay_data.add_roi_list(expected_roi_list, sample)
    test_roi_list = assay_data.get_roi_list(sample)
    assert test_roi_list == expected_roi_list


def test_AssayData_delete_roi_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)
    assay_data.delete_roi(sample)
    test_roi_list = assay_data.get_roi_list(sample)
    expected_roi_list = list()
    assert expected_roi_list == test_roi_list


def test_AssayData_delete_roi_list_no_delete(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    sample2 = create_dummy_sample(path, 2)
    assay_data.add_samples([sample])
    expected_roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(expected_roi_list, sample)
    assay_data.delete_roi(sample2)
    test_roi_list = assay_data.get_roi_list(sample)
    assert expected_roi_list == test_roi_list


def test_AssayData_get_roi_by_id(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    add_dummy_peaks(roi_list, 2)
    assay_data.add_roi_list(roi_list, sample)
    assay_data.add_features(roi_list, sample)

    roi_id = 5
    expected = roi_list[roi_id]
    actual = assay_data.get_roi_by_id(roi_id)
    assert expected == actual


def test_AssayData_get_roi_by_id_invalid_id(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    with pytest.raises(ValueError):
        roi_id = 100
        assay_data.get_roi_by_id(roi_id)


def test_AssayData_add_features(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample)


def test_AssayData_get_features_by_sample(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample))

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample)

    expected_feature_list: list[Peak] = list()
    for roi in roi_list:
        expected_feature_list.extend(roi.features)

    test_feature_list = cast(list[Peak], assay_data.get_features_by_sample(sample))

    assert len(test_feature_list) == len(expected_feature_list)
    for eft, tft in zip(expected_feature_list, test_feature_list):
        assert eft.start == tft.start
        assert eft.apex == tft.apex
        assert eft.end == tft.end
        assert eft.roi == tft.roi


def test_AssayData_get_features_by_label(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add roi and features for sample 1
    sample1 = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample1])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)

    # use roi list with indices
    roi_list = assay_data.get_roi_list(sample1)

    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample1)

    # add roi and features for sample 1
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample2])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample2)
    roi_list = assay_data.get_roi_list(sample2)
    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample2)

    test_label = 1
    test_feature_list = cast(list[Peak], assay_data.get_features_by_label(test_label))

    assert len(test_feature_list) == 2
    for ft in test_feature_list:
        assert ft.annotation.label == test_label


def test_AssayData_get_features_by_label_specify_group(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add roi and features for sample 1
    sample1 = create_dummy_sample(tmp_path, 1, group="group-1")
    assay_data.add_samples([sample1])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)

    # use roi list with indices
    roi_list = assay_data.get_roi_list(sample1)

    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample1)

    # add roi and features for sample 1
    sample2 = create_dummy_sample(tmp_path, 2, group="group-2")
    assay_data.add_samples([sample2])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample2)
    roi_list = assay_data.get_roi_list(sample2)
    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample2)

    test_label = 1
    test_feature_list = cast(
        list[Peak], assay_data.get_features_by_label(test_label, groups=["group-1"])
    )

    assert len(test_feature_list) == 1
    for ft in test_feature_list:
        assert ft.annotation.label == test_label


def test_AssayData_add_features_no_features(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    # rois do not have features
    assay_data.add_features(roi_list, sample)


def test_AssayData_get_features_by_sample_no_features(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    # rois do not have features
    assay_data.add_features(roi_list, sample)

    test_features = assay_data.get_features_by_sample(sample)
    assert len(test_features) == 0


def test_AssayData_get_feature_by_id(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    add_dummy_peaks(roi_list, 2)
    assay_data.add_roi_list(roi_list, sample)
    assay_data.add_features(roi_list, sample)

    expected = roi_list[2].features[1]
    feature_id = 5  # Fifth feature added to the DB. Should have id=5
    actual = cast(Peak, assay_data.get_features_by_id(feature_id))
    assert expected.start == actual.start
    assert expected.apex == actual.apex
    assert expected.end == actual.end
    assert expected.roi == actual.roi
    assert expected.annotation == actual.annotation


def test_AssayData_get_feature_by_id_invalid_id(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    add_dummy_peaks(roi_list, 2)
    assay_data.add_roi_list(roi_list, sample)
    assay_data.add_features(roi_list, sample)

    with pytest.raises(ValueError):
        invalid_feature_id = 1000
        assay_data.get_features_by_id(invalid_feature_id)


def test_AssayData_get_descriptors_one_sample(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample1 = create_dummy_sample(tmp_path, 1)
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample1, sample2])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)
    assay_data.add_roi_list(roi_list, sample2)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample1))

    # add features
    add_dummy_peaks(roi_list, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample2)

    descriptors = assay_data.get_descriptors(sample1)
    for v in descriptors.values():
        assert len(v) == 20


def test_AssayData_get_descriptors_all_samples(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample1 = create_dummy_sample(tmp_path, 1)
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample1, sample2])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)
    assay_data.add_roi_list(roi_list, sample2)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample1))

    # add features
    add_dummy_peaks(roi_list, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample2)

    descriptors = assay_data.get_descriptors()
    for v in descriptors.values():
        assert len(v) == 40


def test_AssayData_get_descriptors_invalid_descriptors(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample1 = create_dummy_sample(tmp_path, 1)
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample1, sample2])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)
    assay_data.add_roi_list(roi_list, sample2)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample1))

    # add features
    add_dummy_peaks(roi_list, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample2)

    requested_descriptors = ["mz", "invalid_descriptor"]
    with pytest.raises(ValueError):
        assay_data.get_descriptors(sample1, descriptors=requested_descriptors)


def test_AssayData_get_descriptors_descriptors_subset(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample1 = create_dummy_sample(tmp_path, 1)
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample1, sample2])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)
    assay_data.add_roi_list(roi_list, sample2)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample1))

    # add features
    add_dummy_peaks(roi_list, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    add_dummy_peaks(roi_list, 2)
    assay_data.add_features(roi_list, sample2)

    requested_descriptors = ["mz", "rt"]
    descriptors = assay_data.get_descriptors(sample1, descriptors=requested_descriptors)
    assert len(descriptors) == len(requested_descriptors)
    for v in descriptors.values():
        assert len(v) == 20


def test_AssayData_load_existing_db(tmp_path: Path):
    db_path = tmp_path / "my_db.db"
    assay_data = AssayData(db_path, LCTrace, Peak)

    # add roi and features for sample 1
    sample1 = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample1])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample1)

    # use roi list with indices
    roi_list = assay_data.get_roi_list(sample1)

    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample1)

    # add roi and features for sample 1
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample2])
    roi_list = [create_dummy_lc_trace() for _ in range(10)]
    assay_data.add_roi_list(roi_list, sample2)
    roi_list = assay_data.get_roi_list(sample2)
    add_dummy_peaks(cast(list[LCTrace], roi_list), 2)
    assay_data.add_features(roi_list, sample2)

    del assay_data

    assay_data = AssayData(db_path, LCTrace, Peak)
    test_label = 1
    test_feature_list = cast(list[Peak], assay_data.get_features_by_label(test_label))

    assert len(test_feature_list) == 2
    for ft in test_feature_list:
        assert ft.annotation.label == test_label


def test_AssayData_search_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(sample_list)

    samples = assay_data.get_samples()
    expected = samples[5]
    actual = assay_data.search_sample(samples[5].id)

    assert expected == actual


def test_AssayData_search_sample_invalid_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData("", LCTrace, Peak)
    sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(sample_list)

    with pytest.raises(ValueError):
        assay_data.search_sample("invalid_sample_id")


def test_AssayData_store_sample_data(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # create roi and features
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    add_dummy_peaks(roi_list, 2)

    expected = SampleData(sample, roi_list)

    assay_data.store_sample_data(expected)
    actual = assay_data.get_sample_data(sample.id)
    assert expected.sample == actual.sample
    assert expected.roi == actual.roi
