import numpy as np
from tidyms.assay.assay_data import AssayData, Sample
from tidyms.lcms import LCTrace, Peak, Annotation
from pathlib import Path
import pytest
from typing import cast


def create_dummy_sample(path: Path, suffix: int) -> Sample:
    file = path / f"sample-{suffix}.mzML"
    sample = Sample(path=str(file), id=file.stem)
    return sample


def create_dummy_lc_trace():
    time = np.linspace(0, 20, 50)
    spint = np.abs(np.random.normal(size=time.size))
    mz = spint
    scans = np.arange(time.size)
    noise = np.zeros_like(mz)
    baseline = np.zeros_like(mz)
    return LCTrace(time, spint, mz, scans, noise=noise, baseline=baseline)


def add_dummy_peaks(lc_trace: LCTrace, n: int):
    features = list()
    for k in range(n):
        annotation = Annotation(label=k)
        ft = Peak(0, 1, 2, lc_trace, annotation)
        features.append(ft)
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


def test_AssayData_flag_samples(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    step = "detect_features"
    assay_data.flag_processed_samples(samples, step)


def test_AssayData_flag_samples_invalid_step(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    step = "invalid_preprocessing_step"
    with pytest.raises(ValueError):
        assay_data.flag_processed_samples(samples, step)


def test_AssayData_get_samples_processing_stage(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    step = "detect_features"
    assay_data.flag_processed_samples(samples, step)

    new_samples = [create_dummy_sample(tmp_path, x) for x in range(20, 30)]
    assay_data.add_samples(new_samples)

    test_samples = assay_data.get_samples(step=step)
    all_samples = assay_data.get_samples()
    assert samples == test_samples
    assert all_samples != test_samples


def test_AssayData_get_samples_invalid_step(tmp_path: Path):
    samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
    assay_data = AssayData("", LCTrace, Peak)
    assay_data.add_samples(samples)
    step = "invalid_preprocessing_step"
    with pytest.raises(ValueError):
        assay_data.get_samples(step)


def test_AssayData_set_processing_parameters():
    assay_data = AssayData("", LCTrace, Peak)
    step = "detect_features"
    parameters = {"a": 1, "b": 2.0, "c": "a"}
    assay_data.set_processing_parameters(step, parameters)


def test_AssayData_set_processing_parameters_invalid_step():
    assay_data = AssayData("", LCTrace, Peak)
    step = "invalid_preprocessing_step"
    parameters = {"a": 1, "b": 2.0, "c": "a"}
    with pytest.raises(ValueError):
        assay_data.set_processing_parameters(step, parameters)


def test_AssayData_get_processing_parameters_invalid_step():
    assay_data = AssayData("", LCTrace, Peak)
    step = "detect_features"
    expected_parameters = {"a": 1, "b": 2.0, "c": "a"}
    assay_data.set_processing_parameters(step, expected_parameters)
    tests_parameters = assay_data.get_processing_parameters(step)
    assert expected_parameters == tests_parameters


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
    assay_data.delete_roi_list(sample)
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
    assay_data.delete_roi_list(sample2)
    test_roi_list = assay_data.get_roi_list(sample)
    assert expected_roi_list == test_roi_list


def test_AssayData_add_features(tmp_path: Path):
    assay_data = AssayData("", LCTrace, Peak)

    # add samples
    sample = create_dummy_sample(tmp_path, 1)
    assay_data.add_samples([sample])

    # add roi
    roi_list = [create_dummy_lc_trace() for _ in range(20)]
    assay_data.add_roi_list(roi_list, sample)

    # add features
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
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
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
    assay_data.add_features(roi_list, sample)

    expected_feature_list: list[Peak] = list()
    for roi in roi_list:
        expected_feature_list.extend(roi.features)

    test_feature_list = cast(list[Peak], assay_data.get_features(sample))

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
    roi_list = [create_dummy_lc_trace() for _ in range(0)]
    assay_data.add_roi_list(roi_list, sample1)
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
    assay_data.add_features(roi_list, sample1)

    # add roi and features for sample 1
    sample2 = create_dummy_sample(tmp_path, 2)
    assay_data.add_samples([sample2])
    roi_list = [create_dummy_lc_trace() for _ in range(0)]
    assay_data.add_roi_list(roi_list, sample2)
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
    assay_data.add_features(roi_list, sample2)

    test_label = 1
    test_feature_list = cast(list[Peak], assay_data.get_features(label=test_label))

    assert len(test_feature_list) == 2
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

    test_features = assay_data.get_features(sample)
    assert len(test_features) == 0


def test_AssayData_get_descriptors_one_samples(tmp_path: Path):
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
    for roi in roi_list:
        add_dummy_peaks(roi, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
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
    for roi in roi_list:
        add_dummy_peaks(roi, 2)

    assay_data.add_features(roi_list, sample1)

    # roi_list with roi.id values
    roi_list = cast(list[LCTrace], assay_data.get_roi_list(sample2))

    # add features
    for roi in roi_list:
        add_dummy_peaks(roi, 2)
    assay_data.add_features(roi_list, sample2)

    descriptors = assay_data.get_descriptors()
    for v in descriptors.values():
        assert len(v) == 40
