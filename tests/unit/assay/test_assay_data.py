import numpy as np
from tidyms.assay.assay_data import AssayData, Sample
from tidyms.lcms import LCTrace, Peak
from pathlib import Path
import pytest


def create_dummy_sample(path: Path, suffix: int) -> Sample:
    file = path / f"sample-{suffix}.mzML"
    sample = Sample(path=str(file), id=file.stem)
    return sample


def create_dummy_roi(index: int):
    time = np.linspace(0, 20, 50)
    spint = np.abs(np.random.normal(size=time.size))
    mz = spint
    scans = np.arange(time.size)
    return LCTrace(time, spint, mz, scans, index=index)


def test_create_AssayData(tmp_path: Path):
    path = tmp_path / "test-assay"
    AssayData(path, LCTrace, Peak)


def test_AssayData_add_samples_add_empty_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    assay_data.add_samples([])


def test_AssayData_get_samples_from_empty_db(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    assay_data.add_samples([])
    sample_list = assay_data.get_samples()
    assert len(sample_list) == 0


def test_AssayData_add_samples_add_single_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(sample)


def test_AssayData_get_samples_single_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    expected_sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(expected_sample)
    test_sample = assay_data.get_samples()
    assert expected_sample == test_sample


def test_AssayData_add_samples_multiple_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(sample_list)


def test_AssayData_get_samples_multiple_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    expected_sample_list = [create_dummy_sample(path, x) for x in range(10, 20)]
    assay_data.add_samples(expected_sample_list)
    test_sample_list = assay_data.get_samples()
    assert test_sample_list == expected_sample_list


def test_AssayData_add_samples_add_existing_sample(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    expected_sample = [create_dummy_sample(path, 1)]
    assay_data.add_samples(expected_sample)
    with pytest.raises(ValueError):
        assay_data.add_samples(expected_sample)


def test_AssayData_delete_samples(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
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


def test_AssayData_add_roi_list(tmp_path: Path):
    path = tmp_path / "test-assay"
    assay_data = AssayData(path, LCTrace, Peak)
    sample = create_dummy_sample(path, 1)
    assay_data.add_samples([sample])
    roi_list = [create_dummy_roi(x) for x in range(20)]
    assay_data.add_roi_list(roi_list, sample)
