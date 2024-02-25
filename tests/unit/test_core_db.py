from pathlib import Path

import pytest
from tidyms.core import exceptions
from tidyms.core.db import AssayData
from tidyms.core.processors import ProcessingPipeline, SampleData

from . import utils
from .utils import ConcreteFeature, ConcreteRoi

# TODO: add multiple samples check


@pytest.fixture
def assay_data() -> AssayData[ConcreteFeature, ConcreteRoi]:
    return AssayData(None, ConcreteRoi, ConcreteFeature)


@pytest.fixture
def pipeline() -> ProcessingPipeline:
    proc_list = [
        utils.DummyRoiExtractor(id="id1"),
        utils.DummyRoiTransformer(id="id2"),
        utils.DummyFeatureExtractor(id="id3"),
        utils.DummyFeatureTransformer(id="id4"),
    ]
    return ProcessingPipeline(id="pipeline", processors=proc_list)


class TestAssayDataCreate:
    """Test assay data constructor."""

    def test_create_on_disk(self, tmp_path: Path):
        path = tmp_path / "assay_data"
        assert not path.exists()
        AssayData(path, ConcreteRoi, ConcreteFeature)
        assert path.exists()


class TestAssayDataSample:
    """test add_sample and get_sample methods."""

    def test_add_samples_empty_tuple(self, assay_data: AssayData):
        expected = list()
        assay_data._add_samples(*expected)
        actual = assay_data.get_samples()
        assert actual == expected

    def test_add_samples_single_sample(self, assay_data: AssayData, tmp_path: Path):
        expected = [utils.create_dummy_sample(tmp_path, 1)]
        assay_data._add_samples(*expected)
        actual = assay_data.get_samples()
        assert actual == expected

    def test_add_samples_multiple_samples(self, assay_data: AssayData, tmp_path: Path):
        expected = [utils.create_dummy_sample(tmp_path, x) for x in range(10, 20)]
        assay_data._add_samples(*expected)
        actual = assay_data.get_samples()
        assert actual == expected

    def test_add_samples_add_existing_sample_raises_error(self, assay_data: AssayData, tmp_path: Path):
        sample = utils.create_dummy_sample(tmp_path, 1)
        assay_data._add_samples(sample)
        with pytest.raises(exceptions.SampleAlreadyInAssay):
            assay_data._add_samples(sample)


class TestAssayDataProcessingPipeline:
    """Test add_pipeline, fetch_pipeline, and delete_pipeline methods."""

    def test_add_pipeline(self, assay_data, pipeline):
        expected = pipeline
        assay_data.add_pipeline(pipeline)
        actual = assay_data.fetch_pipeline(expected.id)
        assert actual == expected

    def test_fetch_non_existing_pipeline_raises_error(self, assay_data):
        pass

    def delete_pipeline(self, assay_data, pipeline):
        pass


class TestAssayDataSampleData:
    """Test add_sample_data, fetch_sample_data and delete_sample_data methods."""

    @pytest.fixture(scope="class")
    def sample_data(self, tmp_path_factory) -> SampleData:
        tmp_path = tmp_path_factory.mktemp("data")
        sample = utils.create_dummy_sample(tmp_path, 1)
        return SampleData(sample=sample)

    @pytest.fixture(scope="class")
    def sample_data_roi(self, sample_data) -> SampleData:
        sample_data = sample_data.model_copy(deep=True)
        sample_data.roi = [utils.create_dummy_roi() for _ in range(5)]
        return sample_data

    @pytest.fixture(scope="class")
    def sample_data_features(self, sample_data_roi) -> SampleData:
        sample_data = sample_data_roi.model_copy(deep=True)
        utils.add_dummy_features(sample_data.roi, 3)
        return sample_data

    def test_add_sample_data_no_roi(self, sample_data, assay_data):
        expected = sample_data
        assay_data.add_sample_data(expected)
        actual = assay_data.fetch_sample_data(expected.sample)
        assert actual == expected

    def test_add_sample_data_with_roi(self, sample_data_roi, assay_data):
        expected = sample_data_roi
        assay_data.add_sample_data(expected)
        actual = assay_data.fetch_sample_data(expected.sample)
        assert actual == expected

    def test_add_sample_data_with_features(self, sample_data_features, assay_data):
        expected = sample_data_features
        assay_data.add_sample_data(expected)
        actual = assay_data.fetch_sample_data(expected.sample)
        assert actual == expected

    def test_delete_sample_data(self, sample_data, assay_data):
        expected = sample_data
        assay_data.add_sample_data(expected)

        all_samples_before_delete = assay_data.get_samples()
        assert expected.sample in all_samples_before_delete

        assay_data.delete_sample_data(sample_data.sample)
        all_samples_after_delete = assay_data.get_samples()
        assert expected.sample not in all_samples_after_delete

        with pytest.raises(exceptions.SampleDataNotFound):
            assay_data.fetch_sample_data(sample_data.sample)

    def test_delete_data_roi(self, sample_data_roi, assay_data):
        expected = sample_data_roi
        assay_data.add_sample_data(expected)

        actual_roi_before_delete = assay_data.get_roi_list(sample_data_roi.sample)
        assert expected.roi == actual_roi_before_delete

        assay_data.delete_sample_data(sample_data_roi.sample)

        actual_roi_after_delete = assay_data.get_roi_list(sample_data_roi.sample)
        assert not actual_roi_after_delete

        with pytest.raises(exceptions.SampleDataNotFound):
            assay_data.fetch_sample_data(sample_data_roi.sample)

    def test_delete_data_feature(self, sample_data_roi, assay_data):
        expected = sample_data_roi
        assay_data.add_sample_data(expected)

        actual_features_before_delete = assay_data.get_features_by_sample(sample_data_roi.sample)
        assert expected.get_features() == actual_features_before_delete

        assay_data.delete_sample_data(sample_data_roi.sample)

        actual_features_after_delete = assay_data.get_features_by_sample(sample_data_roi.sample)
        assert not actual_features_after_delete

        with pytest.raises(exceptions.SampleDataNotFound):
            assay_data.fetch_sample_data(sample_data_roi.sample)

    def test_add_sample_data_with_same_id_raises_error(self, sample_data, assay_data):
        assay_data.add_sample_data(sample_data)
        with pytest.raises(exceptions.SampleAlreadyInAssay):
            assay_data.add_sample_data(sample_data)


# def test_AssayData_delete_samples(assay_data: AssayData, tmp_path: Path):
#     sample_list = [create_dummy_sample(tmp_path, x) for x in range(10, 20)]
#     assay_data.add_samples(sample_list)

#     # check samples before delete
#     expected_before_delete = assay_data.get_samples()
#     assert sample_list == expected_before_delete

#     # delete last sample and check again
#     rm_samples = [str("sample-19")]
#     assay_data.delete_samples(rm_samples)
#     sample_list.pop()
#     test_sample_list = assay_data.get_samples()
#     assert test_sample_list == sample_list


# def test_AssayData_flag_processed(assay_data: AssayData, tmp_path: Path):
#     samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
#     assay_data.add_samples(samples)
#     expected = samples[:3]
#     pipeline = "detect_features"
#     assay_data.flag_processed(expected, pipeline)
#     actual = assay_data.get_processed_samples(pipeline)
#     assert actual == expected


# def test_AssayData_flag_unprocessed(assay_data: AssayData, tmp_path: Path):
#     samples = [create_dummy_sample(tmp_path, x) for x in range(1, 11)]
#     assay_data.add_samples(samples)
#     expected = samples[:3]
#     pipeline = "detect_features"

#     # flag samples and check
#     assay_data.flag_processed(expected, pipeline)
#     actual = assay_data.get_processed_samples(pipeline)
#     assert actual == expected

#     # remove flag
#     assay_data.flag_unprocessed(pipeline)
#     actual = assay_data.get_processed_samples(pipeline)
#     assert len(actual) == 0


# def test_AssayData_set_processing_parameters(assay_data: AssayData):
#     step = "detect_features"
#     pipeline = "pipeline"
#     parameters = {"a": 1, "b": 2.0, "c": "a"}
#     assay_data.set_processing_parameters(step, pipeline, parameters)


# def test_AssayData_get_processing_parameters(assay_data: AssayData):
#     step = "detect_features"
#     pipeline = "pipeline"
#     expected = {"a": 1, "b": 2.0, "c": "a"}
#     assay_data.set_processing_parameters(step, pipeline, expected)
#     actual = assay_data._fetch_processors(step)
#     assert expected == actual


# def test_AssayData_get_processing_parameters_invalid_step_raises_value_error(
#     assay_data: AssayData,
# ):
#     step = "invalid step"
#     with pytest.raises(ValueError):
#         assay_data._fetch_processors(step)


# def test_AssayData_get_pipeline_parameters(assay_data: AssayData):
#     expected = {
#         "step1": {"param1": 20.0, "param2": "value-1"},
#         "step2": {"param3": [1.0, 2.0], "param4": True},
#     }
#     name = "pipeline1"
#     assay_data.add_pipeline_parameters(name, expected)
#     actual = assay_data.get_pipeline_parameters(name)
#     assert actual == expected


# def test_AssayData_update_pipeline_parameters(assay_data: AssayData):
#     expected = {
#         "step1": {"param1": 20.0, "param2": "value-1"},
#         "step2": {"param3": [1.0, 2.0], "param4": True},
#     }
#     name = "pipeline1"
#     assay_data.add_pipeline_parameters(name, expected)
#     expected["step2"]["param4"] = False

#     # update only params of step 1
#     new_params = expected.copy()
#     new_params.pop("step1")
#     assay_data.update_pipeline_parameters(name, new_params)
#     actual = assay_data.get_pipeline_parameters(name)
#     assert expected == actual


# def test_AssayData_add_roi_list(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     roi_list = [create_dummy_roi() for x in range(5)]
#     assay_data.add_roi_list(sample, *roi_list)


# def test_AssayData_get_roi_list(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     expected_roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample, *expected_roi_list)
#     test_roi_list = assay_data.get_roi_list(sample)
#     assert expected_roi_list == test_roi_list


# def test_AssayData_add_roi_list_empty_list(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     roi_list = list()
#     assay_data.add_roi_list(sample, *roi_list)


# def test_AssayData_get_roi_list_empty_list(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     expected_roi_list = list()
#     assay_data.add_roi_list(sample, *expected_roi_list)
#     test_roi_list = assay_data.get_roi_list(sample)
#     assert test_roi_list == expected_roi_list


# @pytest.fixture
# def assay_data_with_roi(assay_data, tmp_path: Path) -> AssayData[ConcreteFeature, ConcreteRoi]:
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     expected_roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample, *expected_roi_list)
#     return assay_data


# def test_AssayData_delete_roi_list(assay_data_with_roi: AssayData):
#     sample = assay_data_with_roi.get_samples()[0]
#     assay_data_with_roi.delete_roi(sample)
#     test_roi_list = assay_data_with_roi.get_roi_list(sample)
#     expected_roi_list = list()
#     assert expected_roi_list == test_roi_list


# def test_AssayData_delete_roi_list_no_delete(
#     assay_data_with_roi: AssayData, tmp_path: Path
# ):
#     sample = assay_data_with_roi.get_samples()[0]
#     expected_roi_list = assay_data_with_roi.get_roi_list(sample)
#     sample2 = create_dummy_sample(tmp_path, 2)
#     assay_data_with_roi.delete_roi(sample2)
#     test_roi_list = assay_data_with_roi.get_roi_list(sample)
#     assert expected_roi_list == test_roi_list


# def test_AssayData_get_roi_by_id(assay_data_with_roi):
#     sample = assay_data_with_roi.get_samples()[0]
#     roi_list = assay_data_with_roi.get_roi_list(sample)
#     roi_id = 3
#     expected = roi_list[roi_id]
#     actual = assay_data_with_roi.get_roi_by_id(roi_id)
#     assert expected == actual


# def test_AssayData_get_roi_by_id_invalid_id(assay_data_with_roi: AssayData):
#     with pytest.raises(ValueError):
#         roi_id = 100
#         assay_data_with_roi.get_roi_by_id(roi_id)


# def test_AssayData_add_features(assay_data_with_roi):
#     # add features
#     sample = assay_data_with_roi.get_samples()[0]
#     roi_list = assay_data_with_roi.get_roi_list(sample)
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data_with_roi.add_features(sample, *feature_list)


# def test_AssayData_get_features_by_sample(assay_data_with_roi):
#     sample = assay_data_with_roi.get_samples()[0]
#     roi_list = assay_data_with_roi.get_roi_list(sample)
#     add_dummy_features(roi_list, 2)
#     expected_feature_list = get_feature_list(roi_list)

#     assay_data_with_roi.add_features(sample, *expected_feature_list)

#     actual_feature_list = assay_data_with_roi.get_features_by_sample(sample)

#     for actual_ft, expected_ft in zip(actual_feature_list, expected_feature_list):
#         assert actual_ft.equal(expected_ft)


# def test_AssayData_get_features_by_label(
#     assay_data_with_roi: AssayData, tmp_path: Path
# ):
#     # add features for sample 1
#     sample = assay_data_with_roi.get_samples()[0]
#     roi_list = assay_data_with_roi.get_roi_list(sample)
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data_with_roi.add_features(sample, *feature_list)

#     # add roi and features for sample 2
#     sample2 = create_dummy_sample(tmp_path, 2)
#     assay_data_with_roi.add_samples([sample2])

#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data_with_roi.add_roi_list(sample2, *roi_list)
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data_with_roi.add_features(sample2, *feature_list)

#     test_label = 1
#     test_feature_list = assay_data_with_roi.get_features_by_label(test_label)

#     for ft in test_feature_list:
#         assert ft.annotation.label == test_label


# def test_AssayData_get_features_by_label_specify_group(
#     assay_data: AssayData, tmp_path: Path
# ):
#     # add roi and features for sample 1
#     sample1 = create_dummy_sample(tmp_path, 1, group="group-1")
#     assay_data.add_samples([sample1])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample1, *roi_list)

#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_features(sample1, *feature_list)

#     # add roi and features for sample 2
#     sample2 = create_dummy_sample(tmp_path, 2, group="group-2")
#     assay_data.add_samples([sample2])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample2, *roi_list)
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_features(sample2, *feature_list)

#     test_label = 1
#     test_feature_list = assay_data.get_features_by_label(test_label, groups=["group-1"])

#     assert len(test_feature_list) == 1
#     for ft in test_feature_list:
#         assert ft.annotation.label == test_label


# def test_AssayData_add_features_no_features(assay_data: AssayData, tmp_path: Path):
#     # add samples
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])

#     # add roi
#     roi_list = [create_dummy_roi() for _ in range(20)]
#     assay_data.add_roi_list(sample, *roi_list)

#     # rois do not have features
#     assay_data.add_features(sample, *())


# def test_AssayData_get_features_by_sample_no_features(
#     assay_data: AssayData, tmp_path: Path
# ):
#     # add samples
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])

#     # add roi
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample, *roi_list)

#     # rois do not have features
#     assay_data.add_features(sample, *())

#     test_features = assay_data.get_features_by_sample(sample)
#     assert len(test_features) == 0


# def test_AssayData_get_feature_by_id(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_roi_list(sample, *roi_list)
#     assay_data.add_features(sample, *feature_list)
#     expected = roi_list[2].features[1]
#     feature_id = 5  # Fifth feature added to the DB. Should have id=5
#     actual = assay_data.get_features_by_id(feature_id)

#     assert actual.equal(expected)


# def test_AssayData_get_feature_by_id_invalid_id(assay_data: AssayData, tmp_path: Path):
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_roi_list(sample, *roi_list)
#     assay_data.add_features(sample, *feature_list)

#     with pytest.raises(ValueError):
#         invalid_feature_id = 1000
#         assay_data.get_features_by_id(invalid_feature_id)


# @pytest.fixture
# def assay_data_with_features(assay_data: AssayData, tmp_path: Path):
#     # add roi and features for sample 1
#     sample1 = create_dummy_sample(tmp_path, 1, group="group-1")
#     assay_data.add_samples([sample1])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample1, *roi_list)

#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_features(sample1, *feature_list)

#     # add roi and features for sample 2
#     sample2 = create_dummy_sample(tmp_path, 2, group="group-2")
#     assay_data.add_samples([sample2])
#     roi_list = [create_dummy_roi() for _ in range(5)]
#     assay_data.add_roi_list(sample2, *roi_list)
#     add_dummy_features(roi_list, 2)
#     feature_list = get_feature_list(roi_list)
#     assay_data.add_features(sample2, *feature_list)

#     return assay_data


# def test_AssayData_get_descriptors_one_sample(assay_data_with_features: AssayData):
#     sample = assay_data_with_features.get_samples()[0]
#     features = assay_data_with_features.get_features_by_sample(sample)

#     n_features = len(features)
#     descriptor_names = assay_data_with_features.feature.descriptor_names()

#     descriptors = assay_data_with_features.get_descriptors(sample)
#     base_descriptor_names = ["sample_id", "roi_id", "id", "label"]

#     for k, v in descriptors.items():
#         if k not in base_descriptor_names:
#             assert len(v) == n_features
#             assert k in descriptor_names
#         elif k == "sample_id":
#             assert all(x == sample.id for x in v)


# def test_AssayData_get_descriptors_all_samples(assay_data_with_features: AssayData):
#     samples = assay_data_with_features.get_samples()
#     n_features = 0
#     for sample in samples:
#         features = assay_data_with_features.get_features_by_sample(sample)
#         n_features += len(features)

#     descriptor_names = assay_data_with_features.feature.descriptor_names()
#     base_descriptor_names = ["sample_id", "roi_id", "id", "label"]

#     descriptors = assay_data_with_features.get_descriptors()
#     for k, v in descriptors.items():
#         if k not in base_descriptor_names:
#             assert len(v) == n_features
#             assert k in descriptor_names


# def test_AssayData_get_descriptors_invalid_descriptors(
#     assay_data_with_features: AssayData,
# ):
#     requested_descriptors = ["mz", "invalid_descriptor"]
#     with pytest.raises(ValueError):
#         assay_data_with_features.get_descriptors(descriptors=requested_descriptors)


# def test_AssayData_get_descriptors_descriptors_subset(
#     assay_data_with_features: AssayData,
# ):
#     sample = assay_data_with_features.get_samples()[0]
#     features = assay_data_with_features.get_features_by_sample(sample)
#     n_features = len(features)

#     all_descriptors = assay_data_with_features.feature.descriptor_names()
#     base_descriptor_names = ["sample_id", "roi_id", "id", "label"]
#     requested_descriptors = list(all_descriptors)[:2]
#     descriptors = assay_data_with_features.get_descriptors(
#         sample=sample, descriptors=requested_descriptors
#     )

#     for k, v in descriptors.items():
#         if k not in base_descriptor_names:
#             assert len(v) == n_features
#             assert k in requested_descriptors


# def test_AssayData_get_descriptors_no_descriptors(
#     assay_data_with_features: AssayData, tmp_path: Path
# ):
#     sample = create_dummy_sample(tmp_path, 3)
#     assay_data_with_features.add_samples([sample])

#     descriptors = assay_data_with_features.get_descriptors(sample)
#     for k, v in descriptors.items():
#         assert v == []


# def test_AssayData_get_descriptors_no_descriptors_all_samples(assay_data: AssayData):
#     descriptors = assay_data.get_descriptors()
#     all_descriptors = assay_data.feature.descriptor_names()
#     base_descriptor_names = ["sample_id", "roi_id", "id", "label"]
#     for k, v in descriptors.items():
#         assert v == []
#         if k not in base_descriptor_names:
#             assert k in all_descriptors


# def test_AssayData_get_descriptors_subset_no_descriptors(
#     assay_data_with_features: AssayData, tmp_path: Path
# ):
#     sample = create_dummy_sample(tmp_path, 3)
#     assay_data_with_features.add_samples([sample])

#     all_descriptors = assay_data_with_features.feature.descriptor_names()
#     base_descriptor_names = ["sample_id", "roi_id", "id", "label"]
#     requested_descriptors = list(all_descriptors)[:2]

#     descriptors = assay_data_with_features.get_descriptors(
#         sample, descriptors=requested_descriptors
#     )
#     for k, v in descriptors.items():
#         assert v == []
#         if k not in base_descriptor_names:
#             assert k in requested_descriptors


# def test_AssayData_update_feature_labels(assay_data_with_features: AssayData):
#     # get descriptors and annotations
#     desc_before = assay_data_with_features.get_descriptors()
#     ann_before = assay_data_with_features.get_annotations()

#     # update feature labels
#     update_labels_dict = {5: 1, 10: 1, 11: 2}
#     assay_data_with_features.update_feature_labels(update_labels_dict)

#     desc_after = assay_data_with_features.get_descriptors()
#     ann_after = assay_data_with_features.get_annotations()

#     # Check annotations before and after update
#     id_to_label_before = {k: v for k, v in zip(ann_before["id"], ann_before["label"])}
#     id_to_label_after = {k: v for k, v in zip(ann_after["id"], ann_after["label"])}

#     for id_, label_before in id_to_label_before.items():
#         label_before = update_labels_dict.get(id_, label_before)
#         label_after = id_to_label_after[id_]
#         assert label_before == label_after

#     # Check descriptors before and after update
#     id_to_label_before = {k: v for k, v in zip(desc_before["id"], desc_before["label"])}
#     id_to_label_after = {k: v for k, v in zip(desc_after["id"], desc_after["label"])}

#     for id_, label_before in id_to_label_before.items():
#         label_before = update_labels_dict.get(id_, label_before)
#         label_after = id_to_label_after[id_]
#         assert label_before == label_after


# def test_AssayData_load_existing_db(tmp_path: Path):
#     db_path = tmp_path / "my_db.db"
#     assay_data = AssayData(db_path, ConcreteRoi, ConcreteFeature)

#     # add roi and features
#     sample = create_dummy_sample(tmp_path, 1)
#     assay_data.add_samples([sample])
#     expected_roi_list = [create_dummy_roi() for _ in range(10)]
#     assay_data.add_roi_list(sample, *expected_roi_list)
#     add_dummy_features(expected_roi_list, 2)
#     expected_feature_list = get_feature_list(expected_roi_list)
#     assay_data.add_features(sample, *expected_feature_list)

#     del assay_data

#     # load assay from disk
#     assay_data = AssayData(db_path, ConcreteRoi, ConcreteFeature)

#     samples = assay_data.get_samples()
#     assert len(samples) == 1

#     actual_roi_list = assay_data.get_roi_list(sample)
#     actual_feature_list = assay_data.get_features_by_sample(sample)
#     assert actual_roi_list == expected_roi_list
#     for expected_ft, actual_ft in zip(expected_feature_list, actual_feature_list):
#         expected_ft.equal(actual_ft)


# def test_AssayData_search_sample(tmp_path: Path):
#     assay_data = AssayData(None, ConcreteRoi, ConcreteFeature)
#     sample_list = [create_dummy_sample(tmp_path, x) for x in range(10, 20)]
#     assay_data.add_samples(sample_list)

#     samples = assay_data.get_samples()
#     expected = samples[5]
#     actual = assay_data.search_sample(samples[5].id)

#     assert expected == actual


# def test_AssayData_search_sample_invalid_sample(tmp_path: Path):
#     assay_data = AssayData(None, ConcreteRoi, ConcreteFeature)
#     sample_list = [create_dummy_sample(tmp_path, x) for x in range(10, 20)]
#     assay_data.add_samples(sample_list)

#     with pytest.raises(ValueError):
#         assay_data.search_sample("invalid_sample_id")


# # TODO: fix test
# # def test_AssayData_store_sample_data(tmp_path: Path):
# #     assay_data = AssayData(None, ConcreteRoi, ConcreteFeature)

# #     # add samples
# #     sample = create_dummy_sample(tmp_path, 1)
# #     assay_data.add_samples([sample])

# #     # create roi and features
# #     roi_list = [create_dummy_roi() for _ in range(5)]
# #     add_dummy_features(roi_list, 2)

# #     expected = SampleData(sample)
# #     processor = ProcessorInformation(
# #         id="dummy-id", pipeline="dummy-pipeline", order=1, parameters=dict()
# #     )
# #     expected.set_roi_list(roi_list, processor)

# #     assay_data.store_sample_data(expected)
# #     actual = assay_data.get_sample_data(sample.id)
# #     assert expected.sample == actual.sample
# #     assert expected._roi_snapshots == actual._roi_snapshots
