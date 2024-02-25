import math
import pathlib

import pydantic
import pytest
from tidyms.core import exceptions, processors
from tidyms.core.models import Annotation, Sample

from . import utils


@pytest.fixture
def proc_params():
    return {"order": None, "pipeline": None}


@pytest.fixture
def roi_extractor():
    return utils.DummyRoiExtractor(id="roi extractor")


@pytest.fixture
def feature_extractor():
    return utils.DummyFeatureExtractor(id="feature extractor")


@pytest.fixture
def roi_transformer():
    return utils.DummyRoiTransformer(id="roi transformer")


@pytest.fixture
def feature_transformer():
    return utils.DummyFeatureTransformer(id="feature transformer")


@pytest.fixture
def pipeline(
    roi_extractor: utils.DummyRoiExtractor,
    roi_transformer: utils.DummyRoiTransformer,
    feature_extractor: utils.DummyFeatureExtractor,
    feature_transformer: utils.DummyFeatureTransformer,
):
    id_ = "my-pipeline"
    steps = [roi_extractor, roi_transformer, feature_extractor, feature_transformer]
    return processors.ProcessingPipeline(id=id_, processors=steps)


@pytest.fixture
def sample_data(tmp_path: pathlib.Path):
    sample = utils.create_dummy_sample(tmp_path, 1)
    return processors.SampleData(sample=sample)


@pytest.fixture
def sample_data_with_roi(sample_data: processors.SampleData, roi_extractor: utils.DummyRoiExtractor):
    roi_extractor.process(sample_data)
    return sample_data


class TestSampleData:
    @pytest.fixture
    def sample(self, tmp_path):
        id_ = "my-sample"
        return Sample(id=id_, path=tmp_path / id_)

    def test_get_features_no_roi(self, sample):
        sample_data = processors.SampleData(sample=sample)
        expected = list()
        actual = sample_data.get_features()
        assert actual == expected

    def test_get_features_with_roi_no_features(self, sample):
        rois = [utils.ConcreteRoi() for _ in range(5)]
        sample_data = processors.SampleData(sample=sample, roi=rois)
        expected = list()
        actual = sample_data.get_features()
        assert actual == expected

    def test_get_features_with_roi_and_features(self, sample):
        rois = [utils.ConcreteRoi() for _ in range(5)]
        n_ft = 4
        roi = rois[2]
        features = [utils.ConcreteFeature(roi=roi, data=1) for _ in range(n_ft)]
        for ft in features:
            roi.add_feature(ft)
        sample_data = processors.SampleData(sample=sample, roi=rois)
        actual = sample_data.get_features()
        assert len(actual) == n_ft


class TestRoiExtractor:
    def test_get_parameters(self, roi_extractor):
        expected = {
            "param1": roi_extractor.param1,
            "param2": roi_extractor.param2,
            "id": roi_extractor.id,
            "type": roi_extractor.type.value,
            "accepts": roi_extractor.accepts.value,
            "order": roi_extractor.order,
            "pipeline": None,
        }
        actual = roi_extractor.get_parameters()
        assert actual == expected

    def test_set_parameters(self, roi_extractor):
        expected = roi_extractor.get_parameters()
        new_values = {"param1": 20, "param2": "new_value"}
        expected.update(new_values)

        roi_extractor.set_parameters(**new_values)
        actual = roi_extractor.get_parameters()
        assert actual == expected

    def test_set_invalid_parameter_raise_ValidationError(self, roi_extractor: utils.DummyRoiExtractor):
        with pytest.raises(pydantic.ValidationError):
            roi_extractor.param2 = 10  # type: ignore

    def test_process(self, roi_extractor, sample_data):
        assert not sample_data.roi
        roi_extractor.process(sample_data)
        assert sample_data.roi

    def test_to_config(self, roi_extractor: utils.DummyRoiExtractor):
        config = roi_extractor.to_config()
        roi_extractor_from_config = processors.ProcessorRegistry.create_from_config(config)
        assert roi_extractor == roi_extractor_from_config


class TestFeatureExtractor:
    def test_set_invalid_filter_range_None_normalizes_to_inf(self, feature_extractor: utils.DummyFeatureExtractor):
        filters = {"height": (None, 10.0)}
        feature_extractor.filters = filters

    def test_process(
        self, sample_data_with_roi: processors.SampleData, feature_extractor: utils.DummyFeatureExtractor
    ):
        sample_data = sample_data_with_roi
        assert not sample_data.get_features()
        feature_extractor.process(sample_data)
        assert sample_data.get_features()


class TestProcessingPipeline:
    def test_validate_processors_with_repeated_names_raises_error(
        self,
        roi_extractor: utils.DummyRoiExtractor,
        feature_extractor: utils.DummyFeatureExtractor,
    ):
        feature_extractor.id = roi_extractor.id
        id_ = "my-pipeline"
        processing_steps = (roi_extractor, feature_extractor)
        with pytest.raises(pydantic.ValidationError):
            processors.ProcessingPipeline(id=id_, processors=processing_steps)

    def test_validate_without_roi_extractor_in_first_step_raises_error(
        self,
        roi_transformer: utils.DummyRoiTransformer,
        feature_extractor: utils.DummyFeatureExtractor,
        feature_transformer: utils.DummyFeatureTransformer,
    ):
        id_ = "my-pipeline"
        steps = (roi_transformer, feature_extractor, feature_transformer)
        with pytest.raises(pydantic.ValidationError):
            processors.ProcessingPipeline(id=id_, processors=steps)

    def test_validate_feature_transformer_without_previous_feature_extractor_raises_error(
        self,
        roi_extractor: utils.DummyRoiExtractor,
        roi_transformer: utils.DummyRoiTransformer,
        feature_transformer: utils.DummyFeatureTransformer,
    ):
        id_ = "my-pipeline"
        steps = (roi_extractor, roi_transformer, feature_transformer)

        with pytest.raises(pydantic.ValidationError):
            processors.ProcessingPipeline(id=id_, processors=steps)

    def test_validate_feature_transformer_with_previous_feature_extractor_ok(
        self,
        roi_extractor: utils.DummyRoiExtractor,
        roi_transformer: utils.DummyRoiTransformer,
        feature_extractor: utils.DummyFeatureExtractor,
        feature_transformer: utils.DummyFeatureTransformer,
    ):
        id_ = "my-pipeline"
        steps = (roi_extractor, roi_transformer, feature_extractor, feature_transformer)
        processors.ProcessingPipeline(id=id_, processors=steps)
        assert True

    def test_get_processor(self, pipeline: processors.ProcessingPipeline):
        processor_id = "feature extractor"
        processor = pipeline.get_processor(processor_id)
        assert isinstance(processor, utils.DummyFeatureExtractor)

    def test_get_processor_invalid_id_raises_error(self, pipeline: processors.ProcessingPipeline):
        processor_id = "invalid processor id"
        with pytest.raises(exceptions.ProcessorNotFound):
            pipeline.get_processor(processor_id)

    def test_to_dict(self, pipeline: processors.ProcessingPipeline):
        d = pipeline.to_dict()
        pipeline_from_dict = processors.ProcessingPipeline.from_dict(d)
        assert pipeline_from_dict == pipeline


def test_is_feature_descriptor_in_valid_range_valid_range():
    filters = {"height": (50.0, 150.0)}
    roi = utils.create_dummy_roi()
    ft = utils.create_dummy_feature(roi, Annotation())
    assert processors._is_feature_descriptor_in_valid_range(ft, filters)


def test_is_feature_descriptor_in_valid_range_invalid_range():
    filters = {"height": (50.0, 75.0)}
    roi = utils.create_dummy_roi()
    ft = utils.create_dummy_feature(roi, Annotation())
    assert not processors._is_feature_descriptor_in_valid_range(ft, filters)


def test__fill_filter_boundaries():
    filters = {"param1": (0, 10.0), "param2": (None, 5.0), "param3": (10.0, None)}
    filled = processors._fill_filter_boundaries(filters)
    expected = {
        "param1": (0, 10.0),
        "param2": (-math.inf, 5.0),
        "param3": (10.0, math.inf),
    }
    assert filled == expected


# def test_ProcessingPipeline_get_parameters(pipeline: assay.ProcessingPipeline):
#     for name, parameters in pipeline.get_parameters():
#         processor = pipeline.get_processor(name)
#         assert parameters == processor.get_parameters()


# def test_ProcessingPipeline_set_parameters(pipeline: assay.ProcessingPipeline):
#     new_parameters = {
#         "roi extractor": {"param1": 25, "param2": "new-value"},
#         "feature extractor": {
#             "param1": 15,
#             "param2": "new-value",
#             "filters": {"height": (10.0, None)},
#         },
#     }
#     pipeline.set_parameters(new_parameters)

#     for processor in pipeline.processors:
#         assert new_parameters[processor.name] == processor.get_parameters()


# def test_ProcessingPipeline_set_default_parameters():
#     processing_steps = [
#         ("ROI extractor", DummyRoiExtractor()),
#         ("Feature extractor", DummyFeatureExtractor()),
#     ]
#     pipeline = assay.ProcessingPipeline(processing_steps)
#     instrument = "qtof"
#     separation = "uplc"
#     pipeline.set_default_parameters(instrument, separation)
#     test_defaults = pipeline.get_parameters()

#     expected_defaults = list()
#     for name, processor in pipeline.processors:
#         processor.set_default_parameters(instrument, separation)
#         params = processor.get_parameters()
#         expected_defaults.append((name, params))
#     assert expected_defaults == test_defaults


# def test_ProcessingPipeline_process():
#     processing_steps = [
#         ("ROI extractor", DummyRoiExtractor()),
#         ("Feature extractor", DummyFeatureExtractor()),
#     ]
#     pipeline = assay.ProcessingPipeline(processing_steps)
#     sample = create_dummy_sample_data()
#     pipeline.process(sample)
