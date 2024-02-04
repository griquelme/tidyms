import math
import pathlib

import pydantic
import pytest

from tidyms.core import processors
from tidyms.core.models import Annotation, SampleData
from tidyms.core import exceptions


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
    steps = (roi_extractor, roi_transformer, feature_extractor, feature_transformer)
    return processors.ProcessingPipeline(id_, *steps)


@pytest.fixture
def sample_data(tmp_path: pathlib.Path):
    sample = utils.create_dummy_sample(tmp_path, 1)
    return SampleData(sample)


@pytest.fixture
def sample_data_with_roi(
    sample_data: SampleData, roi_extractor: utils.DummyRoiExtractor
):
    roi_extractor.process(sample_data)
    return sample_data


def test_BaseRoiExtractor_get_parameters(roi_extractor, proc_params):

    expected = {"param1": 10, "param2": "default"}
    expected.update(proc_params)
    actual = roi_extractor.get_parameters()
    assert actual == expected


def test_BaseRoiExtractor_set_parameters(roi_extractor, proc_params):
    expected = {"param1": 20, "param2": "new_value"}
    expected.update(proc_params)
    roi_extractor.set_parameters(**expected)
    actual = roi_extractor.get_parameters()
    assert actual == expected


def test_BaseRoiExtractor_set_invalid_parameter_raise_ValidationError(
    roi_extractor: utils.DummyRoiExtractor,
):
    with pytest.raises(pydantic.ValidationError):
        roi_extractor.param2 = 10  # type: ignore


def test_BaseRoiExtractor_process(roi_extractor, sample_data):
    assert not sample_data.get_roi_list()
    roi_extractor.process(sample_data)
    assert sample_data.get_roi_list()


def test_BaseFeatureExtractor_get_parameters(feature_extractor, proc_params):
    expected = {"param1": 2, "param2": "default", "filters": dict()}
    expected.update(proc_params)
    actual = feature_extractor.get_parameters()
    assert actual == expected


def test_BaseFeatureExtractor_set_parameters(feature_extractor, proc_params):
    expected = {
        "param1": 10,
        "param2": "new_value",
        "filters": {"height": (10.0, 20.0)},
    }
    expected.update(proc_params)
    feature_extractor.set_parameters(**expected)
    actual = feature_extractor.get_parameters()
    assert actual == expected


def test_BaseFeatureExtractor_set_invalid_parameter_raise_ValidationError(
    feature_extractor: utils.DummyFeatureExtractor,
):
    with pytest.raises(pydantic.ValidationError):
        feature_extractor.filters = 10  # type: ignore


def test_BaseFeatureExtractor_set_invalid_filter_range_None_normalizes_to_inf(
    feature_extractor: utils.DummyFeatureExtractor,
):
    filters = {"height": (None, 10.0)}
    feature_extractor.filters = filters


def test_BaseFeatureExtractor_process(
    sample_data_with_roi: SampleData,
    feature_extractor: utils.DummyFeatureExtractor,
):
    sample_data = sample_data_with_roi
    assert not sample_data.get_feature_list()
    feature_extractor.process(sample_data)
    assert sample_data.get_feature_list()


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


def test_ProcessingPipeline_processors_with_repeated_names_raises_error(
    roi_extractor: utils.DummyRoiExtractor,
    feature_extractor: utils.DummyFeatureExtractor,
):
    feature_extractor.id = roi_extractor.id
    id_ = "my-pipeline"
    processing_steps = (roi_extractor, feature_extractor)
    with pytest.raises(ValueError):
        processors.ProcessingPipeline(id_, *processing_steps)


def test_sample_pipeline_creation(
    roi_extractor: utils.DummyRoiExtractor,
    roi_transformer: utils.DummyRoiTransformer,
    feature_extractor: utils.DummyFeatureExtractor,
    feature_transformer: utils.DummyFeatureTransformer,
):
    id_ = "my-pipeline"
    steps = (roi_extractor, roi_transformer, feature_extractor, feature_transformer)
    processors.ProcessingPipeline(id_, *steps)
    assert True


def test_sample_pipeline_creation_without_roi_extractor_in_first_step_raises_error(
    roi_transformer: utils.DummyRoiTransformer,
    feature_extractor: utils.DummyFeatureExtractor,
    feature_transformer: utils.DummyFeatureTransformer,
):
    id_ = "my-pipeline"
    steps = (roi_transformer, feature_extractor, feature_transformer)
    with pytest.raises(exceptions.IncompatibleProcessorStatus):
        processors.ProcessingPipeline(id_, *steps)


def test_sample_pipeline_creation_without_feature_extractor_raises_error(
    roi_extractor: utils.DummyRoiExtractor,
    roi_transformer: utils.DummyRoiTransformer,
    feature_transformer: utils.DummyFeatureTransformer,
):
    id_ = "my-pipeline"
    steps = (roi_extractor, roi_transformer, feature_transformer)
    with pytest.raises(exceptions.IncompatibleProcessorStatus):
        processors.ProcessingPipeline(id_, *steps)


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
