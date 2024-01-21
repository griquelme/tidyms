import math
import pathlib

import pydantic
import pytest

from tidyms.base import assay
from tidyms.base import Annotation, SampleData
from tidyms.base.exceptions import IncompatibleSampleDataStatus


import dummy


@pytest.fixture
def roi_extractor():
    return dummy.DummyRoiExtractor()


@pytest.fixture
def feature_extractor():
    return dummy.DummyFeatureExtractor()


@pytest.fixture
def pipeline(
    roi_extractor: dummy.DummyRoiExtractor,
    feature_extractor: dummy.DummyFeatureExtractor,
):
    steps = [("roi extractor", roi_extractor), ("feature extractor", feature_extractor)]
    return assay.ProcessingPipeline(steps)


@pytest.fixture
def sample_data(tmp_path: pathlib.Path):
    sample = dummy.create_dummy_sample(tmp_path, 1)
    return SampleData(sample)


@pytest.fixture
def sample_data_with_roi(
    sample_data: SampleData, roi_extractor: dummy.DummyRoiExtractor
):
    roi_extractor.process(sample_data)
    return sample_data


def test_BaseRoiExtractor_get_parameters(roi_extractor: dummy.DummyRoiExtractor):
    expected = {"param1": 10, "param2": "default"}
    actual = roi_extractor.get_parameters()
    assert actual == expected


def test_BaseRoiExtractor_set_parameters(roi_extractor: dummy.DummyRoiExtractor):
    expected = {"param1": 20, "param2": "new_value"}
    roi_extractor.set_parameters(**expected)
    actual = roi_extractor.get_parameters()
    assert actual == expected


def test_BaseRoiExtractor_set_invalid_parameter_raise_ValidationError(
    roi_extractor: dummy.DummyRoiExtractor,
):
    with pytest.raises(pydantic.ValidationError):
        roi_extractor.param2 = 10  # type: ignore


def test_BaseRoiExtractor_process(
    roi_extractor: dummy.DummyRoiExtractor, sample_data: SampleData
):
    assert not sample_data.status.roi
    assert not sample_data.roi
    roi_extractor.process(sample_data)
    assert sample_data.status.roi
    assert sample_data.roi


def test_BaseFeatureExtractor_get_parameters(
    feature_extractor: dummy.DummyRoiExtractor,
):
    expected = {"param1": 2, "param2": "default", "filters": dict()}
    actual = feature_extractor.get_parameters()
    assert actual == expected


def test_BaseFeatureExtractor_set_parameters(
    feature_extractor: dummy.DummyFeatureExtractor,
):
    expected = {
        "param1": 10,
        "param2": "new_value",
        "filters": {"height": (10.0, 20.0)},
    }
    feature_extractor.set_parameters(**expected)
    actual = feature_extractor.get_parameters()
    assert actual == expected


def test_BaseFeatureExtractor_set_invalid_parameter_raise_ValidationError(
    feature_extractor: dummy.DummyFeatureExtractor,
):
    with pytest.raises(pydantic.ValidationError):
        feature_extractor.filters = 10  # type: ignore


def test_BaseFeatureExtractor_set_invalid_filter_range_None_normalizes_to_inf(
    feature_extractor: dummy.DummyFeatureExtractor,
):
    filters = {"height": (None, 10.0)}
    feature_extractor.filters = filters


def test_BaseFeatureExtractor_process(
    sample_data_with_roi: SampleData,
    feature_extractor: dummy.DummyFeatureExtractor,
):
    sample_data = sample_data_with_roi
    assert not sample_data.status.feature
    assert not sample_data.get_feature_list()
    feature_extractor.process(sample_data)
    assert sample_data.status.feature
    assert sample_data.get_feature_list()


def test_BaseFeatureExtractor_process_invalid_sample_data_status_raises_error(
    sample_data: SampleData,
    feature_extractor: dummy.DummyFeatureExtractor,
):
    with pytest.raises(IncompatibleSampleDataStatus):
        feature_extractor.process(sample_data)


def test_is_feature_descriptor_in_valid_range_valid_range():
    filters = {"height": (50.0, 150.0)}
    roi = dummy.create_dummy_roi()
    ft = dummy.create_dummy_feature(roi, Annotation())
    assert assay._is_feature_descriptor_in_valid_range(ft, filters)


def test_is_feature_descriptor_in_valid_range_invalid_range():
    filters = {"height": (50.0, 75.0)}
    roi = dummy.create_dummy_roi()
    ft = dummy.create_dummy_feature(roi, Annotation())
    assert not assay._is_feature_descriptor_in_valid_range(ft, filters)


def test__fill_filter_boundaries():
    filters = {"param1": (0, 10.0), "param2": (None, 5.0), "param3": (10.0, None)}
    filled = assay._fill_filter_boundaries(filters)
    expected = {
        "param1": (0, 10.0),
        "param2": (-math.inf, 5.0),
        "param3": (10.0, math.inf),
    }
    assert filled == expected


def test_ProcessingPipeline_processors_with_repeated_names_raises_error(
    roi_extractor: dummy.DummyRoiExtractor,
    feature_extractor: dummy.DummyFeatureExtractor,
):
    processing_steps = [
        ("ROI extractor", roi_extractor),
        ("ROI extractor", feature_extractor),
    ]
    with pytest.raises(ValueError):
        assay.ProcessingPipeline(processing_steps)


def test_ProcessingPipeline_get_parameters(pipeline: assay.ProcessingPipeline):
    for name, parameters in pipeline.get_parameters():
        processor = pipeline.get_processor(name)
        assert parameters == processor.get_parameters()


def test_ProcessingPipeline_set_parameters(pipeline: assay.ProcessingPipeline):
    new_parameters = {
        "roi extractor": {"param1": 25, "param2": "new-value"},
        "feature extractor": {
            "param1": 15,
            "param2": "new-value",
            "filters": {"height": (10.0, None)},
        },
    }
    pipeline.set_parameters(new_parameters)

    for name, processor in pipeline.processors:
        assert new_parameters[name] == processor.get_parameters()


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
