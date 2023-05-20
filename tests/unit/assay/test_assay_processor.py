from tidyms.assay.assay_data import AssayData, SampleData, Sample
from tidyms.assay import assay_processor
from tidyms.lcms import LCTrace, Peak
import pytest
import numpy as np
from math import inf


def create_dummy_trace() -> LCTrace:
    scans = np.arange(100)
    time = scans.astype(float)
    spint = np.ones_like(time)
    mz = spint
    noise = np.zeros_like(time)
    baseline = noise
    return LCTrace(time, spint, mz, scans, noise, baseline)


def create_dummy_sample_data() -> SampleData:
    sample = Sample("sample1", "sample1")
    roi_list = [create_dummy_trace() for _ in range(10)]
    return SampleData(sample, roi_list)


class DummyProcessor(assay_processor.Processor):
    """
    Dummy class for testing purposes

    """

    def __init__(self, param1: int = 1, param2: int = 0):
        self.param1 = param1
        self.param2 = param2

    def process(self, assay_data: AssayData):
        pass

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass

    def _func(self):
        pass

    def set_default_parameters(self, instrument: str, separation: str):
        pass


class DummyFeatureExtractor(assay_processor.FeatureExtractor):
    """

    Dummy class for testing purposes

    """

    _validation_schema = {}

    def __init__(
        self,
        param1: float = 1.0,
        param2: float = 2.0,
        filters=None,
    ):
        super().__init__(filters)
        self.param1 = param1
        self.param2 = param2

    @staticmethod
    def _check_data(sample_data: SampleData):
        pass

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass

    @staticmethod
    def _extract_features_func(roi: LCTrace, **params):
        roi.features = [Peak(0, 5, 10, roi), Peak(15, 20, 25, roi)]

    def set_default_parameters(self, instrument: str, separation: str):
        params = {"filters": {"mz": (10.0, 20.0), "width": (5.0, 10.0)}}
        self.set_parameters(params)


class DummyRoiExtractor(assay_processor.SingleSampleProcessor):
    def __init__(self, param1: int = 1, param2: int = 0):
        self.param1 = param1
        self.param2 = param2

    @staticmethod
    def _func(sample_data: SampleData):
        sample_data.roi = [create_dummy_trace() for _ in range(10)]

    def set_default_parameters(self, instrument: str, separation: str):
        pass

    @staticmethod
    def _check_data(sample_data: SampleData):
        pass

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass


def test_Processor_var_args_invalid():
    class InvalidProcessor(DummyProcessor):
        def __init__(self, *args, param1: int = 1, param2: int = 2):
            super().__init__(param1=param1, param2=param2)

        @staticmethod
        def _validate_parameters(parameters: dict):
            pass

    processor = InvalidProcessor()
    with pytest.raises(RuntimeError):
        processor.get_parameters()


def test_Processor_get_parameters():
    expected_parameters = {"param1": 5, "param2": 10}
    processor = DummyProcessor(**expected_parameters)
    test_parameters = processor.get_parameters()
    assert test_parameters == expected_parameters


def test_Processor_set_parameters():
    processor = DummyProcessor()
    expected_parameters = {"param1": 5}
    processor.set_parameters(expected_parameters)
    expected_parameters["param2"] = processor.param2
    test_parameters = processor.get_parameters()
    assert test_parameters == expected_parameters


def test_Processor_set_parameters_invalid_parameter():
    processor = DummyProcessor()
    with pytest.raises(ValueError):
        bad_parameters = {"invalid_parameter": 10}
        processor.set_parameters(bad_parameters)


def test_FeatureExtractor_creation():
    DummyFeatureExtractor()
    assert True


def test_FeatureExtractor_filter_values():
    filters = {"descriptor1": (5.0, 8.0)}
    extractor = DummyFeatureExtractor(filters=filters)
    assert extractor.filters == filters


def test_FeatureExtractor_set_default_parameters():
    extractor = DummyFeatureExtractor()
    extractor.set_default_parameters("qtof", "uplc")
    parameters = extractor.get_parameters()
    assert extractor.param1 == parameters["param1"]
    assert extractor.param2 == parameters["param2"]
    assert extractor.filters == parameters["filters"]


def test_FeatureExtractor_extract_features():
    extractor = DummyFeatureExtractor()
    extractor.set_default_parameters("qtof", "uplc")
    sample_data = create_dummy_sample_data()
    extractor.process(sample_data)


def test__is_valid_feature_valid_feature():
    filters = {"mz": (0.5, 1.5), "height": (0.5, 1.5)}
    roi = create_dummy_trace()
    ft = Peak(0, 5, 10, roi)
    assert assay_processor._is_valid_feature(ft, filters)


def test__is_valid_feature_invalid_feature():
    # m/z is outside valid range
    filters = {"mz": (2.0, 3.0), "height": (0.5, 1.5)}
    roi = create_dummy_trace()
    ft = Peak(0, 5, 10, roi)
    assert not assay_processor._is_valid_feature(ft, filters)


def test__fill_filter_boundaries():
    filters = {"param1": (0, 10.0), "param2": (None, 5.0), "param3": (10.0, None)}
    filled = assay_processor._fill_filter_boundaries(filters)
    expected = {"param1": (0, 10.0), "param2": (-inf, 5.0), "param3": (10.0, inf)}
    assert filled == expected


def test_ProcessingPipeline_processors_with_repeated_names_raises_error():
    processing_steps = [
        ("ROI extractor", DummyRoiExtractor()),
        ("ROI extractor", DummyFeatureExtractor()),
    ]
    with pytest.raises(ValueError):
        assay_processor.ProcessingPipeline(processing_steps)


def test_ProcessingPipeline_get_parameters():
    processing_steps = [
        ("ROI extractor", DummyRoiExtractor()),
        ("Feature extractor", DummyFeatureExtractor()),
    ]
    pipeline = assay_processor.ProcessingPipeline(processing_steps)
    for name, parameters in pipeline.get_parameters():
        processor = pipeline.get_processor(name)
        assert parameters == processor.get_parameters()


def test_ProcessingPipeline_set_parameters():
    processing_steps = [
        ("ROI extractor", DummyRoiExtractor()),
        ("Feature extractor", DummyFeatureExtractor()),
    ]
    pipeline = assay_processor.ProcessingPipeline(processing_steps)

    new_parameters = {
        "ROI extractor": {"param1": 25.0, "param2": 23.0},
        "Feature extractor": {"param1": 15.0, "param2": 0.5, "filters": {}},
    }
    pipeline.set_parameters(new_parameters)

    for name, processor in pipeline.processors:
        assert new_parameters[name] == processor.get_parameters()


def test_ProcessingPipeline_set_default_parameters():
    processing_steps = [
        ("ROI extractor", DummyRoiExtractor()),
        ("Feature extractor", DummyFeatureExtractor()),
    ]
    pipeline = assay_processor.ProcessingPipeline(processing_steps)
    instrument = "qtof"
    separation = "uplc"
    pipeline.set_default_parameters(instrument, separation)
    test_defaults = pipeline.get_parameters()

    expected_defaults = list()
    for name, processor in pipeline.processors:
        processor.set_default_parameters(instrument, separation)
        params = processor.get_parameters()
        expected_defaults.append((name, params))
    assert expected_defaults == test_defaults


def test_ProcessingPipeline_process():
    processing_steps = [
        ("ROI extractor", DummyRoiExtractor()),
        ("Feature extractor", DummyFeatureExtractor()),
    ]
    pipeline = assay_processor.ProcessingPipeline(processing_steps)
    sample = create_dummy_sample_data()
    pipeline.process(sample)
