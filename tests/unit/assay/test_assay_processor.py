from tidyms.assay.assay_data import AssayData
from tidyms.assay.assay_processor import AssayProcessor
import pytest


class DummyAssayProcessor(AssayProcessor):
    """
    Dummy class for testing purposes

    """

    _validation_schema = {"param1": {"type": "integer"}, "param2": {"type": "integer"}}

    def __init__(self, param1: int = 1, param2: int = 0):
        self.param1 = param1
        self.param2 = param2

    def process(self, assay_data: AssayData):
        pass

    def _func(self):
        pass

    def set_default_parameters(self, instrument: str, separation: str):
        pass


def test_AssayProcessor_var_args_invalid():
    class InvalidProcessor(DummyAssayProcessor):
        def __init__(self, *args, param1: int = 1, param2: int = 2):
            super().__init__(param1=param1, param2=param2)

    processor = InvalidProcessor()
    with pytest.raises(RuntimeError):
        processor.get_parameters()


def test_AssayProcessor_get_parameters():
    expected_parameters = {"param1": 5, "param2": 10}
    processor = DummyAssayProcessor(**expected_parameters)
    test_parameters = processor.get_parameters()
    assert test_parameters == expected_parameters


def test_AssayProcessor_set_parameters():
    processor = DummyAssayProcessor()
    expected_parameters = {"param1": 5}
    processor.set_parameters(expected_parameters)
    expected_parameters["param2"] = processor.param2
    test_parameters = processor.get_parameters()
    assert test_parameters == expected_parameters


def test_AssayProcessor_set_parameters_invalid_parameter():
    processor = DummyAssayProcessor()
    with pytest.raises(ValueError):
        bad_parameters = {"invalid_parameter": 10}
        processor.set_parameters(bad_parameters)
