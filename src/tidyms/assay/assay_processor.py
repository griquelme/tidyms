"""
Tools to process Assay data.

"""
import inspect
from abc import ABC, abstractmethod
from .assay_data import SampleData
from .. import validation as val


class PreprocessingOrderError(ValueError):
    """
    Exception raised when the preprocessing methods are called in the wrong order.

    """

    pass


class AssayProcessor(ABC):
    """
    The base class for Assay processors.

    Contains functionality to validate Processor parameters and to store parameters
    in the DB managed by AssayData.

    Processor Parameters MUST be defined as key-value parameters on the
    `__init__` and set as attributes. Parameters constraints MUST be specified
    in the _validation_schema class attribute.

    Attributes
    -----------
    _step : str
        (class attribute). Processing step. Must be one of the values defined
        in `_constants.PREPROCESSING_STEPS`.
    _validation_schema : dict
        (class attribute). Dictionary with a cerberus schema used to validate
        processor parameters.
    _roi : Type
        (class attribute). Roi type used (if applicable).
    _feature : Type
        (class attribute). Feature type used (if applicable).

    """

    _validation_schema: dict

    def __repr__(self) -> str:
        name = self.__class__.__name__
        arg_str = ", ".join([f"{k}={v}" for k, v in self.get_parameters().items()])
        return f"{name}({arg_str})"

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""

        init_signature = inspect.signature(cls.__init__)

        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Processors should always specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_parameters(self) -> dict:
        """
        Get the estimator parameters.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        parameters = {k: getattr(self, k) for k in self._get_param_names()}
        validator = val.ValidatorWithLowerThan(self._validation_schema)
        val.validate(parameters, validator)
        return parameters

    def set_parameters(self, parameters: dict):
        """
        Set processor parameters.

        Parameters
        ----------
        parameters : dict

        """
        valid_parameters = self._get_param_names()
        for k, v in parameters.items():
            if k not in valid_parameters:
                msg = f"{k} is not a valid parameter for {self.__class__}"
                raise ValueError(msg)
            setattr(self, k, v)

    @abstractmethod
    def set_default_parameters(self, instrument: str, separation: str):
        ...


class SingleSampleProcessor(AssayProcessor):
    """
    Base class to process samples in an assay independently.

    MUST implement _check_data to check compatibility of data with the processor.
    MUST implement _func to process the sample data. Must modify sample data.

    Processor Parameters MUST be defined as key-value parameters in the
    `__init__` method and set as attributes.

    Parameters constraints MUST be specified in the _validation_schema class attribute.

    """

    def process(self, sample_data: SampleData):
        self._check_data(sample_data)
        self._func(sample_data)

    @staticmethod
    @abstractmethod
    def _check_data(sample_data: SampleData):
        ...

    @abstractmethod
    def _func(self, sample_data: SampleData):
        ...


class MultipleSampleProcessor(AssayProcessor):
    pass
