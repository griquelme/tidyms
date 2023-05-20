"""
Base classes to create processors for assay data.

- SingleSampleProcessor : Base class to process data from a single sample.
- FeatureExtractor : Base class to extract feature from ROI.
- MultipleSampleProcessor : Base class to process multiple samples.

"""
import inspect
from abc import ABC, abstractmethod
from .assay_data import SampleData, AssayData
from ..lcms import Feature
from typing import Optional, Sequence
from math import inf


class Processor(ABC):
    """
    The base class for data processor.

    MUST implement the processor method.

    Processor Parameters MUST be defined as key-value parameters on the
    `__init__` and set as attributes. Parameters constraints MUST be specified
    in the _validation_schema class attribute.

    MUST implement _validate_parameters staticmethod.
    MUST implement set_default_parameters

    """

    def __repr__(self) -> str:
        name = self.__class__.__name__
        arg_str = ", ".join([f"{k}={v}" for k, v in self.get_parameters().items()])
        return f"{name}({arg_str})"

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""

        init_signature = inspect.signature(cls.__init__)

        parameters = list()
        for p in init_signature.parameters.values():
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Processors should always specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                )
            if (
                (not p.name.startswith("_"))
                and (p.kind != p.VAR_KEYWORD)
                and (p.name != "self")
            ):
                parameters.append(p.name)
        return sorted([p for p in parameters])

    def get_parameters(self) -> dict:
        """
        Get the estimator parameters.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        parameters = {k: getattr(self, k) for k in self._get_param_names()}
        return parameters

    def set_parameters(self, parameters: dict):
        """
        Set processor parameters.

        Parameters
        ----------
        parameters : dict

        """
        self._validate_parameters(parameters)
        valid_parameters = self._get_param_names()
        for k, v in parameters.items():
            if k not in valid_parameters:
                msg = f"{k} is not a valid parameter for {self.__class__}"
                raise ValueError(msg)
            setattr(self, k, v)

    @abstractmethod
    def process(self, data):
        pass

    @staticmethod
    @abstractmethod
    def _validate_parameters(parameters: dict):
        """
        Takes a dictionary of parameters to be passed to set_parameters and validate them.

        """
        ...

    @abstractmethod
    def set_default_parameters(self, instrument: str, separation: str):
        """
        Set default values for a processor based on instrument and separation type.

        """
        ...


class SingleSampleProcessor(Processor):
    """
    Base class to process samples in an assay independently.

    MUST implement _check_data to check compatibility of data with the processor.
    MUST implement _func to process the sample data. Must modify sample data.

    Processor Parameters MUST be defined as key-value parameters in the
    `__init__` method and set as attributes.

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


class FeatureExtractor(SingleSampleProcessor):
    """
    The base class for feature extraction. Includes utilities to filter Features
    based on descriptor values.

    MUST implement `_check_data` from SingleSampleProcessor.
    MUST implement the staticmethod `_extract_features_func` that takes a Roi
    and stores detected features in the `features` attribute.

    """

    def __init__(
        self, filters: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None
    ):
        if filters is None:
            filters = dict()
        self.filters = filters

    @property
    def filters(self) -> dict[str, tuple[float, float]]:
        """
        Gets filter boundaries for each feature descriptor.

        Boundaries are expressed using the name of the descriptor as the key
        and lower and upper bounds as a tuple. If only a lower/upper bound
        is desired, ``None`` values must be used (e.g. ``(None, 10.0)`` to
        use only an upper bound). Descriptors not included in the dictionary
        are ignored during filtering.

        Returns
        -------
        dict[str, tuple[float, float]]
            Descriptor boundaries
        """
        return self._filters

    @filters.setter
    def filters(self, value: dict[str, tuple[Optional[float], Optional[float]]]):
        self._filters = _fill_filter_boundaries(value)

    def _func(self, sample_data: SampleData):
        """
        Detect features in all ROI and filters features using the `filters` attribute.

        Parameters
        ----------
        sample_data : SampleData

        """
        params = self.get_parameters()
        filters = params.pop("filters")
        for roi in sample_data.roi:
            self._extract_features_func(roi, **params)
        _filter_features(sample_data, filters)

    @staticmethod
    @abstractmethod
    def _extract_features_func(roi, **params):
        ...


class MultipleSampleProcessor(Processor):
    """
    Base class for multiple class process. Data is managed through the AssayData class.

    """

    @abstractmethod
    def process(self, data: AssayData):
        ...


class ProcessingPipeline:
    """
    Combines a series of Processor objects of the same type into a data
    processing pipeline.

    Attributes
    ----------
    processors : list[tuple[str, Processor]]

    """

    def __init__(self, steps: Sequence[tuple[str, Processor]]):
        self.processors = steps
        self._name_to_processor = {x: y for x, y in steps}
        if len(self.processors) > len(self._name_to_processor):
            msg = "Processor names must be unique."
            raise ValueError(msg)

    def get_processor(self, name: str) -> Processor:
        processor = self._name_to_processor[name]
        return processor

    def get_parameters(self) -> list[tuple[str, dict]]:
        """
        Get parameters from all processors.

        Returns
        -------
        parameters : list[tuple[str, dict]]

        """
        parameters = list()
        for name, processor in self.processors:
            params = processor.get_parameters()
            parameters.append((name, params))
        return parameters

    def set_parameters(self, parameters: dict[str, dict]):
        """
        Set parameters for processors

        Parameters
        ----------
        parameters : dict[str, dict]
            A dictionary from processor name to processor parameters.

        """
        for name, param in parameters.items():
            processor = self._name_to_processor[name]
            processor.set_parameters(param)

    def set_default_parameters(self, instrument: str, separation: str):
        """
        Set the default parameters for each processor.

        Parameters
        ----------
        instrument : str
            Instrument type.
        separation : str
            Separation method.

        """
        for _, processor in self.processors:
            processor.set_default_parameters(instrument, separation)

    def process(self, data):
        """
        Applies the processing pipeline

        Parameters
        ----------
        data : Any
            data to process. Depends on the processor types. For Single sample
            processors, SampleData is expected. For MultipleSampleProcessor
            AssayData is expected.

        """
        for _, processor in self.processors:
            processor.process(data)


def _filter_features(sample_data: SampleData, filters: dict[str, tuple[float, float]]):
    """
    Filter features using feature descriptors bounds.

    Auxiliary function for FeatureExtractor
    """
    for roi in sample_data.roi:
        if roi.features is not None:
            roi.features = [ft for ft in roi.features if _is_valid_feature(ft, filters)]


def _is_valid_feature(ft: Feature, filters: dict[str, tuple[float, float]]) -> bool:
    """
    Check if a feature descriptors are inside bounds defined by filters.

    Parameters
    ----------
    ft : Feature
    filters : dict[str, tuple[float, float]]
        Dictionary from descriptor name to lower and upper bound values.

    Returns
    -------
    bool

    """
    is_valid = True
    for name, (lower, upper) in filters.items():
        is_valid = lower <= ft.get(name) <= upper
        if not is_valid:
            break
    return is_valid


def _fill_filter_boundaries(
    filters: dict[str, tuple[Optional[float], Optional[float]]]
) -> dict[str, tuple[float, float]]:
    """
    Replace ``None`` values in lower and upper bounds of filter.

    """
    d = dict()
    for name, (lower, upper) in filters.items():
        lower = lower if lower is not None else -inf
        upper = upper if upper is not None else inf
        d[name] = lower, upper
    return d
