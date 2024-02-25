"""TidyMS core processors and utilities to process raw data datasets.

ChainedProcessor :
    Concatenate a sequence of processors into a single unit.
FeatureExtractor :
    Base class to extract features from ROI data.
FeatureTransformer :
    Base class to transform feature data.
MultipleSampleProcessor :
    Base class to process multiple samples.
MultipleSampleProcessor :
    Process data from multiple samples stored in an AssayData instance.
ProcessingPipeline :
    Apply a sequence of processors to data.
Processor :
    Base class to process data.
ProcessStatus :
    Store the expected status from data through a processing pipeline.
RoiExtractor :
    Base class to extract ROIs from raw data.
RoiTransformer :
    Base class to transform ROI data.
SampleDataSnapshot :
    Store independent sample data snapshots through a processing pipeline.

"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from math import inf
from typing import Any, Generic, Literal, Mapping, Protocol, Sequence, TypedDict, TypeVar, Union, cast, overload

import pydantic
from typing_extensions import Annotated

from . import constants as c
from . import exceptions
from .models import Feature, Roi, Sample

logger = logging.getLogger(__file__)


FeatureType = TypeVar("FeatureType", bound=Feature)
RoiType = TypeVar("RoiType", bound=Roi)


class DescriptorPatch(TypedDict):
    """Store a descriptor patch."""

    id: int
    descriptor: str
    value: float


class AnnotationPatch(TypedDict):
    """Store an annotation patch."""

    id: int
    field: str
    value: int


class MissingData(TypedDict):
    """Store missing feature data."""

    sample_id: str
    feature_id: int
    value: float


class AssayDataInterface(Protocol, Generic[FeatureType, RoiType]):
    """Public interface of AssayData."""

    def add_sample_data(self, sample_data: SampleData[FeatureType, RoiType]) -> None:
        """Store sample data in DB."""
        ...

    def fetch_sample_data(self, sample_id: str) -> SampleData[FeatureType, RoiType]:
        """Fetch sample data from DB."""
        ...

    def delete_sample_data(self, sample_id: str) -> None:
        """Delete sample data stored in DB."""
        ...

    def add_pipeline(self, pipeline: ProcessingPipeline) -> None:
        """Store pipeline in DB."""
        ...

    def fetch_pipeline(self, pipeline_id: str) -> ProcessingPipeline:
        """Fetch pipeline from DB."""
        ...

    def delete_pipeline(self, sample_id: str) -> None:
        """Delete pipeline stored in DB."""
        ...

    def fetch_descriptors(
        self, samples: str | None = None, descriptors: list[str] | None = None
    ) -> dict[str, list[float]]:
        """Fetch the descriptor table."""
        ...

    def fetch_annotations(self, samples: str | None = None) -> dict[str, list[int]]:
        """Retrieve the annotation table."""
        ...

    def patch_annotations(self, *patches: AnnotationPatch) -> None:
        """Patch feature annotations."""
        ...

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Patch feature descriptors."""
        ...

    def add_missing(self, *missing: MissingData) -> None:
        """Add missing feature values."""
        ...

class SampleData(pydantic.BaseModel, Generic[FeatureType, RoiType]):
    """Stores data state during a sample processing pipeline."""

    sample: Sample
    roi: list[RoiType] = list()
    pipeline: str | None = None

    def get_features(self) -> list[FeatureType]:
        """Update the feature attribute using feature data from ROIs."""
        feature_list = list()
        for roi in self.roi:
            feature_list.extend(roi.features)
        return feature_list


class SampleDataSnapshots:
    """
    Store sample data snapshots down a processing pipeline.

    Attributes
    ----------
    sample : Sample
    copy : bool
        If ``True``, create an independent copy of the data for each processing
        step.

    """

    def __init__(self, sample: Sample) -> None:
        self.sample = sample
        self._snapshots: OrderedDict[str, SampleData] = OrderedDict()
        self.add_snapshot(SampleData(sample=sample), "initial")

    def add_snapshot(self, data: SampleData, snapshot_id: str) -> None:
        """
        Store a new snapshot.

        Parameters
        ----------
        data : SampleData
        snapshot_id: str

        Raises
        ------
        ValueError
            If an existing `snapshot_id` is used.

        """
        if snapshot_id in self._snapshots:
            msg = f"Snapshot with id={snapshot_id} already exists."
            raise ValueError(msg)

        self._snapshots[snapshot_id] = data

    def delete_snapshot(self, snapshot_id: str):
        """
        Delete a snapshot.

        Parameters
        ----------
        snapshot_id : str

        Raises
        ------
        ValueError
            If a non-existing `snapshot_id` is provided.

        """
        try:
            self._snapshots.pop(snapshot_id)
        except KeyError as e:
            msg = f"Snapshot with id={snapshot_id} not found."
            raise ValueError(msg) from e

    def fetch_initial(self) -> SampleData:
        """Fetch a copy of the initial sample data snapshot."""
        initial_id = self.list_snapshots()[0]
        return self.fetch(initial_id)

    def fetch_latest(self) -> SampleData:
        """Fetch a copy of the latest sample data snapshot."""
        latest_id = self.list_snapshots()[-1]
        return self.fetch(latest_id)

    def fetch(self, snapshot_id: str) -> SampleData:
        """
        Create an independent copy of a snapshot.

        Parameters
        ----------
        snapshot_id : str
            The id of the snapshot to retrieve.

        Raises
        ------
        ValueError
            If a non-existing `snapshot_id` is provided.

        """
        try:
            return self._snapshots[snapshot_id].model_copy(deep=True)
        except KeyError as e:
            msg = f"Snapshot with id={snapshot_id} not found."
            raise ValueError(msg) from e

    def get_features(self, snapshot_id: str) -> list[Feature]:
        """Create a list of features in the specified snapshot."""
        return self.fetch(snapshot_id).get_features()

    def get_roi(self, snapshot_id: str) -> list[Roi]:
        """Create a list of ROI in the specified snapshot."""
        return self.fetch(snapshot_id).roi

    def list_snapshots(self) -> list[str]:
        """List snapshots in the order they were created."""
        return list(self._snapshots)


class Processor(ABC, pydantic.BaseModel):
    """
    The base class for data processors.

    Provides functionality to:
    - set default parameters using instrument type, separation type and polarity.
    - set parameters using a dictionary.
    - get default parameters.

    """

    model_config = pydantic.ConfigDict(validate_assignment=True)
    id: str = ""
    order: int = 0
    pipeline: str | None = None

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    @overload
    def process(self, data: AssayDataInterface) -> AssayDataInterface:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayDataInterface
    ) -> SampleData | SampleDataSnapshots | AssayDataInterface:
        """Apply processor function to data."""
        config = self.to_config()
        if isinstance(data, SampleDataSnapshots):
            process_data = data.fetch_latest()
            id_ = process_data.sample.id
        elif isinstance(data, SampleData):
            process_data = data
            id_ = process_data.sample.id
        else:
            process_data = data
            id_ = "dummy_id"  # FIXME: add id to AssayData

        msg = f"Applying {config.id} with parameters={config.parameters} to sample={id_}."
        self._func(process_data)
        logger.info(msg)

        if isinstance(data, SampleDataSnapshots) and isinstance(process_data, SampleData):
            data.add_snapshot(process_data, config.id)

        return data

    @abstractmethod
    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ) -> dict:
        """Get default parameters using instrument, separation type and polarity."""
        ...

    def set_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        """
        Set parameters using instrument type separation type and polarity.

        Parameters
        ----------
        instrument : MSInstrument, default=MSInstrument.QTOF
            The instrument type.
        separation : SeparationMode, default=SeparationMode.UPLC
            The separation analytical platform.
        polarity : Polarity, default=Polarity.POSITIVE
            The measurement polarity.

        See Also
        --------
        get_default_parameters
            Retrieve default parameters using instrument type, separation type
            and polarity mode.

        """
        defaults = self.get_default_parameters(instrument, separation, polarity)
        self.set_parameters(**defaults)

    def get_parameters(self):
        """Get instance parameters."""
        params = self.model_dump()
        params["type"] = params["type"].value
        params["accepts"] = params["accepts"].value
        return params

    def set_parameters(self, **kwargs):
        """Set instance parameters."""
        self.__dict__.update(kwargs)

    def to_config(self) -> ProcessorConfiguration:
        """Convert processor into a configuration dictionary."""
        return ProcessorConfiguration(
            id=self.id,
            type=self.__class__.__name__,
            order=self.order,
            pipeline=self.pipeline,
            parameters=self.model_dump(exclude={"type", "accepts"}),
        )

    @abstractmethod
    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected process status of the sample data."""
        ...

    @staticmethod
    @abstractmethod
    def update_status_out(status_in: ProcessStatus) -> ProcessStatus:
        """Set output process status."""
        ...

    @abstractmethod
    def _func(self, data: SampleData | SampleDataSnapshots | AssayDataInterface):
        """Process data function."""
        ...


class ChainedProcessor(pydantic.BaseModel):
    """Concatenate multiple processors into a single unit."""

    id: str
    type: Literal[c.ProcessorType.CHAINED_PROCESSOR] = c.ProcessorType.CHAINED_PROCESSOR
    processors: Sequence[ChainedProcessorUnit] = list()
    order: int = 0
    pipeline: str | None = None

    @property
    def accepts(self) -> c.DataType:
        """Accepts getter."""
        return self.processors[0].accepts

    @abstractmethod
    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ) -> dict:
        """Get default parameters using instrument, separation type and polarity."""
        ...

    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected process status of the sample data."""
        return self.processors[0].get_expected_process_status_in()

    def get_parameters(self):
        """Get instance parameters."""
        return self.model_dump(exclude={"id", "type"})

    def to_config(self) -> ProcessorConfiguration:
        """Create a processor configuration object."""
        return ProcessorConfiguration(
            id=self.id,
            type=self.__class__.__name__,
            pipeline=self.pipeline,
            order=self.order,
            parameters={"processors": [x.to_config() for x in self.processors]}
        )

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    @overload
    def process(self, data: AssayDataInterface) -> AssayDataInterface:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayDataInterface
    ) -> SampleData | SampleDataSnapshots | AssayDataInterface:
        """Apply processor function to data."""
        for proc in self.processors:
            data = proc.process(data)
        return data

    def set_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        """
        Set parameters using instrument type separation type and polarity.

        Parameters
        ----------
        instrument : MSInstrument, default=MSInstrument.QTOF
            The instrument type.
        separation : SeparationMode, default=SeparationMode.UPLC
            The separation analytical platform.
        polarity : Polarity, default=Polarity.POSITIVE
            The measurement polarity.

        See Also
        --------
        get_default_parameters
            Retrieve default parameters using instrument type, separation type
            and polarity mode.

        """
        defaults = self.get_default_parameters(instrument, separation, polarity)
        self.set_parameters(**defaults)

    def set_parameters(self, **kwargs):
        """Set instance parameters."""
        for proc in self.processors:
            proc_params = {k: v for k, v in kwargs.items() if k in proc.__dict__}
            proc.set_parameters(**proc_params)

    def update_status_out(self, status_in: ProcessStatus) -> ProcessStatus:
        """Set output process status."""
        for proc in self.processors:
            proc.update_status_out(status_in)
        return status_in


class RoiExtractor(Processor):
    """
    Base class to extract ROIs from raw data.

    MUST implement extract_roi_func to extract ROIs from raw data.

    """

    type: Literal[c.ProcessorType.ROI_EXTRACTOR] = c.ProcessorType.ROI_EXTRACTOR
    accepts: Literal[c.DataType.SAMPLE] = c.DataType.SAMPLE

    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected status of the sample data."""
        return ProcessStatus(id=self.id)

    def _func(self, sample_data: SampleData):
        """Add extracted ROIs to sample data."""
        sample_data.roi = self._extract_roi_func(sample_data.sample)

    @abstractmethod
    def _extract_roi_func(self, sample: Sample) -> list[Roi]:
        ...

    @staticmethod
    def update_status_out(status_in: ProcessStatus) -> ProcessStatus:
        """Retrieve output status of sample data."""
        status_in.sample_roi_extracted = True
        return status_in


class RoiTransformer(Processor):
    """
    Base class to transform ROIs.

    MUST implement _transform_roi to transform a ROI.

    """

    type: Literal[c.ProcessorType.ROI_TRANSFORMER] = c.ProcessorType.ROI_TRANSFORMER
    accepts: Literal[c.DataType.SAMPLE] = c.DataType.SAMPLE

    def _func(self, sample_data: SampleData) -> None:
        for roi in sample_data.roi:
            self._transform_roi(roi)

    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected status of the sample data."""
        return ProcessStatus(id=self.id, sample_roi_extracted=True)

    @staticmethod
    def update_status_out(status_in: ProcessStatus) -> ProcessStatus:
        """Retrieve output status of sample data."""
        return status_in

    @abstractmethod
    def _transform_roi(self, roi: Roi):
        ...


class FeatureTransformer(Processor):
    """
    Base class to transform features.

    MUST implement _transform_feature to transform a feature.

    """

    type: Literal[c.ProcessorType.FEATURE_TRANSFORMER] = c.ProcessorType.FEATURE_TRANSFORMER
    accepts: Literal[c.DataType.SAMPLE] = c.DataType.SAMPLE

    def _func(self, sample_data: SampleData):
        for feature in sample_data.get_features():
            self._transform_feature(feature)

    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected status of the sample data."""
        return ProcessStatus(id=self.id, sample_roi_extracted=True, sample_feature_extracted=True)

    @staticmethod
    def update_status_out(status_in: ProcessStatus) -> ProcessStatus:
        """Retrieve output status of sample data."""
        return status_in

    @abstractmethod
    def _transform_feature(self, feature: Feature):
        ...


class FeatureExtractor(Processor):
    """
    Base class to extract features from ROIs.

    MUST implement _extract_features_func to extract features from a ROI.

    The filter field define valid boundaries for each feature descriptor.
    Boundaries are expressed using the name of the descriptor as the key and
    lower and upper bounds as a tuple. If only a lower/upper bound is desired,
    ``None`` values must be used (e.g. ``(None, 10.0)`` to use only an upper
    bound). Descriptors not included in the dictionary are ignored during
    filtering.

    """

    filters: Mapping[str, tuple[float | None, float | None]] = dict()
    type: Literal[c.ProcessorType.FEATURE_EXTRACTOR] = c.ProcessorType.FEATURE_EXTRACTOR
    accepts: Literal[c.DataType.SAMPLE] = c.DataType.SAMPLE

    def get_expected_process_status_in(self) -> ProcessStatus:
        """Get the expected status of the sample data."""
        return ProcessStatus(id=self.id, sample_roi_extracted=True)

    @staticmethod
    def update_status_out(status_in: ProcessStatus) -> ProcessStatus:
        """Set output status of process status."""
        status_in.sample_feature_extracted = True
        return status_in

    def _func(self, sample_data: SampleData):
        """
        Detect features in all ROI and filters features using the `filters` attribute.

        Parameters
        ----------
        sample_data : SampleData

        """
        filters = _fill_filter_boundaries(self.filters)
        for roi in sample_data.roi:
            for ft in self._extract_features_func(roi):
                if _is_feature_descriptor_in_valid_range(ft, filters):
                    roi.add_feature(ft)

    @abstractmethod
    def _extract_features_func(self, roi: Roi) -> Sequence[Feature]:
        ...


class MultipleSampleProcessor(Processor):
    """
    Base class for multiple class process.

    Data is managed through the AssayData class.

    MUST implement the process method.

    """

    @abstractmethod
    def process(self, data: AssayDataInterface):
        """Apply processor to assay data."""
        ...


class FeatureMatcher(MultipleSampleProcessor):
    """
    Base class for feature correspondence.

    MUST implement the `_func` method, which takes an AssayData instance and
    assigns a label to features from different samples if they have the same
    chemical identity.

    """

    def process(self, data: AssayDataInterface):
        """Group features across samples based on their chemical identity."""
        patches = self._func(data)
        data.patch_annotations(*patches)

    @abstractmethod
    def _func(self, data: AssayDataInterface) -> list[AnnotationPatch]:
        """Create a label for each feature based on their chemical identity."""
        ...


ChainedProcessorUnit = Annotated[
    Union[RoiExtractor, RoiTransformer, FeatureExtractor, FeatureTransformer],
    pydantic.Field(discriminator="type"),
]

ProcessorItem = Annotated[
    Union[RoiExtractor, RoiTransformer, FeatureExtractor, FeatureTransformer, ChainedProcessor],
    pydantic.Field(discriminator="type"),
]


class ProcessorConfiguration(pydantic.BaseModel):
    """Stores sample processing step name and parameters."""

    id: str
    pipeline: str | None = None
    order: int
    type: str
    parameters: dict[str, Any]

    @pydantic.field_serializer("parameters")
    def serialize_parameters(self, parameters: dict[str, Any], _info) -> str:
        """Serialize parameters field into a JSON string."""
        return json.dumps(parameters)

    @pydantic.field_validator("parameters", mode="before")
    @classmethod
    def deserialize_json_str(cls, processors: dict[str, Any] | str) -> dict[str, Any]:
        """Deserialize a parameter JSON string into a dictionary."""
        if isinstance(processors, str):
            processors = cast(dict[str, Any], json.loads(processors))
        return processors


class ProcessorRegistry:
    """Maintain a registry of available processors."""

    _PROCESSORS = dict()

    @classmethod
    def list_available(cls) -> list[str]:
        """Retrieve name of all available processors."""
        return list(cls._PROCESSORS)

    @classmethod
    def create_from_config(cls, config: ProcessorConfiguration) -> ProcessorItem:
        """Create a processor instance from a processor configuration."""
        proc_cls = cls.get_processor(config.type)
        if isinstance(proc_cls, ChainedProcessor):
            config.parameters["processors"] = [cls.create_from_config(x) for x in config.parameters["processors"]]
            return ChainedProcessor(**config.parameters)
        return proc_cls(**config.parameters)


    @classmethod
    def get_processor(cls, type_: str) -> type[ProcessorItem]:
        """
        Retrieve a Processor type from the registry.

        Parameters
        ----------
        type_ : str
            The name of the Processor to retrieve.

        Returns
        -------
        type[Processor]

        Raises
        ------
        ProcessorTypeNotRegistered
            If a non-registered Processor name is requested

        """
        try:
            return cls._PROCESSORS[type_]
        except KeyError as e:
            raise exceptions.ProcessorTypeNotRegistered(type_) from e

    @classmethod
    def register(cls, processor: type[ProcessorItem]) -> type[ProcessorItem]:
        """
        Add a processor to the registry.

        Parameters
        ----------
        processor : type[Processor]

        Returns
        -------
        type[Processor]

        """
        name = processor.__name__
        if name in cls._PROCESSORS:
            raise exceptions.RegisteredProcessorError(name)

        cls._PROCESSORS[name] = processor
        return processor


class ProcessingPipeline(pydantic.BaseModel):
    """
    Combines Processor objects into a data processing pipeline.

    Processors must be all subclasses from one of the three:

    - BaseSampleProcessor
    - BaseAssayProcessor
    - BaseMatrixProcessor

    Attributes
    ----------
    processors : list[_Processor]

    """

    id: str
    processors: Sequence[ProcessorItem]

    @pydantic.field_validator("processors")
    @classmethod
    def check_non_empty_processor(cls, processors: Sequence[ProcessorItem]) -> Sequence[ProcessorItem]:
        """Check that the list of processors is not empty."""
        if len(processors) == 0:
            msg = "Pipelines must contain at least one processor."
            raise exceptions.InvalidPipelineConfiguration(msg)
        return processors

    @pydantic.field_validator("processors")
    @classmethod
    def check_unique_processor_ids(cls, processors: Sequence[ProcessorItem]) -> Sequence[ProcessorItem]:
        """Check that the id of each processor is unique."""
        unique_processor_id = {x.id for x in processors}
        if len(unique_processor_id) < len(processors):
            msg = "Found multiple processors with same id."
            raise exceptions.InvalidPipelineConfiguration(msg)
        return processors

    @pydantic.model_validator(mode="before")
    @classmethod
    def attach_pipeline_metadata(cls, data):
        """Attach order value and pipeline id to each processor."""
        for k, proc in enumerate(data["processors"]):
            if isinstance(proc, Processor):
                proc.order = k
                proc.pipeline = data["id"]
            else:
                proc["order"] = k
                proc["pipeline"] = data["id"]
        return data

    @pydantic.field_validator("processors")
    @classmethod
    def check_processors_accept_same_data_type(cls, processors: Sequence[ProcessorItem]) -> Sequence[ProcessorItem]:
        """Check that all processors accept the same data type as input."""
        first = processors[0]
        if isinstance(first, ChainedProcessor) and first.processors:
            expected = first.processors[0].accepts
        else:
            expected = first.accepts
        for proc in processors:
            if proc.accepts != expected:
                msg = (
                    f"Expected all processors to accept data type = {expected.value}. Processor with id={proc.id} "
                    f"accepts data type = {proc.accepts.value}."
                )
                raise exceptions.IncompatibleProcessorStatus(msg)
        return processors

    @pydantic.field_validator("processors")
    @classmethod
    def check_compatible_processors(cls, processors: Sequence[ProcessorItem]) -> Sequence[ProcessorItem]:
        """Check that the Output of a processor matches the expected input of the next processor."""
        accepts = processors[0].accepts
        status = _get_expected_initial_status(accepts)
        for proc in processors:
            proc_status_in = proc.get_expected_process_status_in()
            check_process_status(status, proc_status_in)
            proc.update_status_out(status)

        expected_final_status = _get_expected_final_status(accepts)
        check_process_status(status, expected_final_status)
        return processors

    def get_processor(self, id_: str) -> Processor | ChainedProcessor:
        """Get pipeline processors based on name."""
        for processor in self.processors:
            if processor.id == id_:
                return processor

        raise exceptions.ProcessorNotFound(id_)

    def set_parameters(self, parameters: dict[str, dict]):
        """
        Set parameters for processors.

        Parameters
        ----------
        parameters : dict[str, dict]
            A dictionary that maps processor names to processor parameters.

        """
        for name, param in parameters.items():
            processor = self.get_processor(name)
            processor.set_parameters(**param)

    def get_configuration(self) -> dict:
        """Get the pipeline default parameters."""
        configuration = self.model_dump(exclude={"processors"})
        processors = list()
        for proc in self.processors:
            proc_params = proc.model_dump()
            proc_params["class"] = proc.__class__.__name__
            processors.append(proc_params)
        configuration["processors"] = processors
        return configuration

    def set_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        """
        Set the default parameters for each processor.

        Parameters
        ----------
        instrument : str
            Instrument type.
        separation : str
            Separation method.

        """
        for processor in self.processors:
            processor.set_default_parameters(instrument, separation, polarity)

    def to_dict(self) -> dict:
        """Create a dictionary with pipeline configuration."""
        return {"id": self.id, "processors": [x.to_config() for x in self.processors]}

    @classmethod
    def from_dict(cls, d: dict) -> ProcessingPipeline:
        """Create a new pipeline instance from a configuration dictionary."""
        processors = [ProcessorRegistry.create_from_config(x) for x in d["processors"]]
        return ProcessingPipeline(id=d["id"], processors=processors)

    @overload
    def process(self, data: AssayDataInterface) -> AssayDataInterface:
        ...

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayDataInterface
    ) -> SampleData | SampleDataSnapshots | AssayDataInterface:
        """
        Apply the processing pipeline.

        Parameters
        ----------
        data : Any
            data to process. Depends on the processor types. For Single sample
            processors, SampleData is expected. For MultipleSampleProcessor
            AssayData is expected.

        """
        for processor in self.processors:
            processor.process(data)
        return data


class ProcessStatus(pydantic.BaseModel):
    """
    Report data process status.

    Each attribute is a flag representing the current status of the data through
    a processing pipeline.

    """

    id: str | None = None
    sample_roi_extracted: bool = False
    sample_feature_extracted: bool = False
    sample_isotopologue_annotated: bool = False
    sample_adduct_annotated: bool = False
    all_roi_extracted: bool = False
    all_feature_feature_extracted: bool = False
    all_isotopologue_annotated: bool = False
    all_adduct_annotated: bool = False
    all_feature_matched: bool = False
    all_missing_imputed: bool = False


def check_process_status(status1: ProcessStatus, status2: ProcessStatus) -> None:
    """Check if two process status are compatible."""
    for attr, actual in status1.model_dump(exclude={"id"}).items():
        expected = getattr(status2, attr)
        check_ok = _compare_process_status(actual, expected)
        if not check_ok:
            status = " ".join(attr.split("_"))
            msg = f"Expected {status} to be {expected} in step {status2.id}. Got {actual}."
            raise exceptions.IncompatibleProcessorStatus(msg)


def _compare_process_status(actual: bool, expected: bool) -> bool:
    if expected:
        return expected and actual
    else:
        return True


def _is_feature_descriptor_in_valid_range(ft: Feature, filters: dict[str, tuple[float, float]]) -> bool:
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
    filters: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, tuple[float, float]]:
    """Replace ``None`` values in lower and upper bounds of feature extractor filter."""
    d = dict()
    for name, (lower, upper) in filters.items():
        lower = lower if lower is not None else -inf
        upper = upper if upper is not None else inf
        d[name] = lower, upper
    return d


def _get_expected_initial_status(data_type: c.DataType) -> ProcessStatus:
    if data_type == c.DataType.SAMPLE:
        return ProcessStatus()
    elif data_type == c.DataType.ASSAY:
        return ProcessStatus(all_roi_extracted=True, all_feature_feature_extracted=True)
    else:
        raise NotImplementedError


def _get_expected_final_status(data_type: c.DataType) -> ProcessStatus:
    id_ = "final"
    if data_type is c.DataType.SAMPLE:
        return ProcessStatus(id=id_, sample_roi_extracted=True, sample_feature_extracted=True)
    else:
        raise NotImplementedError
