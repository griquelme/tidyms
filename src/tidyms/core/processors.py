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

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from math import inf
from typing import ClassVar, Literal, Mapping, Sequence, overload

import pydantic

from . import constants as c
from . import exceptions
from .db import AssayData
from .models import Feature, ProcessorInformation, Roi, Sample, SampleData

logger = logging.getLogger(__file__)


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

    def fetch_initial_snapshot(self) -> SampleData:
        """Fetch the initial sample data."""
        initial_id = self.list_snapshots()[0]
        return self.fetch_snapshot(initial_id)

    def fetch_latest_snapshot(self) -> SampleData:
        """Fetch the latest sample data."""
        latest_id = self.list_snapshots()[-1]
        return self.fetch_snapshot(latest_id)

    def fetch_snapshot(self, snapshot_id: str) -> SampleData:
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
        return self.fetch_snapshot(snapshot_id).get_features()

    def get_roi(self, snapshot_id: str) -> list[Roi]:
        """Create a list of ROI in the specified snapshot."""
        return self.fetch_snapshot(snapshot_id).roi

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
    type: ClassVar[c.ProcessorType]
    order: int | None = None
    pipeline: str | None = None

    def get_processor_info(self) -> ProcessorInformation:
        """Create a ProcessingStepInfo instance."""
        return ProcessorInformation(
            id=self.id, pipeline=self.pipeline, order=self.order, parameters=self.get_parameters()
        )

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    @overload
    def process(self, data: AssayData) -> AssayData:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayData
    ) -> SampleData | SampleDataSnapshots | AssayData:
        """Apply processor function to data."""
        info = self.get_processor_info()
        if isinstance(data, SampleDataSnapshots):
            process_data = data.fetch_latest_snapshot()
            id_ = process_data.sample.id
        elif isinstance(data, SampleData):
            process_data = data
            id_ = process_data.sample.id
        else:
            process_data = data
            id_ = "dummy_id"  # FIXME: add id to AssayData

        msg = f"Applying {info.id} with parameters={info.parameters} to sample={id_}."
        self._func(process_data)
        logger.info(msg)

        if isinstance(process_data, SampleData):
            process_data.processing_info.append(info)

        if isinstance(data, SampleDataSnapshots) and isinstance(process_data, SampleData):
            data.add_snapshot(process_data, info.id)

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
        return self.model_dump(exclude={"id", "type"})

    def set_parameters(self, **kwargs):
        """Set instance parameters."""
        self.__dict__.update(kwargs)

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
    def _func(self, data: SampleData | SampleDataSnapshots | AssayData):
        """Process data function."""
        ...


class ChainedProcessor(pydantic.BaseModel):
    """Concatenate multiple processors into a single unit."""

    id: str
    type: ClassVar[Literal[c.ProcessorType.PIPELINE]] = c.ProcessorType.PIPELINE
    processors: Sequence[Processor]
    order: int | None = None
    pipeline: str | None = None

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

    def get_processor_info(self) -> ProcessorInformation:
        """Create a ProcessingStepInfo instance."""
        info = ProcessorInformation(id=self.id, pipeline=self.pipeline, order=self.order, parameters=dict())
        for proc in self.processors:
            proc_info = proc.get_processor_info()
            info.parameters.update(proc_info.parameters)
        return info

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    @overload
    def process(self, data: AssayData) -> AssayData:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayData
    ) -> SampleData | SampleDataSnapshots | AssayData:
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

    type: ClassVar[Literal[c.ProcessorType.SAMPLE]] = c.ProcessorType.SAMPLE

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

    type: ClassVar[Literal[c.ProcessorType.SAMPLE]] = c.ProcessorType.SAMPLE

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

    type: ClassVar[Literal[c.ProcessorType.SAMPLE]] = c.ProcessorType.SAMPLE

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
    type: ClassVar[Literal[c.ProcessorType.SAMPLE]] = c.ProcessorType.SAMPLE

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
    def process(self, data: AssayData):
        """Apply processor to assay data."""
        ...


class FeatureMatcher(MultipleSampleProcessor):
    """
    Base class for feature correspondence.

    MUST implement the `_func` method, which takes an AssayData instance and
    assigns a label to features from different samples if they have the same
    chemical identity.

    """

    def process(self, data: AssayData):
        """Group features across samples based on their chemical identity."""
        labels = self._func(data)
        data.update_feature_labels(labels)

    @abstractmethod
    def _func(self, data: AssayData) -> dict[int, int]:
        """Create a label for each feature based on their chemical identity."""
        ...


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
    processors: Sequence[Processor | ChainedProcessor]

    def validate(self) -> None:
        """
        Validate the pipeline.

        Raises
        ------
        InvalidPipelineConfiguration
            If an empty pipeline is created or if multiple processors have the
            same id.
        IncompatibleProcessorStatus
            If processors with incompatible status are chained together.

        """
        if len(self.processors) == 0:
            msg = f"No processor provided to pipeline with id={self.id}"
            raise exceptions.InvalidPipelineConfiguration(msg)

        unique_processor_id = {x.id for x in self.processors}
        if len(unique_processor_id) < len(self.processors):
            msg = f"Found multiple processors with same id in pipeline {self.id}."
            raise exceptions.InvalidPipelineConfiguration(msg)

        expected_processor_type = self.processors[0].type
        status = _get_expected_initial_status(expected_processor_type)
        for proc in self.processors:
            assert proc.type == expected_processor_type
            proc_status_in = proc.get_expected_process_status_in()
            check_process_status(status, proc_status_in)
            proc.update_status_out(status)

        expected_final_status = _get_expected_final_status(expected_processor_type)
        check_process_status(status, expected_final_status)

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

    @overload
    def process(self, data: AssayData) -> AssayData:
        ...

    @overload
    def process(self, data: SampleData) -> SampleData:
        ...

    @overload
    def process(self, data: SampleDataSnapshots) -> SampleDataSnapshots:
        ...

    def process(
        self, data: SampleData | SampleDataSnapshots | AssayData
    ) -> SampleData | SampleDataSnapshots | AssayData:
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


def _get_expected_initial_status(processor_type: c.ProcessorType) -> ProcessStatus:
    if processor_type == c.ProcessorType.SAMPLE:
        return ProcessStatus()
    elif processor_type == c.ProcessorType.ASSAY:
        return ProcessStatus(all_roi_extracted=True, all_feature_feature_extracted=True)
    else:
        raise NotImplementedError


def _get_expected_final_status(processor_type: c.ProcessorType) -> ProcessStatus:
    id_ = "final"
    if processor_type is c.ProcessorType.SAMPLE:
        return ProcessStatus(id=id_, sample_roi_extracted=True, sample_feature_extracted=True)
    else:
        raise NotImplementedError
