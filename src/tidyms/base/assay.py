"""
Tools to process datasets.

Assay :
    Manages processing of complete datasets.
SingleSampleProcessor :
    Base class to process data from a single sample.
FeatureExtractor :
    Base class to extract feature from ROI.
MultipleSampleProcessor :
    Base class to process multiple samples.
MultipleSampleProcessor :
    Process data from multiple samples stored in an AssayData instance.
ProcessingPipeline :
    Apply several processing steps to data.

"""
from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from math import inf
from multiprocessing.pool import Pool
from typing import Mapping, Sequence, Type

from ..utils import get_progress_bar
from .base import (
    Feature,
    Roi,
    Sample,
    SampleData,
    SampleDataProcessStatus,
    ProcessingStepInfo,
)
from .db import AssayData
from . import constants as c

import pydantic

logger = logging.getLogger(__file__)


# TODO: don't process multiple times the same sample.
# TODO: set_parameters and set_default_parameters.
# TODO: store pipeline parameters using set parameters.


class Assay:
    """
    Manage data processing and storage of datasets.

    See HERE for a tutorial on how to work with Assay objects.

    Parameters
    ----------
    path : str or Path
        Path to store the assay data. If ``None``, the assay data is stored
        on memory. If a Path is passed, the processed data from each sample is
        stored on disk.
    sample_pipeline: ProcessingPipeline
        A processing pipeline to process each sample.
    roi_type: Type[Roi]
        The Roi Type generated during sample processing. This value is passed
        to :py:class:`tidyms.base.AssayData` to enable retrieval of data stored
        on disk.
    feature_type: Type[Feature]
        The Feature Type generated during sample processing. This value is
        passed to :py:class:`tidyms.base.AssayData` to enable retrieval of data
        stored on disk.

    """

    def __init__(
        self,
        path: str | Path,
        sample_pipeline: ProcessingPipeline,
        roi_type: Type[Roi],
        feature_type: Type[Feature],
    ):
        if isinstance(path, str):
            path = Path(path)

        self._data = AssayData(path, roi_type, feature_type)
        self._sample_pipeline = sample_pipeline
        self._multiple_sample_pipeline = list()
        self._data_matrix_pipeline = list()

    @property
    def data(self):
        """Get the Assay data instance."""
        return self._data

    def add_samples(self, samples: list[Sample]):
        """
        Add samples to the Assay.

        Parameters
        ----------
        samples : list[Sample]

        """
        self.data.add_samples(samples)

    def get_samples(self) -> list[Sample]:
        """List all samples in the assay."""
        return self.data.get_samples()

    def get_parameters(self) -> dict:
        """Get the processing parameters of each processing pipeline."""
        parameters = dict()
        parameters["sample pipeline"] = self._sample_pipeline.get_parameters()
        return parameters

    def get_feature_descriptors(
        self, sample_id: str | None = None, descriptors: list[str] | None = None
    ) -> dict[str, list]:
        """
        Get descriptors for features detected in the assay.

        Parameters
        ----------
        sample_id : str or None, default=None
            Retrieves descriptors for the selected sample. If ``None``,
            descriptors for all samples are returned.
        descriptors : list[str] or None, default=None
            Feature  descriptors to retrieve. By default, all available
            descriptors are retrieved.

        Returns
        -------
        dict[str, list]
            A dictionary where each key is a descriptor and values are list of
            values for each feature. Beside all descriptors, four additional
            entries are provided: `id` is an unique identifier for the feature.
            `roi_id` identifies the ROI where the feature was detected.
            `sample_id` identifies the sample where the feature was detected.
            `label` identifies the feature group obtained from feature
            correspondence.

        """
        sample = None if sample_id is None else self.data.search_sample(sample_id)
        return self.data.get_descriptors(descriptors=descriptors, sample=sample)

    def get_features(
        self, key, by: str, groups: list[str] | None = None
    ) -> list[Feature]:
        """
        Retrieve features from the assay.

        Features can be retrieved by sample, feature label or id.

        Parameters
        ----------
        key : str or int
            Key used to search features. If `by` is set
            to ``"id"`` or ``"label"`` an integer must be provided. If `by` is
            set to ``"sample"`` a string with the corresponding sample id must
            be passed.
        by : {"sample", "id", "label"}
            Criteria to select features. ``"sample"`` returns all features from
            a given sample, ``"id"``, retrieves a feature by id and ``"label"``
            retrieves features labelled by a correspondence algorithm.
        groups : list[str] or None, default=None
            Select features from these sample groups. Applied only if `by` is
            set to ``"label"``.

        Returns
        -------
        list[Feature]

        """
        if by == "sample":
            sample = self.data.search_sample(key)
            features = self.data.get_features_by_sample(sample)
        elif by == "label":
            features = self.data.get_features_by_label(key, groups=groups)
        elif by == "id":
            features = [self.data.get_features_by_id(key)]
        else:
            msg = f"`by` must be one of 'sample', 'label' or 'id'. Got {by}."
            raise ValueError(msg)

        return features

    def get_roi(self, key, by: str) -> Sequence[Roi]:
        """
        Retrieve ROIs from the assay. ROIs can be retrieved by sample, or id.

        Parameters
        ----------
        key : str or int
            Key used to search ROIs. If `by` is set
            to ``"id"`` an integer must be provided. If `by` is set to
            ``"sample"`` a string with the corresponding sample id must be
            provided.
        by : {"sample", "id"}
            Criteria to select features.
        groups : list[str] or None, default=None
            Select features from these sample groups. Applied only if `by` is
            set to ``"label"``.

        Returns
        -------
        list[Feature]

        Raises
        ------
        ValueError
            If an invalid value is passed to `by`.
            If an non-existent sample is passed to `key`.
            If an non-existent ROI id is passed to `key`.

        """
        if by == "sample":
            data = self._data.get_sample_data(key)
            roi_list = data._roi
        elif by == "id":
            roi_list = [self._data.get_roi_by_id(key)]
        else:
            msg = f"`by` must be one of 'sample' or 'id'. Got {by}."
            raise ValueError(msg)
        return roi_list

    def process_samples(
        self,
        samples: list[str] | None = None,
        n_jobs: int | None = 1,
        delete_empty_roi: bool = True,
        silent: bool = True,
    ):
        """
        Apply individual samples processing steps.

        See HERE for a detailed explanation of how assay-based sample processing
        works.

        Parameters
        ----------
        samples : list[str] or None, default=None
            List of sample ids to process. If ``None``, process all samples in
            the assay.
        n_jobs : int or None, default=1
            Number of cores to use to process sample in parallel. If ``None``,
            uses all available cores.
        delete_empty_roi: bool, default=True
            Deletes ROI where no feature was detected.
        silent : bool, default=True
            Process samples silently. If set to ``False``, displays a progress
            bar.

        See Also
        --------
        tidyms.base.Assay.get_parameters: returns parameters for all processing steps.
        tidyms.base.Assay.set_parameters: set parameters for all processing steps.

        """
        if samples is None:
            samples = [x.id for x in self.get_samples()]

        def iterator():
            for sample_id in samples:
                sample_data = self._data.get_sample_data(sample_id)
                pipeline = self._sample_pipeline.copy()
                yield pipeline, sample_data

        if not silent:
            tqdm_func = get_progress_bar()
            bar = tqdm_func()
            bar.total = len(samples)
        else:
            bar = None

        with Pool(n_jobs) as pool:
            for sample_data in pool.imap_unordered(_process_sample_worker, iterator()):
                if delete_empty_roi:
                    sample_data.delete_empty_roi()
                self.data.store_sample_data(sample_data)
                if not silent and bar is not None:
                    bar.set_description(f"Processing {sample_data.sample.id}")
                    bar.update()


class _Processor(ABC, pydantic.BaseModel):
    """
    The base class for data processors.

    Provides functionality to:
    - set default parameters using instrument type, separation type and polarity.
    - set parameters using a dictionary.
    - get default parameters.

    """

    model_config = pydantic.ConfigDict(validate_assignment=True)

    @abstractmethod
    def process(self, data):
        ...

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
        return self.model_dump()

    def set_parameters(self, **kwargs):
        """Set instance parameters."""
        self.__dict__.update(kwargs)


class BaseSampleProcessor(_Processor):
    """
    Base class to process independent samples.

    MUST implement _get_expected_sample_data_status to check if the sample
    data status allows applying the processing function.

    MUST implement _update_sample_data_status to update the sample data status
    after applying the processing function.

    MUST implement _func to process the sample data.

    """

    def process(self, sample_data: SampleData):
        """Apply processor function to a sample."""
        expected_status = self._get_expected_sample_data_status()
        sample_data.status.check(expected_status)
        info = self._get_processor_info()
        msg = f"Applying {info.name} with parameters={info.parameters} to sample={sample_data.sample.id}."
        logger.info(msg)
        self._func(sample_data)
        self._update_sample_data_status(sample_data)
        sample_data.add_processing_step_info(info)

    @abstractmethod
    def _func(self, sample_data: SampleData):
        """Processing function that modifies sample data."""
        ...

    def _get_processor_info(self) -> ProcessingStepInfo:
        """Create a ProcessingStepInfo instance."""
        return ProcessingStepInfo(
            name=self.__class__.__name__, parameters=self.get_parameters()
        )

    @staticmethod
    @abstractmethod
    def _get_expected_sample_data_status() -> SampleDataProcessStatus:
        """Provide the expected sample data status to apply the processing function."""
        ...

    @staticmethod
    @abstractmethod
    def _update_sample_data_status(sample_data: SampleData):
        """Update the sample data status after applying the processing function."""
        ...


class BaseRoiExtractor(BaseSampleProcessor):
    """
    Base class to extract ROIs from raw data.

    MUST implement _func to add ROIs to sample data.

    """

    @staticmethod
    def _get_expected_sample_data_status() -> SampleDataProcessStatus:
        return SampleDataProcessStatus()

    @staticmethod
    def _update_sample_data_status(sample_data: SampleData):
        sample_data.status.roi = True


class BaseRoiTransformer(BaseSampleProcessor):
    """
    Base class to transform ROIs.

    MUST implement _transform_roi to transform a ROI.

    """

    @staticmethod
    def _get_expected_sample_data_status() -> SampleDataProcessStatus:
        return SampleDataProcessStatus(roi=True)

    def _func(self, sample_data: SampleData):
        for roi in sample_data.roi:
            self._transform_roi(roi)

    @staticmethod
    def _update_sample_data_status(sample_data: SampleData):
        sample_data.status.roi = True

    @abstractmethod
    def _transform_roi(self, roi: Roi):
        ...


class BaseFeatureTransformer(BaseSampleProcessor):
    """
    Base class to transform features.

    MUST implement _transform_feature to transform a feature.

    """

    @staticmethod
    def _get_expected_sample_data_status() -> SampleDataProcessStatus:
        return SampleDataProcessStatus(roi=True, feature=True)

    def _func(self, sample_data: SampleData):
        for roi in sample_data.roi:
            for ft in roi.features:
                self._transform_feature(ft)

    @staticmethod
    def _update_sample_data_status(sample_data: SampleData):
        sample_data.status.roi = True

    @abstractmethod
    def _transform_feature(self, feature: Feature):
        ...


class BaseFeatureExtractor(BaseSampleProcessor):
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

    @staticmethod
    def _get_expected_sample_data_status() -> SampleDataProcessStatus:
        return SampleDataProcessStatus(roi=True)

    @staticmethod
    def _update_sample_data_status(sample_data: SampleData):
        sample_data.status.feature = True

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


class MultipleSampleProcessor(_Processor):
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


class ProcessingPipeline:
    """
    Combines a Processor objects into a data processing pipeline.

    Attributes
    ----------
    processors : list[tuple[str, Processor]]

    """

    def __init__(self, steps: Sequence[tuple[str, _Processor]]):
        self.processors = steps
        self._name_to_processor = {x: y for x, y in steps}
        if len(self.processors) > len(self._name_to_processor):
            msg = "Processor names must be unique."
            raise ValueError(msg)

    def copy(self):
        """Create a deep copy of the pipeline."""
        return deepcopy(self)

    def get_processor(self, name: str) -> _Processor:
        """Get pipeline processors based on name."""
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
        Set parameters for processors.

        Parameters
        ----------
        parameters : dict[str, dict]
            A dictionary that maps processor names to processor parameters.

        """
        for name, param in parameters.items():
            processor = self.get_processor(name)
            processor.set_parameters(**param)

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
        for _, processor in self.processors:
            processor.set_default_parameters(instrument, separation, polarity)

    def process(self, data):
        """
        Apply the processing pipeline.

        Parameters
        ----------
        data : Any
            data to process. Depends on the processor types. For Single sample
            processors, SampleData is expected. For MultipleSampleProcessor
            AssayData is expected.

        """
        for _, processor in self.processors:
            processor.process(data)
        return data


def _is_feature_descriptor_in_valid_range(
    ft: Feature, filters: dict[str, tuple[float, float]]
) -> bool:
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


def _process_sample_worker(args: tuple[ProcessingPipeline, SampleData]):
    pipeline, sample_data = args
    return pipeline.process(sample_data)


def _fill_filter_boundaries(
    filters: Mapping[str, tuple[float | None, float | None]]
) -> dict[str, tuple[float, float]]:
    """Replace ``None`` values in lower and upper bounds of feature extractor filter."""
    d = dict()
    for name, (lower, upper) in filters.items():
        lower = lower if lower is not None else -inf
        upper = upper if upper is not None else inf
        d[name] = lower, upper
    return d
