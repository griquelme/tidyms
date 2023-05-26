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
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from math import inf
from multiprocessing.pool import Pool
from typing import Optional, Union, Sequence, Type
from ..utils import get_progress_bar
from . import AssayData, Feature, Roi, Sample, SampleData


# TODO: don't process multiple times the same sample.
# TODO: set_parameters and set_default_parameters.
# TODO: store pipeline parameters using set parameters.


class Assay:
    """
    Manage data processing and storage of datasets.

    See HERE for a tutorial on how to work with Assay objects.

    """

    def __init__(
        self,
        path: Union[str, Path],
        sample_pipeline: "ProcessingPipeline",
        roi_type: Type[Roi],
        feature_type: Type[Feature],
    ):
        self._data = AssayData(path, roi_type, feature_type)
        self._sample_pipeline = sample_pipeline
        self._multiple_sample_pipeline = list()
        self._data_matrix_pipeline = list()

    def add_samples(self, samples: list[Sample]):
        """
        Add samples to the Assay.

        Parameters
        ----------
        samples : list[Sample]

        """
        self._data.add_samples(samples)

    def get_samples(self) -> list[Sample]:
        """List all samples in the assay."""
        return self._data.get_samples()

    def get_parameters(self) -> dict:
        """Get the processing parameters of each processing pipeline."""
        parameters = dict()
        parameters["sample pipeline"] = self._sample_pipeline.get_parameters()
        return parameters

    def get_feature_descriptors(
        self, sample_id: Optional[str] = None, descriptors: Optional[list[str]] = None
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
        sample = None if sample_id is None else self._data.search_sample(sample_id)
        return self._data.get_descriptors(descriptors=descriptors, sample=sample)

    def get_features(self, key, by: str, groups: Optional[list[str]] = None) -> list[Feature]:
        """
        Retrieve features from the assay.

        Features can be retrieved by sample, feature label or id.

        Parameters
        ----------
        key : str or int
            Key used to search features. If `by` is set
            to ``"id"`` or ``"label"`` an integer must be provided. If `by` is
            set to ``"sample"`` a string with the corresponding sample id.
        by : {"sample", "id", "label"}
            Criteria to select features.
        groups : list[str] or None, default=None
            Select features from these sample groups. Applied only if `by` is
            set to ``"label"``.

        Returns
        -------
        list[Feature]

        """
        if by == "sample":
            sample = self._data.search_sample(key)
            features = self._data.get_features_by_sample(sample)
        elif by == "label":
            features = self._data.get_features_by_label(key, groups=groups)
        elif by == "id":
            features = [self._data.get_features_by_id(key)]
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
            roi_list = data.roi
        elif by == "id":
            roi_list = [self._data.get_roi_by_id(key)]
        else:
            msg = f"`by` must be one of 'sample' or 'id'. Got {by}."
            raise ValueError(msg)
        return roi_list

    def process_samples(
        self,
        samples: Optional[list[str]] = None,
        n_jobs: Optional[int] = 1,
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
                    sample_data.roi = [x for x in sample_data.roi if x.features is not None]
                self._data.store_sample_data(sample_data)
                if not silent and bar is not None:
                    bar.set_description(f"Processing {sample_data.sample.id}")
                    bar.update()


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
        """Create a string representation of the instance."""
        name = self.__class__.__name__
        arg_str = ", ".join([f"{k}={v}" for k, v in self.get_parameters().items()])
        return f"{name}({arg_str})"

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
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
        """Apply processing function."""
        pass

    @staticmethod
    @abstractmethod
    def _validate_parameters(parameters: dict):
        """Validate processor parameters."""
        ...

    @abstractmethod
    def set_default_parameters(self, instrument: str, separation: str):
        """Set default parameters based on instrument and separation type."""
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
        """Apply processor to a sample."""
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
    The base class for feature extraction.

    Includes utilities to filter Features based on descriptor values.

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
    assigns a label to features from different samples if the have the same
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

    def __init__(self, steps: Sequence[tuple[str, Processor]]):
        self.processors = steps
        self._name_to_processor = {x: y for x, y in steps}
        if len(self.processors) > len(self._name_to_processor):
            msg = "Processor names must be unique."
            raise ValueError(msg)

    def copy(self):
        """Create a deep copy of the pipeline."""
        return deepcopy(self)

    def get_processor(self, name: str) -> Processor:
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
    """Replace ``None`` values in lower and upper bounds of filter."""
    d = dict()
    for name, (lower, upper) in filters.items():
        lower = lower if lower is not None else -inf
        upper = upper if upper is not None else inf
        d[name] = lower, upper
    return d


def _process_sample_worker(args: tuple[ProcessingPipeline, SampleData]):
    pipeline, sample_data = args
    return pipeline.process(sample_data)
