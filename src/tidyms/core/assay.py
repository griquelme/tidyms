"""Utility to process datasets.

Assay :
    Provide functionality to process and store data from complete datasets.
AssayConfiguration :
    Store assay configuration.

"""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
from typing import Generator, Generic, Sequence, TypeVar

import pydantic
import yaml

from ..utils import get_progress_bar
from . import exceptions
from .db import AssayData
from .models import Feature, Roi, Sample
from .processors import ProcessingPipeline, ProcessorRegistry, SampleData, SampleDataSnapshots

logger = logging.getLogger(__file__)


FeatureType = TypeVar("FeatureType", bound=Feature)
RoiType = TypeVar("RoiType", bound=Roi)

class AssayConfiguration(pydantic.BaseModel):
    """
    Store assay configuration.

    Parameters
    ----------
    path : str or Path
        Path to store the assay data. If ``None``, the assay data is stored
        on memory. If Path that does not exist is passed, create a file to store
        assay results on disk. If Path exists, loads existing data from assay.
    sample_pipeline: ProcessingPipeline
        A processing pipeline to process each sample.
    roi: str
        A registered ROI type name that will be used to represent ROIs in the
        assay data.
    feature: str
        A registered feature type name that will be used to represent features
        in the assay data.

    """

    roi: str
    feature: str
    samples: list[Sample] = list()
    path: pathlib.Path | None = None
    sample_pipeline: ProcessingPipeline


class Assay(Generic[FeatureType, RoiType]):
    """
    Manage data processing and storage of datasets.

    See HERE for a tutorial on how to work with Assay objects.

    """

    def __init__(
        self,
        roi_type: type[RoiType],
        feature_type: type[FeatureType],
        sample_pipeline: ProcessingPipeline,
        path: pathlib.Path | None = None,
    ):
        self._config = AssayConfiguration(
            roi=roi_type.__name__,
            feature=feature_type.__name__,
            path=path,
            sample_pipeline=sample_pipeline
        )
        self._data = AssayData(path, roi_type, feature_type)
        self._staging: set[str]


    @property
    def config(self):
        """Config getter."""
        return self._config

    @property
    def data(self):
        """AssayData getter."""
        return self._data

    def add_samples(self, *samples: Sample) -> None:
        """
        Add samples to the Assay staging area.

        Parameters
        ----------
        samples : list[Sample]

        """
        for sample in samples:
            if sample.id in self._staging:
                raise exceptions.SampleAlreadyInAssay(sample.id)
            self.config.samples.append(sample)

    @classmethod
    def from_yaml(cls, yaml_path: pathlib.Path) -> Assay:
        """Create an assay instance from a YAML configuration file."""
        with yaml_path.open("rt") as file:
            d = yaml.load(file, yaml.Loader)

        pipeline_ids = ["sample_pipeline"]

        for pipeline_id in pipeline_ids:
            proc_config_list: list[dict] = d.pop(pipeline_id)
            proc_list = list()
            for proc_config in proc_config_list:
                proc_type_str = proc_config.pop("class")
                proc_type = ProcessorRegistry.get_processor(proc_type_str)
                proc = proc_type(**proc_config)
                proc_list.append(proc)
            d[pipeline_id] = proc_list

        return cls(**d)


    def list_processed_samples(self) -> list[str]:
        """List all samples processed by the single sample pipeline."""
        return self.data.list_samples()

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
        return self.data.fetch_descriptors(descriptors=descriptors, sample_id=sample_id)

    def get_features(self, key, by: str, groups: list[str] | None = None) -> list[FeatureType]:
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
            features = self.data.get_features_by_sample(key)
        elif by == "label":
            features = self.data.get_features_by_label(key, groups=groups)
        elif by == "id":
            features = [self.data.get_features_by_id(key)]
        else:
            msg = f"`by` must be one of 'sample', 'label' or 'id'. Got {by}."
            raise ValueError(msg)

        return features

    def get_roi(self, key, by: str) -> Sequence[RoiType]:
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
            roi_list = self._data.get_roi_list(key)
        elif by == "id":
            roi_list = [self._data.get_roi_by_id(key)]
        else:
            msg = f"`by` must be one of 'sample' or 'id'. Got {by}."
            raise ValueError(msg)
        return roi_list

    def process_samples(
        self,
        samples: list[str] | None = None,
        max_workers: int | None = 1,
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
        valid_samples = self._staging
        if samples is not None:
            valid_samples  = valid_samples & set(samples)
        sample_list = [x for x in self.config.samples if x.id in valid_samples]


        def iterator() -> Generator[tuple[ProcessingPipeline, Sample], None, None]:
            """Provide independent pipeline instances to subprocess worker."""
            for sample in sample_list:
                pipeline = self.config.sample_pipeline.model_copy(deep=True)
                yield pipeline, sample

        def worker(pipeline: ProcessingPipeline, sample: Sample) -> SampleData | SampleDataSnapshots:
            """Apply pipeline to a sample data instance."""
            sample_data = SampleData(sample=sample)
            return pipeline.process(sample_data)

        if not silent:
            tqdm_func = get_progress_bar()
            bar = tqdm_func()
            bar.total = len(sample_list)
        else:
            bar = None

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, pipe, data) for pipe, data in iterator()]
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if bar is not None:
                    bar.set_description(f"Processing {data.sample.id}")
                    bar.update()
        self._staging = self._staging - set(valid_samples)

    def to_yaml(self, yaml_path: pathlib.Path) -> None:
        """Dump model into a YAML file."""
        exclude = {"sample_pipeline"}
        yaml_dict = self.config.model_dump(exclude=exclude)

        yaml_dict["sample_pipeline"] = self.config.sample_pipeline.get_configuration()

        with yaml_path.open("wt") as file_out:
            yaml.dump(yaml_dict, file_out)
