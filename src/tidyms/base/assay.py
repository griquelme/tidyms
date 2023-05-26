"""
Provides the Assay class, which centralizes tools to process datasets.

Manages data processing including:

- Storage of intermediate results and parameters.
- Processing from raw data to Data matrix using ProcessingPipelines.
- Parallelization of processing steps.

"""
from pathlib import Path
from multiprocessing.pool import Pool
from typing import Optional, Union, Sequence, Type
from .assay_data import AssayData
from .assay_processor import ProcessingPipeline
from .base import Feature, Roi, Sample, SampleData
from ..utils import get_progress_bar


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
        sample_pipeline: ProcessingPipeline,
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


def _process_sample_worker(args: tuple[ProcessingPipeline, SampleData]):
    pipeline, sample_data = args
    return pipeline.process(sample_data)
