"""
Tools to work with data in a data matrix format.

DataMatrixColumn : Store feature data detected in different samples.
DataMatrix : Stores data in `tidy` format.

"""

from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
import pydantic

from .models import BaseFeatureGroup, Sample
from .numpy import FloatArray1D, FloatingArray

FeatureGroupType = TypeVar("FeatureGroupType", bound=BaseFeatureGroup)


class FeatureVector(pydantic.BaseModel, Generic[FeatureGroupType]):
    """Data matrix column view."""

    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data: FloatArray1D = pydantic.Field(repr=False)
    """The column values."""

    feature: FeatureGroupType
    """The feature information associated with the matrix column."""


class SampleVector(pydantic.BaseModel):
    """Data matrix row view."""

    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data: FloatArray1D = pydantic.Field(repr=False)
    """The column values."""

    sample: Sample
    """The sample information associated with the matrix row."""


class DataMatrix(Generic[FeatureGroupType]):
    """
    The Data Matrix class.

    Stores data from features across samples.

    Attributes
    ----------
    samples : Sequence[Sample]
        Samples associated with each row in the data matrix.
    features : Sequence[FeatureVector]
        Feature data.

    """

    # TODO: method to search columns by descriptors.
    # TODO: get column by label.
    # TODO: DataContainer functionality

    def __init__(self, samples: list[Sample], features: list[FeatureVector[FeatureGroupType]]):
        n_samples = len(samples)
        n_features = len(features)

        for ft in features:
            if ft.data.size != n_samples:
                id_ = ft.feature.annotation.id
                msg = f"Data matrix feature vector {id_} length does not match the number of samples."
                raise ValueError(msg)

        # store samples and features metadata
        self._samples = samples
        self._features = [x.feature for x in features]
        self._X = np.zeros((n_features, n_samples))

        # store mappings for samples and features for fast retrieval
        self._sample_id_to_sample = {x.id: k for k, x in enumerate(samples)}
        self._feature_id_to_sample = {x.annotation.id: k for k, x in enumerate(self._features)}

        for k, ft in enumerate(features):
            self._X[k] = ft.data

    def get_n_features(self) -> int:
        """Return the number of features in the matrix."""
        return len(self._features)

    def get_n_samples(self) -> int:
        """Return the number of samples in the matrix."""
        return len(self._samples)

    def get_matrix(self) -> FloatingArray:
        """Return a copy of the data matrix as a 2D numpy array."""
        return self._X.copy()

    def sort_features(self, key=Callable[[FeatureGroupType], float]) -> None:
        """Sort features in the data matrix."""

        def default_sort_comparison(x: FeatureGroupType) -> float:
            return x.mz

        sort_func = default_sort_comparison if key is None else key

        sorted_index = sorted(range(self.get_n_features()), key=lambda x: sort_func(self._features[x]))

        self._features = [self._features[x] for x in sorted_index]
        self._X = self._X[sorted_index]
        self._feature_id_to_sample = {x.annotation.id: k for k, x in enumerate(self._features)}

    @property
    def X(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Matrix getter."""
        return self._X

    @property
    def features(self) -> list[FeatureGroupType]:
        """Feature group getter."""
        return self._features

    @property
    def samples(self) -> list[Sample]:
        """Samples getter."""
        return self._samples


def create_data_matrix_from_feature_table(
    descriptors: dict[str, list], samples: list[Sample], value_descriptor: str, aggregate: list[str] | None
):
    """
    Create a DataMatrix using feature descriptors.

    Parameters
    ----------
    descriptors : dict[str, list]
        Feature descriptors obtained from AssayData.get_descriptors.
    samples : list[Sample]
        Samples used in the assay.
    value_descriptor : str
        Descriptor used as value for the data matrix.
    aggregate : list[str] or None, default=None
        Descriptors aggregated for each DataMatrixColumn. If ``None``, computes
        all available descriptors.

    Returns
    -------
    DataMatrix

    """
    samples = sorted(samples, key=lambda x: x.order)
    sample_id: Sequence[str] = descriptors["sample_id"]
    labels: Sequence[int] = descriptors["label"]  # TODO: fix after removing container module
    values: Sequence[float] = descriptors[value_descriptor]

    if aggregate is None:
        ignore = {"sample_id", "label", "roi_id", "feature_id"}
        aggregate = [x for x in descriptors if x not in ignore]

    column_values = _create_column_values(values, sample_id, labels, samples)
    column_descriptors = _create_column_descriptors(descriptors, aggregate)

    columns = list()
    for k, k_descriptors in column_descriptors.items():
        if k in column_values:
            col = FeatureVector(column_values[k], k_descriptors, k)
            columns.append(col)

    return DataMatrix(columns, samples)


def _create_column_values(
    values: list[float], sample_id: list[str], labels: list[int], samples: list[Sample]
) -> dict[int, np.ndarray[Any, np.dtype[np.floating]]]:
    """Compute DataMatrix values from a given descriptor."""
    sample_to_index = {sample.id: k for k, sample in enumerate(samples)}

    # create a 2D array with data
    # n_features is computed using the max value as some values may be missing
    # (eg: ..., 5000, 5001, 5003, 5004, ...)
    n_samples = len(samples)
    n_features = max(labels) + 1
    X = np.empty((n_features, n_samples))
    X[:] = np.nan

    for sample_id, row_index, value in zip(sample_id, labels, values):
        if row_index > -1:  # ignore noise
            col_index = sample_to_index[sample_id]
            X[row_index, col_index] = value

    columns = {k: row for k, row in enumerate(list(X))}

    return columns


def _create_column_descriptors(
    descriptors: dict[str, list], aggregate_list: Sequence[str]
) -> dict[int, dict[str, float]]:
    """Compute aggregated values for each selected descriptor."""
    labels: Sequence[int] = descriptors["labels"]
    unique_labels, inverse_index = np.unique(labels, return_inverse=True)
    grouped_index = dict()  # contains a list with indices of features for each label group
    for i in inverse_index:
        label = unique_labels[i]
        label_index = grouped_index.setdefault(label, list())
        label_index.append(i)

    column_descriptors = dict()

    for d in aggregate_list:
        values: np.ndarray[Any, np.dtype[np.floating]] = np.array(descriptors[d])
        for k in unique_labels:
            if k > -1:
                k_col_dict = column_descriptors.setdefault(k, dict())
                k_index = grouped_index[k]
                k_col_dict[d] = np.mean(values[k_index])
    return column_descriptors
