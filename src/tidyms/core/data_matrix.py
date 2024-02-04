"""
Tools to work with data in a data matrix format.

DataMatrixColumn : Store feature data detected in different samples.
DataMatrix : Stores data in `tidy` format.

"""

import numpy as np
from typing import Any, Optional, Sequence

from .base import Sample
from . import constants as c

# TODO: think about using a dict of labels to columns as data matrix


class DataMatrixColumn:
    """
    Store column data from a DataMatrix associated with a feature.

    Attribute
    ---------
    data : np.ndarray
        ``(p,) floating ndarray`` with feature measurements in different samples.
    descriptors : dict[str, float]
        Feature descriptors.
    label : int
        Feature label obtained from feature correspondence.

    """

    def __init__(
        self,
        data: np.ndarray[Any, np.dtype[np.floating]],
        descriptors: dict[str, float],
        label: int,
    ):
        self.data = data
        self.descriptors = descriptors
        self.label = label

    @property
    def mz(self) -> float:
        """Get the feature m/z."""
        return self.descriptors[c.MZ]


class DataMatrix:
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

    def __init__(self, columns: list[DataMatrixColumn], samples: list[Sample]):
        n_samples = len(samples)
        for column in columns:
            if column.data.size != n_samples:
                msg = "DataMatrixColumns must have a length equal to the number of samples."
                raise ValueError(msg)
        self.samples: Sequence[Sample] = samples
        self.columns: Sequence[DataMatrixColumn] = sorted(columns, key=lambda x: x.mz)
        self._cache = dict()

    @property
    def X(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Create a 2D array with the matrix data."""
        X = self._cache.get("X")
        if X is None:
            X = np.vstack([x.data for x in self.columns]).T
            self._cache["X"] = X
        return X


def create_data_matrix_from_feature_table(
    descriptors: dict[str, list],
    samples: list[Sample],
    value_descriptor: str,
    aggregate: Optional[list[str]],
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
            col = DataMatrixColumn(column_values[k], k_descriptors, k)
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
