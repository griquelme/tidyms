"""
Functions used to match features

"""

import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, Generator, List, Optional, Tuple
from .utils import get_progress_bar
from . import _constants as c


def match_features(
        feature_table: pd.DataFrame,
        samples_per_class: dict,
        include_classes: Optional[List[int]],
        mz_tolerance: float,
        rt_tolerance: float,
        min_fraction: float,
        max_deviation: float,
        n_jobs: Optional[int] = None,
        verbose: bool = False
):
    r"""
    Match features across samples using DBSCAN and GMM.See the
    :ref:`user guide <ft-correspondence>` for a detailed description of the
    algorithm.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table obtained after feature detection.
    samples_per_class : dict
        Maps a class name to the number of samples in the class.
    include_classes : List or None, default=None
        Sample classes used to estimate the minimum cluster size and number of
        chemical species in a cluster.
    mz_tolerance : float
        m/z tolerance used to group close features. Sets the `eps` parameter in
        the DBSCAN algorithm.
    rt_tolerance : float
        Rt tolerance used to group close features. Sets the `eps` parameter in
        the DBSCAN algorithm.
    min_fraction : float
        Minimum fraction of samples of a given group in a cluster. If
        `include_classes` is ``None``, the total number of sample is used
        to compute the minimum fraction.
    max_deviation : float
        The maximum deviation of a feature from a cluster, measured in numbers
        of standard deviations from the cluster.
    n_jobs: int or None, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    verbose : bool
        If True, shows progress bar

    Returns
    -------
    results: dictionary
        `clusters_` Contains the results from the feature mathing, where each
        number is a different ionic species. Features labelled with `-1` are
        considered noise. `indecisiveness` is a metric that counts the fraction
        of features in a cluster that could be potentially assigned to more
        than one cluster. Values close to zero indicate higher quality grouping.

    """
    X = feature_table.loc[:, [c.MZ, c.RT]].to_numpy()
    samples = feature_table[c.SAMPLE].to_numpy()
    classes = feature_table[c.CLASS].to_numpy()
    # scale rt to use the same tolerance in both dimensions
    X[:, 1] *= mz_tolerance / rt_tolerance

    min_samples = _get_min_sample(
        samples_per_class, include_classes, min_fraction
    )
    if include_classes is None:
        include_classes = list(samples_per_class.keys())

    # DBSCAN clustering
    max_size = 100000
    cluster = _cluster_dbscan(X, mz_tolerance, min_samples, max_size)

    species_per_cluster = _estimate_n_species(
        samples,
        cluster,
        classes,
        samples_per_class,
        include_classes,
        min_fraction
    )

    cluster_iterator = _cluster_iterator(
        X, cluster, samples, species_per_cluster
    )

    if verbose:
        progress_bar = get_progress_bar()
        total = _get_progress_bar_total(species_per_cluster)
        cluster_iterator = progress_bar(cluster_iterator, total=total)

    func = partial(_split_cluster_worker, max_deviation=max_deviation)
    func = delayed(func)
    data = Parallel(n_jobs=n_jobs)(func(x) for x in cluster_iterator)
    refined_cluster, score = _build_label(data, feature_table.shape[0])
    results = {
        c.LABEL: refined_cluster,
        "indecisiveness": score
    }
    return results


def _get_min_sample(
    samples_per_class: Dict[int, int],
    include_classes: Optional[List[int]],
    min_fraction: float
) -> int:
    """
    Computes the `min_sample` parameter to use in the DBSCAN model.

    Auxiliary function to feature_correspondence

    Parameters
    ----------
    samples_per_class : Dict[int, int]
    include_classes : List[int] or None
    min_fraction : number between 0 and 1

    Returns
    -------
    min_sample : int

    """
    if include_classes is None:
        min_samples = round(sum(samples_per_class.values()) * min_fraction)
    else:
        min_samples = np.inf
        for k, v in samples_per_class.items():
            if k in include_classes:
                tmp = round(v * min_fraction)
                min_samples = min(tmp, min_samples)
    return min_samples


def _cluster_dbscan(
        X: np.ndarray,
        eps: float,
        min_samples: int,
        max_size: int
) -> np.ndarray:
    """
    Clusterize rows of X using the DBSCAN algorithm. X is split into chunks to
    reduce memory usage. The split is done in a way such that the solution
    obtained is the same as the solution using X.

    Auxiliary function to match_features.

    Parameters
    ----------
    X : array
        m/z and rt values for each feature
    eps : float
        Used to build epsilon parameter of DBSCAN
    min_samples : int
        parameter to pass to DBSCAN
    max_size : int
        maximum number of rows in X. If the number of rows is greater than
        this value, the data is processed in chunks to reduce memory usage.
    Returns
    -------
    cluster : Series
        The assigned cluster by DBSCAN

    """
    n_rows = X.shape[0]

    if n_rows > max_size:
        # sort X based on the values of the columns and find positions to
        # split into smaller chunks of data.
        sorted_index = np.lexsort(tuple(X.T))
        revert_sort_ind = np.arange(X.shape[0])[sorted_index]
        X = X[sorted_index]

        # indices to split X based on max_size
        split_index = np.arange(max_size, n_rows, max_size)

        # find split indices candidates
        # it can be shown that if X is split at one of these points, the
        # points in each one of the chunks are not connected with points in
        # another chunk
        min_diff_x = np.min(np.diff(X.T), axis=0)
        split_candidates = np.where(min_diff_x > eps)[0]
        close_index = np.searchsorted(split_candidates, split_index)

        close_index[-1] = min(split_candidates.size - 1, close_index[-1])
        split_index = split_candidates[close_index] + 1
        split_index = np.hstack((0, split_index, n_rows))
    else:
        split_index = np.array([0, n_rows])

    # clusterize using DBSCAN on each chunk
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    n_chunks = split_index.size - 1
    cluster = np.zeros(X.shape[0], dtype=int)
    cluster_counter = 0
    for k in range(n_chunks):
        start = split_index[k]
        end = split_index[k + 1]
        dbscan.fit(X[start:end, :])
        labels = dbscan.labels_
        n_cluster = (np.unique(labels) >= 0).sum()
        labels[labels >= 0] += cluster_counter
        cluster_counter += n_cluster
        cluster[start:end] = labels

    # revert sort on cluster and X
    if n_rows > max_size:
        cluster = cluster[revert_sort_ind]
    return cluster


def _estimate_n_species(
        samples: np.ndarray,
        clusters: np.ndarray,
        classes: np.ndarray,
        samples_per_class: Dict[int, int],
        include_classes: List[int],
        min_dr: float
) -> Dict[int, int]:
    """
    Estimate the number of species in a cluster.

    Auxiliary function to match_features.

    Returns
    -------
    n_species_per_cluster : dict
        A mapping from cluster label to the number of species estimated.

    """
    n_clusters = clusters.max() + 1
    n_class = len(include_classes)
    species_array = np.zeros((n_class, n_clusters), dtype=int)
    # estimate the number of species in a cluster according to each class
    for k, cl in enumerate(include_classes):
        n_samples = samples_per_class[cl]
        n_min = round(n_samples * min_dr)
        c_mask = classes == cl
        c_samples = samples[c_mask]
        c_clusters = clusters[c_mask]
        c_species = _estimate_n_species_one_class(
            c_samples, c_clusters, n_min, n_clusters)
        species_array[k, :] = c_species
    # keep the estimation with the highest number of species
    species = species_array.max(axis=0)
    n_species_per_cluster = dict(zip(np.arange(n_clusters), species))
    return n_species_per_cluster


def _estimate_n_species_one_class(
        samples: np.ndarray,
        clusters: np.ndarray,
        min_samples: int,
        n_clusters: int
) -> np.ndarray:
    """
    Estimates the number of species in a cluster. Assumes only one class.

    Auxiliary function to _estimate_n_species.

    """
    species = np.zeros(n_clusters, dtype=int)
    for cl in range(n_clusters):
        c_mask = clusters == cl
        c_samples = samples[c_mask]
        # count features per sample in a cluster
        s_unique, s_counts = np.unique(c_samples, return_counts=True)
        # count the number of times a sample has k features
        k_unique, k_counts = np.unique(s_counts, return_counts=True)
        k_mask = k_counts >= min_samples
        k_unique = k_unique[k_mask]
        if k_unique.size:
            species[cl] = k_unique.max()
    return species


def _cluster_iterator(
    X: np.ndarray,
    cluster: np.ndarray,
    samples: np.ndarray,
    species_per_cluster: Dict[int, int]
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields the rows of X associated with a cluster.

    Auxiliary function to match_features.

    Yields
    ------
    X_c : array
        Rows of X associated to a cluster.
    samples_c : array
        Sample labels associated to X_c.
    n_species : int
        Number of species estimated for the cluster.
    index : indices of the rows of `X_c` in `X`

    """
    n_cluster = cluster.max() + 1
    for cl in range(n_cluster):
        n_species = species_per_cluster[cl]
        if n_species > 0:
            index = np.where(cluster == cl)[0]
            X_c = X[index, :]
            samples_c = samples[index]
            yield X_c, samples_c, n_species, index


def _process_cluster(
    X_c: np.ndarray,
    samples_c: np.ndarray,
    n_species: int,
    max_deviation: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process each cluster using GMM.

    Auxiliary function to `match_features`.

    Parameters
    ----------
    X_c : array
    samples_c : array
    max_deviation : float
    n_species: int
        Number of features in the cluster, estimated with
        `estimate_features_per_cluster`.

    Returns
    -------
    label : np.ndarray
    indecisiveness : np.ndarray
    """

    # fit GMM
    gmm = GaussianMixture(n_components=n_species, covariance_type="diag")
    gmm.fit(X_c)

    # compute the deviation of the features respect to each cluster
    deviation = _get_deviation(X_c, gmm.covariances_, gmm.means_)

    # assign each feature in a sample to component in the GMM minimizing the
    # total deviation in the sample.
    label = - np.ones_like(samples_c)   # by default features are set as noise
    unique_samples = np.unique(samples_c)
    # the indecisiveness is a metric that counts the number of samples
    # where more than one feature can be potentially assigned to a species
    # that is, for each species, the number of rows in deviation with values
    # lower than max_deviation are counted as 1 and zero otherwise.
    # This is done for all samples and the indecisiveness is divided by the
    # number of samples.
    indecisiveness = np.zeros(n_species)
    for s in unique_samples:
        sample_mask = samples_c == s
        sample_deviation = deviation[sample_mask, :]
        # indecisiveness
        count = (sample_deviation < max_deviation).sum(axis=0)
        indecisiveness += (count > 1)

        # Find the best option for each feature
        best_row, best_col = linear_sum_assignment(sample_deviation)

        # features with deviation greater than max_deviation are set to noise
        valid_ft_mask = sample_deviation[best_row, best_col] <= max_deviation
        best_col = best_col[valid_ft_mask]
        best_row = best_row[valid_ft_mask]
        sample_label = - np.ones(sample_deviation.shape[0], dtype=int)
        sample_label[best_row] = best_col
        label[sample_mask] = sample_label
    indecisiveness /= unique_samples.size
    return label, indecisiveness


def _get_deviation(
    X: np.ndarray,
    covariances_: np.ndarray,
    means_: np.ndarray,
) -> np.ndarray:
    """
    Compute the deviation of features.

    Auxiliary function to _process_cluster
    
    Parameters
    ----------
    X: array
    covariances_ : array
        output from the GMM fit
    means_ : array
        output from the GMM fit

    Returns
    -------
    deviation : array

    """
    n_species = covariances_.shape[0]
    n_ft = X.shape[0]
    deviation = np.zeros((n_ft, n_species))
    for k, (m, s) in enumerate(zip(means_, covariances_)):
        # the deviation is the absolute value of X after standardization
        Xs = np.abs((X - m) / np.sqrt(s))
        deviation[:, k] = Xs.max(axis=1)
    return deviation


def _split_cluster_worker(args, max_deviation):
    """
    Worker used to parallelize feature clustering.

    """
    Xc, samples_c, n_ft, index = args
    label_c, score = _process_cluster(Xc, samples_c, n_ft, max_deviation)
    return label_c, score, index, n_ft


def _build_label(data, size):
    """
    Merge the data obtained from each cluster.

    Auxiliary function to match_features.

    """
    label = -1 * np.ones(size, dtype=int)
    cluster_count = 0
    score_list = list()
    for c_label, c_score, c_index, c_n_ft in data:
        c_label[c_label > -1] += cluster_count
        label[c_index] = c_label
        cluster_count += c_n_ft
        score_list.append(c_score)
    score_list = np.hstack(score_list)
    return label, score_list


def _get_progress_bar_total(ft_per_cluster: Dict[int, int]) -> int:
    total = 0
    for k, v in ft_per_cluster.items():
        if (k > -1) and (v > 0):
            total += 1
    return total
