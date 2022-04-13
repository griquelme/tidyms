import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, Generator, List, Optional, Tuple
from functools import partial
from .utils import get_progress_bar


def feature_correspondence(
        feature_data: pd.DataFrame,
        samples_per_group: dict,
        include_groups: Optional[List[int]],
        mz_tolerance: float,
        rt_tolerance: float,
        min_fraction: float,
        max_deviation: float,
        n_jobs: Optional[int] = None,
        verbose: bool = False
):
    """
    Match features across samples using a combination of clustering algorithms.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature table obtained after feature detection.
    samples_per_group : dict
        Maps a group name to number of samples in the group.
    include_groups : List[int] or None, default=None
        Sample groups used to estimate the minimum cluster size and number of
        chemical species in a cluster.
    mz_tolerance : float
        m/z tolerance used to set the `eps` parameter in the DBSCAN algorithm.
    rt_tolerance : float
        Rt tolerance used to set the `eps` parameter in the DBSCAN algorithm.
    min_fraction : float
        Minimum fraction of samples of a given group in a cluster.
    max_deviation : float
        The maximum deviation of a feature from a cluster.
    n_jobs: int or None, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    verbose : bool
        If True, shows a progress bar

    Returns
    -------

    """
    X = feature_data.loc[:, ["mz", "rt"]].to_numpy()
    samples = feature_data["sample_"].to_numpy()
    # scale rt to use the same tolerance in both dimensions
    X[:, 1] *= mz_tolerance / rt_tolerance

    # min_samples estimation using the number of samples
    if include_groups is None:
        min_samples = round(sum(samples_per_group.values()) * min_fraction)
        include_groups = list(samples_per_group.keys())
    else:
        min_samples = _get_min_sample(
            samples_per_group, include_groups, min_fraction
        )

    # DBSCAN clustering
    max_size = 100000
    cluster = _make_initial_cluster(X, mz_tolerance, min_samples, max_size)
    feature_data["cluster_"] = cluster

    # estimate the number of chemical species per cluster
    features_per_cluster = _estimate_n_species(
        feature_data.loc[:, ["sample_", "cluster_", "class_"]],
        min_fraction,
        samples_per_group,
        include_groups,
    )

    cluster_iterator = _cluster_iterator(
        X, cluster, samples, features_per_cluster
    )

    if verbose:
        progress_bar = get_progress_bar()
        total = _get_progress_bar_total(features_per_cluster)
        cluster_iterator = progress_bar(cluster_iterator, total=total)

    func = partial(_split_cluster_worker, max_deviation=max_deviation)
    func = delayed(func)
    data = Parallel(n_jobs=n_jobs)(func(x) for x in cluster_iterator)
    refined_cluster = _build_label(data, feature_data.shape[0])
    feature_data["cluster_"] = refined_cluster


def _get_min_sample(
    samples_per_group: Dict[int, int],
    include_groups: List[int],
    min_fraction: float
) -> int:
    min_sample = np.inf
    for k, v in samples_per_group.items():
        if k in include_groups:
            tmp = round(v * min_fraction)
            min_sample = min(tmp, min_sample)
    return min_sample


def _make_initial_cluster(
        X: np.ndarray,
        eps: float,
        min_samples: int,
        max_size: int
) -> np.ndarray:
    """
    First estimation of matched features using the DBSCAN algorithm.
    Data is split to reduce memory usage.

    Auxiliary function to feature_correspondence.

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
        df: pd.DataFrame,
        min_dr: float,
        sample_per_group: Dict[int, int],
        include_groups: Optional[List[int]]
) -> Dict[int, int]:

    # gets a dataframe with four columns: group, sample, cluster and number
    # of features in the cluster.
    n_ft_cluster = (
        df.groupby(["class_"])
        .value_counts()
        .reset_index()
    )

    # computes maximum number of species where the detection rate is greater
    # than min_dr
    n_species_per_cluster = (
        n_ft_cluster.groupby("class_")
        .apply(
            _get_n_species_per_group,
            min_dr,
            sample_per_group,
            include_groups
        )
    )
    if isinstance(n_species_per_cluster.index, pd.MultiIndex):
        # unstack MultiIndex into a DataFrame and get the highest value
        # estimation of the number of species for each cluster
        n_species_per_cluster = n_species_per_cluster.unstack(-1).max()
    elif isinstance(n_species_per_cluster, pd.DataFrame):
        # If multiple groups are used a DataFrame is obtained
        n_species_per_cluster = n_species_per_cluster.max()
    else:
        # DataFrame with only one row is converted to a Series
        n_species_per_cluster = n_species_per_cluster.iloc[0]

    n_species_per_cluster = n_species_per_cluster.astype(int).to_dict()
    return n_species_per_cluster


def _get_n_species_per_group(
        df: pd.DataFrame,
        min_dr: float,
        samples_per_group: Dict[int, int],
        include_groups: List[int]
) -> pd.DataFrame:
    if df.name in include_groups:
        n_samples = samples_per_group[df.name]
        res = (
            # pivot table where the values are the number of features
            # contributed by a sample
            df.pivot(index="sample_", columns="cluster_", values=0)
            .fillna(0)
            # count how many times a given number of features was obtained
            .apply(lambda x: x.value_counts())
            .div(n_samples)  # convert values to detection rates
            .fillna(0)
            .iloc[::-1]
            .cumsum()
            .iloc[::-1]
            # these three previous steps are a trick to pass detection rates
            # values from higher to lower values.
            .ge(min_dr)     # find values where above the min_dr
            .apply(lambda x: x.iloc[::-1].idxmax() if x.any() else 0)
            # get the maximum number of species above the min_dr
        )
    else:
        c = df.cluster_.unique()
        res = pd.Series(data=np.zeros(c.size), index=c)
    return res


def _cluster_iterator(
    X: np.ndarray,
    cluster: np.ndarray,
    samples: np.ndarray,
    features_per_cluster: Dict[int, int]
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    n_cluster = cluster.max() + 1
    for c in range(n_cluster):
        n_ft = features_per_cluster[c]
        if n_ft > 0:
            index = np.where(cluster == c)[0]
            Xc = X[index, :]
            samples_c = samples[index]
            yield Xc, samples_c, n_ft, index


def _split_cluster(
    X: np.ndarray,
    samples: np.ndarray,
    n_species: int,
    max_deviation: float
) -> np.ndarray:
    """
    Process each cluster obtained from DBSCAN. Auxiliary function to
    `feature_correspondence`.

    Parameters
    ----------
    X : array
    samples : array
    max_deviation : float
    n_species: int
        Number of features in the cluster, estimated with
        `estimate_features_per_cluster`.

    Returns
    -------
    label : np.ndarray
    """

    # fit GMM
    gmm = GaussianMixture(n_components=n_species, covariance_type="diag")
    gmm.fit(X)

    # compute the deviation of the features respect to each cluster
    deviation = _get_deviation(X, gmm.covariances_, gmm.means_)

    # assign each feature in a sample to component in the GMM minimizing the
    # total deviation in the sample.
    label = - np.ones_like(samples)
    unique_samples = np.unique(samples)
    for s in unique_samples:
        sample_mask = samples == s
        sample_deviation = deviation[sample_mask, :]
        # Find best option for each feature
        best_row, best_col = linear_sum_assignment(sample_deviation)

        # features with deviation greater than max_deviation are set to noise
        min_proba_mask = sample_deviation[best_row, best_col] <= max_deviation
        best_col = best_col[min_proba_mask]
        best_row = best_row[min_proba_mask]
        sample_label = - np.ones(sample_deviation.shape[0], dtype=int)
        sample_label[best_row] = best_col
        label[sample_mask] = sample_label
    return label


def _get_deviation(
    X: np.ndarray,
    covariances_: np.ndarray,
    means_: np.ndarray,
) -> np.ndarray:
    """
    Compute the deviation of features.
    
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
    Xc, samples_c, n_ft, index = args
    label_c = _split_cluster(Xc, samples_c, n_ft, max_deviation)
    return label_c, index, n_ft


def _build_label(data, size):
    label = -1 * np.ones(size, dtype=int)
    cluster_count = 0
    for c_label, c_index, c_n_ft in data:
        c_label[c_label > -1] += cluster_count
        label[c_index] = c_label
        cluster_count += c_n_ft
    return label


def _get_progress_bar_total(ft_per_cluster: Dict[int, int]) -> int:
    total = 0
    for k, v in ft_per_cluster.items():
        if (k > -1) and (v > 0):
            total += 1
    return total
