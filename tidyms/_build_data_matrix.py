"""
Functions used to create data matrix and feature metadata DataFrames from a
feature table.

"""

import pandas as pd
from numpy import nan
from . import _constants as c
from .consensus_annotation import vote_annotations
from typing import Dict, List, Optional, Set, Tuple


def build_data_matrix(
    feature_table: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    separation: str,
    merge_close_features: bool,
    mz_merge: Optional[float],
    rt_merge: Optional[float],
    merge_threshold: Optional[float],
    annotate_isotopologues: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a data matrix and feature metadata with aggregated descriptors from
    a feature table.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Table with detected and matched features.
    sample_metadata : DataFrame
        Metadata of analyzed samples
    separation : str,
        Separation method used for the data
    merge_close_features : bool
        If ``True`` finds close features and merge them into a single feature.
        The code of the merged features is in the `merged` column of the
        feature metadata. The area in the data matrix is the sum of the merged
        features.
    mz_merge : float or None, default=None
        Merge features only if their mean m/z, as described by the feature
        metadata, are closer than this values.
    rt_merge : float or None, default=None
        Merge features only if their mean Rt, as described by the feature
        metadata, are closer than this values.
    merge_threshold : float or None, default=None
        Number between 0.0 and 1.0. This value is compared against the quotient
        between the number of samples where both features where detected and the
        number of samples where any of the features was detected. If this
        quotient is lower than the threshold, the pair of features is merged
        into a single one.
    annotate_isotopologues : bool
        Include isotopologue annotation in feature metadata.

    Returns
    -------
    data_matrix : pd.DataFrame
    feature_metadata : pd.DataFrame

    """
    # remove noise
    feature_table = feature_table[feature_table[c.LABEL] > -1].copy()
    labels = feature_table[c.LABEL]
    cluster_to_ft = _cluster_to_feature_name(labels)

    data_matrix = _build_data_matrix(feature_table, sample_metadata, cluster_to_ft)

    if separation in c.LC_MODES:
        feature_metadata = _build_lc_feature_metadata(feature_table, cluster_to_ft)
    else:
        raise NotImplementedError

    if merge_close_features:
        if separation in c.LC_MODES:
            merged_dict = _lc_merge_close_features(
                data_matrix, feature_metadata, mz_merge, rt_merge, merge_threshold
            )
        else:
            raise NotImplementedError
        _update_feature_table_merged_labels(feature_table, merged_dict)

    if annotate_isotopologues:
        feature_metadata = _add_annotation_data(feature_table, feature_metadata)

    return data_matrix, feature_metadata


def _build_lc_feature_metadata(
    feature_table: pd.DataFrame, cluster_to_ft_name: Dict[int, str]
) -> pd.DataFrame:
    """
    Computes aggregated descriptors for Features in an LC feature table.

    Parameters
    ----------
    feature_table : DataFrame
        A feature table with noisy features removed.
    cluster_to_ft_name : dict
        Mapping from cluster to feature name.

    Returns
    -------
    feature_metadata : DataFrame

    """
    # compute aggregate statistics for each feature
    estimators = {
        "mz": ["mean", "std", "min", "max"],
        "rt": ["mean", "std", "min", "max"],
        "rt start": ["mean"],
        "rt end": ["mean"],
    }
    feature_metadata: pd.DataFrame = feature_table.groupby(c.LABEL).agg(estimators)

    # flatten MultiIndex column
    columns = feature_metadata.columns  # type: pd.MultiIndex
    level_0 = columns.get_level_values(0)
    level_1 = columns.get_level_values(1)
    col_name_map = {
        "mzmean": "mz",
        "mzstd": "mz std",
        "mzmin": "mz min",
        "mzmax": "mz max",
        "rtmean": "rt",
        "rtstd": "rt std",
        "rtmin": "rt min",
        "rtmax": "rt max",
        "rt startmean": "rt start",
        "rt endmean": "rt end",
    }
    new_columns = [col_name_map[x + y] for x, y in zip(level_0, level_1)]
    feature_metadata.columns = new_columns

    # rename features
    feature_metadata.index = feature_metadata.index.map(cluster_to_ft_name)
    feature_metadata.index.name = c.FEATURE
    return feature_metadata


def _build_data_matrix(
    feature_table: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    cluster_to_ft_name: Dict[int, str],
) -> pd.DataFrame:
    """
    Creates a data matrix from a feature table.

    Parameters
    ----------
    feature_table : DataFrame
        feature table without noisy features
    sample_metadata : DataFrame
        metadata of samples
    cluster_to_ft_name : dict
        Mapping from cluster to feature name.

    Returns
    -------
    data_matrix: DataFrame

    """
    data_matrix = feature_table.pivot(index=c.SAMPLE, columns=c.LABEL, values=c.AREA)
    data_matrix.columns = data_matrix.columns.map(cluster_to_ft_name)

    # add samples without features as rows of missing values
    missing_index = sample_metadata.index.difference(data_matrix.index)
    missing = pd.DataFrame(data=nan, index=missing_index, columns=data_matrix.columns)
    data_matrix = pd.concat((data_matrix, missing))
    # sort data_matrix using sample metadata indices
    data_matrix = data_matrix.loc[sample_metadata.index, :]
    data_matrix.columns.name = c.FEATURE
    data_matrix.index.name = c.SAMPLE

    return data_matrix


def _cluster_to_feature_name(labels: pd.Series) -> Dict[int, str]:
    """
    Creates a mapping of cluster labels to a string of feature names.

    Parameters
    ----------
    labels: Series
        Feature labels obtained after feature correspondence

    Returns
    -------

    """
    # feature names
    unique_cluster = labels.unique()
    n_cluster = unique_cluster[unique_cluster > -1].size
    max_n_chars_cluster = len(str(n_cluster))
    template = "FT-{{:0{}d}}".format(max_n_chars_cluster)
    cluster_to_ft = {k: template.format(k) for k in range(n_cluster)}
    return cluster_to_ft


# def _feature_name_to_cluster(ft: str) -> int:
#     return int(ft.split("-")[-1])


def _lc_merge_close_features(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    mz_merge: float,
    rt_merge: float,
    merge_threshold: float,
) -> Dict[str, List[str]]:
    rt = feature_metadata[c.RT]
    mz = feature_metadata[c.MZ]
    features = feature_metadata.index
    feature_set = set(features)
    merged_dict = dict()
    while len(feature_set):
        ft = feature_set.pop()
        candidates = _find_merge_candidates(
            ft, mz, rt, features, feature_set, mz_merge, rt_merge
        )
        for ft2 in candidates:
            merged_ft = _merge_features(
                data_matrix, merged_dict, feature_set, ft, ft2, merge_threshold
            )
            if merged_ft != ft:
                break
    _add_merged_features_to_feature_metadata(merged_dict, feature_metadata)
    _remove_merged_features(merged_dict, data_matrix, feature_metadata)
    return merged_dict


def _find_merge_candidates(
    ft: str,
    mz: pd.Series,
    rt: pd.Series,
    features: pd.Index,
    feature_set: Set[str],
    mz_merge: float,
    rt_merge: float,
) -> pd.Index:
    ft_mz = mz[ft]
    ft_rt = rt[ft]
    merge_candidates_mask = ((rt - ft_rt).abs() < rt_merge) & (
        (mz - ft_mz).abs() < mz_merge
    )
    merge_candidates_mask[ft] = False
    return features[merge_candidates_mask].intersection(feature_set)


def _merge_features(
    data_matrix: pd.DataFrame,
    merged_dict: Dict[str, List[str]],
    feature_set: Set[str],
    ft1: str,
    ft2: str,
    merge_threshold: float,
) -> Optional[str]:
    x1 = data_matrix[ft1]
    x2 = data_matrix[ft2]
    x1_detected = x1.notnull()
    x2_detected = x2.notnull()
    any_detected = x1_detected | x2_detected
    any_detected_count = any_detected.sum()
    both_detected_count = (x1_detected & x2_detected).sum()
    both_detected_fraction = both_detected_count / any_detected_count
    if both_detected_fraction < merge_threshold:
        # choose which feature to keep and which to merge
        if x1_detected.sum() >= x2_detected.sum():
            main_ft = ft1
            merged_ft = ft2
        else:
            main_ft = ft2
            merged_ft = ft1

        # update values in the data matrix
        data_merged = x1.add(x2, fill_value=0)
        data_merged[~any_detected] = nan
        data_matrix[main_ft] = data_merged

        # store merged features into merged dictionary
        main_ft_merged_list = merged_dict.setdefault(main_ft, list())
        main_ft_merged_list.append(merged_ft)
        if merged_ft in merged_dict:
            ft_merged_list = merged_dict.pop(merged_ft)
            main_ft_merged_list.extend(ft_merged_list)

        if merged_ft in feature_set:
            feature_set.remove(merged_ft)
    else:
        merged_ft = None
    return merged_ft


def _add_merged_features_to_feature_metadata(
    merged_dict: Dict[str, List[str]], feature_metadata: pd.DataFrame
):
    # create a Series with of comma-separated feature names
    s = dict()
    for k, v in merged_dict.items():
        merged_str = ""
        for ft in v:
            if merged_str:
                merged_str += ", "
            merged_str += ft
        s[k] = merged_str
    s = pd.Series(s, dtype=str)
    feature_metadata[c.MERGED] = ""
    feature_metadata.loc[s.index, c.MERGED] = s


def _remove_merged_features(
    merged_dict: Dict[str, List[str]],
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
):
    rm_features = list()
    for v in merged_dict.values():
        rm_features.extend(v)
    data_matrix.drop(columns=rm_features, inplace=True)
    feature_metadata.drop(index=rm_features, inplace=True)


def _update_feature_table_merged_labels(
    feature_table: pd.DataFrame, merged_dict: Dict[str, List[str]]
):
    new_labels = dict()
    for ft, merged in merged_dict.items():
        v = int(ft.split("-")[-1])
        for m in merged:
            k = int(m.split("-")[-1])
            new_labels[k] = v

    new_labels = {c.LABEL: new_labels}
    feature_table.replace(new_labels, inplace=True)


def _add_annotation_data(
    feature_table: pd.DataFrame, feature_metadata: pd.DataFrame
) -> pd.DataFrame:
    _, annotations = vote_annotations(feature_table)
    annotation_df = pd.DataFrame(annotations).T.sort_index()
    n_ft = feature_metadata.shape[0]
    length = len(str(n_ft))
    template = "FT-{{:0{}d}}".format(length)
    annotation_df.index = [template.format(x) for x in annotation_df.index]
    annotation_df = annotation_df.reindex(feature_metadata.index, fill_value=-1)
    feature_metadata = pd.concat((feature_metadata, annotation_df), axis=1)
    return feature_metadata