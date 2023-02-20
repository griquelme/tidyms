import numpy as np
import pandas as pd
from tidyms import _build_data_matrix
from tidyms import _constants as c
import pytest


def create_dummy_sample_metadata(n_samples):
    samples = ["Sample{}".format(x + 1) for x in range(n_samples)]
    # dummy sample metadata, we only care about the index
    return pd.DataFrame(data=1, index=samples, columns=["class", "id"])


def create_dummy_feature_table(n_ft, n_samples):
    n_rows = 1000
    columns = [c.AREA, c.HEIGHT, c.MZ, c.RT, c.RT_START, c.RT_END]
    n_cols = len(columns)
    samples = ["Sample{}".format(x + 1) for x in range(n_samples)]
    X = np.random.normal(size=(n_rows, n_cols))
    feature_table = pd.DataFrame(data=X, columns=columns)

    # assign unique cluster/sample pairs to each feature
    def assign_cluster(df):
        size = df.shape[0]
        clusters = np.arange(n_ft)
        if size > n_ft:
            # excess values are set to -1
            label = -np.ones(size, dtype=int)
            label[:n_ft] = np.random.choice(clusters, replace=False, size=n_ft)

        else:
            label = np.random.choice(clusters, replace=False, size=size)
        df[c.LABEL] = label
        return df

    feature_table[c.SAMPLE] = np.random.choice(samples, replace=True, size=n_rows)
    feature_table = feature_table.groupby(c.SAMPLE).apply(assign_cluster)
    feature_table = feature_table[feature_table[c.LABEL] > -1]

    return feature_table


@pytest.mark.parametrize("n_ft", [5, 50, 200])
def test_cluster_to_feature_name(n_ft):
    n_samples = 20
    feature_table = create_dummy_feature_table(n_ft, n_samples)
    n_rows = feature_table.shape[0]
    feature_table[c.LABEL] = np.random.randint(low=0, high=n_ft, size=n_rows)
    # ensure that all features are present
    feature_table.loc[feature_table.index[:n_ft], c.LABEL] = np.arange(n_ft)
    labels = feature_table[c.LABEL]
    cluster_to_ft = _build_data_matrix._cluster_to_feature_name(labels)
    # check that each feature has a code
    assert len(cluster_to_ft) == n_ft

    # check that the code of each feature has the same length
    ft_str_len = len(cluster_to_ft[0])
    for v in cluster_to_ft.values():
        assert ft_str_len == len(v)


def test__build_data_matrix():
    n_samples = 20
    n_ft = 10
    sample_metadata = create_dummy_sample_metadata(n_samples)
    feature_table = create_dummy_feature_table(n_ft, n_samples)
    labels = feature_table[c.LABEL]
    cluster_to_ft = _build_data_matrix._cluster_to_feature_name(labels)
    data_matrix = _build_data_matrix._build_data_matrix(
        feature_table, sample_metadata, cluster_to_ft
    )
    assert data_matrix.index.name == c.SAMPLE
    assert data_matrix.columns.name == c.FEATURE
    assert data_matrix.shape == (n_samples, n_ft)
    assert data_matrix.index.equals(sample_metadata.index)


def test__build_data_matrix_all_features_missing_in_a_sample():
    n_samples = 20
    n_ft = 10
    sample_metadata = create_dummy_sample_metadata(n_samples + 1)
    feature_table = create_dummy_feature_table(n_ft, n_samples)
    labels = feature_table[c.LABEL]
    cluster_to_ft = _build_data_matrix._cluster_to_feature_name(labels)
    data_matrix = _build_data_matrix._build_data_matrix(
        feature_table, sample_metadata, cluster_to_ft
    )
    assert data_matrix.index.name == c.SAMPLE
    assert data_matrix.columns.name == c.FEATURE
    assert data_matrix.shape == (n_samples + 1, n_ft)
    assert data_matrix.index.equals(sample_metadata.index)


def test__merge_features_not_merge():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    merged_dict = dict()
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]
    results = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert results is None


def test__merge_features_merge_main_ft1():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    merged_dict = dict()
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft2] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert merged_ft == ft2
    assert ft1 in merged_dict
    assert ft2 not in feature_set
    assert (data_matrix.loc[nan_index, ft1] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft1] == 2).all()


def test__merge_features_merge_main_ft1_ft1_in_merged_dict():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]
    merged_dict = {ft1: ["FT5"]}

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft2] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert merged_ft == ft2
    assert ft1 in merged_dict
    assert merged_dict[ft1] == ["FT5", ft2]
    assert ft2 not in feature_set
    assert (data_matrix.loc[nan_index, ft1] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft1] == 2).all()


def test__merge_features_merge_main_ft1_ft2_in_merged_dict():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]
    merged_dict = {ft2: ["FT5"]}

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft2] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert merged_ft == ft2
    assert ft1 in merged_dict
    assert set(merged_dict[ft1]) == {"FT5", ft2}
    assert ft2 not in feature_set
    assert (data_matrix.loc[nan_index, ft1] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft1] == 2).all()


def test__merge_features_merge_main_ft2():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    merged_dict = dict()
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft1] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert ft1 not in merged_dict
    assert merged_ft == ft1
    assert ft1 in merged_dict[ft2]
    assert ft2 in feature_set
    assert (data_matrix.loc[nan_index, ft2] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft2] == 2).all()


def test__merge_features_merge_main_ft2_ft1_in_merged_dict():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]
    merged_dict = {ft1: ["FT5"]}

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft1] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert ft1 not in merged_dict
    assert merged_ft == ft1
    assert {"FT5", ft1} == set(merged_dict[ft2])
    assert ft2 in feature_set
    assert (data_matrix.loc[nan_index, ft2] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft2] == 2).all()


def test__merge_features_merge_main_ft2_ft2_in_merged_dict():
    # features detected in all samples should not be merged
    merge_threshold = 0.8
    n_ft = 4
    n_samples = 20
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    data_matrix = pd.DataFrame(data=np.ones((n_samples, n_ft)), columns=feature_names)
    ft1 = feature_names[1]
    ft2 = feature_names[2]
    merged_dict = {ft2: ["FT5"]}

    nan_index = data_matrix.index[:10]
    not_nan_index = data_matrix.index[10:]
    data_matrix.loc[nan_index, ft1] = np.nan

    merged_ft = _build_data_matrix._merge_features(
        data_matrix, merged_dict, feature_set, ft1, ft2, merge_threshold
    )
    assert ft1 not in merged_dict
    assert merged_ft == ft1
    assert {"FT5", ft1} == set(merged_dict[ft2])
    assert ft2 in feature_set
    assert (data_matrix.loc[nan_index, ft2] == 1).all()
    assert (data_matrix.loc[not_nan_index, ft2] == 2).all()


def test__find_merge_candidates_no_candidates():
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    n_ft = len(feature_names)
    mz = pd.Series(data=np.arange(n_ft), index=feature_names)
    rt = pd.Series(data=np.arange(n_ft), index=feature_names)
    features = pd.Index(feature_names)
    mz_merge = 0.01
    rt_merge = 0.01
    ft = feature_names[0]
    candidates = _build_data_matrix._find_merge_candidates(
        ft, mz, rt, features, feature_set, mz_merge, rt_merge
    )
    assert candidates.size == 0
    assert isinstance(candidates, pd.Index)


def test__find_merge_candidates_one_candidate():
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    n_ft = len(feature_names)
    mz = pd.Series(data=np.arange(n_ft), index=feature_names)
    rt = pd.Series(data=np.arange(n_ft), index=feature_names)
    features = pd.Index(feature_names)
    mz_merge = 1.1
    rt_merge = 1.1
    ft = feature_names[0]
    candidates = _build_data_matrix._find_merge_candidates(
        ft, mz, rt, features, feature_set, mz_merge, rt_merge
    )
    assert candidates.size == 1
    assert isinstance(candidates, pd.Index)


def test__find_merge_candidates_multiple_candidates():
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    n_ft = len(feature_names)
    mz = pd.Series(data=np.arange(n_ft), index=feature_names)
    rt = pd.Series(data=np.arange(n_ft), index=feature_names)
    features = pd.Index(feature_names)
    mz_merge = 2.1
    rt_merge = 2.1
    ft = feature_names[0]
    candidates = _build_data_matrix._find_merge_candidates(
        ft, mz, rt, features, feature_set, mz_merge, rt_merge
    )
    assert candidates.size == 2
    assert isinstance(candidates, pd.Index)


def test__find_merge_candidates_multiple_candidates_one_removed_from_ft_set():
    feature_names = ["FT1", "FT2", "FT3", "FT4"]
    feature_set = set(feature_names)
    # FT3 is a valid candidate, but it is removed from feature set. It should
    # not appear as a candidate.
    feature_set.remove("FT3")
    n_ft = len(feature_names)
    mz = pd.Series(data=np.arange(n_ft), index=feature_names)
    rt = pd.Series(data=np.arange(n_ft), index=feature_names)
    features = pd.Index(feature_names)
    mz_merge = 2.1
    rt_merge = 2.1
    ft = feature_names[0]
    candidates = _build_data_matrix._find_merge_candidates(
        ft, mz, rt, features, feature_set, mz_merge, rt_merge
    )
    assert candidates.size == 1
    assert isinstance(candidates, pd.Index)
    assert "FT3" not in candidates


def test__lc_merge_close_features_none_close():
    n_samples = 10
    n_ft = 4
    ft_names = ["FT-{}".format(x) for x in range(n_ft)]
    samples = ["sample{}".format(x) for x in range(n_samples)]
    data_matrix = pd.DataFrame(
        data=np.ones((n_samples, n_ft)), columns=ft_names, index=samples
    )
    ft_descriptors = [c.MZ, c.RT]
    feature_metadata = pd.DataFrame(
        data=[[1, 1], [2, 2], [3, 3], [4, 4]], columns=ft_descriptors, index=ft_names
    )

    mz_merge = 0.01
    rt_merge = 0.01
    merge_threshold = 1.0

    expected_data_matrix = data_matrix.copy()
    expected_feature_metadata = feature_metadata.copy()
    _build_data_matrix._lc_merge_close_features(
        data_matrix, feature_metadata, mz_merge, rt_merge, merge_threshold
    )
    assert expected_data_matrix.equals(data_matrix)
    merged = feature_metadata.pop(c.MERGED)
    assert expected_feature_metadata.equals(feature_metadata)
    assert (merged == "").all()


def test__lc_merge_close_features_all_features_above_merge_threshold():
    n_samples = 10
    n_ft = 4
    ft_names = ["FT-{}".format(x) for x in range(n_ft)]
    samples = ["sample{}".format(x) for x in range(n_samples)]
    data_matrix = pd.DataFrame(
        data=np.ones((n_samples, n_ft)), columns=ft_names, index=samples
    )
    ft_descriptors = [c.MZ, c.RT]
    feature_metadata = pd.DataFrame(
        data=[[1, 1], [1.005, 1.005], [3, 3], [4, 4]], columns=ft_descriptors, index=ft_names
    )

    mz_merge = 0.01
    rt_merge = 0.01
    merge_threshold = 1.0

    expected_data_matrix = data_matrix.copy()
    expected_feature_metadata = feature_metadata.copy()
    _build_data_matrix._lc_merge_close_features(
        data_matrix, feature_metadata, mz_merge, rt_merge, merge_threshold
    )
    assert expected_data_matrix.equals(data_matrix)
    merged = feature_metadata.pop(c.MERGED)
    assert expected_feature_metadata.equals(feature_metadata)
    assert (merged == "").all()


def test__lc_merge_close_features_merge_one_feature():
    n_samples = 10
    n_ft = 4
    ft_names = ["FT-{}".format(x) for x in range(n_ft)]
    samples = ["sample{}".format(x) for x in range(n_samples)]
    data_matrix = pd.DataFrame(
        data=np.ones((n_samples, n_ft)), columns=ft_names, index=samples
    )
    ft_descriptors = [c.MZ, c.RT]
    feature_metadata = pd.DataFrame(
        data=[[1, 1], [1.005, 1.005], [3, 3], [4, 4]], columns=ft_descriptors, index=ft_names
    )

    # set values in FT2 to nan
    main_ft = ft_names[0]
    merge_ft = ft_names[1]
    data_matrix.loc[samples[5:], merge_ft] = np.nan

    mz_merge = 0.01
    rt_merge = 0.01
    merge_threshold = 1.0

    expected_data_matrix = data_matrix.copy()
    expected_data_matrix[main_ft] = expected_data_matrix[main_ft] + expected_data_matrix[
        merge_ft
    ].fillna(0)
    expected_data_matrix.drop(columns=[merge_ft], inplace=True)
    expected_feature_metadata = feature_metadata.copy()
    expected_feature_metadata.drop(index=[merge_ft], inplace=True)

    _build_data_matrix._lc_merge_close_features(
        data_matrix, feature_metadata, mz_merge, rt_merge, merge_threshold
    )
    assert expected_data_matrix.equals(data_matrix)
    merged = feature_metadata.pop(c.MERGED)
    expected_merged = pd.Series("", index=ft_names)
    expected_merged[main_ft] = merge_ft
    expected_merged.pop(merge_ft)
    assert expected_feature_metadata.equals(feature_metadata)
    assert merged.equals(expected_merged)
