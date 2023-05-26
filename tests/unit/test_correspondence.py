import numpy as np
from tidyms import correspondence
import pytest
from sklearn.cluster import DBSCAN


# test make_initial_cluster


@pytest.mark.parametrize(
    "n,k,max_size", [[20, 2, 10], [100, 4, 125], [200, 25, 1500], [200, 10, 20000]]
)
def test_make_initial_cluster(n, k, max_size):
    # n is the number of samples
    # k is the number of clusters
    # test with several sample sizes and check that the result is the same
    # as using DBSCAN without data split
    X1 = np.arange(n)
    X2 = np.arange(n)
    X = np.vstack((X1, X2)).T
    X = np.repeat(X, k, axis=0)
    X = np.random.permutation(X)
    # k cluster, no noise should be present
    eps = 0.1
    min_samples = round(n * 0.2)
    test_cluster = correspondence._cluster_dbscan(X, eps, min_samples, max_size)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    dbscan.fit(X)
    expected_cluster = dbscan.labels_
    assert np.array_equal(test_cluster, expected_cluster)


# test estimate n species


@pytest.mark.parametrize(
    "min_samples,expected",
    [[1, np.array([2, 2])], [2, np.array([2, 2])], [3, np.array([0, 0])]],
)
def test_estimate_n_species_one_class(min_samples, expected):
    samples = np.array([0] * 4 + [1] * 4)  # 8 features detected in total in two samples
    clusters = np.array(([0] * 2 + [1] * 2) * 2)  # two clusters
    n_clusters = 2
    # two species in two clusters are expected
    res = correspondence._estimate_n_species_one_class(
        samples, clusters, min_samples, n_clusters
    )
    assert np.array_equal(res, expected)


def test_estimate_n_species_multiple_groups():
    samples = np.array([0] * 4 + [1] * 4 + [2] * 4)  # 12 features in three samples
    clusters = np.array(([0] * 2 + [1] * 2) * 3)  # two clusters
    classes = np.array([0] * 8 + [1] * 4)  # two groups
    min_dr = 0.5
    # two species in two clusters are expected
    expected = {0: 2, 1: 2}
    include_classes = [0, 1]
    samples_per_class = {0: 2, 1: 1}

    res = correspondence._estimate_n_species(
        samples, clusters, classes, samples_per_class, include_classes, min_dr
    )
    assert res == expected


# test _get_min_samples


@pytest.fixture
def samples_per_class():
    res = {0: 8, 1: 16, 2: 24}
    return res


def test_get_min_samples_include_classes_none(samples_per_class):
    min_fraction = 0.25
    include_classes = None
    test_min_samples = correspondence._get_min_sample(
        samples_per_class, include_classes, min_fraction
    )
    expected_min_samples = round(sum(samples_per_class.values()) * min_fraction)
    assert expected_min_samples == test_min_samples


def test_get_min_samples_include_classes(samples_per_class):
    min_fraction = 0.25
    include_classes = [0, 1]
    test_min_samples = correspondence._get_min_sample(
        samples_per_class, include_classes, min_fraction
    )
    n_include = [v for k, v in samples_per_class.items() if k in include_classes]
    expected_min_samples = round(min(n_include) * min_fraction)
    assert expected_min_samples == test_min_samples


def test_process_cluster_one_species():
    np.random.seed(1234)
    # features
    n = 200
    X = np.random.normal(size=(n, 2))
    samples = np.arange(n)

    # add noise
    n_noise = 10
    noise = np.random.normal(size=(n_noise, 2), loc=4)
    X = np.vstack((X, noise))
    s_noise = np.random.choice(samples, size=n_noise)
    samples = np.hstack((samples, s_noise))

    expected = np.array([0] * n + [-1] * n_noise)

    n_species = 1
    max_deviation = 4
    labels, score = correspondence._process_cluster(X, samples, n_species, max_deviation)
    assert np.array_equal(labels, expected)


def test_process_cluster_two_species():
    np.random.seed(1234)
    # features
    n = 200
    x_list = list()
    s_list = list()
    for loc in [0, 4]:
        x_list.append(np.random.normal(size=(n, 2), loc=loc))
        s_list.append(np.arange(n))

    # add noise
    n_noise = 10
    x_list.append(np.random.normal(size=(n_noise, 2), loc=8))
    X = np.vstack(x_list)
    s_list.append(np.random.choice(s_list[0], size=n_noise))
    samples = np.hstack(s_list)

    expected = np.array([0] * n + [1] * n + [-1] * n_noise)

    n_species = 2
    max_deviation = 4
    labels, score = correspondence._process_cluster(X, samples, n_species, max_deviation)
    assert np.array_equal(labels, expected)


def test_match_features():
    np.random.seed(1234)
    # features
    n = 200
    x_list = list()
    s_list = list()
    for loc in [0, 4]:
        x_list.append(np.random.normal(size=(n, 2), loc=loc))
        s_list.append(np.arange(n))

    # add noise
    n_noise = 10
    x_list.append(np.random.normal(size=(n_noise, 2), loc=8))
    X = np.vstack(x_list)
    s_list.append(np.random.choice(s_list[0], size=n_noise))
    samples = np.hstack(s_list)
    classes = np.zeros_like(samples)

    samples_per_class = {0: 200}

    expected = np.array([0] * n + [1] * n + [-1] * n_noise)

    include_classes = None
    tolerance = np.array([2.0, 2.0])
    min_fraction = 0.25
    max_deviation = 4.0
    actual = correspondence.match_features(
        X,
        samples,
        classes,
        samples_per_class,
        include_classes,
        tolerance,
        min_fraction,
        max_deviation,
    )

    assert np.array_equal(actual, expected)
