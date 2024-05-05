import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from tidyms.annotation import correspondence
from tidyms.core.constants import IDFeatureDescriptors
from tidyms.core.models import FeatureGroupMap


@pytest.mark.parametrize(
    "n_sample,n_cluster,max_size", [[20, 2, 10], [100, 4, 125], [200, 25, 1500], [200, 10, 20000]]
)
def test_cluster_db_scan_solution_is_equal_to_sklearn_solution(n_sample, n_cluster, max_size):
    X1 = np.arange(n_sample)
    X = np.vstack((X1, X1)).T
    X = np.repeat(X, n_cluster, axis=0)
    X = np.random.permutation(X)

    eps = 0.1
    min_samples = round(n_sample * 0.2)
    test_cluster = correspondence._cluster_dbscan(X, eps, min_samples, max_size)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    dbscan.fit(X)
    expected_cluster = dbscan.labels_
    assert np.array_equal(test_cluster, expected_cluster)


class TestEstimateNSpecies:
    @pytest.mark.parametrize(
        "min_samples,expected",
        [[1, np.array([2, 2])], [2, np.array([2, 2])], [3, np.array([0, 0])]],
    )
    def test_one_class(self, min_samples, expected):
        samples = np.array([0] * 4 + [1] * 4)  # 8 features detected in total in two samples
        clusters = np.array(([0] * 2 + [1] * 2) * 2)  # two clusters
        n_clusters = 2
        # two species in two clusters are expected
        res = correspondence._estimate_n_species_one_group(samples, clusters, min_samples, n_clusters)
        assert np.array_equal(res, expected)

    def test_estimate_n_species_multiple_groups(self):
        samples = np.array([0] * 4 + [1] * 4 + [2] * 4)  # 12 features in three samples
        clusters = np.array(([0] * 2 + [1] * 2) * 3)  # two clusters
        classes = np.array([0] * 8 + [1] * 4)  # two groups
        min_dr = 0.5
        # two species in two clusters are expected
        expected = {0: 2, 1: 2}
        include_classes = [0, 1]
        samples_per_class = {0: 2, 1: 1}

        res = correspondence._estimate_gmm_n_components(
            samples, clusters, classes, samples_per_class, include_classes, min_dr
        )
        assert res == expected


class TestEstimateDBSCANMinSample:
    @pytest.fixture
    def samples_per_class(self):
        res = {0: 8, 1: 16, 2: 24}
        return res

    def test_no_include_groups_ok(self, samples_per_class):
        min_fraction = 0.25
        include_classes = None
        test_min_samples = correspondence._estimate_dbscan_min_sample(samples_per_class, include_classes, min_fraction)
        expected_min_samples = round(sum(samples_per_class.values()) * min_fraction)
        assert expected_min_samples == test_min_samples

    def test_include_groups_ok(self, samples_per_class):
        min_fraction = 0.25
        include_classes = [0, 1]
        test_min_samples = correspondence._estimate_dbscan_min_sample(samples_per_class, include_classes, min_fraction)
        n_include = [v for k, v in samples_per_class.items() if k in include_classes]
        expected_min_samples = round(min(n_include) * min_fraction)
        assert expected_min_samples == test_min_samples


class TestProcessCluster:
    def test_one_species(self):
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

    def test_two_species(self):
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

        n_species = 2
        max_deviation = 4
        labels, score = correspondence._process_cluster(X, samples, n_species, max_deviation)

        # create expected array: it is created after actual computation because
        # it is not possible to know which label is assigned to each cluster
        expected_first_label, expected_second_label = (0, 1) if labels[0] == 0 else (1, 0)
        expected = np.array([expected_first_label] * n + [expected_second_label] * n + [-1] * n_noise)

        assert np.array_equal(labels, expected)


def test_match_features():
    np.random.seed(1234)
    descriptors = dict()
    n_noise = 10
    for n, loc in [(200, 0), (200, 4), (n_noise, 8)]:
        mz_list = descriptors.setdefault("mz", list())
        mz_list.extend(np.random.normal(size=n, loc=loc))

        rt_list = descriptors.setdefault("rt", list())
        rt_list.extend(np.random.normal(size=n, loc=loc))

        sample_list = descriptors.setdefault(IDFeatureDescriptors.SAMPLE_ID.value, list())
        sample_list.extend(list(range(n)))

        group_list = descriptors.setdefault(IDFeatureDescriptors.FEATURE_GROUP.value, list())
        group_list.extend([0] * n)

    descriptors[IDFeatureDescriptors.ID] = list(range(410))

    tolerance = {"mz": 2.0, "rt": 2.0}

    params = correspondence.FeatureCorrespondenceParameters(max_deviation=4.0)
    actual = correspondence.match_features(descriptors, tolerance, params)

    # expected result is created after actual computation because it is not
    # possible to know which label is assigned to each cluster before the algorithm
    # runs
    expected_first_label, expected_second_label = (0, 1) if actual[0]["group"] == 0 else (1, 0)
    expected_labels = np.array([expected_first_label] * 200 + [expected_second_label] * 200 + [-1] * n_noise)
    expected = [FeatureGroupMap(id=x, group=y) for x, y in zip(descriptors[IDFeatureDescriptors.ID], expected_labels)]

    assert actual == expected
