# -*- coding: utf-8 -*-

from ms_feature_validation import _filter_functions
import pandas as pd
import numpy as np
import pytest

np.random.seed(100)


@pytest.fixture
def sample_data():
    n_samples = 50
    n_features = 10
    data = np.zeros((n_samples, n_features))
    sample_names = ["sample_{:02d}".format(x) for x in range(data.shape[0])]
    feature_names = ["feature_{:02d}".format(x) for x in range(data.shape[1])]
    data = pd.DataFrame(data=data, columns=feature_names, index=sample_names)
    class_names = ["blank", "qc", "sample_class_1",
                   "sample_class_2", "sample_class_3"]
    classes = list()
    for k in range(n_samples):
        if k <= 2:
            classes.append("blank")
        elif (k > 2) and (k <= 5):
            classes.append("qc")
        elif (k >= (n_samples - 4)):
            classes.append("qc")
        elif k % 5 == 0:
            classes.append("qc")
        elif k % 3 == 0:
            classes.append("sample_class_1")
        elif k % 3 == 1:
            classes.append("sample_class_2")
        elif k % 3 == 2:
            classes.append("sample_class_3")
    
    order = list(range(n_samples))
    batch = ([1] * (n_samples // 3) +
             [2] * (n_samples // 3) +
             [3] * (n_samples - 2 * (n_samples // 3)))
    metadata = pd.DataFrame(data=(classes, batch, order)).T
    metadata.index = sample_names
    metadata.columns = ["class", "batch", "order"]
    
    # different mean and variance for each class
    mean_dict = {"blank": 20,
                 "qc" : 50,
                 "sample_class_1": 80,
                 "sample_class_2": 60,
                 "sample_class_3": 40,}
    variance_dict = {"blank": 5,
                     "qc" : np.arange(n_features) * 0.05,
                     "sample_class_1": 20,
                     "sample_class_2": 20,
                     "sample_class_3": 20,}
    
    for class_name in class_names:
        mask = metadata["class"] == class_name
        mask = metadata[mask].index
        data.loc[mask, :] = np.random.normal(loc=mean_dict[class_name],
            scale=variance_dict[class_name],
            size=data.loc[mask, :].shape)
    return data, metadata


# prevalence for different features


def test_normalizer_global(sample_data):
    # test normalizer function with intraclass set to False
    data, metadata = sample_data
    intraclass = False
    normalizer = _filter_functions.normalizer(metadata["class"], intraclass)
    assert (data / data.shape[0]).equals(normalizer(data))


def test_normalizer_intraclass(sample_data):
    # test normalizer function with intraclass set to True
    data, metadata = sample_data
    intraclass = True
    normalizer = _filter_functions.normalizer(metadata["class"], intraclass)
    class_counts = metadata["class"].value_counts()
    print(data / class_counts)
    assert data.divide(class_counts, axis=0).equals(normalizer(data))


def test_grouper_global(sample_data):
    # test grouper with intraclass set to False
    data, metadata = sample_data
    intraclass = False
    grouper = _filter_functions.grouper(metadata["class"], intraclass)
    assert data.equals(grouper(data))


def test_grouper_intraclass(sample_data):
    # test grouper with intraclass set to True
    data, metadata = sample_data
    intraclass = True
    grouper = _filter_functions.grouper(metadata["class"], intraclass)
    grouped_data = grouper(data)
    # check if there is a better way to check for equality
    for class_name in metadata["class"].unique():
        assert (grouped_data.get_group(class_name)
        .equals(data[metadata["class"] == class_name]))
        

@pytest.mark.parametrize("lb,ub,expected",
                         [(0, 3, [False, False, False, False]),
                          (1, 3, [False, False, False, False]),
                          (1, 2, [False, False, False, False]),
                          (0, 1, [False, False, True, True])])
def test_bound_checker_intraclass(lb, ub, expected):
    data = [1, 1, 2, 2]
    ft_names = ["FT{:02d}".format(x) for x in range(4)]
    data = pd.Series(data=data, index=ft_names)
    expected = pd.Series(data=expected, index=ft_names)
    intraclass = False
    bounds_checker = _filter_functions.bounds_checker(lb, ub, intraclass)
    print(bounds_checker(data))
    assert bounds_checker(data).equals(expected)
    
    
    