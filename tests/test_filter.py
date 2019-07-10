from ms_feature_validation import filter
import numpy as np
import pandas as pd
import pytest

n_features = 20
n_samples = 100
sample_names = ["sample{:03d}".format(x) for x in range(1, n_samples + 1)]
feature_names = ["FT{:03d}".format(x) for x in range(1, n_features + 1)]
classes_names = ["class1", "class2"]
classes = ["class1"] * (n_samples // 2) + ["class2"] * (n_samples // 2)
classes = pd.Series(data=classes, index=sample_names)


def test_prevalence_filter_empty():
    data = pd.DataFrame([])
    classes = pd.Series([])
    lb = 0.5
    ub = 0.8
    intra_class = True
    result = filter.prevalence_filter(data, classes, classes_names,
                                      lb, ub, intra_class)
    assert result.equals(pd.Index([]))


def test_prevalence_filter_intraclass():
    prevalence_data_class1 = np.random.rand(n_samples // 2, n_features)
    prevalence_data_class1[0:25, 0:5] = 0
    prevalence_data_class2 = np.random.rand(n_samples // 2, n_features)
    prevalence_data_class2[0:23, 0:5] = 0
    pdata = np.vstack((prevalence_data_class1, prevalence_data_class2))
    prevalence_df = pd.DataFrame(data=pdata,
                                 index=sample_names,
                                 columns=feature_names)
    lb = 0.6
    ub = 1
    intra_class = True
    result = filter.prevalence_filter(prevalence_df, classes, classes_names,
                                      lb, ub, intra_class)
    print(result)
    indices = pd.Index(["FT001", "FT002", "FT003", "FT004", "FT005"])
    assert indices.equals(result)


def test_prevalence_filter_global():
    prevalence_data = np.random.rand(n_samples, n_features)
    prevalence_data[0:41, 0:5] = 0
    prevalence_df = pd.DataFrame(data=prevalence_data,
                                 index=sample_names,
                                 columns=feature_names)
    lb = 0.6
    ub = 1
    intra_class = False
    result = filter.prevalence_filter(prevalence_df, classes, classes_names,
                                      lb, ub, intra_class)
    print(result)
    indices = pd.Index(["FT001", "FT002", "FT003", "FT004", "FT005"])
    assert indices.equals(result)


def test_variation_filter():
    lb = 0
    ub = 0.4
    intra_class = True
    robust = False
    data_class1 = np.random.normal(size=(n_samples // 2, n_features),
                                    loc = 1,
                                    scale=np.array([0.2] * 10 + [0.6] * 10))
    data_class2 = np.random.normal(size=(n_samples // 2, n_features),
                                    loc = 1,
                                    scale=0.6)
    data = np.vstack((data_class1, data_class2))
    df = pd.DataFrame(data=data, index=sample_names, columns=feature_names)
    result = filter.variation_filter(df, classes, classes_names,
                                     lb, ub, intra_class, robust)
    print(result)
    indices = pd.Index(["FT{:03d}".format(x) for x in range(11, n_features + 1)])
    assert result.equals(indices)

