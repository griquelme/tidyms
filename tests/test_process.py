from ms_feature_validation import process
import numpy as np
import pandas as pd
import pytest


# simulated examples used for tests
ft_names = ["FT{:02d}".format(x) for x in range(1, 7)]
sample_names = ["sample{:2d}".format(x) for x in range(1, 9)]
classes = ["SV", "SV", "disease", "disease", "healthy",
           "healthy", "healthy", "SV"]
data_matrix = pd.DataFrame(data=np.random.normal(loc=10,
                                                 size=(len(sample_names),
                                                       len(ft_names))),
                           columns=ft_names,
                           index=sample_names)
sample_information = pd.DataFrame(data=classes,
                                  index=sample_names,
                                  columns=["class"])
batch = [1, 1, 1, 1, 2, 2, 2, 2]
order = [1, 2, 3, 4, 5, 6, 7, 8]
sample_information["batch"] = batch
sample_information["order"] = order
feature_definitions = pd.DataFrame(data=np.random.normal(loc=200,
                                                         scale=30,
                                                         size=(len(ft_names), 2)),
                                   columns=["mz", "rt"],
                                   index=ft_names)
data = process.DataContainer(data_matrix,
                             feature_definitions,
                             sample_information)


def test_data_path_setter_unexistent_path():
    with pytest.raises(FileNotFoundError):
        data.data_path = "wrong_path"


def test_class_getter():
    class_series = pd.Series(data=classes, index=sample_names)
    assert data.classes.equals(class_series)


def test_class_setter():
    class_series = pd.Series(data=classes, index=sample_names)
    #set classes to an arbitrary value
    data.classes = 4
    data.classes = class_series
    assert data.classes.equals(class_series)


def test_batch_getter():
    batch_series = pd.Series(data=batch, index=sample_names)
    assert data.batch.equals(batch_series)


def test_batch_setter():
    batch_series = pd.Series(data=batch, index=sample_names)
    #set classes to an arbitrary value
    data.batch = 4
    data.batch = batch_series
    assert data.batch.equals(batch_series)


def test_order_getter():
    order_series = pd.Series(data=order, index=sample_names)
    assert data.order.equals(order_series)


def test_order_setter():
    order_series = pd.Series(data=order, index=sample_names)
    #set classes to an arbitrary value
    data.order = 4
    data.order = order_series
    assert data.order.equals(order_series)


def test_is_valid_class_name_with_valid_names():
        assert data.is_valid_class_name("healthy")
        assert data.is_valid_class_name(["healthy", "disease"])


def test_is_valid_class_name_with_invalid_names():
        assert not data.is_valid_class_name("invalid_name")
        assert not data.is_valid_class_name(["healthy", "invalid_name"])


def test_mapping_setter():
    mapping = {"sample": ["healthy", "disease"],
               "blank": ["SV"]}
    expected_mapping = {"sample": ["healthy", "disease"],
                        "blank": ["SV"], "qc": None, "zero": None,
                        "suitability": None}
    data.mapping = mapping
    assert data.mapping == expected_mapping

def test_mapping_setter_bad_sample_type():
    mapping = {"sample": ["healthy", "disease"],
               "blank": ["SV"], "bad_sample_type": ["healthy"]}
    with pytest.raises(ValueError):
        data.mapping = mapping


def test_mapping_setter_bad_sample_class():
    mapping = {"sample": ["healthy", "disease"],
               "blank": ["SV", "bad_sample_class"]}
    with pytest.raises(ValueError):
        data.mapping = mapping


def test_remove_empty_feature_list():
    features = data.data_matrix.columns.copy()
    data.remove([], "features")
    assert data.data_matrix.columns.equals(features)


def test_remove_empty_sample_list():
    samples = data.data_matrix.index.copy()
    data.remove([], "samples")
    assert data.data_matrix.index.equals(samples)


def test_remove_correct_samples():
    samples = data.data_matrix.index.copy()
    rm_samples = ["sample 1", "sample 2"]
    data.remove(rm_samples, "samples")
    assert data.data_matrix.index.equals(samples.difference(rm_samples))


def test_remove_correct_features():
    features = data.data_matrix.columns.copy()
    rm_features = ["FT01", "FT02"]
    data.remove(rm_features, "features")
    assert data.data_matrix.columns.equals(features.difference(rm_features))
    

def test_equal_feature_index():
    assert data.feature_definitions.index.equals(data.data_matrix.columns)


def test_equal_sample_index():
    assert data.data_matrix.index.equals(data.sample_information.index)


def test_remove_nonexistent_feature():
    with pytest.raises(KeyError):
        data.remove("bad_feature_name", "features")


def test_remove_nonexistent_sample():
    with pytest.raises(KeyError):
        data.remove("bad_sample_name", "samples")


# TODO: think about how to test data_path and get_available_samples
