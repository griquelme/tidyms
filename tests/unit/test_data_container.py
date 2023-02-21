from tidyms import container
from tidyms.container import DataContainer
from tidyms import _constants as c
import numpy as np
import pandas as pd
import pytest
import os


def test_class_getter(data_container_with_order):
    data = data_container_with_order
    class_series = pd.Series(data=data.classes, index=data.classes.index)
    assert data.classes.equals(class_series)


def test_class_setter(data_container_with_order):
    data = data_container_with_order
    class_series = pd.Series(data=data.classes, index=data.classes.index)
    # set classes to an arbitrary value
    data.classes = 4
    data.classes = class_series
    assert data.classes.equals(class_series)


def test_batch_getter(data_container_with_order):
    data = data_container_with_order
    batch_series = pd.Series(data=data.batch, index=data.batch.index)
    assert data.batch.equals(batch_series)


def test_batch_getter_no_batch_information(data_container_without_order):
    data = data_container_without_order
    with pytest.raises(container.BatchInformationError):
        data.batch


def test_batch_setter(data_container_with_order: DataContainer):
    data = data_container_with_order
    b = np.arange(data.data_matrix.shape[0])
    batch_series = pd.Series(data=b, index=data.batch.index)
    data.batch = batch_series
    assert data.batch.equals(batch_series)


def test_order_getter(data_container_with_order):
    data = data_container_with_order
    order_series = pd.Series(data=data.order, index=data.order.index)
    assert data.order.equals(order_series)


def test_order_getter_no_order_information(data_container_without_order):
    data = data_container_without_order
    with pytest.raises(container.RunOrderError):
        data.order


def test_order_setter(data_container_with_order):
    data = data_container_with_order
    # new order value
    order_series = pd.Series(
        data=data.order + 1, index=data.order.index, dtype=data.order.dtype
    )
    data.order = order_series
    assert (data.order == order_series).all()


def test_id_getter(data_container_with_order):
    data = data_container_with_order
    assert np.equal(data.data_matrix.index.values, data.id.values).all()


def test_is_valid_class_name_with_valid_names(data_container_with_order):
    data = data_container_with_order
    assert data.is_valid_class_name("healthy")


def test_is_valid_class_name_with_invalid_names(data_container_with_order):
    data = data_container_with_order
    assert not data.is_valid_class_name("invalid_name")


def test_is_valid_class_name_several_invalid_names(data_container_with_order):
    data = data_container_with_order
    classes = ["bad_class_name_1", "bad_class_name_2"]
    assert not data.is_valid_class_name(classes)


def test_mapping_setter(data_container_with_order):
    data = data_container_with_order
    mapping = {c.STUDY_TYPE: ["healthy", "disease"], c.BLANK_TYPE: ["blank"]}
    expected_mapping = {
        c.STUDY_TYPE: ["healthy", "disease"],
        c.BLANK_TYPE: ["blank"],
        c.QC_TYPE: None,
        c.DQC_TYPE: None,
    }
    data.mapping = mapping
    assert data.mapping == expected_mapping


def test_mapping_setter_bad_sample_type(data_container_with_order):
    data = data_container_with_order
    mapping = {
        "sample": ["healthy", "disease"],
        "blank": ["SV"],
        "bad_sample_type": ["healthy"],
    }
    with pytest.raises(ValueError):
        data.mapping = mapping


def test_mapping_setter_bad_sample_class(data_container_with_order):
    data = data_container_with_order
    mapping = {"sample": ["healthy", "disease"], "blank": ["SV", "bad_sample_class"]}
    with pytest.raises(ValueError):
        data.mapping = mapping


def test_remove_invalid_axis(data_container_with_order):
    data = data_container_with_order
    with pytest.raises(ValueError):
        data.remove([], "bad_axis_name")


def test_remove_empty_feature_list(data_container_with_order):
    data = data_container_with_order
    features = data.data_matrix.columns.copy()
    data.remove([], "features")
    assert data.data_matrix.columns.equals(features)


def test_remove_empty_sample_list(data_container_with_order):
    data = data_container_with_order
    samples = data.data_matrix.index.copy()
    data.remove([], "samples")
    assert data.data_matrix.index.equals(samples)


def test_remove_correct_samples(data_container_with_order):
    data = data_container_with_order
    samples = data.data_matrix.index.copy()
    rm_samples = data.data_matrix.index[[3, 4]]
    data.remove(rm_samples, "samples")
    assert data.data_matrix.index.equals(samples.difference(rm_samples))


def test_remove_correct_features(data_container_with_order):
    data = data_container_with_order
    features = data.data_matrix.columns.copy()
    rm_features = data.data_matrix.columns[[1, 2]]
    data.remove(rm_features, "features")
    assert data.data_matrix.columns.equals(features.difference(rm_features))


def test_equal_feature_index(data_container_with_order):
    data = data_container_with_order
    assert data.feature_metadata.index.equals(data.data_matrix.columns)


def test_equal_sample_index(data_container_with_order):
    data = data_container_with_order
    assert data.data_matrix.index.equals(data.sample_metadata.index)


def test_remove_nonexistent_feature(data_container_with_order):
    data = data_container_with_order
    with pytest.raises(ValueError):
        data.remove(["bad_feature_name"], "features")


def test_remove_nonexistent_sample(data_container_with_order):
    data = data_container_with_order
    with pytest.raises(ValueError):
        data.remove(["bad_sample_name"], "samples")


def test_is_valid_with_invalid_samples(data_container_with_order):
    data = data_container_with_order
    invalid_samples = ["bad_sample1", "bad_sample2"]
    assert not data._is_valid(invalid_samples, "samples")


def test_is_valid_with_invalid_features(data_container_with_order):
    data = data_container_with_order
    invalid_samples = ["bad_feature1", "bad_feature2"]
    assert not data._is_valid(invalid_samples, "samples")


def test_is_valid_invalid_axis(data_container_with_order):
    data = data_container_with_order
    samples = ["sample01", "sample02"]
    with pytest.raises(ValueError):
        data._is_valid(samples, "bad_axis")


def test_diagnose_empty(data_container_with_order):
    data = data_container_with_order
    assert not data.diagnose()["empty"]
    # remove all features
    data.remove(data.data_matrix.columns, "features")
    assert data.diagnose()["empty"]


def test_diagnose_missing(data_container_with_order):
    data = data_container_with_order
    assert not data.diagnose()["missing"]
    # set one column to nan
    data._data_matrix[data.data_matrix.columns[0]] = np.nan
    assert data.diagnose()["missing"]
    # reset data values
    data.reset()


def test_diagnose_qc(data_container_with_order):
    data = data_container_with_order
    assert data.diagnose()[c.QC_TYPE]
    # remove mapping info
    data.mapping = None
    assert not data.diagnose()[c.QC_TYPE]


def test_diagnose_blank(data_container_with_order):
    data = data_container_with_order
    assert data.diagnose()[c.BLANK_TYPE]
    # remove mapping info
    data.mapping = None
    assert not data.diagnose()[c.BLANK_TYPE]


def test_diagnose_sample(data_container_with_order):
    data = data_container_with_order
    assert data.diagnose()[c.STUDY_TYPE]
    # remove mapping info
    data.mapping = None
    assert not data.diagnose()[c.STUDY_TYPE]


def test_diagnose_run_order(data_container_with_order):
    data = data_container_with_order
    assert data.diagnose()["order"]


def test_diagnose_invalid_run_order(data_container_without_order):
    data = data_container_without_order
    assert not data.diagnose()["order"]


def test_diagnose_batch(data_container_with_order):
    data = data_container_with_order
    assert data.diagnose()["batch"]


def test_diagnose_invalid_batch(data_container_without_order):
    data = data_container_without_order
    assert not data.diagnose()["batch"]


def test_reset(data_container_with_order):
    data = data_container_with_order
    original_dm = data.data_matrix.copy()
    original_fm = data.feature_metadata.copy()
    original_sm = data.sample_metadata.copy()

    # modify original data
    data.remove(data.data_matrix.columns[0:], "features")
    data.remove(data.data_matrix.index[0:], "samples")
    data.data_matrix = original_dm + 10

    assert not data.data_matrix.equals(original_dm)
    assert not data.feature_metadata.equals(original_fm)
    assert not data.sample_metadata.equals(original_sm)

    # reset data and compare again
    data.reset()

    assert data.data_matrix.equals(original_dm)
    assert data.feature_metadata.equals(original_fm)
    assert data.sample_metadata.equals(original_sm)


def test_select_features(data_container_with_order):
    data = data_container_with_order
    # manually selected features
    mz_tol = 0.1
    mz_mask = np.abs(data.feature_metadata["mz"] - 200) < mz_tol
    rt_tol = 2
    rt_mask = np.abs(data.feature_metadata["rt"] - 200) < rt_tol
    ft_mask = mz_mask & rt_mask
    ft_name = ft_mask[ft_mask].index
    assert data.select_features(200, 200, 0.1, 2).equals(ft_name)


def test_set_default_order(data_container_without_order):
    data = data_container_without_order
    data.set_default_order()
    # this should not raise any exception
    data.order
    data.batch


def test_sort_samples(data_container_with_order):
    data = data_container_with_order
    classes_copy = data.classes.copy().sort_values()
    data.sort("class", "samples")
    assert data.classes.equals(classes_copy)


def test_sort_features(data_container_with_order):
    data = data_container_with_order
    mz = data.feature_metadata["mz"].copy().sort_values()
    data.sort("mz", "features")
    assert data.feature_metadata["mz"].equals(mz)


def test_sort_invalid_axis(data_container_with_order):
    data = data_container_with_order
    with pytest.raises(ValueError):
        data.sort("class", "invalid_axis")


def test_save(tmpdir, data_container_with_order):
    data = data_container_with_order
    save_path = os.path.join(tmpdir, "data.pickle")
    data.save(save_path)
    data_loaded = container.DataContainer.from_pickle(save_path)
    assert data.data_matrix.equals(data_loaded.data_matrix)
    assert data.sample_metadata.equals(data_loaded.sample_metadata)
    assert data.feature_metadata.equals(data_loaded.feature_metadata)
    assert data.mapping == data_loaded.mapping


def test_set_plot_mode(data_container_with_order):
    data = data_container_with_order

    # test seaborn
    data.set_plot_mode("seaborn")
    assert isinstance(data.plot, container.SeabornPlotMethods)

    # test bokeh
    data.set_plot_mode("bokeh")
    assert isinstance(data.plot, container.BokehPlotMethods)

    # test invalid mode
    with pytest.raises(ValueError):
        data.set_plot_mode("bad-plot-mode")


def test_add_run_order_from_csv(tmpdir, data_container_without_order):
    data = data_container_without_order
    order_data = pd.DataFrame(data=np.arange(data.data_matrix.shape[0]), columns=["order"])
    order_data["sample"] = data.data_matrix.index
    order_data["batch"] = 1
    save_path = os.path.join(tmpdir, "order.csv")
    order_data.to_csv(save_path)
    data.add_order_from_csv(save_path)
    assert True


# test metrics


def test_metrics_cv_no_groupby(data_container_with_order):
    data = data_container_with_order
    cv = data.metrics.cv(groupby=None)
    assert cv.size == data.data_matrix.shape[1]


def test_metrics_cv_groupby_class(data_container_with_order):
    data = data_container_with_order
    cv = data.metrics.cv(groupby="class")
    n_class = data.classes.unique().size
    n_ft = data.data_matrix.shape[1]
    assert cv.shape == (n_class, n_ft)


def test_metrics_cv_groupby_multiple_columns(data_container_with_order):
    data = data_container_with_order
    cv = data.metrics.cv(groupby=["class", "batch"])
    n_batch = data.batch.unique().size
    n_class = data.classes.unique().size
    n_ft = data.data_matrix.shape[1]
    assert cv.shape == (n_class * n_batch, n_ft)


def test_metrics_cv_robust(data_container_with_order):
    data = data_container_with_order
    cv = data.metrics.cv(groupby="class", robust=True)
    n_class = data.classes.unique().size
    n_ft = data.data_matrix.shape[1]
    assert cv.shape == (n_class, n_ft)


def test_metrics_dratio(data_container_with_order):
    data = data_container_with_order
    dratio = data.metrics.dratio()
    assert dratio.size == data.data_matrix.shape[1]


def test_metrics_dratio_robust(data_container_with_order):
    data = data_container_with_order
    dratio = data.metrics.dratio(robust=True)
    assert dratio.size == data.data_matrix.shape[1]


def test_metrics_dratio_no_sample_mapping(data_container_with_order):
    data = data_container_with_order
    data.mapping["sample"] = None
    with pytest.raises(ValueError):
        data.metrics.dratio()


def test_metrics_dratio_no_qc_mapping(data_container_with_order):
    data = data_container_with_order
    data.mapping["qc"] = None
    with pytest.raises(ValueError):
        data.metrics.dratio()


def test_metrics_detection_rate_no_grouping(data_container_with_order):
    data = data_container_with_order
    dr = data.metrics.detection_rate(groupby=None)
    assert dr.size == data.data_matrix.shape[1]


def test_metrics_detection_rate_group_class(data_container_with_order):
    data = data_container_with_order
    n_class = data.classes.unique().size
    n_ft = data.data_matrix.shape[1]
    dr = data.metrics.detection_rate(groupby="class")
    assert dr.shape == (n_class, n_ft)


def test_metrics_detection_rate_group_multiple(data_container_with_order):
    data = data_container_with_order
    n_class = data.classes.unique().size
    n_batch = data.batch.unique().size
    n_ft = data.data_matrix.shape[1]
    dr = data.metrics.detection_rate(groupby=["class", "batch"])
    assert dr.shape == (n_class * n_batch, n_ft)


def test_metrics_pca(data_container_with_order):
    data = data_container_with_order
    scores, loadings, pc_variance, total_variance = data.metrics.pca(n_components=None)
    # shape check for scores and loadings
    assert scores.shape[0] == data.data_matrix.shape[0]
    assert loadings.shape[0] == data.data_matrix.shape[1]
    # check variance calculation
    assert np.isclose(pc_variance.sum(), total_variance)


def test_metrics_pca_n_components(data_container_with_order):
    n_comp = 3
    data = data_container_with_order
    scores, loadings, pc_variance, total_variance = data.metrics.pca(n_components=n_comp)
    # shape check for scores and loadings
    assert scores.shape[1] == n_comp
    assert scores.shape[0] == data.data_matrix.shape[0]
    assert loadings.shape[0] == data.data_matrix.shape[1]


def test_metrics_pca_normalization(data_container_with_order):
    n_comp = 2
    data = data_container_with_order
    scores, loadings, pc_variance, total_variance = data.metrics.pca(
        n_components=n_comp, normalization="sum"
    )
    # shape check for scores and loadings
    assert scores.shape[1] == n_comp
    assert scores.shape[0] == data.data_matrix.shape[0]
    assert loadings.shape[0] == data.data_matrix.shape[1]


def test_metrics_pca_scaling(data_container_with_order):
    n_comp = 2
    data = data_container_with_order
    scores, loadings, pc_variance, total_variance = data.metrics.pca(
        n_components=n_comp, scaling="autoscaling"
    )
    # shape check for scores and loadings
    assert scores.shape[1] == n_comp
    assert scores.shape[0] == data.data_matrix.shape[0]
    assert loadings.shape[0] == data.data_matrix.shape[1]


def test_metrics_pca_ignore_classes(data_container_with_order):
    n_comp = 2
    data = data_container_with_order
    scores, loadings, pc_variance, total_variance = data.metrics.pca(
        n_components=n_comp, ignore_classes=["blank"]
    )
    # shape check for scores and loadings
    assert scores.shape[1] == n_comp
    assert loadings.shape[0] == data.data_matrix.shape[1]


def test_metrics_correlation_spearman(data_container_with_order):
    data = data_container_with_order
    r = data.metrics.correlation("order", mode="spearman")
    assert r.size == data.data_matrix.shape[1]


def test_metrics_correlation_ols(data_container_with_order):
    data = data_container_with_order
    r = data.metrics.correlation("order", mode="ols")
    assert r.shape[1] == data.data_matrix.shape[1]


def test_metrics_correlation_class(data_container_with_order):
    data = data_container_with_order
    r = data.metrics.correlation("order", mode="ols", classes=["healthy", "disease"])
    assert r.shape[1] == data.data_matrix.shape[1]


def test_metrics_correlation_invalid_mode(data_container_with_order):
    data = data_container_with_order
    with pytest.raises(ValueError):
        data.metrics.correlation("order", mode="invalid-mode")
