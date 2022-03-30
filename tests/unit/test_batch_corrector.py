import numpy as np
from tidyms import _batch_corrector
# import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess


def test_correct_batches(data_container_with_order):
    data_matrix = data_container_with_order.data_matrix
    sample_metadata = data_container_with_order.sample_metadata
    sample_class = ["healthy", "disease"]
    qc_class = ["QC"]
    _batch_corrector.correct_batches(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        verbose=False
    )
    assert True


def test_correct_batches_frac(data_container_with_order):
    # test specifying a frac value
    data_matrix = data_container_with_order.data_matrix
    sample_metadata = data_container_with_order.sample_metadata
    sample_class = ["healthy", "disease"]
    qc_class = ["QC"]
    _batch_corrector.correct_batches(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        frac=0.7,
        verbose=False
    )
    assert True


def test_correct_batches_first_n(data_container_with_order):
    # test specifying a frac value
    data_matrix = data_container_with_order.data_matrix
    sample_metadata = data_container_with_order.sample_metadata
    sample_class = ["healthy", "disease"]
    qc_class = ["QC"]
    _batch_corrector.correct_batches(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        first_n=1,
        verbose=False
    )
    assert True


def test_lowess_min_n_samples():
    # check that the if n is lower or equal than 3 lowess return the same value
    n = 4
    for k in range(2, n):
        x = np.arange(k)
        y = np.random.normal(size=k)
        y_fit = lowess(y, x, is_sorted=True, return_sorted=False)
        assert np.allclose(y, y_fit)


def test_split_data_matrix(data_container_with_order):
    # Test if we can rebuild the matrix from the fragments
    data_matrix = data_container_with_order.data_matrix
    sample_metadata = data_container_with_order.sample_metadata
    sample_class = ["healthy", "disease"]
    qc_class = ["QC"]
    iterator = _batch_corrector._split_data_matrix(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        0.0
    )
    rebuilt = np.zeros(shape=data_matrix.shape, dtype=float)
    for start, k, order, xgk, _, _ in iterator:
        rebuilt[start + np.arange(xgk.size), k] = xgk.flatten()
    assert np.array_equal(data_matrix.to_numpy(), rebuilt)


def test_rebuild_data_matrix(data_container_with_order):
    # Test if we can rebuild the matrix from the fragments
    data_matrix = data_container_with_order.data_matrix
    sample_metadata = data_container_with_order.sample_metadata
    sample_class = ["healthy", "disease"]
    qc_class = ["QC"]
    iterator = _batch_corrector._split_data_matrix(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        0.0
    )

    # compute index used to rebuild the matrix but don't modify the values
    def process_chunk(args):
        start_index, column, order, x, train_index, predict_index = args
        index = np.arange(x.size) + start_index
        return x, index, column

    chunks = [process_chunk(x) for x in iterator]
    shape = data_matrix.shape
    rebuilt = _batch_corrector._rebuild_data_matrix(shape, chunks)
    X = data_matrix.to_numpy()
    assert np.array_equal(X, rebuilt)


def test_find_invalid_samples(data_container_with_order):
    data = data_container_with_order
    sample_metadata = data.sample_metadata
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    invalid_samples = _batch_corrector.find_invalid_samples(
        sample_metadata,
        sample_class,
        qc_class
    )
    assert invalid_samples.size == 0

def test_find_invalid_samples_remove_first_block(data_container_with_order):
    # check if study samples with order lower than qc samples are removed
    data = data_container_with_order
    sample_metadata = data.sample_metadata.copy()
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    # modify one value at the beginning
    sample_metadata.at[sample_metadata.index[0], "class"] = sample_class[0]
    invalid_samples = _batch_corrector.find_invalid_samples(
        sample_metadata,
        sample_class,
        qc_class
    )
    assert invalid_samples.size == 1


def test_find_invalid_samples_remove_last_block(data_container_with_order):
    # check if study samples with order lower than qc samples are removed
    data = data_container_with_order
    sample_metadata = data.sample_metadata.copy()
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    # modify one value at the beginning
    sample_metadata.at[sample_metadata.index[-1], "class"] = sample_class[0]
    invalid_samples = _batch_corrector.find_invalid_samples(
        sample_metadata,
        sample_class,
        qc_class
    )
    assert invalid_samples.size == 1


def test_find_invalid_samples_invalid_batch(
        data_container_with_order_single_qc):
    # check if study samples with order lower than qc samples are removed
    data = data_container_with_order_single_qc
    sample_metadata = data.sample_metadata
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    # the third batch have only two QC samples and must be removed.
    n_invalid = sample_metadata["batch"].value_counts()[3]
    invalid_samples = _batch_corrector.find_invalid_samples(
        sample_metadata,
        sample_class,
        qc_class
    )
    assert invalid_samples.size == n_invalid


def test_find_invalid_features(data_container_with_order):
    data = data_container_with_order
    data_matrix = data.data_matrix
    sample_metadata = data.sample_metadata
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    threshold = 0.0
    min_detection_rate = 1.0
    invalid_features = _batch_corrector.find_invalid_features(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        threshold,
        min_detection_rate
    )
    assert invalid_features.size == 0


def test_find_invalid_features_threshold(data_container_with_order):
    # using high threshold, all features should be removed
    data = data_container_with_order
    data_matrix = data.data_matrix
    sample_metadata = data.sample_metadata
    sample_class = data.mapping["sample"]
    qc_class = data.mapping["qc"]
    threshold = 10000000.0
    min_detection_rate = 1.0
    invalid_features = _batch_corrector.find_invalid_features(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        threshold,
        min_detection_rate
    )
    assert invalid_features.size == data_matrix.shape[1]