import pandas as pd

from tidyms.simulation import simulate_dataset
from tidyms.container import DataContainer
import numpy as np
import pytest


# simulated data used for tests

@pytest.fixture
def data_container_with_order():
    population = {"healthy": 20, "disease": 35}
    mean = {"healthy": np.array([50, 100, 150]),
            "disease": np.array([150, 200, 300])}
    cov = {"healthy": np.array([1, 1, 1]),
           "disease": np.array([2, 2, 2])}
    blank_contribution = np.array([3, 5, 10])
    mz = np.array([100, 200, 300])
    rt = np.array([50, 60, 70])
    dc = simulate_dataset(population, mean, cov, mz, rt, blank_contribution,
                          prepend_blank=1, append_blank=1)
    return dc


@pytest.fixture
def data_container_with_batch_effect():
    population = {"healthy": 24, "disease": 24}
    mean = {"healthy": np.array([50, 100, 150]),
            "disease": np.array([150, 200, 300])}
    cov = {"healthy": np.array([1, 1, 1]),
           "disease": np.array([2, 2, 2])}
    blank_contribution = np.array([3, 5, 10])
    mz = np.array([100, 200, 300])
    rt = np.array([50, 60, 70])
    dc = simulate_dataset(population, mean, cov, mz, rt, blank_contribution,
                          prepend_blank=1, append_blank=1)

    def add_batch_effect(x):
        n_samples = x.shape[0]
        batch_effect = pd.Series(data=np.linspace(1, 0.8, n_samples),
                                 index=x.index)
        return x.multiply(batch_effect, axis=0)
    dc._data_matrix = (dc.data_matrix.groupby(dc.batch)
                       .apply(add_batch_effect))
    return dc


@pytest.fixture
def data_container_without_order(data_container_with_order):
    dc = data_container_with_order
    dm = dc.data_matrix.copy()
    sm = dc.sample_metadata.copy()
    sm.pop("order")
    sm.pop("batch")
    fm = dc.feature_metadata.copy()
    mapping = {k: v for k, v in dc.mapping.items() if v is not None}
    dcwo = DataContainer(dm, fm, sm, mapping)
    return dcwo
