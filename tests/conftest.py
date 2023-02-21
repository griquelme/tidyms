import pandas as pd

from tidyms.simulation import simulate_dataset
from tidyms.container import DataContainer
from tidyms import fileio
from tidyms.utils import get_tidyms_path
import numpy as np
import pytest
import os


# simulated data used for tests
def pytest_sessionstart(session):
    for dataset in fileio.list_available_datasets(False):
        fileio.download_dataset(dataset)


@pytest.fixture
def data_container_with_order():
    population = {"healthy": 20, "disease": 35}
    mean = {"healthy": np.array([50, 100, 150]), "disease": np.array([150, 200, 300])}
    cov = {"healthy": np.array([1, 1, 1]), "disease": np.array([2, 2, 2])}
    blank_contribution = np.array([3, 5, 10])
    mz = np.array([100, 200, 300])
    rt = np.array([50, 60, 70])
    dc = simulate_dataset(
        population, mean, cov, mz, rt, blank_contribution, prepend_blank=1, append_blank=1
    )
    return dc


@pytest.fixture
def data_container_with_order_single_qc():
    population = {"healthy": 20, "disease": 35}
    mean = {"healthy": np.array([50, 100, 150]), "disease": np.array([150, 200, 300])}
    cov = {"healthy": np.array([1, 1, 1]), "disease": np.array([2, 2, 2])}
    blank_contribution = np.array([3, 5, 10])
    mz = np.array([100, 200, 300])
    rt = np.array([50, 60, 70])
    dc = simulate_dataset(
        population,
        mean,
        cov,
        mz,
        rt,
        blank_contribution,
        prepend_blank=1,
        append_blank=1,
        triple_qc=False,
    )
    return dc


@pytest.fixture
def data_container_with_batch_effect():
    population = {"healthy": 24, "disease": 24}
    mean = {"healthy": np.array([50, 100, 150]), "disease": np.array([150, 200, 300])}
    cov = {"healthy": np.array([1, 1, 1]), "disease": np.array([2, 2, 2])}
    blank_contribution = np.array([3, 5, 10])
    mz = np.array([100, 200, 300])
    rt = np.array([50, 60, 70])
    dc = simulate_dataset(
        population, mean, cov, mz, rt, blank_contribution, prepend_blank=1, append_blank=1
    )

    def add_batch_effect(x):
        n_samples = x.shape[0]
        batch_effect = pd.Series(data=np.linspace(1, 0.8, n_samples), index=x.index)
        return x.multiply(batch_effect, axis=0)

    dc._data_matrix = dc.data_matrix.groupby(dc.batch).apply(add_batch_effect)
    mapping = {"sample": list(population), "qc": ["QC"]}
    dc.mapping = mapping
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
    return DataContainer(dm, fm, sm, mapping)


@pytest.fixture
def centroid_mzml():
    cache_path = get_tidyms_path()
    dataset_name = "test-raw-data"
    filename = "centroid-data-zlib-indexed-compressed.mzML"
    data_path = os.path.join(cache_path, dataset_name, filename)
    ms_data = fileio.MSData.create_MSData_instance(data_path, ms_mode="profile")
    return ms_data


@pytest.fixture
def profile_mzml():
    cache_path = get_tidyms_path()
    filename = "profile-data-zlib-indexed-compressed.mzML"
    data_path = os.path.join(cache_path, "test-raw-data", filename)
    ms_data = fileio.MSData.create_MSData_instance(data_path, ms_mode="profile")
    return ms_data
