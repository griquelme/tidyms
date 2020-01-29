from ms_feature_validation import data_container
import numpy as np
import pandas as pd
import pytest


# simulated data used for tests

ft_names = ["FT{:02d}".format(x) for x in range(1, 7)]
sample_names = ["sample{:2d}".format(x) for x in range(1, 19)]
classes = ["SV", "SV", "SV"] + ["disease", "healthy", "QC"] * 5
n_features = len(ft_names)
n_samples = len(sample_names)
batch = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
order = list(range(1, 19))
dm_data = np.vstack((np.random.normal(loc=3, size=(3, n_features)),
                     np.random.normal(loc=10, size=(n_samples - 3, n_features)))
                    )
ft_data = np.random.normal(loc=200, scale=30, size=(n_features, 2))
mapping = {"sample": ["healthy", "disease"], "blank": ["SV"],
           "qc": ["QC"]}


@pytest.fixture
def data_container_example():
    dm = pd.DataFrame(data=dm_data, columns=ft_names, index=sample_names)
    sample_information = pd.DataFrame(data=classes,
                                      index=sample_names,
                                      columns=["class"])
    feature_definitions = pd.DataFrame(data=ft_data,
                                       columns=["mz", "rt"],
                                       index=ft_names)
    data = data_container.DataContainer(dm,
                                        feature_definitions,
                                        sample_information,
                                        mapping=mapping)
    data.batch = batch
    data.order = order
    return data