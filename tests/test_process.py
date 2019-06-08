from ms_feature_validation import process
import numpy as np
import pandas as pd


# simulated examples used for tests
ft_names = ["FT{:02d}".format(x) for x in range(1, 7)]
sample_names = ["sample{:2d}".format(x) for x in range(1, 9)]
classes = ["SV", "SV", "disease", "disease", "healthy", "healthy", "healthy", "SV"]
data_matrix = pd.DataFrame(data=np.random.normal(loc=10, size=(len(sample_names), len(ft_names))),
                           columns=ft_names,
                           index=sample_names)
sample_information = pd.DataFrame(data=classes, index=sample_names, columns=["class"])
feature_definitions = pd.DataFrame(data=np.random.normal(loc=200, scale=30, size=(len(ft_names), 2)),
                                   columns=["m/z", "Retention time (min)"],
                                   index=ft_names)
data = process.DataContainer(data_matrix, feature_definitions, sample_information)


def test_group_by_class():
    data_class = data.sample_information["class"]
    for class_group, grouped_data in data.group_by_class():
        assert grouped_data.index.equals(data_class[data_class == class_group].index)


def test_select_classes():
    example_class = "SV"
    assert (data.select_classes([example_class]).
            equals(data.data_matrix[data.sample_information["class"] == example_class]))