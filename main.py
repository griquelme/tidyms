import ms_feature_validation as mfv
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # simulated data used for tests
    ft_names = ["FT{:02d}".format(x) for x in range(1, 7)]
    sample_names = ["sample{:2d}".format(x) for x in range(1, 9)]
    classes = ["SV", "SV", "SV", "disease", "disease", "healthy",
               "healthy", "healthy"]
    n_features = len(ft_names)
    n_samples = len(sample_names)
    batch = [1, 1, 1, 1, 2, 2, 2, 2]
    order = [1, 2, 3, 4, 5, 6, 7, 8]
    dm_data = np.vstack((np.random.normal(loc=5, size=(3, n_features)),
                         np.random.normal(loc=10, size=(n_samples - 3,
                                                        n_features)))
                        )

    ft_data = np.random.normal(loc=200, scale=30, size=(n_features, 2))
    # data container creation
    dm = pd.DataFrame(data=dm_data, columns=ft_names, index=sample_names)
    sample_information = pd.DataFrame(data=classes,
                                      index=sample_names,
                                      columns=["class"])
    feature_definitions = pd.DataFrame(data=ft_data,
                                       columns=["mz", "rt"],
                                       index=ft_names)
    mapping = {"sample": ["healthy", "disease"], "blank": ["SV"]}
    data = mfv.data_container.DataContainer(dm,
                                            feature_definitions,
                                            sample_information,
                                            mapping=mapping)
    data.batch = batch
    data.order = order
    # bc = mfv.filters.BlankCorrector(verbose=True)
    # bc.process(data)
    process_classes = None
    lb = 0
    ub = 1
    intraclass = True
    threshold = 0
    pf = mfv.filter.PrevalenceFilter(process_classes=process_classes,
                                     lb=lb,
                                     ub=ub,
                                     intraclass=intraclass,
                                     threshold=threshold)
    pf.process(data)