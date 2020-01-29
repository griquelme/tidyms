import ms_feature_validation as mfv
import numpy as np
import pandas as pd
import pytest


def test_prevalence_filter_remove_none(data_container_example):
    data = data_container_example
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
    assert True


def test_prevalence_filter_remove_one_feature(data_container_example):
    data = data_container_example
    rm_ft = "FT01"
    data._data_matrix.loc[:, rm_ft] = 0
    process_classes = None
    lb = 0.1
    ub = 1
    intraclass = True
    threshold = 0
    pf = mfv.filter.PrevalenceFilter(process_classes=process_classes,
                                     lb=lb,
                                     ub=ub,
                                     intraclass=intraclass,
                                     threshold=threshold)
    pf.process(data)
    assert rm_ft in pf.remove



def test_blank_filter_custom_func(data_container_example):
    data = data_container_example
    func = lambda x: 20     # value higher than mean in samples
    bc = mfv.filter.BlankCorrector(mode=func)
    bc.process(data)
    assert (data._data_matrix[data.classes
            .isin(bc.params["process_classes"])] == 0).all().all()


def test_variation_filter(data_container_example):
    data = data_container_example
    vf = mfv.filter.VariationFilter(lb=0,
                                    ub=0.2,
                                    process_classes=None)
    vf.process(data)
    print(vf.remove)
    assert vf.remove.empty
