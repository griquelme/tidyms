import tidyms as ms


def test_prevalence_filter_remove_none(data_container_with_order):
    data = data_container_with_order
    process_classes = None
    lb = 0
    ub = 1
    intraclass = True
    threshold = 0
    pf = ms.filter.PrevalenceFilter(process_classes=process_classes, lb=lb,
                                    ub=ub, intraclass=intraclass,
                                    threshold=threshold)
    pf.process(data)
    assert True


def test_prevalence_filter_remove_one_feature(data_container_with_order):
    data = data_container_with_order
    rm_ft = "FT01"
    data._data_matrix.loc[:, rm_ft] = 0
    process_classes = None
    lb = 0.1
    ub = 1
    intraclass = True
    threshold = 0
    pf = ms.filter.PrevalenceFilter(process_classes=process_classes,
                                    lb=lb,
                                    ub=ub,
                                    intraclass=intraclass,
                                    threshold=threshold)
    pf.process(data)
    assert rm_ft in pf.remove


def test_blank_filter_custom_func(data_container_with_order):
    data = data_container_with_order
    bc = ms.filter.BlankCorrector(mode=lambda x: 20)
    bc.process(data)
    assert (data._data_matrix[data.classes
            .isin(bc.params["process_classes"])] == 0).all().all()


def test_variation_filter(data_container_with_order):
    data = data_container_with_order
    vf = ms.filter.VariationFilter(lb=0,
                                   ub=0.2,
                                   process_classes=None)
    vf.process(data)
    print(vf.remove)
    assert vf.remove.empty
