import pytest

import tidyms as ms


def test_class_remover(data_container_with_order):
    rm = ["QC"]
    data = data_container_with_order
    n_qc_samples = (data.sample_metadata["class"] == 'QC').sum()
    n_samples = data.data_matrix.shape[0]
    crm = ms.filter.ClassRemover(rm)
    crm.process(data)
    assert data.data_matrix.shape[0] == (n_samples - n_qc_samples)


def test_class_remover_invalid_class(data_container_with_order):
    rm = ["invalid_class"]
    data = data_container_with_order
    crm = ms.filter.ClassRemover(rm)
    n_samples = data.data_matrix.shape[0]
    crm.process(data)
    assert data.data_matrix.shape[0] == n_samples


def test_batch_corrector(data_container_with_batch_effect):
    # in this dataset the batch effect is simulated as a linear loss of
    # sensitivity. After the correction the QCs CV should be close to zero
    data = data_container_with_batch_effect
    cv_before = data.metrics.cv().loc["QC"]     # qc cv before correction
    bc = ms.filter.BatchCorrector()
    bc.process(data)
    cv_after = data.metrics.cv().loc["QC"]
    assert (cv_before > cv_after).all()

# def test_prevalence_filter_remove_none(data_container_with_order):
#     data = data_container_with_order
#     process_classes = None
#     lb = 0
#     ub = 1
#     intraclass = True
#     threshold = 0
#     pf = ms.filter.PrevalenceFilter(process_classes=process_classes, lb=lb,
#                                     ub=ub, intraclass=intraclass,
#                                     threshold=threshold)
#     pf.process(data)
#     assert True
#
#
# def test_prevalence_filter_remove_one_feature(data_container_with_order):
#     data = data_container_with_order
#     rm_ft = "FT01"
#     data._data_matrix.loc[:, rm_ft] = 0
#     process_classes = None
#     lb = 0.1
#     ub = 1
#     intraclass = True
#     threshold = 0
#     pf = ms.filter.PrevalenceFilter(process_classes=process_classes,
#                                     lb=lb,
#                                     ub=ub,
#                                     intraclass=intraclass,
#                                     threshold=threshold)
#     pf.process(data)
#     assert rm_ft in pf.remove
#
#
# def test_blank_filter_custom_func(data_container_with_order):
#     data = data_container_with_order
#     bc = ms.filter.BlankCorrector(mode=lambda x: 20)
#     bc.process(data)
#     assert (data._data_matrix[data.classes
#             .isin(bc.params["process_classes"])] == 0).all().all()
#
#
# def test_variation_filter(data_container_with_order):
#     data = data_container_with_order
#     vf = ms.filter.VariationFilter(lb=0,
#                                    ub=0.2,
#                                    process_classes=None)
#     vf.process(data)
#     print(vf.remove)
#     assert vf.remove.empty
