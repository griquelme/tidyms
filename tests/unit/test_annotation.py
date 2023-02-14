import numpy as np
import pandas as pd
import pytest

from tidyms import annotation
from tidyms.raw_data_utils import make_roi
from tidyms import _constants as c
from tidyms.fileio import SimulatedMSData
from tidyms.lcms import Peak, LCRoi
from tidyms.chem import get_chnops_bounds, Formula
from math import isclose

# Due to the complexity


def test__overlap_ratio_overlapping_peaks():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[25:55]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(5, 10, 15)
    ft2 = Peak(5, 10, 15)
    test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
    expected_result = 0.5
    assert isclose(expected_result, test_result)


def test__overlap_ratio_non_overlapping_peaks():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(5, 10, 15)
    ft2 = Peak(15, 16, 20)
    test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
    expected_result = 0.0
    assert isclose(expected_result, test_result)


def test__overlap_ratio_perfect_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(10, 15, 20)
    ft2 = Peak(0, 5, 10)
    test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__get_overlap_index_partial_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[25:55]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(5, 10, 15)
    ft2 = Peak(5, 10, 15)
    test_result = annotation._get_overlap_index(roi1, ft1, roi2, ft2)
    expected_result = 10, 15, 5, 10
    assert test_result == expected_result


def test__get_overlap_index_perfect_overlap():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[25:55]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(5, 10, 15)
    ft2 = Peak(0, 5, 10)
    test_result = annotation._get_overlap_index(roi1, ft1, roi2, ft2)
    expected_result = 5, 15, 0, 10
    assert test_result == expected_result


def test__overlap_ratio_ft2_contained_in_ft1():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    scans_roi2 = scans[30:50]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft1 = Peak(10, 15, 20)
    ft2 = Peak(2, 5, 8)
    test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
    # if ft2 is contained in ft1, the overlap ratio is 1.0
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_same_features():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = Peak(10, 15, 20)
    min_overlap = 0.5
    test_result = annotation._feature_similarity_lc(
        roi1, ft1, roi1, ft1, min_overlap)
    expected_result = 1.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_non_overlapping_features():
    scans = np.arange(100)
    scans_roi1 = scans[20:40]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = Peak(10, 15, 20)
    scans_roi2 = scans[50:70]
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft2 = Peak(10, 15, 20)
    min_overlap = 0.5
    test_result = annotation._feature_similarity_lc(
        roi1, ft1, roi2, ft2, min_overlap)
    expected_result = 0.0
    assert isclose(expected_result, test_result)


def test__feature_similarity_non_overlapping_features_ft1_starts_after_ft2():
    scans = np.arange(100)
    scans_roi1 = scans[50:70]
    roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
    ft1 = Peak(10, 15, 20)
    scans_roi2 = scans[20:40]
    roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
    ft2 = Peak(10, 15, 20)
    min_overlap = 0.5
    test_result = annotation._feature_similarity_lc(
        roi1, ft1, roi2, ft2, min_overlap)
    expected_result = 0.0
    assert isclose(expected_result, test_result)


@pytest.fixture
def annotator():
    bounds = get_chnops_bounds(1000)
    bounds.update({"Cl": (0, 2)})
    params = {
        "bounds": bounds,
        "max_mass": 2500,
        "max_length": 10,
        "max_charge": 3,
        "min_M_tol": 0.005,
        "max_M_tol": 0.01,
        "p_tol": 0.05,
        "min_similarity": 0.9,
        "min_p": 0.01
    }
    return annotation.create_annotator(**params)


def test__IsotopologueAnnotator_load_data_empty_feature_table(annotator):
    roi_list = list()
    feature_table = pd.DataFrame(data=None)
    annotator.load_data(feature_table, roi_list)


def test_annotate_empty_feature_table(annotator):
    roi_list = list()
    feature_table = pd.DataFrame(data=None)
    annotation.annotate(feature_table, roi_list, annotator)


@pytest.fixture
def compound_data():
    compounds = [
        "[C10H20O2]-",
        "[C10H20SO3]-",
        "[C20H40SO5]2-",
        "[C18H19N2O3]-",
        "[C18H20N2O3Cl]-"
    ]
    rt_list = [50, 75, 150, 200, 200]
    amp_list = [1000, 2000, 3000, 2500, 2500]
    return compounds, rt_list, amp_list


@pytest.fixture
def simulated_data(compound_data):
    compounds, rt_list, amp_list = compound_data
    mz_grid = np.linspace(100, 1200, 20000)
    rt_grid = np.arange(300)
    rt_params = list()
    mz_params = list()
    width = 4
    for comp, c_amp, c_rt in zip(compounds, amp_list, rt_list):
        f = Formula(comp)
        cM, cp = f.get_isotopic_envelope(4)
        cmz = [[x, y] for x, y in zip(cM, cp)]
        crt = [[c_rt, width, c_amp] for _ in cM]
        rt_params.append(crt)
        mz_params.append(cmz)
    mz_params = np.vstack(mz_params)
    rt_params = np.vstack(rt_params)
    ms_data = SimulatedMSData(mz_grid, rt_grid, mz_params, rt_params, noise=0.025)

    roi_list = make_roi(ms_data, tolerance=0.01)
    ft_list = list()
    roi_index = list()
    ft_index = list()
    for k, r in enumerate(roi_list):
        r.extract_features()
        rft = r.describe_features()
        ft_list.extend(rft)
        roi_index.extend([k] * len(rft))
        ft_index.extend(range(len(rft)))
    feature_table = pd.DataFrame(ft_list)
    feature_table[c.ROI_INDEX] = roi_index
    feature_table[c.FT_INDEX] = ft_index
    return roi_list, feature_table


def test_annotate(simulated_data, annotator):
    roi_list, feature_table = simulated_data
    annotation.annotate(feature_table, roi_list, annotator)

    # there should be 6 isotopologue groups, (including -1)
    label = feature_table[c.ENVELOPE_LABEL].unique()
    label = label[label > -1]
    assert len(label) == 5
