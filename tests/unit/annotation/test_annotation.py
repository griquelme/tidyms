import numpy as np
import pytest

from tidyms.annotation import annotation
from tidyms.raw_data_utils import make_roi
from tidyms.fileio import MSData_simulated
from tidyms.lcms import Peak
from tidyms.chem import Formula


@pytest.fixture
def annotation_tools_params():
    bounds = {
        "C": (0, 50),
        "H": (0, 100),
        "O": (0, 20),
        "N": (0, 20),
        "Cl": (0, 2),
        "B": (0, 1),
    }
    params = {
        "bounds": bounds,
        "max_mass": 2500,
        "max_length": 10,
        "max_charge": 3,
        "min_M_tol": 0.005,
        "max_M_tol": 0.01,
        "p_tol": 0.05,
        "min_similarity": 0.9,
        "min_p": 0.01,
    }
    return params


def test__annotate_empty_feature_list(annotation_tools_params):
    tools = annotation.create_annotation_tools(**annotation_tools_params)
    feature_list = list()
    annotation.annotate(feature_list, *tools)


@pytest.fixture
def compound_data():
    compounds = [
        "[C10H20O2]-",
        "[C10H20BO3]-",
        "[C20H40BO5]2-",
        "[C18H19N2O3]-",
        "[C18H20N2O3Cl]-",
        "[C10H20Cl]-",
    ]
    rt_list = [50, 75, 150, 200, 200, 175]
    amp_list = [10000, 20000, 30000, 25000, 25000, 20000]
    return compounds, rt_list, amp_list


@pytest.fixture
def feature_list(compound_data) -> list[Peak]:
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
    ms_data = MSData_simulated(mz_grid, rt_grid, mz_params, rt_params, noise=0.025)

    roi_list = make_roi(ms_data, tolerance=0.01, pad=2)
    ft_list = list()
    for k, r in enumerate(roi_list):
        r.extract_features()
        r.id = k
        if r.features:
            for j, ft in enumerate(r.features):
                ft.id = j
            ft_list.extend(r.features)
    return ft_list


def test_annotate(feature_list, annotation_tools_params):
    tools = annotation.create_annotation_tools(**annotation_tools_params)
    annotation.annotate(feature_list, *tools)

    # group features by isotopologue label.
    annotation_check = dict()
    for ft in feature_list:
        group_list = annotation_check.setdefault(ft.annotation.isotopologue_label, list())
        group_list.append(ft)
    if -1 in annotation_check:
        annotation_check.pop(-1)
    assert len(annotation_check) == 6
    for v in annotation_check.values():
        assert len(v) == 4  # features where generated with 4 isotopologues.
