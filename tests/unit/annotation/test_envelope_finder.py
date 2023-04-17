from tidyms.annotation import envelope_finder as ef
from tidyms.annotation.annotation_data import AnnotationData
from tidyms.chem import PeriodicTable
from tidyms.chem import Formula
from tidyms.lcms import LCTrace, Peak
import pytest
import numpy as np
from collections.abc import Sequence


@pytest.fixture
def formulas():
    formulas = {
        "cho": [
            "C27H34O9",
            "C62H120O6",
            "C59H114O6",
            "C62H120O6",
            "C56H42O10",
            "C17H20O4",
            "C54H104O6",
            "C48H92O6",
            "C52H100O6",
            "C54H104O6",
            "C47H90O6",
            "C50H96O6",
            "C56H108O6",
            "C21H19O13",
            "C57H94O6",
            "C58H112O6",
            "C64H124O6",
            "C24H20O8",
            "C17H12O6",
            "C61H118O6",
            "C47H90O6",
            "C6H12O6",
            "C63H106O6",
            "C40H52O4",
            "C61H118O6",
            "C61H118O6",
            "C57H96O6",
            "C37H72O5",
            "C28H44O2",
            "C29H24O12",
            "C51H98O6",
            "C39H72O5",
            "C46H78O7",
            "C54H104O6",
            "C63H110O6",
            "C21H18O13",
            "C53H102O6",
            "C62H120O6",
            "C59H114O6",
            "C41H78O6",
            "C25H30O6",
            "C51H98O6",
            "C53H102O6",
            "C43H68O13",
            "C37H72O5",
            "C59H114O6",
            "C15H12O4",
            "C16H18O4",
            "C61H110O6",
            "C58H112O6",
        ],
        "chnops": [
            "C41H80NO8P",
            "C54H104O6",
            "C27H40O2",
            "C24H26O12",
            "C55H106O6",
            "C45H80O16P2",
            "C50H96O6",
            "C8H13NO",
            "C35H36O15",
            "C48H92O6",
            "C63H98O6",
            "C15H14O5",
            "C18H23N3O6",
            "C44H80NO8P",
            "C47H90O6",
            "C47H84O16P2",
            "C14H14O4",
            "C46H80NO10P",
            "C35H64O9",
            "C51H98O6",
            "C6H12O6",
            "C26H34O7",
            "C17H18O4",
            "C6H8O9S",
            "C63H100O6",
            "C51H98O6",
            "C6H12O",
            "C50H96O6",
            "C56H108O6",
            "C61H114O6",
            "C57H110O6",
            "C44H76NO8P",
            "C63H110O6",
            "C41H71O8P",
            "C16H16O10",
            "C21H20O15",
            "C4H6O3",
            "C16H18O9",
            "C51H98O6",
            "C57H94O6",
            "C4H9NO2",
            "C56H108O6",
            "C6H8O7",
            "C57H98O6",
            "C63H110O6",
            "C58H112O6",
            "C12H16O7S",
            "C27H30O12",
            "C26H28O16",
            "C27H38O12",
        ],
    }
    return formulas


@pytest.fixture
def elements():
    elements = {"cho": ["C", "H", "O"], "chnops": ["C", "H", "N", "O", "P", "S"]}
    return elements


def create_feature_list_from_formula(f_str: str) -> Sequence[Peak]:
    f = Formula(f_str)
    M, _ = f.get_isotopic_envelope()
    if f.charge:
        mz = M / abs(f.charge)
    else:
        mz = M
    feature_list = list()
    for k_mz in mz:
        size = 30
        time = np.linspace(0, size, size)
        scan = np.arange(size)
        spint = np.ones(size)
        roi = LCTrace(time, spint, spint * k_mz, scan)
        peak = Peak(10, 15, 20, roi)
        feature_list.append(peak)
    return feature_list


@pytest.mark.parametrize("element_set", ["cho", "chnops"])
def test__make_exact_mass_difference_bounds(elements, element_set):
    # test bounds for different element combinations
    elements = elements[element_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    # m and M are the bounds for each nominal mass increment
    for e in elements:
        nom, ex, ab = e.get_abundances()
        nom = nom - nom[0]
        ex = ex - ex[0]
        for i, mi in zip(nom[1:], ex[1:]):
            m_min, m_max = bounds[i]
            assert m_min <= mi
            assert m_max >= mi


@pytest.mark.parametrize("element_set", ["cho", "chnops"])
def test__get_next_mz_search_interval_mz(elements, formulas, element_set):
    elements = elements[element_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    dM_bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    # test bounds for different formulas
    for f_str in formulas[element_set]:
        feature_list = create_feature_list_from_formula(f_str)
        length = len(feature_list)
        for k in range(1, length - 1):
            k_ft = feature_list[k]
            min_mz, max_mz = ef._get_next_mz_search_interval(
                feature_list[:k], dM_bounds, 1, 0.005
            )
            assert (min_mz < k_ft.mz) and (k_ft.mz < max_mz)


@pytest.mark.parametrize("charge", list(range(1, 6)))
def test_get_k_bounds_multiple_charges(elements, formulas, charge):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    for f_str in formulas:
        features = create_feature_list_from_formula(f"[{f_str}]{charge}+")
        length = len(features)
        for k in range(1, length - 1):
            m_min, m_max = ef._get_next_mz_search_interval(
                features[:k], bounds, charge, 0.005
            )
            assert (m_min < features[k]) and (features[k] < m_max)


@pytest.mark.parametrize(
    "elements_set,charge", [["cho", 1], ["cho", 2], ["chnops", 1], ["chnops", 2]]
)
def test__find_envelopes(formulas, elements, elements_set, charge):
    # test that the function works using as a list m/z values generated from
    # formulas.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    max_length = 10
    mz_tol = 0.005
    min_similarity = 0.9
    for f_str in formulas:
        f_str = f"[{f_str}]{charge}+"
        features = create_feature_list_from_formula(f_str)
        data = AnnotationData(features)
        mmi = data.features[0]
        results = ef._find_envelopes(
            data.features,
            mmi,
            data.non_annotated,
            data.similarity_cache,
            charge,
            max_length,
            mz_tol,
            min_similarity,
            bounds,
        )
        expected = features
        assert results[0] == expected


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test__find_envelopes_no_charge(formulas, elements, elements_set):
    # test that the function works using as a list m/z values generated from
    # formulas.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    max_length = 10
    charge = 0
    mz_tol = 0.005
    min_similarity = 0.9
    for f_str in formulas:
        features = create_feature_list_from_formula(f_str)
        data = AnnotationData(features)
        mmi = features[0]
        results = ef._find_envelopes(
            features,
            mmi,
            data.non_annotated,
            data.similarity_cache,
            charge,
            max_length,
            mz_tol,
            min_similarity,
            bounds,
        )
        expected = features
        assert results[0] == expected


def test_EnvelopeFinder(elements, formulas):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    envelope_finder = ef.EnvelopeFinder(elements, 0.005, max_length=10)
    charge = 1
    for f_str in formulas:
        features = create_feature_list_from_formula(f_str)
        mmi = features[0]
        data = AnnotationData(features)
        results = envelope_finder.find(data, mmi, charge)
        expected = features
        assert len(results) == 1
        assert results[0] == expected
