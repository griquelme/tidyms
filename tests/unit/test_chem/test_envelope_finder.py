from tidyms.chem import envelope_finder as ef
from tidyms.chem import PeriodicTable
from tidyms.chem import Formula
import pytest
import numpy as np


@pytest.fixture
def formulas():
    formulas = {
        "cho": [
            'C27H34O9', 'C62H120O6', 'C59H114O6', 'C62H120O6', 'C56H42O10',
            'C17H20O4', 'C54H104O6', 'C48H92O6', 'C52H100O6', 'C54H104O6',
            'C47H90O6', 'C50H96O6', 'C56H108O6', 'C21H19O13', 'C57H94O6',
            'C58H112O6', 'C64H124O6', 'C24H20O8', 'C17H12O6', 'C61H118O6',
            'C47H90O6', 'C6H12O6', 'C63H106O6', 'C40H52O4', 'C61H118O6',
            'C61H118O6', 'C57H96O6', 'C37H72O5', 'C28H44O2', 'C29H24O12',
            'C51H98O6', 'C39H72O5', 'C46H78O7', 'C54H104O6', 'C63H110O6',
            'C21H18O13', 'C53H102O6', 'C62H120O6', 'C59H114O6', 'C41H78O6',
            'C25H30O6', 'C51H98O6', 'C53H102O6', 'C43H68O13', 'C37H72O5',
            'C59H114O6', 'C15H12O4', 'C16H18O4', 'C61H110O6', 'C58H112O6'
        ],
        "chnops": [
            'C41H80NO8P', 'C54H104O6', 'C27H40O2', 'C24H26O12', 'C55H106O6',
            'C45H80O16P2', 'C50H96O6', 'C8H13NO', 'C35H36O15', 'C48H92O6',
            'C63H98O6', 'C15H14O5', 'C18H23N3O6', 'C44H80NO8P', 'C47H90O6',
            'C47H84O16P2', 'C14H14O4', 'C46H80NO10P', 'C35H64O9', 'C51H98O6',
            'C6H12O6', 'C26H34O7', 'C17H18O4', 'C6H8O9S', 'C63H100O6',
            'C51H98O6', 'C6H12O', 'C50H96O6', 'C56H108O6', 'C61H114O6',
            'C57H110O6', 'C44H76NO8P', 'C63H110O6', 'C41H71O8P', 'C16H16O10',
            'C21H20O15', 'C4H6O3', 'C16H18O9', 'C51H98O6', 'C57H94O6',
            'C4H9NO2', 'C56H108O6', 'C6H8O7', 'C57H98O6', 'C63H110O6',
            'C58H112O6', 'C12H16O7S', 'C27H30O12', 'C26H28O16', 'C27H38O12'
        ]
    }
    return formulas


@pytest.fixture
def elements():
    elements = {
        "cho": ["C", "H", "O"],
        "chnops": ["C", "H", "N", "O", "P", "S"]
    }
    return elements


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
        f = Formula(f_str)
        mz, _ = f.get_isotopic_envelope()
        length = len(mz)
        # test bounds for different mz lengths
        for k in range(1, length - 1):
            mz_l = mz[k]
            min_mz, max_mz = ef._get_next_mz_search_interval(
                mz[:k], dM_bounds, 1, 0.005)
            assert (min_mz < mz_l) and (mz_l < max_mz)


@pytest.mark.parametrize("charge", list(range(1, 6)))
def test_get_k_bounds_multiple_charges(elements, formulas, charge):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        mz, _ = f.get_isotopic_envelope()
        mz = mz / charge
        length = len(mz)
        for k in range(1, length - 1):
            m_min, m_max = ef._get_next_mz_search_interval(
                mz[:k], bounds, charge, 0.005)
            assert (m_min < mz[k]) and (mz[k] < m_max)


@pytest.mark.parametrize(
    "elements_set,charge",
    [["cho", 1], ["cho", 2], ["chnops", 1], ["chnops", 2]]
)
def test__find_envelopes(formulas, elements, elements_set, charge):
    # test that the function works using as a list m/z values generated from
    # formulas.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    max_length = 5
    mz_tol = 0.005
    expected = list(range(max_length))
    valid_indices = set(expected)
    for f_str in formulas:
        f_str = "[{}]{}+".format(f_str, charge)
        f = Formula(f_str)
        M, _ = f.get_isotopic_envelope()
        mz = M / f.charge
        results = ef._find_envelopes(mz, 0, valid_indices, charge, max_length, mz_tol, bounds)
        assert results[0] == expected


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test__find_envelopes_no_charge(formulas, elements, elements_set):
    # test that the function works using as a list m/z values generated from
    # formulas.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PeriodicTable().get_element(x) for x in elements]
    bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
    max_length = 5
    charge = 0
    mz_tol = 0.005
    expected = list(range(max_length))
    valid_indices = set(expected)
    for f_str in formulas:
        f = Formula(f_str)
        mz, _ = f.get_isotopic_envelope()
        results = ef._find_envelopes(mz, 0, valid_indices, charge, max_length, mz_tol, bounds)
        assert results[0] == expected


def test_EnvelopeFinder(elements, formulas):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    envelope_finder = ef.EnvelopeFinder(elements, 0.005)
    max_length = envelope_finder.max_length
    mmi_index = 0
    charge = 1
    for f_str in formulas:
        f = Formula(f_str)
        mz, _ = f.get_isotopic_envelope(n=max_length)
        expected = np.arange(mz.size)
        valid_indices = set(expected)
        results = envelope_finder.find(mz, mmi_index, charge, valid_indices)
        assert len(results) == 1
        assert np.array_equal(results[0], expected)
