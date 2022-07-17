from tidyms.chem import envelope_finder as ef
from tidyms.chem import PTABLE
from tidyms.chem import Formula
import pytest
import numpy as np


@pytest.fixture
def formulas():
    formulas = dict()
    formulas["cho"] = ['C27H34O9', 'C62H120O6', 'C59H114O6', 'C62H120O6',
                       'C56H42O10', 'C17H20O4', 'C54H104O6', 'C48H92O6',
                       'C52H100O6', 'C54H104O6', 'C47H90O6', 'C50H96O6',
                       'C56H108O6', 'C21H19O13', 'C57H94O6', 'C58H112O6',
                       'C64H124O6', 'C24H20O8', 'C17H12O6', 'C61H118O6',
                       'C47H90O6', 'C6H12O6', 'C63H106O6', 'C40H52O4',
                       'C61H118O6', 'C61H118O6', 'C57H96O6', 'C37H72O5',
                       'C28H44O2', 'C29H24O12', 'C51H98O6', 'C39H72O5',
                       'C46H78O7', 'C54H104O6', 'C63H110O6', 'C21H18O13',
                       'C53H102O6', 'C62H120O6', 'C59H114O6', 'C41H78O6',
                       'C25H30O6', 'C51H98O6', 'C53H102O6', 'C43H68O13',
                       'C37H72O5', 'C59H114O6', 'C15H12O4', 'C16H18O4',
                       'C61H110O6', 'C58H112O6']
    formulas["chnops"] = ['C41H80NO8P', 'C54H104O6', 'C27H40O2', 'C24H26O12',
                          'C55H106O6', 'C45H80O16P2', 'C50H96O6', 'C8H13NO',
                          'C35H36O15', 'C48H92O6', 'C63H98O6', 'C15H14O5',
                          'C18H23N3O6', 'C44H80NO8P', 'C47H90O6', 'C47H84O16P2',
                          'C14H14O4', 'C46H80NO10P', 'C35H64O9', 'C51H98O6',
                          'C6H12O6', 'C26H34O7', 'C17H18O4', 'C6H8O9S',
                          'C63H100O6', 'C51H98O6', 'C6H12O', 'C50H96O6',
                          'C56H108O6', 'C61H114O6', 'C57H110O6', 'C44H76NO8P',
                          'C63H110O6', 'C41H71O8P', 'C16H16O10', 'C21H20O15',
                          'C4H6O3', 'C16H18O9', 'C51H98O6', 'C57H94O6',
                          'C4H9NO2', 'C56H108O6', 'C6H8O7', 'C57H98O6',
                          'C63H110O6', 'C58H112O6', 'C12H16O7S', 'C27H30O12',
                          'C26H28O16', 'C27H38O12']
    return formulas


@pytest.fixture
def elements():
    elements = {"cho": ["C", "H", "O"],
                "chnops": ["C", "H", "N", "O", "P", "S"]}
    return elements


@pytest.mark.parametrize("element_set", ["cho", "chnops"])
def test_make_m_bounds(elements, element_set):
    # test bounds for different element combinations
    elements = elements[element_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
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
def test_get_k_bounds_float_mz(elements, element_set):
    elements = elements[element_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    f = Formula("H2O")
    _, mz, _ = f.get_isotopic_envelope()
    mz0 = mz[0]
    mz1 = mz[1]
    min_mz, max_mz = ef._get_k_bounds(mz0, bounds, 1, 1, 0.005)
    assert (min_mz < mz1) and (mz1 < max_mz)


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test_get_k_bounds_different_k_values(elements, formulas, elements_set):
    # test that the bounds for each k value make sense
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        for k in range(1, mz.size):
            m_min, m_max = ef._get_k_bounds(mz, bounds, k, 1, 0.005)
            assert (m_min < mz[k]) and (mz[k] < m_max)


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test_get_k_bounds_few_mz_values(elements, formulas, elements_set):
    # test that the bounds for each k value make sense when there are
    # few previous mz values.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        for k in range(1, 5):
            m_min, m_max = ef._get_k_bounds(mz[:2], bounds, k, 1, 0.005)
            assert (m_min < mz[k]) and (mz[k] < m_max)


@pytest.mark.parametrize("charge", list(range(1, 6)))
def test_get_k_bounds_multiple_charges(elements, formulas, charge):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        mz = mz / charge
        for k in range(1, mz.size):
            m_min, m_max = ef._get_k_bounds(mz, bounds, k, charge, 0.005)
            assert (m_min < mz[k]) and (mz[k] < m_max)


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test_find_envelope_candidates_aux(formulas, elements, elements_set):
    # test that the function works using as a list m/z values generated from
    # formulas.
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        n_isotopes = mz.size
        for k in range(1, n_isotopes):
            envelopes = ef._find_envelopes_aux(mz, 0, bounds, n_isotopes,
                                               0, 1, 0.005)
            assert envelopes[0] == list(range(n_isotopes))


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test_find_envelope_candidates_aux_max_nominal(formulas, elements,
                                                  elements_set):
    # test that the values obtained match the specified max_nominal value
    max_nominal = 3
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        envelopes = ef._find_envelopes_aux(mz, 0, bounds, max_nominal, 0, 1,
                                           0.005)
        assert envelopes[0][-1] == max_nominal


@pytest.mark.parametrize("elements_set", ["cho", "chnops"])
def test_find_envelope_candidates_aux_max_missing(formulas, elements,
                                                  elements_set):
    # test that the values when missing values = 1
    max_missing = 1
    elements = elements[elements_set]
    formulas = formulas[elements_set]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        n_isotopes = mz.size
        mz[2] += 0.5    # convert one peak into an invalid value.
        envelopes = ef._find_envelopes_aux(mz, 0, bounds, n_isotopes,
                                           max_missing, 1, 0.005)
        expected = np.arange(n_isotopes)
        expected = expected[expected != 2]
        assert (envelopes[0] == expected).all()


def test_find_envelope_candidates_aux_non_zero_index(formulas, elements):
    # test if the function works when the monoisotopic index is greater than 0
    elements = elements["chnops"]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    f_str = formulas["chnops"][0]
    f = Formula(f_str)
    _, mz, _ = f.get_isotopic_envelope()
    n_isotopes = mz.size
    mz_min = mz[0]
    mz_prepend = np.arange(mz_min - 10, mz_min)
    mz = np.hstack((mz_prepend, mz))
    monoisotopic_index = mz_prepend.size
    expected = np.arange(monoisotopic_index, monoisotopic_index + n_isotopes)
    envelopes = ef._find_envelopes_aux(mz, monoisotopic_index, bounds,
                                       n_isotopes, 0, 1, 0.005)
    assert (expected == envelopes[0]).all()


def tests_find_envelope_candidates(elements, formulas):
    # test that a dictionary with multiple charges is obtained
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    elements = [PTABLE[x] for x in elements]
    bounds = ef._make_m_bounds(elements, 0.0)
    charge_list = [1, 2, 3]
    for f_str in formulas:
        q = np.random.choice(charge_list)
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope()
        n_isotopes = mz.size
        mz /= q
        envelopes = ef._find_envelopes(mz, 0, charge_list, bounds, n_isotopes,
                                       0, 0.005)
        for charge in charge_list:
            if charge == q:
                assert len(envelopes[charge][0]) == mz.size


def test_ef(elements, formulas):
    elements = elements["chnops"]
    formulas = formulas["chnops"]
    envelope_finder = ef.EnvelopeFinder(elements, 0.005)
    max_length = envelope_finder._max_length
    for f_str in formulas:
        f = Formula(f_str)
        _, mz, _ = f.get_isotopic_envelope(n=max_length)
        envelopes = envelope_finder.find(mz, 0)
        assert np.array_equal(envelopes[1][0], np.arange(mz.size))
