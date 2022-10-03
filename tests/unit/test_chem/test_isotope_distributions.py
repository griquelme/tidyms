import pytest
import numpy as np
from tidyms.chem import _envelope_utils as ids
from tidyms.chem import Formula, PeriodicTable
from itertools import product


@pytest.mark.parametrize(
    "isotope_symbol,n,max_length",
    product(["2H", "31P"], [0, 1, 5], [1, 2, 5]))
def test__get_n_isotopes_envelope(isotope_symbol: str, n: int, max_length: int):
    isotope = PeriodicTable().get_isotope(isotope_symbol)
    M, p = ids._get_n_isotopes_envelope(isotope, n, max_length)
    M_expected = np.zeros(max_length)
    M_expected[0] = n * isotope.m
    p_expected = np.zeros(max_length)
    p_expected[0] = 1.0
    assert np.array_equal(M, M_expected)
    assert np.array_equal(p, p_expected)


def test__validate_abundance_valid_value():
    symbol = "C"
    c = PeriodicTable().get_element(symbol)
    mc, _, _ = c.get_abundances()
    p = np.array([0.8, 0.2])
    ids._validate_abundance(p, mc, symbol)


def test__validate_abundance_negative_values():
    symbol = "C"
    c = PeriodicTable().get_element(symbol)
    mc, _, _ = c.get_abundances()
    p = np.array([0.8, -0.01])
    with pytest.raises(ValueError):
        ids._validate_abundance(p, mc, symbol)


def test__validate_abundance_non_normalized():
    symbol = "C"
    c = PeriodicTable().get_element(symbol)
    mc, _, _ = c.get_abundances()
    p = np.array([0.8, 0.21])
    with pytest.raises(ValueError):
        ids._validate_abundance(p, mc, symbol)


def test__validate_abundance_invalid_length():
    symbol = "C"
    c = PeriodicTable().get_element(symbol)
    mc, _, _ = c.get_abundances()
    p = np.array([0.8, 0.015, 0.05])
    with pytest.raises(ValueError):
        ids._validate_abundance(p, mc, symbol)


@pytest.mark.parametrize(
    "n_isotopes,n",
    [[1, 1], [1, 2], [1, 5], [1, 10], [2, 1], [2, 5], [2, 20], [5, 1], [5, 10]]
)
def test__find_n_isotopes_combination(n_isotopes, n):
    comb = ids._find_n_isotope_combination(n_isotopes, n)
    expected = [x for x in product(range(n + 1), repeat=n_isotopes) if sum(x) == n]
    expected = np.array(expected)
    # check that the row content is equal
    for x in expected:
        assert x in comb
    for x in comb:
        assert x in expected


@pytest.mark.parametrize(
    "element,max_length",
    product(["C", "S"], [2, 5, 10]))
def test__get_n_atoms_envelope_aux_n_1(element: str, max_length: int):
    element = PeriodicTable().get_element(element)
    me, Me, pe = element.get_abundances()
    M, p = ids._get_n_atoms_envelope_aux(me, Me, pe, 1, max_length)
    Me, pe = ids._fill_missing_nominal(me, Me, pe, max_length)
    assert np.allclose(M, Me)
    assert np.allclose(p, pe / np.sum(pe))


def test__get_n_atoms_envelope_aux_c_n_3_max_length_3():
    element = PeriodicTable().get_element("C")
    m_c12 = 12
    m_c13 = element.isotopes[13].m
    me, Me, pe = element.get_abundances()
    n = 3
    max_length = 3
    M, p = ids._get_n_atoms_envelope_aux(me, Me, pe, n, max_length)
    M_expected = np.array([3 * m_c12, 2 * m_c12 + m_c13, 12 + 2 * m_c13])
    assert np.allclose(M, M_expected)
    assert np.allclose(np.sum(pe), 1.0)


def test__get_n_atoms_envelope_aux_c_n_3_max_length_5():
    element = PeriodicTable().get_element("C")
    m_c12 = 12
    m_c13 = element.isotopes[13].m
    me, Me, pe = element.get_abundances()
    n = 3
    max_length = 5
    M, p = ids._get_n_atoms_envelope_aux(me, Me, pe, n, max_length)
    M_expected = np.array([3 * m_c12, 2 * m_c12 + m_c13, 12 + 2 * m_c13, 3 * m_c13, 0])
    assert np.allclose(M, M_expected)
    assert np.allclose(np.sum(pe), 1.0)


def test__get_n_atoms_envelope_aux_s_n_2_max_length_3():
    element = PeriodicTable().get_element("S")
    me, Me, pe = element.get_abundances()
    n = 2
    max_length = 3
    M, p = ids._get_n_atoms_envelope_aux(me, Me, pe, n, max_length)
    assert np.array_equal(M.round().astype(int), np.array([64, 65, 66]))
    assert np.allclose(np.sum(pe), 1.0)


def test__get_n_atoms_envelope_aux_s_n_2_max_length_10():
    element = PeriodicTable().get_element("S")
    me, Me, pe = element.get_abundances()
    n = 2
    max_length = 10
    M, p = ids._get_n_atoms_envelope_aux(me, Me, pe, n, max_length)
    M_rounded = np.array([64, 65, 66, 67, 68, 69, 70,  0, 72, 0])
    assert np.array_equal(M.round().astype(int), M_rounded)
    assert np.allclose(np.sum(pe), 1.0)


def test__get_n_atoms_envelope():
    element = PeriodicTable().get_element("C")
    c12 = element.isotopes[12]
    me, Me, pe = element.get_abundances()
    M, p = ids._get_n_atoms_envelope(c12, 1, 2)
    assert np.allclose(M, Me)
    assert np.allclose(p, pe)


def test__get_n_atoms_envelope_custom_abundance():
    element = PeriodicTable().get_element("C")
    c12 = element.isotopes[12]
    me, Me, pe = element.get_abundances()
    pe = np.array([0.8, 0.2])
    M, p = ids._get_n_atoms_envelope(c12, 1, 2, p=pe)
    assert np.allclose(M, Me)
    assert np.allclose(p, pe)


def test__fill_missing_nominal_no_fill():
    # carbon element do not need to feel missing values.
    max_length = 5
    m = np.array([24, 25, 26, 0, 0])
    M = np.array([24.1, 24.2, 24.3, 0, 0])
    p = np.array([0.5, 0.3, 0.2, 0, 0])
    M_fill, p_fill = ids._fill_missing_nominal(m, M, p, max_length)
    assert np.allclose(M_fill, M)
    assert np.allclose(p_fill, p)


def test__fill_missing_nominal_fill():
    # Cl  does not have an M + 1 isotope and must be filled.
    max_length = 5
    m = np.array([105, 107, 109])
    M = np.array([105.1, 107.2, 109.3])
    p = np.array([0.5, 0.3, 0.2])
    M_fill, p_fill = ids._fill_missing_nominal(m, M, p, max_length)
    M_expected = np.array([M[0], 0, M[1], 0, M[2]])
    p_expected = np.array([p[0], 0, p[1], 0, p[2]])
    assert np.allclose(M_fill, M_expected)
    assert np.allclose(p_fill, p_expected)


def test__combine_envelopes_one_row_array():
    c12 = PeriodicTable().get_isotope("12C")
    max_length = 10
    n1 = 2
    n2 = 5
    n = n1 + n2
    M1, p1 = ids._get_n_atoms_envelope(c12, n1, max_length)
    M1 = M1.reshape((1, M1.size))
    p1 = p1.reshape((1, p1.size))
    M2, p2 = ids._get_n_atoms_envelope(c12, n2, max_length)
    M2 = M2.reshape((1, M1.size))
    p2 = p2.reshape((1, p1.size))
    M, p = ids.combine_envelopes(M1, p1, M2, p2)
    M_expected, p_expected = ids._get_n_atoms_envelope(c12, n, max_length)
    M_expected = M_expected.reshape((1, M_expected.size))
    p_expected = p_expected.reshape((1, p_expected.size))
    assert np.allclose(M, M_expected)
    assert np.allclose(p, p_expected)


def test__combine_envelopes_multiple_row_array():
    c12 = PeriodicTable().get_isotope("12C")
    n_rep = 5
    max_length = 10
    n1 = 2
    n2 = 5
    n = n1 + n2
    M1, p1 = ids._get_n_atoms_envelope(c12, n1, max_length)
    M1 = np.tile(M1, (n_rep, 1))
    p1 = np.tile(p1, (n_rep, 1))
    M2, p2 = ids._get_n_atoms_envelope(c12, n2, max_length)
    M2 = np.tile(M2, (n_rep, 1))
    p2 = np.tile(p2, (n_rep, 1))
    M, p = ids.combine_envelopes(M1, p1, M2, p2)
    M_expected, p_expected = ids._get_n_atoms_envelope(c12, n, max_length)
    M_expected = np.tile(M_expected, (n_rep, 1))
    p_expected = np.tile(p_expected, (n_rep, 1))
    assert np.allclose(M, M_expected)
    assert np.allclose(p, p_expected)


def test_find_formula_abundances():
    f = Formula("CO2")
    max_length = 10
    ids.find_formula_envelope(f.composition, max_length)
