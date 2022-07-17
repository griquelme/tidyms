import pytest
from tidyms.chem import _isotope_distributions as ids
from tidyms.chem.atoms import find_isotope
import numpy as np
from itertools import product


@pytest.mark.parametrize("isotope", ["12C", "35Cl", "1H", "2H", "13C", "10B"])
def test_find_n_atoms_abundances_zero_atoms(isotope):
    isotope = find_isotope(isotope)
    length = 5
    nom, exact, ab = ids._find_n_atoms_abundances(isotope, 0, length)
    expected_nominal = np.zeros(length, dtype=int)
    expected_exact = np.zeros(length)
    expected_abundance = np.zeros_like(expected_exact)
    expected_abundance[0] = 1
    assert np.array_equal(nom, expected_nominal)
    assert np.allclose(exact, expected_exact)
    assert np.allclose(ab, expected_abundance)

comb = list(product(["12C", "1H", "35Cl"], [1, 3, 5, 7]))


@pytest.mark.parametrize("isotope,n", comb)
def test_find_n_atoms_abundance_one_atom_most_abundant_isotope(isotope, n):
    isotope = find_isotope(isotope)
    element = isotope.get_element()
    length = 15
    test_nom, test_exact, test_ab = \
        ids._find_n_atoms_abundances(isotope, n, length)
    element_nom, element_exact, element_ab = element.get_abundances()

    dexact = element_exact[1] - element_exact[0]
    dnom = element_nom[1] - element_nom[0]
    # nominal mass values that are not possible (eg. M + 1 with Cl)
    # are set to zero. Also if the array was extended to match length
    # the additional elements are set to zero.
    expected_exact = element_exact[0] * n + np.arange(0, length) * dexact / dnom
    expected_nom = element_nom[0] * n + np.arange(0, length)
    set_to_zero_mask = ~np.isin(np.arange(length), np.arange(0, length, dnom))
    expected_exact[set_to_zero_mask] = 0.0
    expected_nom[set_to_zero_mask] = 0
    if n * dnom < length:
        expected_nom[n * dnom + 1:] = 0
        expected_exact[n * dnom + 1:] = 0.0


    assert np.array_equal(test_nom, expected_nom)
    assert np.allclose(test_exact, expected_exact)
    assert np.isclose(test_ab.sum(), 1.0)


@pytest.mark.parametrize("isotope", ["13C", "37Cl", "2H", "10B"])
def test_find_n_atoms_abundance_one_atom(isotope):
    isotope = find_isotope(isotope)
    n_atoms = 1
    length = 5
    test_nom, test_exact, test_ab = \
        ids._find_n_atoms_abundances(isotope, n_atoms, length)
    exp_nom = np.zeros(length, dtype=int)
    exp_nom[0] = isotope.a
    exp_exact = np.zeros(length, dtype=float)
    exp_exact[0] = isotope.m
    exp_ab = np.zeros(length, dtype=float)
    exp_ab[0] = 1.0
    assert np.array_equal(test_nom[:exp_nom.size], exp_nom)
    assert np.allclose(test_exact[:exp_nom.size], exp_exact)
    assert np.allclose(test_ab[:exp_nom.size], exp_ab)


def test_find_n_atoms_abundance_custom_abundance():
    isotope = find_isotope("12C")
    custom_ab = np.array([0.9, 0.1])
    element = isotope.get_element()
    n = 4
    length = 5
    test_nom, test_exact, test_ab = \
        ids._find_n_atoms_abundances(isotope, n, length, abundance=custom_ab)
    element_nom, element_exact, _ = element.get_abundances()

    dexact = element_exact[1] - element_exact[0]
    dnom = element_nom[1] - element_nom[0]
    # nominal mass values that are not possible (eg. M + 1 with Cl)
    # are set to zero. Also if the array was extended to match length
    # the additional elements are set to zero.
    expected_exact = element_exact[0] * n + np.arange(0, length) * dexact / dnom
    expected_nom = element_nom[0] * n + np.arange(0, length)
    set_to_zero_mask = ~np.isin(np.arange(length), np.arange(0, length, dnom))
    expected_exact[set_to_zero_mask] = 0.0
    expected_nom[set_to_zero_mask] = 0
    if n * dnom < length:
        expected_nom[n * dnom + 1:] = 0
        expected_exact[n * dnom + 1:] = 0.0

    assert np.array_equal(test_nom, expected_nom)
    assert np.allclose(test_exact, expected_exact)
    assert np.isclose(test_ab.sum(), 1.0)

@pytest.mark.parametrize("isotope,n", comb)
def test_combine_element_abundance(isotope, n):
    # combine equal abundances values should be equivalent to compute
    # the abundances using find_n_atoms_abundances using 2 * n
    length = 5
    min_p = 1e-10
    isotope = find_isotope(isotope)
    expected_nom, expected_exact, expected_ab = \
        ids._find_n_atoms_abundances(isotope, 2 * n, length)

    expected_ab[expected_ab < min_p] = 0.0
    expected_exact[expected_ab < min_p] = 0.0
    expected_nom[expected_ab < min_p] = 0

    nom1, ex1, ab1 = ids._find_n_atoms_abundances(isotope, n, length)
    test_nom, test_exact, test_ab = \
        ids._combine_element_abundances(nom1, ex1, ab1, nom1, ex1, ab1,
                                        min_p=min_p)

    assert np.allclose(expected_nom, test_nom)
    assert np.allclose(expected_exact, test_exact)
    assert np.allclose(expected_ab, test_ab)

@pytest.mark.parametrize("n_min,n_max", [[0, 0], [1, 1], [0, 1], [0, 5],
                                         [10, 20]])
def test_make_element_envelope_array(n_min, n_max):
    isotope = find_isotope("12C")
    element = isotope.get_element()
    nom = element.nominal_mass
    length = 5
    test_nom, test_ex, test_ab = \
        ids._make_element_abundance_array(isotope, n_min, n_max, length)

    n_array = np.arange(n_min, n_max + 1)
    assert np.array_equal(n_array * nom, test_nom[:, 0])


@pytest.mark.parametrize("isotope,n_min,n_max", [["12C", 0, 0], ["12C", 0, 1],
                                                 ["1H", 0, 3], ["11B", 0, 5],
                                                 ["12C", 10, 20], ["32S", 0, 5]]
                         )
def test_combine_array_abundance(isotope, n_min, n_max):
    # combine equal abundances values should be equivalent to compute
    # the abundances using make_elements_abundance_array using 2 * n_min and
    # 2 * n_max
    length = 5
    min_p = 1e-10
    isotope = find_isotope(isotope)
    expected_nom, expected_exact, expected_ab = \
        ids._make_element_abundance_array(isotope, n_min * 2, n_max * 2, length)

    # remove odd values that are not computed when combining arrays
    expected_ab = expected_ab[::2, :]
    expected_nom = expected_nom[::2, :]
    expected_exact = expected_exact[::2, :]

    expected_ab[expected_ab < min_p] = 0.0
    expected_exact[expected_ab < min_p] = 0.0
    expected_nom[expected_ab < min_p] = 0

    nom1, ex1, ab1 = ids._make_element_abundance_array(isotope, n_min, n_max,
                                                       length)
    test_nom, test_exact, test_ab = \
        ids._combine_array_abundances(nom1, ex1, ab1, nom1, ex1, ab1,
                                      min_p=min_p)

    assert np.allclose(expected_nom, test_nom)
    assert np.allclose(expected_exact, test_exact)
    assert np.allclose(expected_ab, test_ab)
