from tidyms.chem import atoms
import numpy as np
import pytest


@pytest.fixture
def isotope_data():
    isotopes = [atoms.PTABLE["C"].isotopes[12],
                atoms.PTABLE["H"].isotopes[2],
                atoms.PTABLE["P"].isotopes[31]]
    str_repr = ["12C", "2H", "31P"]
    symbol = ["C", "H", "P"]
    return isotopes, str_repr, symbol


# Test Isotope


def test_get_symbol_str(isotope_data):
    isotopes, _, symbol = isotope_data
    for i, s in zip(isotopes, symbol):
        assert i.get_symbol() == s


def test_is_most_abundant(isotope_data):
    isotopes, _, _ = isotope_data
    is_most_abundant = [True, False, True]
    for isotopes, expected_result in zip(isotopes, is_most_abundant):
        assert isotopes.is_most_abundant() == expected_result


def test_get_element(isotope_data):
    isotopes, _, _ = isotope_data
    expected_elements = [atoms.PTABLE[x] for x in ["C", "H", "P"]]
    for i, e in zip(isotopes, expected_elements):
        assert i.get_element() is e

# Test Element


@pytest.fixture
def element_data():
    elements = [atoms.PTABLE["C"], atoms.PTABLE["H"], atoms.PTABLE["P"]]
    symbols = ["C", "H", "P"]
    return elements, symbols


def test_element_repr(element_data):
    elements, expected_symbol = element_data
    for e, s in zip(elements, expected_symbol):
        assert e.symbol == s


def test_get_abundance(element_data):
    elements, _ = element_data
    for e in elements:
        nominal, exact, abundance = e.get_abundances()
        assert np.isclose(abundance.sum(), 1.0)
        assert (np.abs(nominal - exact) < 1).all()


def test_get_most_abundant_isotope(element_data):
    elements, _ = element_data
    expected_nominal_mass = [12, 1, 31]
    for e, em in zip(elements, expected_nominal_mass):
        i = e.get_most_abundant_isotope()
        assert i.a == em


def test_find_isotopes(isotope_data):
    isotope, isotope_str, _ = isotope_data
    for i, i_str in zip(isotope, isotope_str):
        assert atoms.find_isotope(i_str) is i


def test_find_isotopes_bad_input():
    with pytest.raises(ValueError):
        atoms.find_isotope("AA")
