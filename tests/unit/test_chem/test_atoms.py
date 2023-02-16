from tidyms.chem import atoms
import pytest


def test_PeriodicTable_get_element_from_symbol():
    ptable = atoms.PeriodicTable()
    c = ptable.get_element("C")
    assert c.z == 6
    assert c.symbol == "C"


def test_PeriodicTable_get_element_from_z():
    ptable = atoms.PeriodicTable()
    p = ptable.get_element(15)
    assert p.symbol == "P"
    assert p.z == 15


def test_PeriodicTable_get_isotope_from_symbol():
    ptable = atoms.PeriodicTable()
    cl37 = ptable.get_isotope("37Cl")
    assert cl37.a == 37
    assert cl37.get_symbol() == "Cl"


def test_PeriodicTable_get_isotope_copy():
    ptable = atoms.PeriodicTable()
    isotope_str = "37Cl"
    cl37_copy = ptable.get_isotope(isotope_str, copy=True)
    cl37 = ptable.get_isotope(isotope_str)
    assert cl37.a == cl37_copy.a
    assert cl37.m == cl37_copy.m
    assert cl37.z == cl37_copy.z
    assert cl37 is not cl37_copy


@pytest.mark.parametrize(
    "z,a,m,abundance,expected_symbol",
    [
        [6, 12, 12.0, 0.9, "C"],    # Carbon. Dummy abundances and exact mass are used.
        [1, 1, 1.0078, 0.9, "H"],   # Hydrogen
        [15, 31, 30.099, 1.0, "P"]  # Phosphorus
    ]
)
def test_Isotope_get_symbol(z, a, m, abundance, expected_symbol):
    isotope = atoms.Isotope(z, a, m, abundance)
    assert isotope.get_symbol() == expected_symbol


def test_Element_get_monoisotope():
    element = atoms.PeriodicTable().get_element("B")
    monoisotope = element.get_monoisotope()
    assert monoisotope.a == 11


def test_Element_get_mmi():
    element = atoms.PeriodicTable().get_element("B")
    mmi = element.get_mmi()
    assert mmi.a == 10
