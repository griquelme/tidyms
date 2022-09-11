from tidyms.chem import mmi_finder
from tidyms.chem import PTABLE
import pytest
import numpy as np


@pytest.mark.parametrize("e2", ["Na", "P"])
def test__is_distort_envelope_e2_monoisotope(e2):
    e1 = PTABLE["C"]
    e2 = PTABLE[e2]
    assert not mmi_finder._is_distort_envelope(e1, e2)


@pytest.mark.parametrize("e2", ["O", "S"])
def test__is_distort_envelope_different_n_isotopes(e2):
    e1 = PTABLE["C"]
    e2 = PTABLE[e2]
    assert mmi_finder._is_distort_envelope(e1, e2)


@pytest.mark.parametrize("e2", ["Cl", "Br"])
def test__is_distort_envelope_different_nominal_mass_increments(e2):
    e1 = PTABLE["C"]
    e2 = PTABLE[e2]
    assert mmi_finder._is_distort_envelope(e1, e2)
    

@pytest.mark.parametrize("e2", ["H", "N"])
def test__is_distort_envelope_equal_envelopes_e2_dont_distort(e2):
    e1 = PTABLE["C"]
    e2 = PTABLE[e2]
    assert not mmi_finder._is_distort_envelope(e1, e2)


@pytest.mark.parametrize("e2", ["B", "Li"])
def test__is_distort_envelope_equal_envelopes_e2_distort(e2):
    e1 = PTABLE["C"]
    e2 = PTABLE[e2]
    assert mmi_finder._is_distort_envelope(e1, e2)


def test__get_relevant_elements():
    e_list = ["C", "H", "N", "O", "P", "S", "Li", "Na"]
    e_list = [PTABLE[x] for x in e_list]
    expected = ["C", "O", "S", "Li"]
    expected = [PTABLE[x] for x in expected]
    res = mmi_finder._get_relevant_elements(e_list)
    assert set(res) == set(expected)


@pytest.mark.parametrize("max_charge", [1, 2, 3, 4])
def test__get_monoisotopic_mass_candidates_positive_max_charge(max_charge):
    max_mass = 2000.0
    mono_M = 250.0
    polarity = 1
    # all possible values should be lower than max_mass
    expected_charge = np.arange(1, abs(max_charge) + 1)
    expected_M = expected_charge * mono_M
    test_M, test_charge = mmi_finder._get_valid_mono_mass(
        mono_M, max_charge, polarity, max_mass)
    assert np.allclose(expected_M, test_M)
    assert np.array_equal(expected_charge, test_charge)


@pytest.mark.parametrize("max_charge", [1, 2, 3, 4])
def test__get_monoisotopic_mass_candidates_negative_max_charge(max_charge):
    max_mass = 2000.0
    mono_M = 250.0
    polarity = -1
    # all possible values should be lower than max_mass
    expected_charge = np.arange(1, abs(max_charge) + 1)
    expected_M = np.abs(expected_charge) * mono_M
    test_M, test_charge = mmi_finder._get_valid_mono_mass(
        mono_M, max_charge, polarity, max_mass)
    assert np.allclose(expected_M, test_M)
    assert np.array_equal(expected_charge, test_charge)


def test__get_monoisotopic_mass_candidates_mono_mass_greater_than_max_mass():
    max_mass = 2000.0
    mono_M = 900.0
    # valid charges should be 1 and 2.
    max_charge = 3
    expected_charge = np.arange(1, 3)
    polarity = 1
    expected_M = expected_charge * mono_M
    test_M, test_charge = mmi_finder._get_valid_mono_mass(
        mono_M, max_charge, polarity, max_mass)
    assert np.allclose(expected_M, test_M)
    assert np.array_equal(expected_charge, test_charge)


@pytest.fixture
def rules():
    bounds = {
        "C": (0, 108),
        "H": (0, 100),
        "S": (0, 8),
        "Cl": (0, 2)
    }
    max_mass = 2000.0
    length = 5
    bin_size = 100
    p_tol = 0.05
    r = mmi_finder._create_rules_dict(bounds, max_mass, length, bin_size, p_tol)
    return r, max_mass, length, bin_size


def test__find_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    _, M_cl, _ = PTABLE["Cl"].get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    charge = 1
    mono_index = 3
    mz = np.array([100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0])
    sp = np.array([100.0, 200.0, 500.0, 501.0, 34.0])
    mono_sp = sp[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05

    test_candidates = mmi_finder._find_candidate(
        mz, sp, mono_mz, charge, mono_sp, i_rules, mz_tol, p_tol)
    expected_candidates = [(2, 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_multiple_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    _, M_cl, _ = PTABLE["Cl"].get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    charge = 1
    mono_index = 4
    M01 = mono_mz - dm_cl
    M02 = M01 + 0.00001
    mz = np.array([100.0, 300.0, M01, M02, mono_mz, 456.0])
    sp = np.array([100.0, 200.0, 500.0, 500.5, 501.0, 34.0])
    mono_sp = sp[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05

    test_candidates = mmi_finder._find_candidate(
        mz, sp, mono_mz, charge, mono_sp, i_rules, mz_tol, p_tol)
    expected_candidates = [(2, 1), (3, 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_no_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    _, M_cl, _ = PTABLE["Cl"].get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    charge = 1
    mono_index = 2
    mz = np.array([100.0, 300.0, mono_mz, 456.0])
    sp = np.array([100.0, 200.0, 501.0, 34.0])
    mono_sp = sp[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05

    test_candidates = mmi_finder._find_candidate(
        mz, sp, mono_mz, charge, mono_sp, i_rules, mz_tol, p_tol)
    assert len(test_candidates) == 0

def test_MMIFinder():
    bounds = {
        "C": (0, 108),
        "H": (0, 100),
        "S": (0, 8),
        "Cl": (0, 2)
    }
    max_mass = 2000.0
    length = 5
    bin_size = 100
    max_charge = 3
    mz_tol = 0.005
    p_tol = 0.05
    finder = mmi_finder.MMIFinder(
        bounds, max_mass, max_charge, length, bin_size, mz_tol, p_tol)

    _, M_cl, _ = PTABLE["Cl"].get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    mono_index = 3
    mz = np.array([100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0])
    sp = np.array([100.0, 200.0, 500.0, 501.0, 34.0])
    test_mmi_index = finder.find(mz, sp, mono_index)
    expected_mmi_index = [(3, 1), (3, 2), (3, 3), (2, 1)]
    assert test_mmi_index == expected_mmi_index
