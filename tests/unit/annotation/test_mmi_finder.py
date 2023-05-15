from tidyms.annotation import mmi_finder
from tidyms.annotation.annotation_data import AnnotationData
from tidyms.chem import PeriodicTable
from tidyms.lcms import LCTrace, Peak
import pytest
import numpy as np
from typing import Sequence


def test__select_two_isotope_elements_dm_1_p0_greater_than_pi():
    elements = ["C", "H", "N", "O", "P", "S"]
    expected = ["C"]
    custom_abundances = dict()
    dm = 1
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_1_p0_greater_than_pi_custom_abundance():
    elements = ["C", "H", "N", "O", "P", "S"]
    expected = ["H"]
    custom_abundances = {"H": np.array([0.95, 0.05])}
    dm = 1
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_1_no_elements():
    elements = ["O", "P", "S"]
    custom_abundances = {}
    dm = 1
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == 0


def test__select_two_isotope_elements_dm_1_p0_lower_than_pi():
    elements = ["B", "Li", "O", "P", "S"]
    expected = ["B", "Li"]
    dm = 1
    custom_abundances = dict()
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_1_p0_lower_and_higher_than_pi():
    elements = ["C", "H", "B", "Li", "O", "P", "S"]
    expected = ["C", "B", "Li"]
    dm = 1
    custom_abundances = dict()
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_2_p0_greater_than_pi():
    elements = ["Cl", "H", "N", "O", "P", "S"]
    expected = ["Cl"]
    custom_abundances = dict()
    dm = 2
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_2_p0_greater_than_pi_custom_abundance():
    elements = ["Cl", "Br", "N", "O", "P", "S"]
    expected = ["Cl"]
    # Br abundance adjusted to force the result to be Cl
    custom_abundances = {"Br": np.array([0.9, 0.1])}
    dm = 2
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_2_no_elements():
    elements = ["O", "P", "S"]
    custom_abundances = {}
    dm = 2
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == 0


def test__select_two_isotope_elements_dm_2_p0_lower_than_pi():
    elements = ["In", "H", "O", "P", "S"]
    expected = ["In"]
    dm = 2
    custom_abundances = dict()
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_two_isotope_elements_dm_2_p0_lower_and_higher_than_pi():
    elements = ["Cl", "In", "Br", "O", "P", "S"]
    expected = ["Br", "In"]
    dm = 2
    custom_abundances = dict()
    res = mmi_finder._select_two_isotope_element(elements, dm, custom_abundances)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_multiple_isotope_elements():
    elements = ["Cl", "H", "N", "O", "P", "S"]
    expected = ["O", "S"]
    res = mmi_finder._select_multiple_isotope_elements(elements)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


def test__select_multiple_isotope_elements_no_elements():
    elements = ["Cl", "H", "N", "P"]
    expected = []
    res = mmi_finder._select_multiple_isotope_elements(elements)
    assert len(res) == len(expected)
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "elements,expected",
    [
        [["C", "H", "N", "O", "P", "S"], ["C", "O", "S"]],
        [["C", "H", "N", "O", "P", "S", "Cl", "Li", "Na"], ["C", "O", "S", "Li", "Cl"]],
    ],
)
def test__select_elements(elements, expected):
    res = mmi_finder._select_elements(elements)
    res = [x.symbol for x in res]
    assert len(res) == len(expected)
    assert set(res) == set(expected)


@pytest.fixture
def rules():
    bounds = {"C": (0, 108), "H": (0, 100), "S": (0, 8), "Cl": (0, 2)}
    max_mass = 2000.0
    length = 5
    bin_size = 100
    p_tol = 0.05
    r = mmi_finder._create_rules_dict(bounds, max_mass, length, bin_size, p_tol, None)
    return r, max_mass, length, bin_size


def create_peak_list(mz: list[float], sp: list[float]) -> Sequence[Peak]:
    peak_list = list()
    size = 30
    time = np.linspace(0, size, size)
    scan = np.arange(size)
    spint = np.ones(size)
    noise = np.zeros_like(time)
    for k_mz, k_sp in zip(mz, sp):
        roi = LCTrace(time.copy(), spint * k_sp, spint * k_mz, scan, noise, noise)
        peak = Peak(10, 15, 20, roi)
        peak_list.append(peak)
    return peak_list


def test__find_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.

    _, M_cl, _ = PeriodicTable().get_element("Cl").get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    charge = 1
    mono_index = 3
    mz = [100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 501.0, 34.0]
    peak_list = create_peak_list(mz, sp)
    monoisotopologue = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05
    min_similarity = 0.9

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(
        data, monoisotopologue, charge, i_rules, mz_tol, p_tol, max_mass, min_similarity
    )
    mmi = peak_list[2]
    expected_candidates = [(mmi, 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_multiple_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    _, M_cl, _ = PeriodicTable().get_element("Cl").get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    charge = 1
    mono_index = 4
    M01 = mono_mz - dm_cl
    M02 = M01 + 0.00001
    mz = [100.0, 300.0, M01, M02, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 500.5, 501.0, 34.0]
    peak_list = create_peak_list(mz, sp)
    monoisotopologue = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05
    min_similarity = 0.9

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(
        data, monoisotopologue, charge, i_rules, mz_tol, p_tol, max_mass, min_similarity
    )
    expected_candidates = [(peak_list[2], 1), (peak_list[3], 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_no_candidates(rules):
    rules, max_mass, length, bin_size = rules
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    _, M_cl, _ = PeriodicTable().get_element("Cl").get_abundances()
    mono_mz = 400.0
    charge = 1
    mono_index = 2
    mz = [100.0, 300.0, mono_mz, 456.0]
    sp = [100.0, 200.0, 501.0, 34.0]
    peak_list = create_peak_list(mz, sp)
    monoisotopologue = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // bin_size)
    i_rules = rules.get(m_bin)[0]
    mz_tol = 0.005
    p_tol = 0.05
    min_similarity = 0.9

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(
        data, monoisotopologue, charge, i_rules, mz_tol, p_tol, max_mass, min_similarity
    )
    assert len(test_candidates) == 0


def test_MMIFinder():
    bounds = {"C": (0, 108), "H": (0, 100), "S": (0, 8), "Cl": (0, 2)}
    max_mass = 2000.0
    length = 5
    bin_size = 100
    max_charge = 3
    mz_tol = 0.005
    p_tol = 0.05
    min_similarity = 0.9
    finder = mmi_finder.MMIFinder(
        bounds, max_mass, max_charge, length, bin_size, mz_tol, p_tol, min_similarity
    )

    _, M_cl, _ = PeriodicTable().get_element("Cl").get_abundances()
    dm_cl = M_cl[1] - M_cl[0]
    mono_mz = 400.0
    mz = [100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 501.0, 34.0]
    peak_list = create_peak_list(mz, sp)
    data = AnnotationData(peak_list)
    monoisotopologue = data.get_monoisotopologue()
    test_mmi_index = finder.find(data)
    expected_mmi_index = [
        (monoisotopologue, 1),
        (monoisotopologue, 2),
        (monoisotopologue, 3),
        (peak_list[2], 1),
    ]
    # check with set because features may be in a different order
    assert set(test_mmi_index) == set(expected_mmi_index)
