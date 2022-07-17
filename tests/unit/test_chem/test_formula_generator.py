"""
tests for formula module from formula-generator
"""
import tidyms.chem.formula_generator as fg
from tidyms.chem import atoms
from tidyms.chem import Formula
import pytest
from math import isclose
import numpy as np

invalid_tolerance_input = [(0, "ppm"), (-0.001, "da"), (0.005, "bad-unit")]


@pytest.mark.parametrize("invalid_values,invalid_units",
                         invalid_tolerance_input)
def test_tolerance_invalid_input(invalid_values, invalid_units):
    with pytest.raises((ValueError, TypeError)):
        fg._Tolerance(invalid_values, invalid_units)


tolerance_test_input = [(0.001, "da", None, 0.001),
                        (0.001, "da", 350, 0.001),
                        (5.2, "mda", None, 0.0052),
                        (5.2, "mda", 350, 0.0052)]


@pytest.mark.parametrize("values, units, mass, expected", tolerance_test_input)
def test_get_absolute_tolerance_da_and_mda(values, units, mass, expected):
    tol = fg._Tolerance(values, units).get_abs_tolerance(mass)
    assert isclose(tol, expected)


def test_get_absolute_tolerance():
    ppm = 5
    mass = 200
    tol = fg._Tolerance(ppm, "ppm")
    expected = ppm * mass / 1e6
    assert isclose(tol.get_abs_tolerance(mass), expected)


@pytest.mark.parametrize("mass", [None, "asd"])
def test_invalid_ppm_mass(mass):
    tol = fg._Tolerance(5, "ppm")
    with pytest.raises(TypeError):
        tol.get_abs_tolerance(mass)


# Tests for _Bounds


@pytest.fixture
def chnops_bounds_params():
    isotopes_str = ["C", "H", "N", "O", "P", "S"]
    isotopes = [atoms.find_isotope(x) for x in isotopes_str]
    mass = 300
    # s = atoms.find_isotope("S")
    # mass based bounds for each isotope
    bounds_list = [(0, int(mass / x.m) + 1) for x in isotopes]
    valid_params = {"mass": mass,
                    "isotopes_str": isotopes_str,
                    "isotopes": isotopes,
                    "bounds_list": bounds_list,
                    "user_bounds":  ("S", (0, 5))}
    return valid_params


@pytest.fixture
def chnops_bounds(chnops_bounds_params):
    mass = chnops_bounds_params["mass"]
    isotopes = chnops_bounds_params["isotopes"]
    bounds_list = chnops_bounds_params["bounds_list"]
    return fg.Bounds(isotopes, bounds_list, mass)


@pytest.fixture
def chn_bounds_params():
    isotopes_str = ["C", "H", "N"]
    isotopes = [atoms.find_isotope(x) for x in isotopes_str]
    mass = 300
    isotope = atoms.find_isotope("N")
    # mass based bounds for each isotope
    bounds_list = [(0, int(mass / x.m) + 1) for x in isotopes]
    valid_params = {"mass": mass,
                    "isotope_str": isotopes_str,
                    "isotopes": isotopes,
                    "bounds_list": bounds_list,
                    "user_bounds":  (isotope, (0, 5))}
    return valid_params


@pytest.fixture
def chn_bounds(chn_bounds_params):
    mass = chn_bounds_params["mass"]
    isotopes = chn_bounds_params["isotopes"]
    bounds_list = chn_bounds_params["bounds_list"]
    return fg.Bounds(isotopes, bounds_list, mass)


def test_bounds_constructor(chnops_bounds_params):
    mass = chnops_bounds_params["mass"]
    isotopes = chnops_bounds_params["isotopes"]
    bounds_list = chnops_bounds_params["bounds_list"]
    bounds = fg.Bounds(isotopes, bounds_list, mass)
    assert True


@pytest.mark.parametrize("lb,ub", [[0, 0], [0, 10], [0, 2], [1, 2], [2, 2]])
def test_validate_bounds_valid_bounds(lb, ub):
    fg._validate_bounds(lb, ub)
    assert True


@pytest.mark.parametrize("lb,ub", [[0.3, 1], [-1, 5], [3, 2], [0, 0.3]])
def test_validate_bounds_invalid_bounds(lb, ub):
    with pytest.raises(fg.InvalidBound):
        fg._validate_bounds(lb, ub)


def test_bounds_update_bounds(chnops_bounds_params, chnops_bounds):
    bounds = chnops_bounds
    isotope, user_bounds = chnops_bounds_params["user_bounds"]
    isotope = atoms.find_isotope(isotope)
    bounds.update_bounds(isotope, *user_bounds)
    assert user_bounds == bounds[isotope]


@pytest.mark.parametrize("n_cluster", [1, 3, 5, 10])
def test_bounds_refine_h_bounds(chnops_bounds, n_cluster):
    bounds = chnops_bounds
    bounds.refine_h_upper_bounds(n_cluster=n_cluster)
    h = atoms.find_isotope("1H")
    ub_expected = int(bounds.mass / 7) + 1 + 2 * n_cluster
    assert ub_expected == bounds[h][1]


def test_bounds_split_pos_neg(chnops_bounds, chn_bounds):
    # also test with chn_bounds to evaluate the case where isotopes with
    # positive mass defect are present
    for bounds in [chnops_bounds, chn_bounds]:
        pos, neg = bounds.split_pos_neg()
        # isotopes in pos has positive mass defect and that the bounds are the
        # same
        for p in pos:
            assert p.defect > 0
            assert pos[p] == bounds[p]

        for n in neg:
            assert n.defect < 0
            assert neg[n] == bounds[n]

        # 12C is not in neither por or neg
        c12 = atoms.find_isotope("12C")
        for isotope in bounds:
            if not (isotope is c12):
                assert ((isotope in pos) or (isotope in neg))


@pytest.mark.parametrize("m", [100, 200, 300])
def test_bounds_get_mass_query_bounds(chnops_bounds, m):
    bounds = chnops_bounds
    m_bounds = bounds.get_mass_query_bounds(m)
    # check upper bounds for each isotope
    for i, (_, ub) in m_bounds.items():
        assert ub <= bounds[i][1]


def test_bounds_from_isotope_string(chnops_bounds_params, chnops_bounds):
    isotopes_str = chnops_bounds_params["isotopes_str"]
    mass = chnops_bounds_params["mass"]
    test_bounds = fg.Bounds.from_isotope_str(isotopes_str, mass)
    for i in test_bounds:
        assert test_bounds[i] == chnops_bounds[i]


def test_bounds_from_isotope_string_with_user_bounds(chnops_bounds,
                                                     chnops_bounds_params):
    isotopes_str = chnops_bounds_params["isotopes_str"]
    mass = chnops_bounds_params["mass"]
    user_i, i_bounds = chnops_bounds_params["user_bounds"]
    user_bounds = {user_i: i_bounds}
    test_bounds = fg.Bounds.from_isotope_str(isotopes_str, mass,
                                             user_bounds=user_bounds)
    for i in test_bounds:
        if i == atoms.find_isotope(user_i):
            assert i_bounds == test_bounds[i]
        else:
            assert test_bounds[i] == chnops_bounds[i]


def test_bounds_from_isotope_str_invalid_isotope():
    isotopes = ["bad-isotope", "C", "H"]
    m = 200
    with pytest.raises(ValueError):
        fg.Bounds.from_isotope_str(isotopes, m)


def test_bounds_from_isotope_str_invalid_mass():
    isotopes = ["C", "H", "O"]
    m = -50
    with pytest.raises(ValueError):
        fg.Bounds.from_isotope_str(isotopes, m)


# test mass query

@pytest.mark.parametrize("m", [0, -0.5, "bad-value"])
def test_mass_query_invalid_mass(chnops_bounds, m):
    tolerance = 0.001
    with pytest.raises(fg.InvalidMass):
        fg._MassQuery(m, tolerance, chnops_bounds)


@pytest.fixture
def formula_generator_params():
    valid_params = {"elements": ["C", "H", "N", "O", "P"],
                    "mass": 170,
                    "tolerance": 0.001,
                    "tolerance_units": "da",
                    "min_defect": 0,
                    "max_defect": 0.5,
                    "refine_h": True}
    return valid_params


# Test correctness of the algorithm checking that the set of formulas obtained
# is the same as the set obtained using a brute force approach.

formula_str_list = ["C10H10O2", "C9H14", "C7H8N2"]


@pytest.mark.parametrize("formula_str", formula_str_list)
def test_bruteforce(formula_str, formula_generator_params):
    formula = Formula(formula_str)
    mass = formula.get_exact_mass()
    # parameters
    elements = formula_generator_params["elements"]
    max_mass = formula_generator_params["mass"]
    tol = formula_generator_params["tolerance"]
    units = formula_generator_params["tolerance_units"]
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.Bounds.from_isotope_str(elements, max_mass + tol)
    bf_coeff = bounds.make_coefficients(False)
    sorted_mass_index = np.argsort(bf_coeff.monoisotopic)
    sorted_monoisotopic = bf_coeff.monoisotopic[sorted_mass_index]
    start_valid, end_valid = np.searchsorted(sorted_monoisotopic,
                                             [mass - tol, mass + tol])
    sorted_mass_index = sorted_mass_index[start_valid:end_valid]
    valid_coeff_bf = bf_coeff.coefficients[sorted_mass_index, :]

    # mass defect based solution
    f = fg.FormulaGenerator(elements, max_mass, tol, tolerance_units=units,
                            refine_h=False, min_defect=None)
    f.generate_formulas(mass)
    valid_coeff, valid_elements, valid_monoisotopic = f.results_to_array()
    valid_coeff = valid_coeff[np.argsort(valid_monoisotopic), :]
    assert np.array_equal(valid_coeff, valid_coeff_bf)


def test_formula_generator_only_positive_defect_elements():
    elements = ["C", "H", "N"]
    formula = Formula("C2H3N4")
    mass = formula.get_exact_mass()
    # parameters
    max_mass = 200
    tol = 0.001
    units = "da"
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.Bounds.from_isotope_str(elements, max_mass + tol)
    bf_coeff = bounds.make_coefficients(False)
    sorted_mass_index = np.argsort(bf_coeff.monoisotopic)
    sorted_monoisotopic = bf_coeff.monoisotopic[sorted_mass_index]
    start_valid, end_valid = np.searchsorted(sorted_monoisotopic,
                                             [mass - tol, mass + tol])
    sorted_mass_index = sorted_mass_index[start_valid:end_valid]
    valid_coeff_bf = bf_coeff.coefficients[sorted_mass_index, :]

    # mass defect based solution
    f = fg.FormulaGenerator(elements, max_mass, tol, tolerance_units=units,
                            refine_h=False, min_defect=None)
    f.generate_formulas(mass)
    valid_coeff, valid_elements, valid_monoisotopic = f.results_to_array()
    valid_coeff = valid_coeff[np.argsort(valid_monoisotopic), :]
    assert np.array_equal(valid_coeff, valid_coeff_bf)


def test_formula_generator_only_negative_defect_elements():
    elements = ["C", "S", "O"]
    formula = Formula("CS2")
    mass = formula.get_exact_mass()
    # parameters
    max_mass = 200
    tol = 0.001
    units = "da"
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.Bounds.from_isotope_str(elements, max_mass + tol)
    bf_coeff = bounds.make_coefficients(False)
    sorted_mass_index = np.argsort(bf_coeff.monoisotopic)
    sorted_monoisotopic = bf_coeff.monoisotopic[sorted_mass_index]
    start_valid, end_valid = np.searchsorted(sorted_monoisotopic,
                                             [mass - tol, mass + tol])
    sorted_mass_index = sorted_mass_index[start_valid:end_valid]
    valid_coeff_bf = bf_coeff.coefficients[sorted_mass_index, :]

    # mass defect based solution
    f = fg.FormulaGenerator(elements, max_mass, tol, tolerance_units=units,
                            refine_h=False, min_defect=None)
    f.generate_formulas(mass)
    valid_coeff, valid_elements, valid_monoisotopic = f.results_to_array()
    valid_coeff = valid_coeff[np.argsort(valid_monoisotopic), :]
    assert np.array_equal(valid_coeff, valid_coeff_bf)


def test_formula_generator_no_carbon():
    elements = ["H", "N", "O"]
    formula = Formula("N2H4")
    mass = formula.get_exact_mass()
    # parameters
    max_mass = 200
    tol = 0.001
    units = "da"
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.Bounds.from_isotope_str(elements, max_mass + tol)
    bf_coeff = bounds.make_coefficients(False)
    sorted_mass_index = np.argsort(bf_coeff.monoisotopic)
    sorted_monoisotopic = bf_coeff.monoisotopic[sorted_mass_index]
    start_valid, end_valid = np.searchsorted(sorted_monoisotopic,
                                             [mass - tol, mass + tol])
    sorted_mass_index = sorted_mass_index[start_valid:end_valid]
    valid_coeff_bf = bf_coeff.coefficients[sorted_mass_index, :]

    # mass defect based solution
    f = fg.FormulaGenerator(elements, max_mass, tol, tolerance_units=units,
                            refine_h=False, min_defect=None)
    f.generate_formulas(mass)
    valid_coeff, valid_elements, valid_monoisotopic = f.results_to_array()
    valid_coeff = valid_coeff[np.argsort(valid_monoisotopic), :]
    assert np.array_equal(valid_coeff, valid_coeff_bf)


formula_str_list = [("C11H12N2O2", [11, 12, 2, 2, 0]),
                    ("C6H12O6", [6, 12, 0, 6, 0]),
                    ("C27H46O", [27, 46, 0, 1, 0])]

@pytest.mark.parametrize("f_str,f_coeff", formula_str_list)
def test_formula_generator_refine_h(formula_generator_params, f_str, f_coeff):
    formula_generator_params["mass"] = 500
    f = fg.FormulaGenerator(**formula_generator_params)
    m = Formula(f_str).get_exact_mass()
    f.generate_formulas(m)
    coeff, _, _ = f.results_to_array()
    assert f_coeff in coeff

