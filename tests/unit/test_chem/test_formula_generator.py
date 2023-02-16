"""
tests for formula module from formula-generator
"""
import tidyms.chem._formula_generator as fg
from tidyms.chem import Formula, PeriodicTable
from tidyms.chem.atoms import InvalidIsotope
import pytest
import numpy as np


@pytest.fixture
def chnops_bounds() -> fg.FormulaCoefficientBounds:
    chnops_str = ["C", "H", "N", "O", "P", "S"]
    bounds = {x: (0, 5) for x in chnops_str}
    return fg.FormulaCoefficientBounds.from_isotope_str(bounds)


def test__FormulaCoefficientBounds_from_isotope_str_negative_bounds_error():
    chnops_str = ["C", "H", "N", "O", "P", "S"]
    bounds = {x: [-1, 5] for x in chnops_str}
    with pytest.raises(ValueError):
        fg.FormulaCoefficientBounds.from_isotope_str(bounds)


def test__FormulaCoefficientBounds_from_isotope_str_ub_lt_lb_error():
    # check that an ValueError is raised if the lower bound is greater than the
    # upper bound
    chnops_str = ["C", "H", "N", "O", "P", "S"]
    bounds = {x: [7, 5] for x in chnops_str}
    with pytest.raises(ValueError):
        fg.FormulaCoefficientBounds.from_isotope_str(bounds)


def test__FormulaCoefficientBounds_from_isotope_str_invalid_isotope():
    # check that an ValueError is raised if the lower bound is greater than the
    # upper bound
    isotopes = ["14C", "H", "N", "O", "P", "S"]
    bounds = {x: [0, 5] for x in isotopes}
    with pytest.raises(InvalidIsotope):
        fg.FormulaCoefficientBounds.from_isotope_str(bounds)


def test__FormulaCoefficientBounds_from_isotope_str_invalid_element():
    # check that an ValueError is raised if the lower bound is greater than the
    # upper bound
    isotopes = ["X", "H", "N", "O", "P", "S"]
    bounds = {x: [0, 5] for x in isotopes}
    with pytest.raises(InvalidIsotope):
        fg.FormulaCoefficientBounds.from_isotope_str(bounds)


def test__FormulaCoefficientBounds_bound_from_mass(chnops_bounds):
    M = 100.0

    # check bounds for 32S. 32 * 4 == 128 > M, 32 * 3 == 96 < M. the upper bound
    # should be 3.
    s32 = PeriodicTable().get_isotope("32S")
    bounds = chnops_bounds.bounds_from_mass(M)
    min_ns, max_ns = bounds[s32]
    assert min_ns == 0
    assert max_ns == 3

    # check bounds for 1HS. The mass-based bound is ~100 >> 5. The result should
    # be 5
    h1 = PeriodicTable().get_isotope("1H")
    min_nh, max_nh = bounds[h1]
    assert min_nh == 0
    assert max_nh == 5


def test__FormulaCoefficientBounds_bound_negative_positive_defect_only_pos_d():
    # check results when using only isotopes with positive mass defect
    isotopes = ["H", "N"]
    bounds = {x: (0, 5) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    d = 0.01
    tol = 0.005
    dp_min, dp_max, dn_min, dn_max = bounds.bound_negative_positive_defect(d, tol)
    assert np.isclose(dn_min, -tol)
    assert np.isclose(dn_max, tol)
    assert np.isclose(dp_min, d - tol)
    assert np.isclose(dp_max, d + tol)


def test__FormulaCoefficientBounds_bound_negative_positive_defect_only_neg_d():
    # check results when using only isotopes with positive mass defect
    isotopes = ["Cl", "O"]
    bounds = {x: (0, 5) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    d = -0.01
    tol = 0.005
    dp_min, dp_max, dn_min, dn_max = bounds.bound_negative_positive_defect(d, tol)
    assert np.isclose(dn_min, d - tol)
    assert np.isclose(dn_max, d + tol)
    assert np.isclose(dp_min, -tol)
    assert np.isclose(dp_max, tol)


def test__FormulaCoefficientBounds_get_nominal_defect_candidates(chnops_bounds):
    M = 100.01
    m_candidates, d_candidates = chnops_bounds.get_nominal_defect_candidates(M)
    assert len(m_candidates) == 1
    assert m_candidates[0] == 100
    assert np.isclose(d_candidates[0], 0.01)


def test__FormulaCoefficientBounds_get_nominal_defect_candidates_d_gt_one():
    # check nominal mass and mass defect candidates with mass defect greater
    # than one
    isotopes = ["H"]
    bounds = {x: (190, 200) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    M = 195.5
    m_candidates, d_candidates = bounds.get_nominal_defect_candidates(M)
    assert len(m_candidates) == 1
    assert m_candidates[0] == 194
    assert np.isclose(d_candidates[0], 1.5)


def test__FormulaCoefficientBounds_split_pos_neg(chnops_bounds):
    pos, neg = chnops_bounds.split_pos_neg()
    for x in pos.bounds:
        assert x.defect > 0.0

    for x in neg.bounds:
        assert x.defect < 0.0


def test__FormulaCoefficientBounds_split_pos_neg_no_positive_d():
    isotopes = ["Cl", "O"]
    bounds = {x: (0, 5) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    pos, neg = bounds.split_pos_neg()
    assert len(pos.bounds) == 0

    for x in neg.bounds:
        assert x.defect < 0.0


def test__FormulaCoefficientBounds_split_pos_neg_no_negative_d():
    isotopes = ["H", "N"]
    bounds = {x: (0, 5) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    pos, neg = bounds.split_pos_neg()
    assert len(neg.bounds) == 0

    for x in pos.bounds:
        assert x.defect > 0.0


@pytest.mark.parametrize("isotope", ["H", "O"])
def test__FormulaCoefficients_one_isotope_empty(isotope):
    bounds = {x: (0, 0) for x in [isotope]}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = False
    return_reversed = False
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (1, 1)
    assert coeff.M.shape == (1,)
    assert coeff.q.shape == (1,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12


@pytest.mark.parametrize("isotope", ["H", "O"])
def test__FormulaCoefficients_one_isotope(isotope):
    n = 10
    bounds = {x: (0, n) for x in [isotope]}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = False
    return_reversed = False
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n + 1, 1)
    assert coeff.M.shape == (n + 1,)
    assert coeff.q.shape == (n + 1,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12


@pytest.mark.parametrize("isotope", ["H", "O"])
def test__FormulaCoefficients_one_isotope_return_sorted(isotope):
    n = 10
    bounds = {x: (0, n) for x in [isotope]}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = True
    return_reversed = False
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n + 1, 1)
    assert coeff.M.shape == (n + 1,)
    assert coeff.q.shape == (n + 1,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12
    for k in range(12):
        assert coeff.r_to_d[k].size == coeff.r_to_index[k].size
        # sort check for mass defect
        diff = np.diff(coeff.r_to_d[k])
        assert (diff >= 0).all()


@pytest.mark.parametrize("isotope", ["H", "O"])
def test__FormulaCoefficients_one_isotope_return_sorted_return_reversed(isotope):
    n = 10
    bounds = {x: (0, n) for x in [isotope]}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = True
    return_reversed = True
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n + 1, 1)
    assert coeff.M.shape == (n + 1,)
    assert coeff.q.shape == (n + 1,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12
    for k in range(12):
        assert coeff.r_to_d[k].size == coeff.r_to_index[k].size
        # sort check
        diff = np.diff(coeff.r_to_d[k])
        assert (diff <= 0).all()


@pytest.mark.parametrize(
    "isotopes",
    [["H", "N"], ["O", "P"]]
)
def test__FormulaCoefficients_multiple_isotopes(isotopes):
    n = 10
    n_isotopes = len(isotopes)
    n_comb = (n + 1) ** n_isotopes
    bounds = {x: (0, n) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = False
    return_reversed = False
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n_comb, n_isotopes)
    assert coeff.M.shape == (n_comb,)
    assert coeff.q.shape == (n_comb,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12


@pytest.mark.parametrize(
    "isotopes",
    [["H", "N"], ["O", "P"]]
)
def test__FormulaCoefficients_multiple_isotopes_return_sorted(isotopes):
    n = 10
    n_isotopes = len(isotopes)
    n_comb = (n + 1) ** n_isotopes
    bounds = {x: (0, n) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = True
    return_reversed = False
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n_comb, n_isotopes)
    assert coeff.M.shape == (n_comb,)
    assert coeff.q.shape == (n_comb,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12
    for k in range(12):
        assert coeff.r_to_d[k].size == coeff.r_to_index[k].size
        # sort check
        diff = np.diff(coeff.r_to_d[k])
        assert (diff >= 0).all()


@pytest.mark.parametrize(
    "isotopes",
    [["H", "N"], ["O", "P"]]
)
def test__FormulaCoefficients_multiple_isotopes_return_reversed(isotopes):
    n = 10
    n_isotopes = len(isotopes)
    n_comb = (n + 1) ** n_isotopes
    bounds = {x: (0, n) for x in isotopes}
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(bounds)
    max_mass = 1000
    return_sorted = True
    return_reversed = True
    coeff = fg.FormulaCoefficients(bounds, max_mass, return_sorted, return_reversed)
    assert coeff.coefficients.shape == (n_comb, n_isotopes)
    assert coeff.M.shape == (n_comb,)
    assert coeff.q.shape == (n_comb,)
    assert len(coeff.r_to_index) == 12
    assert len(coeff.r_to_d) == 12
    for k in range(12):
        assert coeff.r_to_d[k].size == coeff.r_to_index[k].size
        # sort check
        diff = np.diff(coeff.r_to_d[k])
        assert (diff <= 0).all()


@pytest.mark.parametrize("formula_str", ["C10H10O2", "C9H14", "C7H8N2"])
def test_FormulaGenerator_bruteforce(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"C": (7, 10), "H": (8, 14), "O": (0, 2), "N": (0, 2)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


@pytest.mark.parametrize("formula_str", ["C9H14", "C7H8N2"])
def test_FormulaGenerator_bruteforce_no_negative_isotopes(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"C": (7, 10), "H": (8, 14), "N": (0, 2)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


@pytest.mark.parametrize("formula_str", ["C9O6", "C7P4"])
def test_FormulaGenerator_bruteforce_no_positive_isotopes(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"C": (7, 10), "O": (0, 10), "P": (0, 5)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


@pytest.mark.parametrize("formula_str", ["H2O", "(13C)O2"])
def test_FormulaGenerator_bruteforce_no_c12(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"13C": (0, 5), "O": (0, 5), "H": (0, 5)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


@pytest.mark.parametrize("formula_str", ["H3N", "(13C)H4"])
def test_FormulaGenerator_bruteforce_no_c12_no_negative(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"13C": (0, 5), "N": (0, 5), "H": (0, 5)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


@pytest.mark.parametrize("formula_str", ["ClO2", "SO3"])
def test_FormulaGenerator_bruteforce_no_c12_no_positive(formula_str):
    # Test correctness of the algorithm by comparing the results with the
    # bruteforce solution.
    formula = Formula(formula_str)
    M = formula.get_exact_mass()
    # parameters
    isotope_bounds = {"Cl": (0, 5), "O": (0, 5), "S": (0, 5)}
    max_mass = 1000
    tol = 0.005
    # brute force solution
    # compute al formula coefficients, sort coefficients by exact mass
    # and search the minimum and maximum valid mass using bisection search.
    bounds = fg.FormulaCoefficientBounds.from_isotope_str(isotope_bounds)
    bf_coeff = bounds.make_coefficients(max_mass)
    bf_isotopes = bf_coeff.isotopes
    bf_coeff_M = bf_coeff.M
    bf_sorted_M_index = np.argsort(bf_coeff_M)
    bf_coeff = bf_coeff.coefficients[bf_sorted_M_index]
    bf_coeff_M = bf_coeff_M[bf_sorted_M_index]
    start_valid, end_valid = np.searchsorted(bf_coeff_M, [M - tol, M + tol])
    bf_coeff = bf_coeff[start_valid:end_valid, :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in bf_isotopes])
    bf_coeff = bf_coeff[:, col_sort]

    # mass defect based solution
    f = fg.FormulaGenerator(isotope_bounds, max_mass)
    f.generate_formulas(M, tol)
    coeff, isotopes, coeff_M = f.results_to_array()
    coeff = coeff[np.argsort(coeff_M), :]
    # sort columns by isotope str
    col_sort = np.argsort([str(x) for x in isotopes])
    coeff = coeff[:, col_sort]
    assert np.array_equal(coeff, bf_coeff)


def test_FormulaGenerator_generate_formulas_invalid_mass():
    formula_generator = fg.FormulaGenerator.from_hmdb(500)
    with pytest.raises(ValueError):
        formula_generator.generate_formulas(-100, 0.005)


def test_FormulaGenerator_generate_formulas_invalid_tolerance():
    formula_generator = fg.FormulaGenerator.from_hmdb(500)
    with pytest.raises(ValueError):
        formula_generator.generate_formulas(100, -0.005)


@pytest.mark.parametrize("mass", [500, 1000, 1500, 2000])
def test_FormulaGenerator_from_hmdb(mass):
    fg.FormulaGenerator.from_hmdb(mass)


def test_FormulaGenerator_from_hmdb_invalid_mass():
    with pytest.raises(ValueError):
        fg.FormulaGenerator.from_hmdb(2450)