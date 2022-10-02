from tidyms.chem import formula
from tidyms.chem.atoms import InvalidIsotope, PeriodicTable
import pytest


@pytest.mark.parametrize(
    "formula_str,p_open,p_close",
    [
        ("[Cr[H2O]6]3+", 0, 9),
        ("[C9H11NO2]", 0, 9),
        ("C9H11N(17O)2", 6, 10),
        ("[Cr[(2H)2O]6]3+", 3, 10),
    ],
)
def test_find_matching_parenthesis_valid_input(formula_str, p_open, p_close):
    test_p_close = formula._find_matching_parenthesis(formula_str, p_open)
    assert test_p_close == p_close


@pytest.mark.parametrize(
    "formula_str,formula_without_charge,charge",
    [
        ("H2O", "H2O", 0),
        ("(13C)", "(13C)", 0),
        ("[CO3]2-", "CO3", -2),
        ("[Cr[H2O]6]3+", "Cr[H2O]6", 3),
        ("[C9H11NO2]", "[C9H11NO2]", 0),
        ("CO-", "CO", -1),
        ("[H2O]+", "[H2O]", 1),
        ("H2O+", "H2O", 1),
    ],
)
def test_parse_charge_valid_input(formula_str, formula_without_charge, charge):
    test_formula_without_charge, q = formula._parse_charge(formula_str)
    assert test_formula_without_charge == formula_without_charge
    assert charge == q


@pytest.mark.parametrize("formula_str", ["SO42-"])
def test_parse_charge_invalid_input(formula_str):
    with pytest.raises(formula.InvalidFormula):
        _, q = formula._parse_charge(formula_str)


@pytest.mark.parametrize(
    "formula_str,ind,token_type",
    [
        ("H2O", 0, 0),
        ("H2(34S)O4", 2, 1),
        ("[Cr(H2O)6]3+", 3, 2),
        ("[Fe[CN]6]4-", 3, 2),
    ],
)
def test_get_token_type(formula_str, ind, token_type):
    test_token_type = formula._get_token_type(formula_str, ind)
    assert test_token_type == token_type


@pytest.mark.parametrize(
    "formula_str,ind,coeff,new_ind",
    [
        ("H2O", 3, 1, 3),
        ("CO2", 1, 1, 1),
        ("C9H11NO2", 3, 11, 5),
    ]
)
def test_get_coefficient_valid_input(formula_str, ind, coeff, new_ind):
    test_coeff, test_ind = formula._get_coefficient(formula_str, ind)
    assert coeff == test_coeff
    assert test_ind == new_ind


@pytest.mark.parametrize(
    "formula_str,ind,new_ind,element",
    [
        ("H2O", 0, 2, "H"),
        ("H2O", 2, 3, "O"),
        ("C9H11NO2", 5, 6, "N"),
        ("C9H11N(17O)2", 5, 6, "N"),
        ("Cr(H2O)6", 0, 2, "Cr"),
    ]
)
def test_tokenize_element_valid_input(formula_str, ind, new_ind, element):
    token, test_index = formula._tokenize_element(formula_str, ind)
    assert test_index == new_ind
    isotope = PeriodicTable().get_element(element).get_monoisotope()
    assert isotope in token


@pytest.mark.parametrize(
    "formula_str,ind,isotope_str,new_ind",
    [
        ("(13C)O2", 0, "13C", 5),
        ("C9H11(15N)2O2", 5, "15N", 11),
        ("C6H12O5(18O)", 7, "18O", 12),
        ("C6H12O4(18O)2", 7, "18O", 13),
    ]
)
def test_tokenize_isotope_valid_input(formula_str, ind, isotope_str, new_ind):
    token, test_index = formula._tokenize_isotope(formula_str, ind)
    isotope = PeriodicTable().get_isotope(isotope_str)
    assert test_index == new_ind
    assert isotope in token


@pytest.mark.parametrize(
    "f_str,composition",
    [
        ("H2O", {"1H": 2, "16O": 1}),
        ("(13C)O2", {"13C": 1, "16O": 2}),
        ("C9H11(15N)2O2", {"12C": 9, "1H": 11, "15N": 2, "16O": 2}),
        ("C9H11N2O2", {"12C": 9, "1H": 11, "14N": 2, "16O": 2}),
        ("Cr[(2H)2O]6", {"52Cr": 1, "2H": 12, "16O": 6})
    ]
)
def test_tokenize_formula(f_str, composition):
    composition = {PeriodicTable().get_isotope(k): v for k, v in composition.items()}
    test_composition = formula._parse_formula(f_str)
    for isotope in composition:
        assert composition[isotope] == test_composition[isotope]


def test_arg_sort_elements():
    symbols = ["Cd", "C", "H", "H", "O", "O", "S", "B"]
    a = [60, 12, 2, 1, 16, 17, 32, 7]
    sorted_ind = [7, 1, 0, 3, 2, 4, 5, 6]
    assert sorted_ind == formula._arg_sort_elements(symbols, a)


@pytest.mark.parametrize(
    "charge,charge_str",
    [
        (1, "+"),
        (2, "2+"),
        (-1, "-"),
        (-4, "4-")
    ]
)
def test_get_charge_str(charge, charge_str):
    test_charge_str = formula._get_charge_str(charge)
    assert test_charge_str == charge_str


@pytest.mark.parametrize(
    "f,f_str",
    [
        (formula.Formula("CO2"), "CO2"),
        (formula.Formula("(13C)C2H6O3"), "C2(13C)H6O3"),
        (formula.Formula("C24H46SPN(18O)2"), "C24H46N(18O)2PS"),
        (formula.Formula("[Cr(H2O)6]3+"), "[H12CrO6]3+"),
        (formula.Formula("CH3CH2CH3"), "C3H8"),
        (formula.Formula("F2"), "F2"),
    ]
)
def test_get_formula_str(f, f_str):
    test_f_str = str(f)
    assert test_f_str == f_str


@pytest.mark.parametrize("f_str", ["(CO2", "#H2O"])
def test_parse_formula_invalid_formula(f_str):
    with pytest.raises(formula.InvalidFormula):
        formula.Formula(f_str)


@pytest.mark.parametrize("f_str", ["(14C)O2", "(3H)2O"])
def test_parse_formula_invalid_isotope(f_str):
    with pytest.raises(InvalidIsotope):
        formula.Formula(f_str)


@pytest.fixture
def formula_data():
    formula_str = ["CO2", "H2O", "F2"]
    nominal = [44, 18, 38]
    exact = [43.9898, 18.0106, 37.9968]
    return formula_str, nominal, exact


def test_get_exact_mass(formula_data):
    formula_str, _, exact = formula_data
    for f_str, e in zip(formula_str, exact):
        assert abs(formula.Formula(f_str).get_exact_mass() - e) < 0.0001


def test_get_nominal_mass(formula_data):
    formula_str, nominal, _ = formula_data
    for f_str, n in zip(formula_str, nominal):
        assert formula.Formula(f_str).get_nominal_mass() == n


def test_formula_from_dictionary():
    composition = {"C": 1, "17O": 2, "H": 2}
    charge = 1
    f = formula.Formula(composition, charge)
    for k in composition:
        assert PeriodicTable().get_isotope(k) in f.composition
    assert charge == f.charge


def test_formula_from_dictionary_invalid_isotope():
    composition = {"C": 1, "G": 4}
    charge = 1
    with pytest.raises(InvalidIsotope):
        formula.Formula(composition, charge)


def test_formula_from_dictionary_invalid_isotope_type():
    composition = {4: 1, "G": 4}
    charge = 1
    with pytest.raises(ValueError):
        formula.Formula(composition, charge)


@pytest.mark.parametrize(
    "composition,q",
    [
        [{"C": -1, "H": 4}, 1],
        [{"C": 1, "H": 4}, 0.5],
    ]
)
def test_formula_from_dictionary_invalid_coefficient(composition, q):
    with pytest.raises(ValueError):
        formula.Formula(composition, q)


def test_Formula_add():
    f1 = formula.Formula("H2O")
    f2 = formula.Formula("CO2")
    f_sum = f1 + f2
    expected = formula.Formula("H2CO3")
    assert expected == f_sum


def test_Formula_add_invalid_type():
    f1 = formula.Formula("H2O")
    f2 = "CO2"
    with pytest.raises(ValueError):
        f1 + f2


def test_Formula_subtract_valid():
    f1 = formula.Formula("C6H12O6")
    f2 = formula.Formula("CO2")
    f_diff = f1 - f2
    expected = formula.Formula("C5H12O4")
    assert expected == f_diff


def test_Formula_subtract_invalid_type():
    f1 = formula.Formula("C6H12O6")
    f2 = "CO2"
    with pytest.raises(ValueError):
        f1 - f2


def test_Formula_subtract_valid_zero_coeff():
    f1 = formula.Formula("C4H8O2")
    f2 = formula.Formula("CO2")
    f_diff = f1 - f2
    expected = formula.Formula("C3H8")
    assert expected == f_diff


def test_Formula_subtract_invalid_coeff():
    f1 = formula.Formula("C4H8O")
    f2 = formula.Formula("CO2")
    with pytest.raises(ValueError):
        f1 - f2
