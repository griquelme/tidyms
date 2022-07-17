from tidyms.chem import formula
from tidyms.chem import PTABLE
from tidyms.chem.atoms import InvalidIsotope
import pytest


# test find_matching parenthesis

find_matching_parenthesis_valid_input = \
    [("[Cr[H2O]6]3+", 0, 9), ("[C9H11NO2]", 0, 9),
     ("C9H11N(17O)2", 6, 10), ("[Cr[(2H)2O]6]3+", 3, 10)]

@pytest.mark.parametrize("formula_str,p_open,p_close",
                         find_matching_parenthesis_valid_input)
def test_find_matching_parenthesis_valid_input(formula_str, p_open, p_close):
    test_p_close = formula._find_matching_parenthesis(formula_str, p_open)
    assert test_p_close == p_close


## test parse charge

valid_formulas_charge = [("H2O", "H2O", 0), ("(13C)", "(13C)", 0),
                         ("[CO3]2-", "CO3", -2),
                         ("[Cr[H2O]6]3+", "Cr[H2O]6", 3),
                         ("[C9H11NO2]", "[C9H11NO2]", 0), ("CO-", "CO", -1),
                         ("[H2O]+", "[H2O]", 1), ("H2O+", "H2O", 1)]

invalid_formulas_charge = ["SO42-"]


@pytest.mark.parametrize("formula_str,formula_without_charge,charge",
                         valid_formulas_charge)
def test_parse_charge_valid_input(formula_str, formula_without_charge, charge):
    test_formula_without_charge, q = formula._parse_charge(formula_str)
    assert test_formula_without_charge == formula_without_charge
    assert charge == q


@pytest.mark.parametrize("formula_str", invalid_formulas_charge)
def test_parse_charge_invalid_input(formula_str):
    with pytest.raises(formula.InvalidFormula):
        _, q = formula._parse_charge(formula_str)

# test get_token_type

get_token_type_input = [("H2O", 0, 0,), ("H2(34S)O4", 2, 1),
                        ("[Cr(H2O)6]3+", 3, 2), ("[Fe[CN]6]4-", 3, 2)]

@pytest.mark.parametrize("formula_str,ind,token_type", get_token_type_input)
def test_get_token_type(formula_str, ind, token_type):
    test_token_type = formula._get_token_type(formula_str, ind)
    assert test_token_type == token_type


# test get_coefficient
get_coefficient_valid_input = [("H2O", 3, 1, 3), ("CO2", 1, 1, 1),
                               ("C9H11NO2", 3, 11, 5)]

@pytest.mark.parametrize("formula_str,ind,coeff,new_ind",
                         get_coefficient_valid_input)
def test_get_coefficient_valid_input(formula_str, ind, coeff, new_ind):
    test_coeff, test_ind = formula._get_coefficient(formula_str, ind)
    assert coeff == test_coeff
    assert test_ind == new_ind

# test tokenize element
tokenize_element_test_input = [("H2O", 0, 2, "H"),
                               ("H2O", 2, 3, "O"),
                               ("C9H11NO2", 5, 6, "N"),
                               ("C9H11N(17O)2", 5, 6, "N"),
                               ("Cr(H2O)6", 0, 2, "Cr")]

@pytest.mark.parametrize("formula_str,ind,new_ind,element",
                         tokenize_element_test_input)
def test_tokenize_element_valid_input(formula_str, ind, new_ind, element):
    token, test_index = formula._tokenize_element(formula_str, ind)
    assert test_index == new_ind
    isotope = PTABLE[element].get_most_abundant_isotope()
    assert isotope in token


# test tokenize isotope

tokenize_isotope_test_input = [("(13C)O2", 0, "C", 13, 5),
                               ("C9H11(15N)2O2", 5, "N", 15, 11),
                               ("C6H12O5(18O)", 7, "O", 18, 12),
                               ("C6H12O4(18O)2", 7, "O", 18, 13)]

@pytest.mark.parametrize("formula_str,ind,symbol,mass_number,new_ind",
                         tokenize_isotope_test_input)
def test_tokenize_isotope_valid_input(formula_str, ind, symbol,
                                      mass_number, new_ind):
    token, test_index = formula._tokenize_isotope(formula_str, ind)
    isotope = PTABLE[symbol].isotopes[mass_number]
    assert test_index == new_ind
    assert isotope in token


# test tokenize formula

@pytest.fixture
def tokenize_formula_input():
    formula_input = \
        [("H2O", {PTABLE["H"].isotopes[1]: 2,
                  PTABLE["O"].isotopes[16]: 1}),
         ("(13C)O2", {PTABLE["C"].isotopes[13]: 1,
                      PTABLE["O"].isotopes[16]: 2}),
         ("C9H11(15N)2O2", {PTABLE["C"].isotopes[12]: 9,
                            PTABLE["H"].isotopes[1]: 11,
                            PTABLE["N"].isotopes[15]: 2,
                            PTABLE["O"].isotopes[16]: 2}),
         ("C9H11N2O2", {PTABLE["C"].isotopes[12]: 9,
                        PTABLE["H"].isotopes[1]: 11,
                        PTABLE["N"].isotopes[14]: 2,
                        PTABLE["O"].isotopes[16]: 2}),
         ("Cr[(2H)2O]6", {PTABLE["Cr"].isotopes[52]: 1,
                           PTABLE["H"].isotopes[2]: 12,
                           PTABLE["O"].isotopes[16]: 6}),
         ]
    return formula_input

def test_tokenize_formula(tokenize_formula_input):
    for f, composition in tokenize_formula_input:
        test_composition = formula._parse_formula(f)
        for isotope in composition:
            assert composition[isotope] == test_composition[isotope]


def test_arg_sort_elements():
    symbols = ["Cd", "C", "H", "H", "O", "O", "S", "B"]
    a = [60, 12, 2, 1, 16, 17, 32, 7]
    sorted_ind = [7, 1, 0, 3, 2, 4, 5, 6]
    assert sorted_ind == formula._arg_sort_elements(symbols, a)


symbol_to_subformula_string_params = {
    ("C", 12, 1, True, "C"), ("C", 12, 3, True, "C3"),
    ("C", "13", 1, False, "(13C)"), ("C", 13, 3, False, "(13C)3")
}

@pytest.mark.parametrize("symbol,mass,coefficient,include_mass,subformula",
                         symbol_to_subformula_string_params)
def test_symbol_to_subformula_string(symbol, mass, coefficient, include_mass,
                                     subformula):
    test_subformula = \
        formula._symbol_to_subformula_str(symbol, mass, coefficient,
                                          is_most_abundant=include_mass)
    assert subformula == test_subformula


def test_composition_to_list():
    composition = formula.Formula("H(13C)5N3C12").composition
    symbols = ["C", "C", "H", "N"]
    a = [12, 13, 1, 14]
    coefficients = [12, 5, 1, 3]
    is_most_abundant = [True, False, True, True]
    test_symbols, test_a, test_coefficients, test_is_most_abundant = \
        formula._composition_to_list(composition)
    assert test_symbols == symbols
    assert test_a == a
    assert test_coefficients == coefficients
    assert test_is_most_abundant == is_most_abundant


ch_string_params = [("C6H12O6", "C6H12"), ("H6C3O4N3P4", "C3H6"),
                    ("C6(13C)H7O8NPS", "C6(13C)H7"),
                    ("(13C)4C7(2H)5NOP6", "C7(13C)4(2H)5")]

@pytest.mark.parametrize("f_str,ch_str", ch_string_params)
def test_get_ch_string(f_str, ch_str):
    composition = formula.Formula(f_str).composition
    symbols, a, coefficients, is_most_abundant = \
        formula._composition_to_list(composition)
    test_ch_str = formula._get_ch_string(symbols, a, coefficients,
                                         is_most_abundant)
    assert test_ch_str == ch_str


heteroatom_string_params = [("C6H12O6", "C6H12O6"),
                            ("H6C3O4N3P4", "C3H6N3O4P4"),
                            ("C6(13C)H7O8NPS", "C6(13C)H7NO8PS"),
                            ("(13C)4C7(2H)5NOP6", "C7(13C)4(2H)5NOP6")]

@pytest.mark.parametrize("f_str,heteroatom_str", heteroatom_string_params)
def test_get_ch_string(f_str, heteroatom_str):
    composition = formula.Formula(f_str).composition
    symbols, a, coefficients, is_most_abundant = \
        formula._composition_to_list(composition)
    test_heteroatom_str = formula._get_heteroatom_str(symbols, a, coefficients,
                                                      is_most_abundant)
    assert test_heteroatom_str == heteroatom_str

@pytest.mark.parametrize("charge,charge_str",
                         [(1, "+"), (2, "2+"), (-1, "-"), (-4, "4-")])
def test_get_charge_str(charge, charge_str):
    test_charge_str = formula._get_charge_str(charge)
    assert test_charge_str == charge_str


# test get_formula_str
formula_str_params = [(formula.Formula("CO2"), "CO2"),
                      (formula.Formula("(13C)C2H6O3"), "C2(13C)H6O3"),
                      (formula.Formula("C24H46SPN(18O)2"), "C24H46N(18O)2PS"),
                      (formula.Formula("[Cr(H2O)6]3+"), "[H12CrO6]3+"),
                      (formula.Formula("CH3CH2CH3"), "C3H8"),
                      (formula.Formula("F2"), "F2")]

@pytest.mark.parametrize("f,f_str", formula_str_params)
def test_get_formula_str(f, f_str):
    test_f_str = str(f)
    assert test_f_str == f_str


invalid_formulas = ["(CO2", "C(3H)4", "GCH2", "#H2O"]
@pytest.mark.parametrize("f_str", invalid_formulas)
def test_parse_formula_invalid_formula(f_str):
    with pytest.raises((formula.InvalidFormula, InvalidIsotope)):
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
        assert formula.find_isotope(k) in f.composition
    assert charge == f.charge

formula_from_dictionary_test_values = \
    [[{"C": 1, "G": 4}, 1], [{"C": -1, "H": 4}, 1], [{"C": 1, "H": 4}, 0.5]]
@pytest.mark.parametrize("d,q", formula_from_dictionary_test_values)
def test_formula_from_dictionary_bad_input(d, q):
    with pytest.raises((InvalidIsotope, ValueError)):
        formula.Formula(d, q)