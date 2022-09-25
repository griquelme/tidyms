"""
Tools for working with chemical formulas

Objects
-------

- Formula

Exceptions
----------

- InvalidFormula

"""


import numpy as np
import string
from collections import Counter
from copy import copy
from typing import List, Tuple, Dict, Optional
from .atoms import EM, Isotope, PeriodicTable
from ._isotope_distributions import find_formula_abundances


_abundance_type = Optional[Dict[str, np.ndarray]]


class Formula:
    """
    Represents a chemical formula as a mapping from isotopes to formula
    coefficients.

    Attributes
    ----------
    composition: Counter
        A mapping of Isotopes to formula coefficients.
    charge: int
        The numerical charge of the formula.

    Methods
    -------
    get_exact_mass()
        Computes the exact mass of the formula.
    get_nominal_mass()
        Computes the nominal mass of the formula.
    get_formula_str()
        Return a string representation of the Formula.
    get_isotopic_envelope(n=10, natural_abundance=True, custom_abundance=None)
        Computes nominal mass, exact mass and abundance of the first n
        isotopologues.

    Examples
    --------
    >>> Formula("H2O")
    Formula(H2O)
    >>> Formula("(13C)O2")
    Formula((13C)O2)
    >>> Formula("[Cr(H2O)6]3+")
    Formula([H12CrO6]3+)
    >>> Formula("CH3CH2CH3")
    Formula(C3H8)
    >>> Formula({"C": 1, "17O": 2})
    Formula(C(17O)2)

    """

    def __init__(self, *args):
        if len(args) == 1:
            formula_str = args[0]
            formula_str, charge = _parse_charge(formula_str)
            composition = _parse_formula(formula_str)
        else:
            composition, charge = args
            ptable = PeriodicTable()
            composition = {ptable.get_isotope(k): v for k, v in composition.items()}
            composition = Counter(composition)
            for v in composition.values():
                if not isinstance(v, int) or (v < 1):
                    msg = "Formula coefficients must be positive integers"
                    raise ValueError(msg)
            if not isinstance(charge, int):
                msg = "Charge must be an integer"
                raise ValueError(msg)
        self.charge = charge
        self.composition = composition

    def __add__(self, other: "Formula"):
        if not isinstance(other, Formula):
            msg = "sum operation is defined only for Formula objects"
            raise ValueError(msg)
        else:
            # copy Formula object and composition
            f_comp = copy(self.composition)
            new_f = copy(self)
            new_f.composition = f_comp
            new_f.composition.update(other.composition)
            new_f.charge += other.charge
            return new_f

    def __sub__(self, other: "Formula"):
        if not isinstance(other, Formula):
            msg = "subtraction operation is defined only for Formula objects"
            raise ValueError(msg)
        else:
            f_comp = copy(self.composition)
            new_f = copy(self)
            new_f.composition = f_comp
            new_f.composition.subtract(other.composition)
            new_f.charge += other.charge
            for i, c in new_f.composition.items():
                if c == 0:
                    new_f.composition.pop(i)
                elif c < 0:
                    msg = "subtraction cannot generate negative coefficients"
                    raise ValueError(msg)
            return new_f

    def get_exact_mass(self):
        """
        Computes the exact mass of the formula.

        Returns
        -------
        exact_mass: float
        """
        exact_mass = sum(x.m * k for x, k in self.composition.items())
        exact_mass -= EM * self.charge
        return exact_mass

    def get_nominal_mass(self):
        """
        Computes the nominal mass of the formula.

        Returns
        -------
        nominal_mass: float
        """
        nominal_mass = sum(x.a * k for x, k in self.composition.items())
        return nominal_mass

    def get_isotopic_envelope(
        self, n: int = 10, abundance: _abundance_type = None, min_p: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the isotopic distribution of the formula.

        Parameters
        ----------
        n: int
            the number of isotopes to include in the results.
        abundance: bool
            if True, uses the natural abundances when using the most abundant
            isotope in the formula. If false assumes that each isotope in the
            formula is pure.
        abundance: dict, optional
            used to pass custom abundances for elements. each key, value pair
            is composed of an element symbol str and a numpy array with the
            abundances. The sum of the abundances must be 1 and the size of
            the array must be the same as the number of stable isotopes for the
            given element.
        min_p: float
            Isotopes are included until the abundance is lower than this value

        Returns
        -------
        nominal: numpy.ndarray
            nominal mass of each isotopologue
        exact: numpy.ndarray
            exact mass of each isotopologue
        abundance: numpy.ndarray
            abundance of each isotopologue
        """
        nominal, exact, abundance = find_formula_abundances(
            self.composition, n, p=abundance
        )
        exact -= EM * self.charge
        abundance_mask = abundance > min_p
        nominal = nominal[abundance_mask]
        exact = exact[abundance_mask]
        abundance = abundance[abundance_mask]
        return nominal, exact, abundance

    def __repr__(self):
        return "Formula({})".format(str(self))

    def __str__(self):
        return _get_formula_str(self.composition, self.charge)


_matching_parenthesis = {"(": ")", "[": "]"}


def _multiply_formula_coefficients(composition: Counter, multiplier: int):
    if multiplier != 1:
        for k in composition:
            composition[k] *= multiplier


def _parse_charge(formula: str) -> Tuple[str, int]:
    """
    compute the charge state of a formula and remove the charge from the formula
    string

    Parameters
    ----------
    formula: str
        molecular formula
    Returns
    -------
    formula_without_charge, charge: str, int
    """

    # check if there's a charge in the formula
    if formula[-1] == "+":
        sign = 1
    elif formula[-1] == "-":
        sign = -1
    else:
        sign = 0

    # for charge with an absolute value greater than 1 enforces the use
    # of parenthesis to prevent ambiguity with a formula coefficient
    if sign and (formula[-2] in string.digits):
        try:
            matching = _matching_parenthesis[formula[0]]
            start = 1
            end = formula.rfind(matching)
            charge = sign * int(formula[end + 1 : -1])
            return formula[start:end], charge
        except (KeyError, ValueError):
            raise InvalidFormula
    elif sign:
        return formula[:-1], sign
    else:
        return formula, 0


def _get_token_type(formula: str, ind: int) -> int:
    """
    assigns 0 to elements, 1 to isotopes and 2 to expressions.
    Return the token type and a matching parenthesis if necessary
    """
    c = formula[ind]
    if c in string.ascii_uppercase:
        token_type = 0
    elif c in "(":
        if formula[ind + 1] in string.digits:
            token_type = 1
        else:
            token_type = 2
    elif c == "[":
        token_type = 2
    else:
        raise InvalidFormula
    return token_type


def _find_matching_parenthesis(formula: str, ind: int):
    parenthesis_open = formula[ind]
    parenthesis_close = _matching_parenthesis[parenthesis_open]
    match_ind = ind + 1
    level = 1
    try:
        while level > 0:
            c = formula[match_ind]
            if c == parenthesis_open:
                level += 1
            elif c == parenthesis_close:
                level -= 1
            match_ind += 1
        return match_ind - 1
    except IndexError:
        msg = "non matching parenthesis"
        raise InvalidFormula(msg)


def _get_coefficient(formula: str, ind: int):
    """
    traverses a formula string to compute a coefficient. ind is a position
    after an element or expression.

    Returns
    -------
    coefficient : int
    new_ind : int, new index to continue parsing the formula
    """
    length = len(formula)
    if (ind >= length) or (formula[ind] not in string.digits):
        coefficient = 1
        new_ind = ind
    else:
        end = ind + 1
        while (end < length) and (formula[end] in string.digits):
            end += 1
        coefficient = int(formula[ind:end])
        new_ind = end
    return coefficient, new_ind


def _tokenize_element(formula: str, ind: int):
    length = len(formula)
    if (ind < length - 1) and (formula[ind + 1] in string.ascii_lowercase):
        end = ind + 2
    else:
        end = ind + 1
    symbol = formula[ind:end]
    isotope = PeriodicTable().get_isotope(symbol)
    coefficient, end = _get_coefficient(formula, end)
    token = {isotope: coefficient}
    return token, end


def _tokenize_isotope(formula: str, ind: int):
    """
    Convert an isotope substring starting at `ind` index into a token.

    Returns
    -------
    token, new_ind

    """
    end = _find_matching_parenthesis(formula, ind)
    isotope = PeriodicTable().get_isotope(formula[ind + 1 : end])
    coefficient, end = _get_coefficient(formula, end + 1)
    token = {isotope: coefficient}
    return token, end


def _parse_formula(formula: str):
    """
    Parse a formula string into a Counter that maps isotopes to formula
    coefficients.
    """
    ind = 0
    n = len(formula)
    composition = Counter()
    while ind < n:
        token_type = _get_token_type(formula, ind)
        if token_type == 0:
            token, ind = _tokenize_element(formula, ind)
        elif token_type == 1:
            token, ind = _tokenize_isotope(formula, ind)
        else:
            # expression type evaluated recursively
            exp_end = _find_matching_parenthesis(formula, ind)
            token = _parse_formula(formula[ind + 1 : exp_end])
            exp_coefficient, ind = _get_coefficient(formula, exp_end + 1)
            _multiply_formula_coefficients(token, exp_coefficient)
        composition.update(token)
    return composition


# functions to get a formula string from a Formula


def _arg_sort_elements(symbol_list: List[str], mass_number_list: List[int]):
    """
    Return the sorted index for a list of elements symbols and mass numbers. If
    there are repeated elements, they are sorted by mass number.
    """
    zipped = list(zip(symbol_list, mass_number_list))
    return sorted(range(len(symbol_list)), key=lambda x: zipped[x])


class InvalidFormula(ValueError):
    pass


def _get_formula_str(composition, charge):

    # get C str
    c12 = PeriodicTable().get_isotope("12C")
    c13 = PeriodicTable().get_isotope("13C")
    h1 = PeriodicTable().get_isotope("1H")
    h2 = PeriodicTable().get_isotope("2H")
    ch = [c12, c13, h1, h2]

    # ADD C and H to formula str
    f_str = ""
    for i in ch:
        if i in composition:
            f_str += isotope_coeff_to_f_str(i, composition[i])

    # add other elements, sorted alphabetically
    isotopes = set(composition)
    isotopes = isotopes.difference(ch)
    for i in sorted(isotopes, key=lambda x: x.get_symbol() + str(x.a)):
        f_str += isotope_coeff_to_f_str(i, composition[i])

    if charge:
        charge_str = _get_charge_str(charge)
        f_str = "[{}]{}".format(f_str, charge_str)

    return f_str


def isotope_coeff_to_f_str(isotope: Isotope, coeff: int) -> str:
    coeff_str = str(coeff) if coeff > 1 else ""
    element = isotope.get_element()
    if isotope.a == element.nominal_mass:
        isotope_str = element.symbol
    else:
        isotope_str = "({})".format(isotope)
    return "{}{}".format(isotope_str, coeff_str)


def _get_charge_str(q: int):
    qa = abs(q)
    q_sign = "+" if q > 0 else "-"
    if qa > 1:
        q_str = str(qa)
    else:
        q_str = ""
    q_str = q_str + q_sign
    return q_str
