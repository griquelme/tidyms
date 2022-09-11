# -*- coding: utf-8 -*-
"""
functions to calculate compatible molecular formulas within a given tolerance.
"""

import numpy as np
from collections import namedtuple
from .utils import cartesian_product_from_range_list
from typing import Union, Iterable, Dict, Optional, List, Tuple
from .atoms import Isotope, find_isotope

# from .formula_generator_utils import *


_Coefficients = namedtuple(
    "Coefficients",
    [
        "isotopes",
        "coefficients",
        "monoisotopic",
        "nominal",
        "nominal_q",
        "nominal_r",
        "defect",
        "r_to_defect",
        "r_to_index",
    ],
)
_Coefficients.__doc__ = """\
Named tuple with elements with positive/negative mass defect.

Fields
------
coefficients: np.array[int]
    Formula coefficients. Each row is a formula, each column is an element.
isotopes: List[Isotopes]
    element associated to each column of coefficients.
defects: np.array[float]
    mass defect associated to each row of coefficients.
monoisotopic: np.array[float]
    monoisotopic mass associated to each row of coefficients.
nominal : np.array[int]
    nominal mass associated to each row of coefficients.
nominal_q: np.array[int]
    quotient between the nominal mass and 12
nominal_r: np.array[int]
    remainder  between the nominal mass and 12
r_to_index: np.array[int]
    a 2D array where each row is a possible remainder of division by 12, and
    each column is an index to a formula in coefficients.
r_to_defect: np.array[float]
    A 2D array with the same shape es ri_array. each element is the
    corresponding mass defect.
"""


class Bounds(dict):
    """
    Mapping from isotopes to upper and lower bounds
    """

    def __init__(
        self,
        isotopes: Iterable[Isotope],
        bounds_range: Iterable[Tuple[int, int]],
        mass: float,
    ):
        dict.__init__(self, zip(isotopes, bounds_range))
        self.mass = mass

    def update_bounds(self, isotope: Union[Isotope, str], lower: int, upper: int):
        """
        modify or add new elements to bounds.

        Parameters
        ----------
        isotope: Isotope or str
            An Isotope Object or a string representation of an isotope.
        lower: int
        upper: int

        """
        if isotope not in self:
            isotope = find_isotope(isotope)
        _validate_bounds(lower, upper)
        self[isotope] = (lower, upper)

    def refine_h_upper_bounds(self, n_cluster: int = 5):
        """
        obtains a better guess on the number of H.
        Estimation of the number of H atoms is made assuming an alkane like
        formula.

        Parameters
        ----------
        n_cluster: positive integer, optional
            when working with mass values that are originated from multiple
            molecules, n_cluster is used a security margin for the estimation
            of the max number of H atoms. See Notes.

        Notes
        -----
        The number of Hydrogen atoms estimation is made as follows: if m is
        composed of `n_cluster` alkane-like species, each one with mass
        :math: `m_{i}` and a number of hydrogen atoms equal to
        :math: `n_{Hi} = 2 n_{Ci} + 2`, then an upper bound for the total number
        of hydrogen is:math: `n_{H} < \frac{m}{7} + 2 n_{cluster}`

        """
        nh_guess = int(self.mass / 7) + 1 + 2 * n_cluster
        h = find_isotope("1H")
        if h in self:
            nh_guess = min(nh_guess, self[h][1])
            lb, _ = self[h]
            self.update_bounds("H", lb, nh_guess)

    def split_pos_neg(self) -> Tuple["Bounds", "Bounds"]:
        """
        creates two new Bound objects, one with elements with positive mass
        defect and other with elements with negative mass defects.
        Returns
        -------
        pos: Bounds
        neg: Bounds

        """
        pos_isotopes = list()
        pos_restrictions = list()
        neg_isotopes = list()
        neg_restrictions = list()
        for isotope, restrictions in self.items():
            defect = isotope.defect
            if defect > 0:
                pos_isotopes.append(isotope)
                pos_restrictions.append(restrictions)
            elif defect < 0:
                neg_isotopes.append(isotope)
                neg_restrictions.append(restrictions)
        pos = Bounds(pos_isotopes, pos_restrictions, self.mass)
        neg = Bounds(neg_isotopes, neg_restrictions, self.mass)
        return pos, neg

    def get_mass_query_bounds(self, mass: float) -> "Bounds":
        """
        Creates a Bounds object to use with a MassQuery Object.

        Parameters
        ----------
        mass: float

        Returns
        -------
        query_bounds: Bounds

        """
        isotopes = self.keys()
        bounds_range = self.values()
        query_bounds = Bounds(isotopes, bounds_range, mass)
        for isotope in query_bounds:
            lower = max(0, self[isotope][0])
            upper = min(int(mass / isotope.m) + 1, self[isotope][1])
            query_bounds.update_bounds(isotope, lower, upper)
        return query_bounds

    def get_defect_bounds(self) -> Tuple[float, float, float, float]:
        """
        Computes the minimum and maximum value of the mass defect.

        Returns
        -------
        min_positive, max_positive, min_negative, max_negative: tuple

        """
        min_positive, max_positive, min_negative, max_negative = 0, 0, 0, 0
        for isotope in self:
            defect = isotope.defect
            min_tmp = defect * self[isotope][0]
            max_tmp = defect * self[isotope][1]
            if defect > 0:
                min_positive += min_tmp
                max_positive += max_tmp
            else:
                # min and max switched because mass defect is < 0
                min_negative += max_tmp
                max_negative += min_tmp
        return min_positive, max_positive, min_negative, max_negative

    def make_coefficients(
        self, return_sorted: bool = True, mode: Optional[str] = None
    ) -> _Coefficients:
        """
        Generate coefficients for FormulaGenerator

        Parameters
        ----------
        return_sorted: bool
            sort results according to mass defect
        mode: {"positive", "negative"}
            used for sorting. If positive, use reversed order.
        Returns
        -------
        coefficients: Coefficients

        """
        coefficients = _make_coefficients(self, mode=mode, return_sorted=return_sorted)
        return coefficients

    @staticmethod
    def from_isotope_str(
        isotopes: Iterable[str], mass: float, user_bounds: Optional[dict] = None
    ):
        """
        Creates a _Bounds instance from a list of isotope strings.

        Parameters
        ----------
        isotopes : List[str]
        mass : positive number
            Defines the upper bound for each isotope
        user_bounds : dict[str, (int, int]
            A dictionary of isotopes to lower and upper bounds. Overwrite
            defaults values for each isotope.

        Returns
        -------
        Bounds
        """

        if mass <= 0:
            msg = "mass must be a positive number"
            raise InvalidMass(msg)

        isotope_list = list()
        bounds_range = list()
        for element in isotopes:
            isotope = find_isotope(element)
            isotope_list.append(isotope)
            bounds_range.append((0, int(mass / isotope.m) + 1))
        bounds = Bounds(isotope_list, bounds_range, mass)

        if user_bounds is not None:
            for isotope, isotope_bounds in user_bounds.items():
                isotope = find_isotope(isotope)
                bounds.update_bounds(isotope, *isotope_bounds)
        return bounds


class _MassQuery(object):
    """
    Manages mass input to FormulaGenerator.

    Attributes
    ----------
    mass: positive number
    bounds: Bounds

    """

    def __init__(self, mass: float, tolerance: float, bounds: Bounds):
        """
        Parameters
        ----------
        mass: float or int
        tolerance: float
        bounds: Bounds

        """
        if (not isinstance(mass, (float, int))) or (mass <= 0):
            msg = "mass must be a positive number"
            raise InvalidMass(msg)
        self.mass = mass
        self.tolerance = tolerance
        self.bounds = bounds

    def split_nominal_defect(self) -> Tuple[List[int], List[float]]:
        """
        Split mass into possible values of nominal mass and mass defect.

        Returns
        -------
        nominal, defect: List[int], List[float]

        """
        _, d_max_p, d_min_n, _ = self.bounds.get_defect_bounds()
        min_nom = int(self.mass - d_max_p) + 1
        max_nom = int(self.mass - d_min_n)
        nominal = list(range(min_nom, max_nom + 1))
        defect = [self.mass - x for x in nominal]
        return nominal, defect

    def bound_negative_positive(self, defect: float) -> Tuple:
        """
        Bounds positive and negative contributions to mass defect.

        Parameters
        ----------
        defect: float

        Returns
        -------
        min_pos, max_pos, min_neg, max_neg: Tuple[float]

        """
        min_p, max_p, min_n, max_n = self.bounds.get_defect_bounds()
        max_pos = float(min(defect - min_n + self.tolerance, max_p))
        min_neg = float(max(defect - max_p - self.tolerance, min_n))
        min_pos = float(max(defect - max_n - self.tolerance, min_p))
        max_neg = float(min(defect - min_p + self.tolerance, max_n))
        return min_pos, max_pos, min_neg, max_neg


class FormulaGenerator:
    """
    Computes sum formulas based on exact mass.

    Attributes
    ----------
    n_results: int
    _results: dict
        a mapping of nominal masses of the results to a tuple of three arrays.
        The first array has the row index of positive coefficients, the second
        one has the row index of negative coefficients and the third array
        stores the number of carbons of the formula.

    Methods
    -------
    generate_formulas: find all compatible formulas using a species exact mass.
    results_to_array: return the results as an array where each row is a vector
        of formulas coefficients and each column is an isotope.
    results_to_str: return the results as a list of formula strings.
    """

    def __init__(
        self,
        elements: Iterable[str],
        mass: float,
        refine_h: bool = True,
        n_cluster: int = 5,
        user_bounds: Optional[dict] = None,
        min_defect: Optional[float] = None,
        max_defect: Optional[float] = None,
    ):
        """
        FormulaGenerator constructor.

        Parameters
        ----------
        elements: list[str].
            Each string can be an element symbol (eg: "C") or an isotope string
            representation (eg: "13C"). In the first case, the element is
            converted to the most abundant isotope ("C" ->  "12C").
        mass: float.
            Maximum mass to evaluate. This value is used to estimate the maximum
             coefficient possible for each element used (eg: with a mass of 500
             for 31P, the maximum value is floor(500 / 30.97376) = 16.
        refine_h: bool, True
            Used to find a better bound for the maximum number of H atoms.
            See Notes.
        n_cluster: int, 5.
            Used as a parameter to refine the H bounds.
        user_bounds: dict, optional
            A mapping from element symbols (or isotope representations) to
            a tuple of integers representing the lower and upper bounds. These
            values overwrite the default values only if they are a better bound
            that the default ones (i.e.: the new lower bounds are greater than
            the old ones). This parameter is used to restrict the values of
            elements that are expected to appear with low coefficient values,
            (i.e.: sulphur coefficient in small molecules will probably be lower
             than 10).
        min_defect: float, optional
            minimum mass defect allowed for the results. If None, all values are
            allowed. Default value is -1
        max_defect: float, optional
            maximum mass defect allowed for the results. If None, all values
            are allowed. Default is None.

        Notes
        -----
        The number of Hydrogen atoms estimation is made as follows: if m is
        composed of `n_cluster` alkane-like species, each one with mass
        :math: `m_{i}` and a number of hydrogen atoms equal to
        :math: `n_{Hi} = 2 n_{Ci} + 2`, then an upper bound for the total number
        of hydrogen is:math: `n_{H} < \frac{m}{7} + 2 n_{cluster}`
        """

        max_mass = mass
        self.bounds = Bounds.from_isotope_str(
            elements, max_mass, user_bounds=user_bounds
        )
        self._refine_h = refine_h
        self._n_cluster = n_cluster
        self._min_defect = min_defect
        self._max_defect = max_defect
        if refine_h:
            self.bounds.refine_h_upper_bounds(n_cluster=n_cluster)

        c12 = find_isotope("12C")
        self.has_carbon = c12 in self.bounds
        self._query = None
        self._results = None
        self.n_results = None

        # build Coefficients
        pos_restrictions, neg_restrictions = self.bounds.split_pos_neg()
        # add a dummy negative (or positive) isotope when only positive
        # (negative) isotopes are being used. This is a hack that prevents
        # making changes in the formula generation code...
        _add_dummy_isotope(pos_restrictions, neg_restrictions)
        self.pos = pos_restrictions.make_coefficients(mode="positive")
        self.neg = neg_restrictions.make_coefficients(mode="negative")

    def _add_query(self, mass: float, tolerance: float):
        bounds = self.bounds.get_mass_query_bounds(mass + tolerance)
        if self._refine_h:
            bounds.refine_h_upper_bounds(n_cluster=self._n_cluster)
        self._query = _MassQuery(mass, tolerance, bounds)

    def generate_formulas(self, mass: float, tolerance: float):
        """
        Computes formulas compatibles with the given query mass.

        Computes formulas for neutral species. If charged species are used, mass
        values must be corrected using the electron mass.
        Results are stored in an internal format, use `results_to_array` or
        `results_to_str` to obtain the compatible formulas.

        Parameters
        ----------
        mass : float
            Exact mass of the species.
        tolerance : float
            Tolerance to search compatible formulas.
        """
        self._add_query(mass, tolerance)
        min_d = self._min_defect
        max_d = self._max_defect
        self._results, self.n_results = _generate_formulas2(
            self._query, self.pos, self.neg, min_defect=min_d, max_defect=max_d
        )

    def results_to_array(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Convert results to an array of coefficients.

        Returns
        -------
        coefficients: np.array
            Formula coefficients. Each row is a formula, each column is an
            element.
        elements: list[str]
            Elements symbols associated to each column of coefficients.
        mass: array
            Mass associated to each row of coefficients.
        """
        if self._results:
            return _results_to_array(
                self._results, self.n_results, self.pos, self.neg, self.has_carbon
            )

    def results_to_str(self, return_isotopes: bool = False):
        """
        converts results to molecular formulas.

        Parameters
        ----------
        return_isotopes : bool
            If True include the mass number of each isotope

        Returns
        -------
        res: list[str]
            list of molecular formulas.
        """
        coefficients, isotopes, _ = self.results_to_array()
        if not return_isotopes:
            isotopes = [find_isotope(x).get_symbol() for x in isotopes]
        res = list()
        for k_rows in coefficients:
            k_formula = ""
            for e, c in zip(isotopes, k_rows):
                if c == 1:
                    k_formula += e
                elif c > 1:
                    k_formula += e + str(c)
            res.append(k_formula)
        return res

    @staticmethod
    def from_hmdb(mass: int, **kwargs):
        """
        Creates a FormulaGenerator instance using elemental bounds obtained
        from molecules present in the Human Metabolome database.

        Parameters
        ----------
        mass : {500, 1000, 1500, 2000}
            Creates a FormulaGenerator using molecular mass values lower than
            500, 1000, 1500 or 2000 respectively.
        kwargs: key value parameters to pass to the FormulaGenerator
            constructor.

        Returns
        -------
        FormulaGenerator

        """
        if mass == 500:
            bounds = {
                "C": (0, 34),
                "H": (0, 70),
                "N": (0, 10),
                "O": (0, 18),
                "P": (0, 4),
                "S": (0, 7),
            }
        elif mass == 1000:
            bounds = {
                "C": (0, 70),
                "H": (0, 128),
                "N": (0, 15),
                "O": (0, 31),
                "P": (0, 8),
                "S": (0, 7),
            }
        elif mass == 1500:
            bounds = {
                "C": (0, 100),
                "H": (0, 164),
                "N": (0, 23),
                "O": (0, 46),
                "P": (0, 8),
                "S": (0, 7),
            }
        elif mass == 2000:
            bounds = {
                "C": (0, 108),
                "H": (0, 190),
                "N": (0, 23),
                "O": (0, 61),
                "P": (0, 8),
                "S": (0, 8),
            }
        else:
            msg = "Valid mass values are 500, 1000, 1500 or 2000"
            raise ValueError(msg)

        if "user_bounds" in kwargs:
            bounds.update(kwargs["user_bounds"])
        kwargs["user_bounds"] = bounds
        elements = [str(x) for x in bounds]

        return FormulaGenerator(elements, mass, **kwargs)


class InvalidMass(ValueError):
    pass


class InvalidElement(KeyError):
    pass


class InvalidBound(ValueError):
    pass


def _validate_bounds(n_min: int, n_max: int):
    """
    Validates bounds for a given element

    Parameters
    ----------
    n_min: int
        lower bound.
    n_max: int
        upper bound.
    """
    if (not isinstance(n_min, int)) or (n_min < 0):
        raise InvalidBound("restrictions must be non-negative integers.")
    if not isinstance(n_max, int):
        raise InvalidBound("restrictions must be non-negative integers.")
    if n_min > n_max:
        raise InvalidBound("lower bound greater than upper bound.")


def _add_dummy_isotope(pos: Bounds, neg: Bounds):
    if len(pos) == 0:
        h1 = find_isotope("1H")
        pos[h1] = (0, 0)

    if len(neg) == 0:
        o16 = find_isotope("16O")
        neg[o16] = (0, 0)


def _results_to_array(
    results: Dict, n: int, pos: _Coefficients, neg: _Coefficients, has_carbon: bool
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Converts results from guess_formula to a numpy array of coefficients.
    """
    isotopes = pos.isotopes + neg.isotopes
    if has_carbon:
        isotopes = [find_isotope("12C")] + isotopes
        ic = 0
        i_pos = 1
        i_neg = i_pos + len(pos.isotopes)
    else:
        ic = 0
        i_pos = 0
        i_neg = i_pos + len(pos.isotopes)
    res = np.zeros((n, len(isotopes)), dtype=int)
    mass = np.zeros(n)
    start = 0
    for k in results:
        end = start + results[k][0].size
        res[start:end, i_pos:i_neg] = pos.coefficients[results[k][0], :]
        res[start:end, i_neg:] = neg.coefficients[results[k][1], :]
        if has_carbon:
            res[start:end, ic] = results[k][2]
            mass[start:end] = (
                pos.monoisotopic[results[k][0]]
                + 12 * np.array(results[k][2])
                + neg.monoisotopic[results[k][1]]
            )
        else:
            mass[start:end] = (
                pos.monoisotopic[results[k][0]] + neg.monoisotopic[results[k][1]]
            )
        start = end
    isotopes = [str(x) for x in isotopes]
    res, isotopes = _remove_dummy_isotopes(pos, neg, has_carbon, isotopes, res)
    return res, isotopes, mass


def _remove_dummy_isotopes(
    pos: _Coefficients,
    neg: _Coefficients,
    has_carbon: bool,
    isotopes: List[str],
    res: np.array,
) -> Tuple[np.array, List[str]]:

    has_pos = pos.coefficients.max() > 0
    has_neg = neg.coefficients.max() > 0

    if not has_pos:
        pop_ind = 1 if has_carbon else 0
        isotopes.pop(pop_ind)
        res = np.delete(res, pop_ind, axis=1)

    if not has_neg:
        pop_ind = res.shape[1] - 1
        isotopes.pop(pop_ind)
        res = np.delete(res, pop_ind, axis=1)

    return res, isotopes


def _make_coefficients_range(bounds: Bounds) -> List[range]:
    """
    Generate ranges to make formula coefficients
    Returns
    -------
    range_coeff: List(range)
    """
    coeff_ranges = list()
    for lb, ub in bounds.values():
        coeff_ranges.append(range(lb, ub + 1))
    return coeff_ranges


def _make_coefficients(
    bounds: Bounds, return_sorted: bool = True, mode: Optional[str] = None
) -> _Coefficients:
    """
    make Coefficients for positive and negative mode.

    Parameters
    ----------
    bounds: Bounds
    return_sorted: bool
        Sort coefficients by mass defect.
    mode: {"positive", "negative"}, optional
        if `positive`, reverses the order of the array created.

    Returns
    -------
    Coefficients namedtuple.
    """
    elements = list(bounds.keys())
    e_mono = np.array([isotope.m for isotope in bounds])
    e_nominal = np.array([isotope.a for isotope in bounds])
    e_defect = np.array([isotope.defect for isotope in bounds])
    # coefficient_range = _make_coefficients_range(bounds)
    range_list = _make_coefficients_range(bounds)
    # coefficients = cartesian_product(*coefficient_range)
    coefficients = cartesian_product_from_range_list(range_list)
    defect = np.matmul(coefficients, e_defect)
    if return_sorted:
        sorted_index = np.argsort(defect)
        if mode == "positive":
            sorted_index = sorted_index[::-1]
        coefficients = coefficients[sorted_index, :]
        defect = defect[sorted_index]
    monoisotopic = np.matmul(coefficients, e_mono)
    # remove coefficients with mass higher than the maximum mass
    valid_mass = monoisotopic <= bounds.mass
    coefficients = coefficients[valid_mass, :]
    defect = defect[valid_mass]
    monoisotopic = monoisotopic[valid_mass]
    # --------------------------------------
    nominal = np.matmul(coefficients, e_nominal)
    nominal_q, nominal_r = np.divmod(nominal, 12)

    if mode == "positive":
        fill = -np.inf
    else:
        fill = 0
    rd_array = _make_remainder_arrays(defect, nominal_r, fill_with=fill)
    ri_array = _make_remainder_arrays(np.arange(defect.size), nominal_r)

    coefficients = (
        elements,
        coefficients,
        monoisotopic,
        nominal,
        nominal_q,
        nominal_r,
        defect,
        rd_array,
        ri_array,
    )
    return _Coefficients(*coefficients)


def _make_remainder_arrays(x, r, fill_with=0):
    """
    Auxiliary function of _make_coefficients. makes a 2d array where each row
    has the same remainder.
    Parameters
    ----------
    x: array
    r: array
        array of remainders.

    Returns
    -------
    numpy.array with shape (12, max_size), where max_size is maximum number of
    elements of x with the same remainder. empty values are filled with np.inf
    if dtype of x is float or with max_size if dtype is int.
    """
    if (x.size == 1) and (np.isclose(x[0], 0.0)):
        # dummy array when no positive or negative defect elements are used
        return np.zeros(shape=(12, 1), dtype=x.dtype)
    else:
        r_to_x = dict()
        max_size = 0
        for k in range(12):
            xk = x[r == k]
            max_size = max(max_size, xk.size)
            r_to_x[k] = xk
        max_size += 1
        res = np.ones((12, max_size), dtype=x.dtype)
        res *= fill_with
        for k in range(12):
            xk = r_to_x[k]
            res[k, : xk.size] = xk
        return res


def _solve_generate_formulas_i(
    i: int,
    r: int,
    q: int,
    d: float,
    defect_bounds: Tuple[float, float, float, float],
    c_bounds: Tuple[int, int],
    tol: float,
    pos: _Coefficients,
    neg: _Coefficients,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # bound positive and negative mass defects
    min_dp, max_dp, min_dn, max_dn = defect_bounds
    dp_bounds = d - max_dp, d - min_dp
    # bounds for number of C atoms
    min_qc, max_qc = c_bounds

    # compute the matching values of positive and negative remainders to use
    rp = i
    rn = (r - i) % 12

    # find valid positive mass defect values
    pos_r_to_d = d - pos.r_to_defect[rp]
    p_start, p_end = np.searchsorted(pos_r_to_d, dp_bounds)
    p_end += 1

    # filter values based on valid number of C
    p_index = pos.r_to_index[rp, p_start:p_end]
    qp = pos.nominal_q[p_index]
    valid_qp = (q - qp) >= min_qc
    p_index = p_index[valid_qp]

    # find valid negative mass defect values
    neg_r_to_d = neg.r_to_defect[rn]
    neg_index = neg.r_to_index[rn]
    n_start = np.searchsorted(neg_r_to_d, pos_r_to_d[p_start:p_end][valid_qp] - tol)

    n_end = np.searchsorted(neg_r_to_d, pos_r_to_d[p_start:p_end][valid_qp] + tol)

    n_index_size = n_end - n_start
    valid_dn = n_index_size > 0
    n_start = n_start[valid_dn]
    n_end = n_end[valid_dn]
    n_index_size = n_index_size[valid_dn]
    p_index = p_index[valid_dn]
    if p_index.size:
        p_index = np.repeat(p_index, n_index_size)
        n_index = np.hstack([neg_index[s:e] for s, e in zip(n_start, n_end)])

        qp = pos.nominal_q[p_index]
        qn = neg.nominal_q[n_index]
        extra_c = int((rp + rn) >= 12)
        qc = q - qp - qn - extra_c

        # valid results
        valid_qc = (qc >= min_qc) & (qc <= max_qc)
        p_index = p_index[valid_qc]
        n_index = n_index[valid_qc]
        qc = qc[valid_qc]
    else:
        p_index = np.array([], dtype=int)
        n_index = np.array([], dtype=int)
        qc = np.array([], dtype=int)
    return p_index, n_index, qc


def _generate_formulas2(
    mq: _MassQuery,
    pos: _Coefficients,
    neg: _Coefficients,
    min_defect: Optional[float] = -1,
    max_defect: Optional[float] = None,
):
    """
    Computes formulas compatible with a given mass
    Parameters
    ----------
    mq: _MassQuery
    pos: _Coefficients
    neg: _Coefficients

    Returns
    -------
    res: dict
        a dictionary were each key is a possible nominal mass, and the values
        are a tuple of positive coefficients, negative coefficients and the
        carbon coefficient.
    n: integer
        Number of formulas generated.
    """

    # Algorithm
    # ---------
    # The formula search is based on splitting the mass into nominal mass and
    # mass defect: M = m + d (M: Monoisotopic mass, m: nominal mass, d: mass
    # defect). Several combinations of m and d are possible, but only one of
    # them is valid. The problem is solved for every combination of m and d.
    # m = k_1 * m_1 + ... + k_n * m_n, where m_i is the nominal mass of each
    # element, and n is the number of elements used. k_i is the formula
    # coefficient for each element. In the same way, the defect d can be written
    # as: d = k_1 * d_1 + ... + k_n * d_n.
    #
    # we want to find all k_1, ..., k_n such that:
    # |M - k_1 * m_1 + ... + k_n * m_n - k_1 * d_1 + ... + k_n * d_n| <= tol
    # If we suppose that we have estimated correctly m and d, the problem is
    # reduced to |d - k_1 * d_1 + ... + k_n * d_n| <= tol
    # To find all such K, the approach used is based on decomposing the
    # contribution of d into positive contributions and negative contributions:
    # d = dp + dn (dp: mass defect contribution of elements with positive mass
    # defect; dn: is the analogous with negative mass defects).
    # if we know all possible combinations of dn and dp, and sort them,
    # finding the valid values is equivalent to search for each dp value in
    # the interval [d - dn - tol, d - dn + tol]. Using a bisection search
    # this can be achieved in O(#dp * log(#dn)) where #dp is the cardinality
    # of dp.
    # To perform this step, all combinations of K with their associated values
    # of M, m, and d are generated for all elements, sorted by d and stored
    # in `Coefficients` (More on the data stored in coefficients later on...)
    #
    # Once we know all combinations with valid mass defects, we need to check
    # the validity of the nominal mass. This is done using the division
    # algorithm and modular arithmetic properties of the integers.
    # We can rewrite the nominal mass as m = mp + mn + mc (m: nominal mass,
    # mn: negative nominal mass, mp, positive nominal mass, mc: carbon mass).
    # If we use the algorithm of division we have:
    # m = 12 * q + r; 0 <= r < 12
    # mc = 12 * qc
    # mn = 12 * qn + rn; 0 <= rn < 12
    # mp = 12 * qp + rp; 0 <= rp < 12
    # If we replace mp, mc, and mn we have:
    # m = 12 * (qc + qp + qc) + rn + rp
    # We have two cases:
    #      1. rn + rp < 12
    #      2. rn + rp >= 12
    # In the first case, we have that q = qc + qp + qn. From this expression
    # we can recover the number of carbons.
    # In the other case, we have that
    # m = 12 * (qc + qp + qc + 1) + (rn + rp) % 12
    # From the uniqueness of q and r, we have that q = qc + qp + qn + 1
    # and r = (rn + rp) % 12.
    # It's easy to see that r < rp if and only if r < rn. If this is true,
    # we are in the second case. Using this, the test for a valid number of
    # carbon atoms is going to be:
    # if r < rp then the formula is valid if r - rp + 12 == rn
    # if r >= rp then the formula is valid if r - rp == rn
    # to speed up calculations, `Coefficients` has a dictionary where each key
    # is the remainder of the nominal mass and the values are the mass defects
    # sorted. In this way, we can check in a quick way that the mass defect and
    # the remainder is correct. The last step to check that the formula is valid
    # is to examine the value of qc. If the molecule doesn't have carbon,
    # qc = 0. Otherwise, qc > 0. qc value is checked based on restrictions
    # on the number of carbons.

    nominal, defects = mq.split_nominal_defect()
    tol = mq.tolerance
    c12 = find_isotope("12C")
    if c12 in mq.bounds:
        c_bounds = mq.bounds[c12]
    else:
        c_bounds = 0, 0

    if min_defect is None:
        min_defect = -np.inf

    if max_defect is None:
        max_defect = np.inf

    res = dict()
    n = 0  # number of valid formulas
    for nom, d in zip(nominal, defects):
        if (d < min_defect) or (d > max_defect):
            continue
        q, r = divmod(nom, 12)

        # min and max possible max defects for negative and positive elements
        defect_bounds = mq.bound_negative_positive(d)

        # pos_coeff_index is an array of indices to positive coefficients with
        # valid mass defects. neg_ri_slices is an array that for each has a
        # range of indices of valid negative coefficients in the r_to_index
        # array. neg_r is the remainder associated to each slice.

        pos_index = list()
        neg_index = list()
        qc = list()
        for i in range(12):
            i_pos_coeff, i_neg_coeff, i_qc = _solve_generate_formulas_i(
                i, r, q, d, defect_bounds, c_bounds, tol, pos, neg
            )
            i_n = i_pos_coeff.size
            if i_n:
                n += i_n
                pos_index.append(i_pos_coeff)
                neg_index.append(i_neg_coeff)
                qc.append(i_qc)
        if pos_index:
            res[nom] = (np.hstack(pos_index), np.hstack(neg_index), np.hstack(qc))
    return res, n
