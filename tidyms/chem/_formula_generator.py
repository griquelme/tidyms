# -*- coding: utf-8 -*-
"""
Functions to calculate compatible molecular formulas within a given tolerance.

"""

import numpy as np
from .utils import cartesian_product
from typing import Dict, List, Optional, Tuple
from .atoms import Isotope, PeriodicTable


# name conventions used:
# M is used for molecular mass
# m is used for nominal mass
# d is used for mass defect
# e.g. for (12C)(1H)2(16O)2, M = 46.0055; m = 46; d = 0.0055
# dp is used for the contribution to d of isotopes with positive mass defect
# dn is used for the contribution to d of isotopes with negative mass defect
# d = dp + dn
# for (12C)(1H)2(16O)2 dp = 0.0157, dn = -0.0102.
# q and r are used for the quotient and remainder of the division of m by 12
# for (12C)(1H)2(16O)2 q = 3; r = 10
# mp is used for the contribution to m of isotopes with positive mass defect
# mn is used for the contribution to m of isotopes with negative mass defect
# qp and rp are the quotient and remainder of the division of mp by 12
# qn and rn are the quotient and remainder of the division of mn by 12
# mc is used fot the contribution to m of 12C.
# qc is the quotient of division of mc by 12.
# rc is not used but it is always zero.


class FormulaGenerator:
    """
    Generates sum formulas based on exact mass values.

    Attributes
    ----------
    n_results: int
        Number of valid formulas generated.
    results: dict
        a mapping of nominal masses of the results to a tuple of three arrays:
        1. the row index of positive coefficients.
        2. the row index of negative coefficients.
        3. the number of 12C in the formula.

    Methods
    -------
    generate_formulas
    results_to_array
    from_hmdb

    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[int, int]],
        max_M: Optional[float] = None,
        min_defect: Optional[float] = None,
        max_defect: Optional[float] = None,
    ):
        """
        FormulaGenerator constructor.

        Parameters
        ----------
        bounds: Dict
            A dictionary from strings with isotopes to lower and upper bounds of
            formulas coefficients. Isotope strings can be an element symbol (eg:
            "C") or an isotope string representation (eg: "13C"). In the first
            case, the element is converted to the most abundant isotope ("12C").
        max_M : float or None, default=None
            Maximum mass value for generated formulas. If specified it is used
            to update the bounds. For examples is ``max_M=300`` and the bounds
            for 32S are ``(0, 10)``, then they are updated to ``(0, 9)``.
        min_defect: float or None, default=None
            Minimum mass defect allowed for the results. If None, all values are allowed.
        max_defect: float or None, default=None
            Maximum mass defect allowed for the results. If None, all values are allowed.

        Examples
        --------
        >>> import tidyms as ms
        >>> fg_bounds = {"C": (0, 5), "H": (0, 10), "O": (0, 4)}
        >>> fg = ms.chem.FormulaGenerator(fg_bounds)

        """
        self.bounds = FormulaCoefficientBounds.from_isotope_str(bounds)
        if max_M is not None:
            self.bounds = self.bounds.bounds_from_mass(max_M)
        else:
            max_M = max(v[1] * k.m for k, v in self.bounds.bounds.items())
        dp_min, dp_max, dn_min, dn_max = self.bounds.get_defect_bounds()

        if min_defect is None:
            min_defect = dn_min + dp_min
        self._min_defect = min_defect

        if max_defect is None:
            max_defect = dp_max + dn_max
        self._max_defect = max_defect

        c12 = PeriodicTable().get_isotope("12C")
        self._has_carbon = c12 in self.bounds.bounds
        self._query = None
        self.results = None
        self.n_results = None

        # build Coefficients
        pos_restrictions, neg_restrictions = self.bounds.split_pos_neg()
        # add a dummy negative (or positive) isotope when only positive
        # (negative) isotopes are being used. This is a hack that prevents
        # making changes in the formula generation code...
        _add_dummy_isotope(pos_restrictions, neg_restrictions)
        self.pos = pos_restrictions.make_coefficients(max_M, reverse=True)
        self.neg = neg_restrictions.make_coefficients(max_M)

    def __repr__(self):
        str_repr = "FormulaGenerator(bounds={}, min_defect={}, max_defect={})"
        return str_repr.format(self.bounds.bounds, self._min_defect, self._max_defect)

    def generate_formulas(self, M: float, tolerance: float):
        """
        Computes formulas compatibles with the given query mass. The formulas
        are computed assuming neutral species. If charged species are used, mass
        values must be corrected using the electron mass.

        Results are stored in an internal format, use `results_to_array` to
        obtain the compatible formulas.

        Parameters
        ----------
        M : float
            Exact mass used for formula generation.
        tolerance : float
            Tolerance to search compatible formulas.

        Examples
        --------
        >>> import tidyms as ms
        >>> fg_bounds = {"C": (0, 5), "H": (0, 10), "O": (0, 4)}
        >>> fg = ms.chem.FormulaGenerator(fg_bounds)
        >>> fg.generate_formulas(46.042, 0.005)

        """
        if M <= 0.0:
            msg = "`M` must be a positive number. Got {}".format(M)
            raise ValueError(msg)

        if tolerance <= 0.0:
            msg = "`tolerance` must be a positive number. Got {}".format(tolerance)
            raise ValueError(msg)

        self.results, self.n_results = _generate_formulas(
            M,
            tolerance,
            self.bounds,
            self.pos,
            self.neg,
            min_defect=self._min_defect,
            max_defect=self._max_defect,
        )

    def results_to_array(self) -> Tuple[np.ndarray, List[Isotope], np.ndarray]:
        """
        Convert results to an array of coefficients.

        Returns
        -------
        coefficients: np.array
            Formula coefficients. Each row is a formula, each column is an isotope.
        isotopes: list[Isotopes]
            Isotopes associated to each column of `coefficients`.
        M: array
            Exact mass associated to each row of `coefficients`.

        Examples
        --------
        >>> import tidyms as ms
        >>> fg_bounds = {"C": (0, 5), "H": (0, 10), "O": (0, 4)}
        >>> fg = ms.chem.FormulaGenerator(fg_bounds)
        >>> fg.generate_formulas(46.042, 0.005)
        >>> coeff, isotopes, M = fg.results_to_array()

        """
        if self.results:
            return _results_to_array(
                self.results, self.n_results, self.pos, self.neg, self._has_carbon
            )

    @staticmethod
    def from_hmdb(
        mass: int,
        bounds: Optional[Dict[str, Tuple[int, int]]] = None,
        min_defect: Optional[float] = None,
        max_defect: Optional[float] = None,
    ):
        """
        Creates a FormulaGenerator using elemental bounds obtained from
        molecules present in the Human Metabolome database. By default, bounds
        for CHNOPS elements are included.

        Parameters
        ----------
        mass : {500, 1000, 1500, 2000}
            Bounds are created using molecules with molecular mass lower than this value.
        bounds: Dict[str, Tuple[int, int]] or None, default=None
            Passes additional isotopes to the generator.
        min_defect: float or None, default=None
            minimum mass defect allowed for the results. If None, all values are allowed.
        max_defect: float or None, default=None
            maximum mass defect allowed for the results. If None, all values are allowed.

        Returns
        -------
        FormulaGenerator

        Examples
        --------
        >>> import tidyms as ms
        # creates a formula generator using a max mass of 500.
        # Also include chlorine to the bounds.
        >>> fg = ms.chem.FormulaGenerator.from_hmdb(500, bounds={"Cl": (0, 2)})

        """
        if mass == 500:
            hmdb_bounds = {
                "C": (0, 34),
                "H": (0, 70),
                "N": (0, 10),
                "O": (0, 18),
                "P": (0, 4),
                "S": (0, 7),
            }
        elif mass == 1000:
            hmdb_bounds = {
                "C": (0, 70),
                "H": (0, 128),
                "N": (0, 15),
                "O": (0, 31),
                "P": (0, 8),
                "S": (0, 7),
            }
        elif mass == 1500:
            hmdb_bounds = {
                "C": (0, 100),
                "H": (0, 164),
                "N": (0, 23),
                "O": (0, 46),
                "P": (0, 8),
                "S": (0, 7),
            }
        elif mass == 2000:
            hmdb_bounds = {
                "C": (0, 108),
                "H": (0, 190),
                "N": (0, 23),
                "O": (0, 61),
                "P": (0, 8),
                "S": (0, 8),
            }
        else:
            msg = "Valid mass values are 500, 1000, 1500 or 2000. Got {}."
            raise ValueError(msg.format(mass))
        if bounds is not None:
            hmdb_bounds.update(bounds)
        return FormulaGenerator(hmdb_bounds, min_defect=min_defect, max_defect=max_defect)


class FormulaCoefficientBounds:
    """
    Mapping from isotopes to upper and lower bounds

    Attributes
    ----------
    bounds : Dict[Isotope, Tuple[int, int]]

    """

    def __init__(self, bounds: Dict[Isotope, Tuple[int, int]]):
        self.bounds = bounds

    def __repr__(self):
        return "FormulaCoefficientBounds({})".format(self.bounds)

    def __getitem__(self, item):
        return self.bounds[item]

    def bounds_from_mass(self, M: float) -> "FormulaCoefficientBounds":
        """
        Compute the mass-based bounds for each isotope.

        The bounds are refined using the values for each isotope.

        """
        bounds = dict()
        for i, (lb, ub) in self.bounds.items():
            lower = max(0, lb)
            upper = min(int(M / i.m), ub)
            bounds[i] = lower, upper
        return FormulaCoefficientBounds(bounds)

    def bound_negative_positive_defect(
        self, defect: float, tolerance: float
    ) -> Tuple[float, float, float, float]:
        """
        Bounds positive and negative contributions to mass defect.

        Parameters
        ----------
        defect : float
            Mass defect value of the Query.
        tolerance : float

        Returns
        -------
        min_pos, max_pos, min_neg, max_neg: Tuple[float]

        """
        min_p, max_p, min_n, max_n = self.get_defect_bounds()
        max_pos = float(min(defect - min_n, max_p) + tolerance)
        min_neg = float(max(defect - max_p, min_n) - tolerance)
        min_pos = float(max(defect - max_n, min_p) - tolerance)
        max_neg = float(min(defect - min_p, max_n) + tolerance)
        return min_pos, max_pos, min_neg, max_neg

    def get_nominal_defect_candidates(self, M: float) -> Tuple[List[int], List[float]]:
        """
        Split mass into possible values of nominal mass and mass defect.

        Returns
        -------
        nominal, defect: List[int], List[float]

        """
        dp_min, dp_max, dn_min, dn_max = self.get_defect_bounds()
        m_min = int(M - dn_max - dp_max) + 1
        m_max = int(M - dp_min - dn_min) + 1
        m_candidates = list(range(m_min, m_max))
        d_candidates = [M - x for x in m_candidates]
        return m_candidates, d_candidates

    def split_pos_neg(
        self,
    ) -> Tuple["FormulaCoefficientBounds", "FormulaCoefficientBounds"]:
        """
        creates two new Bound objects, one with elements with positive mass
        defect and other with elements with negative mass defects.

        Returns
        -------
        pos: FormulaCoefficientBounds
        neg: FormulaCoefficientBounds

        """
        pos_bounds = dict()
        neg_bounds = dict()
        for isotope, bounds in self.bounds.items():
            defect = isotope.defect
            if defect > 0:
                pos_bounds[isotope] = bounds
            elif defect < 0:
                neg_bounds[isotope] = bounds
        pos = FormulaCoefficientBounds(pos_bounds)
        neg = FormulaCoefficientBounds(neg_bounds)
        return pos, neg

    def get_defect_bounds(self) -> Tuple[float, float, float, float]:
        """
        Computes the minimum and maximum value of the mass defect.

        Returns
        -------
        min_positive, max_positive, min_negative, max_negative: tuple

        """
        min_positive, max_positive, min_negative, max_negative = 0, 0, 0, 0
        for isotope, (lb, ub) in self.bounds.items():
            defect = isotope.defect
            min_tmp = defect * lb
            max_tmp = defect * ub
            if defect > 0:
                min_positive += min_tmp
                max_positive += max_tmp
            else:
                min_negative += max_tmp
                max_negative += min_tmp
        return min_positive, max_positive, min_negative, max_negative

    def make_coefficients(
        self, max_M: float, reverse: bool = False
    ) -> "FormulaCoefficients":
        """
        Generate coefficients for FormulaGenerator

        Parameters
        ----------
        max_M: float
        reverse: bool

        Returns
        -------
        coefficients: Coefficients

        """
        return FormulaCoefficients(self, max_M, True, reverse)

    @staticmethod
    def from_isotope_str(
        bounds: Dict[str, Tuple[int, int]]
    ) -> "FormulaCoefficientBounds":
        """
        Creates a _Bounds instance from a list of isotope strings.

        Parameters
        ----------
        bounds : Dict[str, Tuple[int, int]]

        Returns
        -------
        FormulaCoefficientBounds
        """

        ptable = PeriodicTable()
        res = dict()
        for i, ib in bounds.items():
            lb, ub = ib
            isotope = ptable.get_isotope(i)
            invalid_lb = (not isinstance(lb, int)) or (lb < 0)
            invalid_ub = (not isinstance(ub, int)) or (ub < lb)
            if invalid_lb or invalid_ub:
                msg = "Invalid bounds for {}. Bounds must be non-negative integers. Got {}.".format(
                    i, ib
                )
                raise ValueError(msg)
            res[isotope] = ib
        return FormulaCoefficientBounds(res)


class FormulaCoefficients:
    """
    Named tuple with elements with positive/negative mass defect.

    Attributes
    ----------
    coefficients : np.array[int]
        Formula coefficients. Each row is a formula, each column is an isotope.
    isotopes : List[Isotopes]
        element associated to each column of coefficients.
    M : array[float]
        monoisotopic mass associated to each row of coefficients.
    q : array[int]
        quotient between the nominal mass and 12.
    r_to_index : Dict[int, array]
        Maps remainders of division of m by 12 to rows of `coefficients`.
    r_to_d : Dict[int, array]
        Maps remainders of division of m by 12 to mass defect  values. Each
        value match to the corresponding index in `r_to_index`.

    """

    def __init__(
        self,
        bounds: FormulaCoefficientBounds,
        max_mass: float,
        return_sorted: bool,
        return_reversed: bool,
    ):
        self.isotopes = list(bounds.bounds)
        i_M = np.array([isotope.m for isotope in self.isotopes])
        i_m = np.array([isotope.a for isotope in self.isotopes])
        i_d = np.array([isotope.defect for isotope in self.isotopes])

        # create coefficients array
        range_list = [range(lb, ub + 1) for lb, ub in bounds.bounds.values()]
        coefficients = cartesian_product(*range_list)

        # sort coefficients by mass defect
        d = np.matmul(coefficients, i_d)
        if return_sorted:
            sorted_index = np.argsort(d)
            if return_reversed:
                sorted_index = sorted_index[::-1]
            coefficients = coefficients[sorted_index, :]
            d = d[sorted_index]

        # remove coefficients with mass higher than the maximum mass
        M = np.matmul(coefficients, i_M)
        valid_M = M <= max_mass
        coefficients = coefficients[valid_M, :]
        d = d[valid_M]
        M = M[valid_M]

        # Compute nominal mass, quotient and remainder
        m = np.matmul(coefficients, i_m)
        q, r = np.divmod(m, 12)

        # group mass defects and coefficient row index by remainder value
        r_to_d = _make_remainder_arrays(d, r)
        r_to_index = _make_remainder_arrays(np.arange(d.size), r)

        self.M = M
        self.coefficients = coefficients
        self.q = q
        self.r_to_index = r_to_index
        self.r_to_d = r_to_d


class _MassQuery(object):
    """
    Stores values for a Mass query.

    Attributes
    ----------
    m : int
        nominal mass of the query.
    d : float
        mass defect of the query.
    q : int
        quotient of the division of `m` by 12.
    r : int
        remainder of the division of `m` by 12.
    nc_min : int
        Minimum number of 12C in the generated formulas.
    nc_max : int
        Maximum number of 12C in the generated formulas.
    dn_min : float
        Minimum value of dn in generated formulas.
    dn_max : float
        Maximum value of dn in generated formulas.
    dp_min : float
        Minimum value of dp in generated formulas.
    dp_max : float
        Maximum value of dp in generated formulas.

    """

    def __init__(self, m: int, d: float, tol: float, bounds: FormulaCoefficientBounds):
        self.m = m
        self.d = d
        self.q, self.r = divmod(m, 12)
        self.tol = tol
        d_bounds = bounds.bound_negative_positive_defect(d, tol)
        self.dp_min, self.dp_max, self.dn_min, self.dn_max = d_bounds

        c12 = PeriodicTable().get_isotope("12C")
        if c12 in bounds.bounds:
            self.nc_min, self.nc_max = bounds.bounds[c12]
        else:
            self.nc_min, self.nc_max = 0, 0


def _add_dummy_isotope(pos: FormulaCoefficientBounds, neg: FormulaCoefficientBounds):
    """
    Add dummy isotopes to positive or negative elements to solve the mass
    defect problem in cases that there aren't any positive/negative isotopes.

    """
    ptable = PeriodicTable()
    if len(pos.bounds) == 0:
        h1 = ptable.get_isotope("1H")
        pos.bounds[h1] = (0, 0)

    if len(neg.bounds) == 0:
        o16 = ptable.get_isotope("16O")
        neg.bounds[o16] = (0, 0)


def _results_to_array(
    results: Dict,
    n: int,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    has_carbon: bool,
) -> Tuple[np.ndarray, List[Isotope], np.ndarray]:
    """
    Converts results from guess_formula to a numpy array of coefficients.
    """
    isotopes = pos.isotopes + neg.isotopes
    if has_carbon:
        isotopes = [PeriodicTable().get_isotope("12C")] + isotopes
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
                pos.M[results[k][0]]
                + 12 * np.array(results[k][2])
                + neg.M[results[k][1]]
            )
        else:
            mass[start:end] = pos.M[results[k][0]] + neg.M[results[k][1]]
        start = end
    res, isotopes = _remove_dummy_isotopes(pos, neg, has_carbon, isotopes, res)
    return res, isotopes, mass


def _remove_dummy_isotopes(
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    has_carbon: bool,
    isotopes: List[Isotope],
    res: np.array,
) -> Tuple[np.array, List[Isotope]]:

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


def _make_remainder_arrays(x, r) -> Dict[int, np.ndarray]:
    """
    Creates a dictionary where each key is a value of r and the values are the
    corresponding x values.

    Auxiliary function of _make_coefficients.

    Parameters
    ----------
    x: array
    r: array
        array of remainders.

    Returns
    -------
    r_to_x : Dict

    """
    r_to_x = dict()
    for k in range(12):
        xk = x[r == k]
        r_to_x[k] = xk
    return r_to_x


def _generate_formulas(
    M: float,
    tol: float,
    bounds: FormulaCoefficientBounds,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    min_defect: float,
    max_defect: float
):
    """
    Finds formulas compatible with a given mass.

    Parameters
    ----------
    M : float
    tol : float
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

    bounds = bounds.bounds_from_mass(M)
    # possible values of nominal mass and mass defect based on the coefficient bounds.
    m_candidates, d_candidates = bounds.get_nominal_defect_candidates(M)
    res = dict()
    n = 0  # number of valid formulas
    for m, d in zip(m_candidates, d_candidates):

        if (d < min_defect) or (d > max_defect):
            continue

        query = _MassQuery(m, d, tol, bounds)

        pos_index = list()
        neg_index = list()
        qc = list()
        for i in range(12):
            results = _generate_formulas_i(i, query, pos, neg)
            if results is not None:
                pos_index_i, neg_index_i, qc_i = results
                n += pos_index_i.size
                pos_index.append(pos_index_i)
                neg_index.append(neg_index_i)
                qc.append(qc_i)
        if pos_index:
            res[m] = (np.hstack(pos_index), np.hstack(neg_index), np.hstack(qc))
    return res, n


def _generate_formulas_i(
    i: int,
    query: _MassQuery,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Solve the mass defect problem for fixed rp and rn values.

    Auxiliary function to _generate_formulas.

    """

    # solves the mass defect problem for fixe rp and rn values.
    # Finds all positive coefficients with rp == i and their matching negative
    # such that |d - dp - dn | <= tol.
    # These values are filtered taking into account the number of 12C, qc, as
    # follows: min_nc <= qc <= max_nc
    # the results are organized into three arrays
    # p_index contains the index to a row of pos.coeff
    # n_index contains the index to a row of neg.coeff
    # q_c contains the number of 12C in the formula

    rp = i
    rn = (query.r - i) % 12

    # find valid positive mass defect values
    rel_dp = query.d - pos.r_to_d[rp]
    rel_dp_bounds = query.d - query.dp_max, query.d - query.dp_min
    p_start, p_end = np.searchsorted(rel_dp, rel_dp_bounds)
    rel_dp = rel_dp[p_start:p_end]

    # filter values based on valid number of C
    p_index = pos.r_to_index[rp][p_start:p_end]
    qp = pos.q[p_index]
    valid_qp = (query.q - qp) >= query.nc_min
    rel_dp = rel_dp[valid_qp]
    p_index = p_index[valid_qp]

    # find valid negative mass defect values
    dn = neg.r_to_d[rn]
    n_index = neg.r_to_index[rn]
    n_start = np.searchsorted(dn, rel_dp - query.tol)
    n_end = np.searchsorted(dn, rel_dp + query.tol)

    # create three arrays, where each element corresponds to an index of valid
    # positive coeff, negative coeff and number of 12C atoms.
    n_index_size = n_end - n_start
    valid_dn = n_index_size > 0
    n_start = n_start[valid_dn]
    n_end = n_end[valid_dn]
    n_index_size = n_index_size[valid_dn]
    p_index = p_index[valid_dn]
    if p_index.size:
        p_index = np.repeat(p_index, n_index_size)
        n_index = np.hstack([n_index[s:e] for s, e in zip(n_start, n_end)])

        qp = pos.q[p_index]
        qn = neg.q[n_index]
        extra_c = int((rp + rn) >= 12)
        qc = query.q - qp - qn - extra_c

        # valid results
        valid_qc = (qc >= query.nc_min) & (qc <= query.nc_max)
        p_index = p_index[valid_qc]
        n_index = n_index[valid_qc]
        qc = qc[valid_qc]
        if p_index.size:
            results = p_index, n_index, qc
        else:
            results = None
    else:
        results = None
    return results
