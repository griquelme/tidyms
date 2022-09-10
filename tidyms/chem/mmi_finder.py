import numpy as np
from math import gcd
import bisect
from typing import Dict, List, Tuple
from .atoms import Element, PTABLE, EM
from .formula import Formula
from ._isotope_distributions import make_coeff_abundances
from .utils import cartesian_product_from_range_list


class MMIFinder:
    """
    Finds Minimum Mass Isotopologue (MMI) candidates using an array of feature
    m/z and an array of feature area.

    """
    def __init__(
        self,
        bounds: Dict[str, Tuple[int, int]],
        max_mass: float,
        max_charge: int,
        length: int,
        bin_size: int,
        mz_tol: float,
        p_tol: float
    ):
        """
        Constructor method.

        Parameters
        ----------
        bounds : Dict
            Mapping from an element symbol str to the minimum and maximum
            allowed values in formulas.
        max_mass : float
            Maximum mass to build rules.
        length : int
            length of the theoretical envelopes used to compute the search
            rules.
        bin_size : int
            Mass interval used to build the rules.
        mz_tol : float
            m/z tolerance to search candidates.
        p_tol : float
            abundance tolerance used to search candidates.

        """
        self.rules = _create_rules_dict(bounds, max_mass, length, bin_size)
        self.bin_size = bin_size
        self.max_charge = abs(max_charge)
        self.polarity = 1 if max_charge >= 0 else -1
        self.max_mass = max_mass
        self.mz_tol = mz_tol
        self.p_tol = p_tol

    def find(
        self,
        mz: np.ndarray,
        sp: np.ndarray,
        mono_index: int
    ) -> List[Tuple[int, int]]:

        """
        Search MMI candidates using m/z and area information from a feature
        list.

        Parameters
        ----------
        mz : array
            Sorted array of m/z values of features
        sp : array
            Area array of features.
        mono_index : int
            Index of the most intense value in mz

        Returns
        -------
        mmi_candidates: List[Tuple[int, int]]
            List of candidates assuming that the monoisotopic index is part of
            the envelope but not the MMI.

        """
        mono_sp = sp[mono_index]
        mono_mz = mz[mono_index]
        # monoisotopic m/z to possible mass values
        # charge sign and electron mass can be ignored as computations are done
        # using mass differences
        mono_M_list, charge_list = _get_valid_mono_mass(
            mono_mz, self.max_charge, self.polarity, self.max_mass
        )
        candidates = [(mono_index, q) for q in charge_list]
        for M, charge in zip(mono_M_list, charge_list):
            M_bin = int(M // self.bin_size)
            mmi_rules = self.rules.get(M_bin)
            if mmi_rules is not None:
                for i_rules in mmi_rules:
                    i_candidates = _find_candidate(
                        mz, sp, M, charge, mono_sp, i_rules, self.mz_tol,
                        self.p_tol
                    )
                    candidates.extend(i_candidates)
        return candidates


def _get_valid_mono_mass(
    mz_mono: float,
    max_charge: int,
    polarity: int,
    max_mass: float
) -> Tuple[List[float], List[int]]:
    charge_list = list()
    mono_M_list = list()
    for q in range(1, max_charge + 1):
        M = q * mz_mono - polarity * q * EM
        if M <= max_mass:
            charge_list.append(q)
            mono_M_list.append(M)
    return mono_M_list, charge_list


def _find_candidate(
    mz: np.ndarray,
    sp: np.ndarray,
    mono_M: float,
    charge: int,
    mono_sp: float,
    i_rules: Dict,
    mz_tol: float,
    p_tol: float
) -> List[Tuple[int, int]]:
    # search valid m/z values
    min_dM, max_dM = i_rules["dM"]
    min_mz = (mono_M - max_dM) / charge - mz_tol
    max_mz = (mono_M - min_dM) / charge + mz_tol
    min_qp = i_rules["qp"][0] - p_tol
    max_qp = i_rules["qp"][1] + p_tol
    start = bisect.bisect(mz, min_mz)
    end = bisect.bisect(mz, max_mz)
    # if valid m/z where found, check if the abundance quotient qp is valid
    if start < end:
        candidates = np.arange(start, end)
        qp_ind = sp[candidates] / mono_sp
        valid_mask = (qp_ind >= min_qp) & (qp_ind <= max_qp)
        candidates = [(x, charge) for x in candidates[valid_mask]]
    else:
        candidates = list()
    return candidates


def _create_rules_dict(
    bounds: Dict[str, Tuple[int, int]],
    max_mass: float,
    length: int,
    bin_size: int
) -> Dict[int, List[Dict[str, Tuple[float, float]]]]:
    _, Ma, pa = _create_envelope_arrays(bounds, max_mass, length)
    # find the monoisotopic index, its Mass difference with the MMI (dM) and
    # its abundance quotient with the MMI (qp)
    mono_index = np.argmax(pa, axis=1)
    dM = Ma[np.arange(Ma.shape[0]), mono_index] - Ma[:, 0]
    qp = pa[np.arange(Ma.shape[0]), mono_index] / pa[:, 0]
    bins = (Ma[:, 0] // bin_size).astype(int)

    # find unique values for bins and monoisotopic index that will be used
    # as key for the rule dictionary
    unique_bins = np.unique(bins)
    unique_mono_index = np.unique(mono_index)
    unique_mono_index = unique_mono_index[unique_mono_index > 0]

    rules = dict()

    for b in unique_bins:
        b_rules = list()
        for mi in unique_mono_index:
            mask = (mono_index == mi) & (bins == b)
            if mask.any():
                mi_rules = dict()
                dM_b_mi = dM[mask]
                qp_b_mi = qp[mask]
                mi_rules["dM"] = dM_b_mi.min(), dM_b_mi.max()
                mi_rules["qp"] = qp_b_mi.min(), qp_b_mi.max()
                b_rules.append(mi_rules)
        if b_rules:
            rules[b] = b_rules
    return rules


def _create_envelope_arrays(
    bounds: Dict[str, Tuple[int, int]],
    max_mass: float,
    length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    elements = [PTABLE[x] for x in bounds]
    elements = _get_relevant_elements(elements)
    isotopes = [min(x.isotopes.values(), key=lambda x: x.a) for x in elements]
    bounds = [bounds[x.symbol] for x in elements]
    bound_range = [range(x, y + 1) for x, y in bounds]
    coeff = cartesian_product_from_range_list(bound_range)
    ma, Ma, pa = make_coeff_abundances(bounds, coeff, isotopes, length)
    mask = Ma[:, 0] <= max_mass
    ma = ma[mask, :]
    Ma = Ma[mask, :]
    pa = pa[mask, :]
    return ma, Ma, pa


def _get_relevant_elements(e_list: List[Element]) -> List[Element]:
    res = set(e_list)
    for e in e_list:
        if len(e.isotopes) > 1:
            tmp = _get_different_elements(e_list, e)
            res = res.intersection(tmp)
    res = list(res)
    return res


def _get_different_elements(
    e_list: List[Element],
    e_ref: Element
) -> List[Element]:
    """
    Filter elements based on if they affect the envelope of formulas in a
    different way than e_ref
    Parameters
    ----------
    e_list : List[Element]
    e_ref : Element

    Returns
    -------
    List[Element]
    """
    res = list()
    for e in e_list:
        if e == e_ref:
            res.append(e)
        elif _is_distort_envelope(e_ref, e):
            res.append(e)
    return res


def _is_distort_envelope(e1: Element, e2: Element) -> bool:
    """
    Checks if an element distorts the isotopic envelope of a formula with in a
    different way than e1.

    Parameters
    ----------
    e1 : Element
    e2 : Element

    Returns
    -------
    bool

    """
    e1_m, _, e1_p = e1.get_abundances()
    e2_m, _, e2_p = e2.get_abundances()

    if e2_p.size == 1:
        # if the length is 1, no contribution is made to the envelope shape
        res = False
    elif e1_p.size != e2_p.size:
        # different shapes imply different contributions to the envelope shape
        res = True
    else:
        e1_m0 = e1_m[0]
        e2_m0 = e2_m[0]
        e1_dm = e1_m - e1_m0
        e2_dm = e2_m - e2_m0
        if np.array_equal(e1_dm, e2_dm):
            res = _compare_abundance_equal_envelopes(e1, e2)
        else:
            # If the relative nominal mass differences are different, the
            # contribution to the shape is different
            res = True
    return res


def _compare_abundance_equal_envelopes(e1: Element, e2: Element) -> bool:
    # computes a formula with the same nominal mass using only e1 and e2
    # and compares the abundances
    e1_m0 = min(e1.isotopes)
    e2_m0 = min(e2.isotopes)
    lcm = _lcm(e1_m0, e2_m0)
    f1 = Formula(e1.symbol + str(lcm // e1_m0))
    _, _, f1_p = f1.get_isotopic_envelope()
    f2 = Formula(e2.symbol + str(lcm // e2_m0))
    _, _, f2_p = f2.get_isotopic_envelope()
    f1_p_argmax = np.argmax(f1_p)
    different_max = f1_p_argmax != np.argmax(f2_p)
    if different_max:
        res = True
    else:
        length = min(f1_p.size, f2_p.size)
        different_shape = f1_p[:length] < f2_p[:length]
        different_shape[f1_p_argmax] = False
        res = different_shape.any()
    return res


def _lcm(a: int, b: int) -> int:
    """
    Computes the least common multiple between a and b.

    """
    return abs(a * b) // gcd(a, b)
