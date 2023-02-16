import numpy as np
import bisect
from typing import Dict, List, Optional, Tuple
from .atoms import Element, PeriodicTable, EM
from ._formula_generator import FormulaCoefficientBounds
from .envelope_tools import make_formula_coefficients_envelopes


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
        p_tol: float,
        custom_abundances: Optional[Dict[str, np.ndarray]] = None
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
        custom_abundances : dict, optional
            Provides custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.

        """
        self.rules = _create_rules_dict(
            bounds, max_mass, length, bin_size, p_tol, custom_abundances)
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
    min_qp = i_rules["qp"][0]
    max_qp = i_rules["qp"][1] + p_tol
    start = bisect.bisect(mz, min_mz)
    end = bisect.bisect(mz, max_mz)
    # if valid m/z where found, check if the abundance quotient qp is valid
    if start < end:
        candidates = np.arange(start, end)
        qp_ind = mono_sp / sp[candidates]
        valid_mask = (qp_ind >= min_qp) & (qp_ind <= max_qp)
        candidates = [(x, charge) for x in candidates[valid_mask]]
    else:
        candidates = list()
    return candidates


def _create_rules_dict(
    bounds: Dict[str, Tuple[int, int]],
    max_mass: float,
    length: int,
    bin_size: int,
    p_tol: float,
    custom_abundances: Optional[Dict[str, np.ndarray]]
) -> Dict[int, List[Dict[str, Tuple[float, float]]]]:
    Ma, pa = _create_envelope_arrays(bounds, max_mass, length, custom_abundances)
    # find the monoisotopic index, its Mass difference with the MMI (dM) and
    # its abundance quotient with the MMI (qp)
    bins = (Ma[:, 0] // bin_size).astype(int)

    # find unique values for bins and monoisotopic index that will be used
    # as key for the rule dictionary
    unique_bins = np.unique(bins)
    # unique_mono_index = np.unique(mono_index)
    # unique_mono_index = unique_mono_index[unique_mono_index > 0]

    rules = dict()
    for b in unique_bins:
        b_rules = list()
        bin_mask = bins == b
        for mi in range(1, length):
            qp = pa[bin_mask, mi] / pa[bin_mask, 0]
            dM = Ma[bin_mask, mi] - Ma[bin_mask, 0]
            qp_mask = qp >= (1.0 - p_tol)
            if qp_mask.any():
                mi_rules = dict()
                dM_b_mi = dM[qp_mask]
                qp_b_mi = qp[qp_mask]
                mi_rules["dM"] = dM_b_mi.min(), dM_b_mi.max()
                mi_rules["qp"] = qp_b_mi.min(), qp_b_mi.max()
                b_rules.append(mi_rules)
        if b_rules:
            rules[b] = b_rules
    return rules


def _create_envelope_arrays(
    bounds: Dict[str, Tuple[int, int]],
    M_max: float,
    max_length: int,
    custom_abundances: Optional[Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    elements = _select_elements(list(bounds), custom_abundances)
    isotopes = [x.get_mmi() for x in elements]
    f_bounds = FormulaCoefficientBounds({x: bounds[x.get_symbol()] for x in isotopes})
    coeff = f_bounds.make_coefficients(M_max)
    envelope = make_formula_coefficients_envelopes(bounds, coeff, max_length, custom_abundances)
    M = envelope.M
    p = envelope.p
    return M, p


def _select_two_isotope_element(
    e_list: List[str],
    dm: int,
    custom_abundances: Dict[str, np.ndarray]
) -> List[str]:
    selected = list()
    p_dm_max = 0
    best_p0_greater_than_pi = None
    for s in e_list:
        e = PeriodicTable().get_element(s)
        n_isotopes = len(e.isotopes)
        m, _, p = e.get_abundances()
        if n_isotopes == 2:
            e_dm = m[-1] - m[0]
            if e_dm == dm:
                p0, pi = custom_abundances.get(s, p)
                if pi > p0:
                    selected.append(s)
                elif pi > p_dm_max:
                    p_dm_max = pi
                    best_p0_greater_than_pi = s
    if best_p0_greater_than_pi is not None:
        selected.append(best_p0_greater_than_pi)
    return selected


def _select_multiple_isotope_elements(e_list: List[str]) -> List[str]:
    selected = list()
    for s in e_list:
        e = PeriodicTable().get_element(s)
        n_isotopes = len(e.isotopes)
        if n_isotopes > 2:
            selected.append(s)
    return selected


def _select_elements(
        e_list: List[str], custom_abundances: Optional[Dict[str, np.ndarray]] = None
) -> List[Element]:
    if custom_abundances is None:
        custom_abundances = dict()
    two_isotope_dm1 = _select_two_isotope_element(e_list, 1, custom_abundances)
    two_isotope_dm2 = _select_two_isotope_element(e_list, 2, custom_abundances)
    selected = _select_multiple_isotope_elements(e_list)
    if two_isotope_dm1 is not None:
        selected.extend(two_isotope_dm1)
    if two_isotope_dm2 is not None:
        selected.extend(two_isotope_dm2)
    selected = [PeriodicTable().get_element(x) for x in selected]
    return selected
