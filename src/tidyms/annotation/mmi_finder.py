import numpy as np
import bisect
from typing import Optional
from .annotation_data import AnnotationData
from ..chem.atoms import Element, PeriodicTable, EM
from ..chem._formula_generator import FormulaCoefficientBounds
from ..chem.envelope_tools import make_formula_coefficients_envelopes
from ..base import Feature


class MMIFinder:
    """
    Finds Minimum Mass Isotopologue (MMI) candidates using an array of feature
    m/z and an array of feature area.

    """

    def __init__(
        self,
        bounds: dict[str, tuple[int, int]],
        max_mass: float,
        max_charge: int,
        length: int,
        bin_size: int,
        mz_tol: float,
        p_tol: float,
        min_similarity: float,
        custom_abundances: Optional[dict[str, np.ndarray]] = None,
    ):
        """
        Constructor method.

        Parameters
        ----------
        bounds : dict
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
        min_similarity : float, default=0.9
            Minimum similarity to create candidates.
        custom_abundances : dict, optional
            Provides custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.

        """
        self.rules = _create_rules_dict(
            bounds, max_mass, length, bin_size, p_tol, custom_abundances
        )
        self.bin_size = bin_size
        self.max_charge = abs(max_charge)
        self.polarity = 1 if max_charge >= 0 else -1
        self.max_mass = max_mass
        self.mz_tol = mz_tol
        self.p_tol = p_tol
        self.min_similarity = min_similarity

    def find(self, data: AnnotationData) -> list[tuple[Feature, int]]:
        """
        Search MMI candidates using m/z and area information from a feature
        list.

        Parameters
        ----------
        features : list[Features]
            list of features sorted by m/z.
        mono: Feature
            Candidate to monoisotopic feature.

        Returns
        -------
        mmi_candidates: list[tuple[int, int]]
            list of candidates assuming that the monoisotopic index is part of
            the envelope but not the MMI.

        """
        mono = data.get_monoisotopologue()
        candidates = list()

        if mono is None:
            return candidates

        for charge in range(1, self.max_charge + 1):
            M_mono = mono.mz * charge - self.polarity * charge * EM
            if M_mono < self.max_mass:
                candidates.append((mono, charge))
            M_bin = int(M_mono // self.bin_size)
            mmi_rules = self.rules.get(M_bin)
            if mmi_rules is not None:
                for i_rules in mmi_rules:
                    i_candidates = _find_candidate(
                        data,
                        mono,
                        charge,
                        i_rules,
                        self.mz_tol,
                        self.p_tol,
                        self.max_mass,
                        self.min_similarity,
                    )
                    candidates.extend(i_candidates)
        return candidates


def _find_candidate(
    data: AnnotationData,
    mono: Feature,
    charge: int,
    i_rules: dict,
    mz_tol: float,
    p_tol: float,
    max_mass: float,
    min_similarity: float,
) -> list[tuple[int, int]]:
    # search valid m/z values
    min_dM, max_dM = i_rules["dM"]
    min_mz = mono.mz - max_dM / charge - mz_tol
    max_mz = mono.mz - min_dM / charge + mz_tol
    min_qp = i_rules["qp"][0] - p_tol
    max_qp = i_rules["qp"][1] + p_tol

    if (mono.mz * charge) < max_mass:
        start = bisect.bisect(data.features, min_mz)
        end = bisect.bisect(data.features, max_mz)
    else:
        start, end = 0, 0  # dummy values

    # if valid m/z where found, check if the abundance quotient qp is valid
    candidates = list()
    if start < end:
        for k in range(start, end):
            candidate = data.features[k]
            is_valid = _check_candidate(data, mono, candidate, min_similarity, min_qp, max_qp)
            if is_valid:
                candidates.append((candidate, charge))
    return candidates


def _check_candidate(
    data: AnnotationData,
    mono: Feature,
    candidate: Feature,
    min_similarity: float,
    min_qp: float,
    max_qp: float,
) -> bool:
    if candidate not in data.non_annotated:
        return False

    similarity = data.similarity_cache.get_similarity(mono, candidate)

    if similarity < min_similarity:
        return False

    mmi_mono_pair = [candidate, mono]
    _, p = mono.compute_isotopic_envelope(mmi_mono_pair)
    qp = p[1] / p[0]
    is_valid_qp = (qp >= min_qp) & (qp <= max_qp)

    return is_valid_qp


def _create_rules_dict(
    bounds: dict[str, tuple[int, int]],
    max_mass: float,
    length: int,
    bin_size: int,
    p_tol: float,
    custom_abundances: Optional[dict[str, np.ndarray]],
) -> dict[int, list[dict[str, tuple[float, float]]]]:
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
    bounds: dict[str, tuple[int, int]],
    M_max: float,
    max_length: int,
    custom_abundances: Optional[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    elements = _select_elements(list(bounds), custom_abundances)
    isotopes = [x.get_mmi() for x in elements]
    f_bounds = FormulaCoefficientBounds({x: bounds[x.get_symbol()] for x in isotopes})
    coeff = f_bounds.make_coefficients(M_max)
    envelope = make_formula_coefficients_envelopes(
        bounds, coeff, max_length, custom_abundances
    )
    M = envelope.M
    p = envelope.p
    return M, p


def _select_two_isotope_element(
    e_list: list[str], dm: int, custom_abundances: dict[str, np.ndarray]
) -> list[str]:
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


def _select_multiple_isotope_elements(e_list: list[str]) -> list[str]:
    selected = list()
    for s in e_list:
        e = PeriodicTable().get_element(s)
        n_isotopes = len(e.isotopes)
        if n_isotopes > 2:
            selected.append(s)
    return selected


def _select_elements(
    e_list: list[str], custom_abundances: Optional[dict[str, np.ndarray]] = None
) -> list[Element]:
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
