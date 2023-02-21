# -*- coding: utf-8 -*-
"""
Utilities to compute isotopic envelopes

"""

import numpy as np
from functools import cache
from scipy.stats import multinomial
from typing import Dict, Optional, Tuple
from .atoms import Isotope, PeriodicTable
from . import utils


def make_envelope_arrays(
    isotope: Isotope, n_min: int, n_max: int, max_length: int, p=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an array of exact mass and abundance for homonuclear formulas.

    Parameters
    ----------
    isotope : Isotope
    n_min : int
        Minimum formula coefficient
    n_max : int
        Maximum formula coefficient
    max_length : int
        Length of the envelope
    p : array or None, default=None
        Element abundance. If None, the natural abundance is used.

    Returns
    -------
    M : (n_max - n_min + 1, max_length) array
        Coefficients exact mass.
    p : (n_max - n_min + 1, max_length) array
        Coefficients abundance.


    """
    rows = n_max - n_min + 1
    M_arr = np.zeros((rows, max_length))
    p_arr = np.zeros((rows, max_length))
    for k in range(n_min, n_max + 1):
        Mk, pk = _get_n_atoms_envelope(isotope, k, max_length, p=p)
        M_arr[k - n_min] = Mk
        p_arr[k - n_min] = pk
    return M_arr, p_arr


def find_formula_envelope(
    composition: Dict[Isotope, int],
    max_length: int,
    p: Optional[Dict[str, np.ndarray]] = None,
    min_p: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the isotopic envelope for a formula.

    """
    if p is None:
        p = dict()

    # initialize an empty envelope for the formula
    Mf = np.zeros((1, max_length), dtype=float)
    pf = np.zeros((1, max_length), dtype=float)
    pf[0, 0] = 1

    for i, coeff in composition.items():
        i_p = p.get(i.get_symbol())
        Mi, pi = _get_n_atoms_envelope(i, coeff, max_length, p=i_p)
        Mi = Mi.reshape((1, Mi.size))
        pi = pi.reshape((1, pi.size))
        Mf, pf = combine_envelopes(Mf, pf, Mi, pi)
    valid_p_mask = pf >= min_p
    pf = pf[valid_p_mask].flatten()
    Mf = Mf[valid_p_mask].flatten()
    return Mf, pf


def combine_envelopes(
    M1: np.ndarray,
    p1: np.ndarray,
    M2: np.ndarray,
    p2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines exact mass and abundance of two envelopes.

    All arrays must be 2-dimensional and have the same shape.

    """
    shape = M1.shape
    M = np.zeros(shape, dtype=float)
    p = np.zeros(shape, dtype=float)
    # Ignore zero division errors when normalizing by pk
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(shape[1]):
            pk = (p1[:, : k + 1] * p2[:, k::-1]).sum(axis=1)
            k1 = k + 1
            k2 = k
            Mk = (p1[:, :k1] * M1[:, :k1] * p2[:, k2::-1]) + (
                p1[:, :k1] * M2[:, k2::-1] * p2[:, k2::-1]
            )
            M[:, k] = Mk.sum(axis=1) / pk
            p[:, k] = pk
    np.nan_to_num(M, copy=False)
    return M, p


def _get_n_atoms_envelope(
    isotope: Isotope, n: int, max_length: int, p: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the nominal mass, exact mass and abundance of n atoms.

    If the isotope is the monoisotope and p is ``None``, the natural abundances
    for the element are used.

    If the isotope is the monoisotope and custom abundance `p` is provided, the
    envelope is computed using this value instead of the natural abundances.

    If the isotopes is not the monoisotope, it is assumed that only this
    isotope contributes to the envelope.

    """
    symbol = isotope.get_symbol()
    element = PeriodicTable().get_element(symbol)
    is_monoisotope = isotope.a == element.get_monoisotope().a
    n_isotopes = len(element.isotopes)
    if is_monoisotope and (n_isotopes > 1):
        if n == 0:
            M, p = _get_n_isotopes_envelope(isotope, n, max_length)
        elif p is None:
            M, p = _get_n_atoms_natural_abundance(symbol, n, max_length)
        else:
            m, M, _ = element.get_abundances()
            _validate_abundance(p, m, symbol)
            M, p = _get_n_atoms_envelope_aux(m, M, p, n, max_length)
    else:
        M, p = _get_n_isotopes_envelope(isotope, n, max_length)
    return M, p


@cache
def _get_n_atoms_natural_abundance(symbol: str, n: int, max_length: int):
    """
    Computes the envelope of n atoms using the natural abundance.

    aux function to _get_n_atoms_envelope

    """
    m, M, p = PeriodicTable().get_element(symbol).get_abundances()
    return _get_n_atoms_envelope_aux(m, M, p, n, max_length)


def _get_n_atoms_envelope_aux(
    m: np.ndarray, M: np.ndarray, p: np.ndarray, n: int, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the envelope of n atoms.

    aux function to _get_n_atoms_envelope.

    """
    n_isotopes = p.size
    # find combinations of isotopes that sum n
    combinations = _find_n_isotope_combination(n_isotopes, n)

    # find m, M and p for each combination of isotopes
    multinomial_dist = multinomial(n, p)
    m = np.matmul(combinations, m)
    M = np.matmul(combinations, M)
    p = multinomial_dist.pmf(combinations)

    # sort by exact mass
    sorted_index = np.argsort(M)
    m, M, p = m[sorted_index], M[sorted_index], p[sorted_index]

    # merge values with the same nominal mass
    _, first_occurrence = np.unique(m, return_index=True)
    m_unique = np.zeros(max_length, dtype=m.dtype)
    M_unique = np.zeros(max_length, dtype=M.dtype)
    p_unique = np.zeros(max_length, dtype=p.dtype)
    # add the length of m_unique to include all nominal mass values
    n_unique = first_occurrence.size
    first_occurrence = list(first_occurrence)
    first_occurrence.append(m.size)
    m0 = m[0]
    for k in range(max_length):
        if k < n_unique:
            start = first_occurrence[k]
            end = first_occurrence[k + 1]
            mk = m[start]
            i = mk - m0
            if i < max_length:
                m_unique[i] = mk
                pk = np.sum(p[start:end])
                p_unique[i] = pk
                M_unique[i] = np.sum(M[start:end] * p[start:end]) / pk
    p_unique = p_unique / np.sum(p_unique)
    return M_unique, p_unique


def _fill_missing_nominal(
    m: np.ndarray, M: np.ndarray, p: np.ndarray, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    rel_m = m - m[0]
    dm = np.arange(max_length)
    M_filled = np.zeros(max_length, dtype=M.dtype)
    p_filled = np.zeros(max_length, dtype=p.dtype)
    if not np.array_equal(rel_m, dm):
        for k, rel_m_k in enumerate(rel_m):
            if 0 <= rel_m_k < max_length:
                M_filled[rel_m_k] = M[k]
                p_filled[rel_m_k] = p[k]
            else:
                break
        M, p = M_filled, p_filled
    return M, p


def _find_n_isotope_combination(n_isotopes, n):
    """
    Finds combinations of isotopes such that the sum is n.

    aux function to _find_n_atoms_abundances.

    """
    n_ranges = [range(x) for x in ([n + 1] * n_isotopes)]
    combinations = utils.cartesian_product(*n_ranges).astype(int)
    valid_combinations = combinations.sum(axis=1) == n
    combinations = combinations[valid_combinations, :]
    return combinations


def _validate_abundance(p: np.ndarray, m: np.ndarray, symbol: str):
    """
    Checks that user-created abundances are non-negative, normalized to 1 and
    has the same length as the number of stable isotopes.

    aux function to _get_n_atoms_envelope.

    """
    is_all_non_negative = (p >= 0.0).all()
    is_normalized = np.isclose(p.sum(), 1.0)
    is_same_size = p.size == m.size
    if not is_same_size:
        msg = "{} has {} stable isotopes. `p` must have the same size."
        raise ValueError(msg.format(symbol, m.size))
    elif not (is_normalized and is_all_non_negative):
        msg = "`p` elements must be non-negative and their sum normalized to 1."
        raise ValueError(msg)


def _get_n_isotopes_envelope(
    isotope: Isotope, n: int, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates the isotopic envelope for n isotopes.

    aux function to _get_n_atoms_envelope.

    """
    M = np.zeros(max_length, dtype=float)
    p = np.zeros(max_length, dtype=float)
    M[0] = isotope.m * n
    p[0] = 1.0
    return M, p
