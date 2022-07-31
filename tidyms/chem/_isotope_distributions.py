# -*- coding: utf-8 -*-
"""
tools to compute isotopic distributions
"""

import numpy as np
from scipy.stats import multinomial
from .atoms import Isotope
from . import utils
from typing import Tuple, Optional, Dict, List

# TODO: it is not necessary to compute the coefficients of the solution to the
#   formula generator problem, they can be computed in an implicit way and
#   obtain just the envelopes.


def _find_n_atoms_abundances(isotope: Isotope, n: int, max_length: int,
                             abundance: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    computes the abundances of n atoms of a given element
    
    Arguments
    ---------
    isotope : Isotope
    n : int
    max_length : int
        first l peaks considered.
    abundance: numpy.ndarray, optional
        custom element abundances, when using elements with non natural
        abundances.
        
    Returns
    -------
    nominal: np.ndarray
        nominal mass of the possibles isotopologues of n atoms
    exact: np.ndarray
        exact mass of the possibles isotopologues of n atoms
    abundances: np.array
        abundances of the possibles isotopologues of n atoms
    """
    if n == 0:
        abundance = np.zeros(max_length, dtype=float)
        abundance[0] = 1.0
        exact = np.zeros(max_length, dtype=float)
        nominal = np.zeros(max_length, dtype=int)
        return nominal, exact, abundance

    if isotope.is_most_abundant():
        element = isotope.get_element()
        element_nominal, element_exact, tmp_abundance = element.get_abundances()
    else:
        return _abundances_from_isotope(isotope, n, max_length)

    if abundance is not None:
        _validate_abundances(abundance, element_nominal)
    else:
        abundance = tmp_abundance
    n_isotopes = element_nominal.size
    # find combinations of isotopes such that its sum is n
    combinations = _find_n_combinations(n_isotopes, n)
    nominal = np.matmul(combinations, element_nominal)
    exact = np.matmul(combinations, element_exact)

    # find abundances for each combination
    multinomial_dist = multinomial(n, abundance)
    abundance = multinomial_dist.pmf(combinations)

    nominal, exact, abundance = _sort_by_exact_mass(nominal, exact, abundance)

    # merge values with the same nominal mass
    if not utils.has_unique_elements(nominal):
        nominal, exact, abundance = \
            _merge_same_nominal_mass(nominal, exact, abundance)

    # fill in missing nominal mass values (eg. M + 1 in Cl). This missing
    # values are set to zero. This makes easier to combine array abundances.
    nominal, exact, abundance = _add_missing_nominal(nominal, exact, abundance)

    # extend results with zeros to match max_length
    nominal, exact, abundance = _resize_to_max_length(nominal, exact, abundance,
                                                      max_length)

    return nominal, exact, abundance


def _find_n_combinations(n_isotopes, n):
    """
    Auxiliary function to _find_n_atoms_abundances. Finds combinations of
    isotopes such that the sum is n.

    """
    n_ranges = [range(x) for x in ([n + 1] * n_isotopes)]
    combinations = utils.cartesian_product(*n_ranges).astype(int)
    valid_combinations = (combinations.sum(axis=1) == n)
    combinations = combinations[valid_combinations, :]
    return combinations


def _sort_by_exact_mass(nominal: np.ndarray, exact: np.ndarray,
                        abundance: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_index = np.argsort(exact)
    nominal = nominal[sorted_index]
    exact = exact[sorted_index]
    abundance = abundance[sorted_index]
    return nominal, exact, abundance


def _resize_to_max_length(nominal: np.ndarray, exact: np.ndarray,
                          abundance: np.ndarray, max_length: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = nominal.size
    if size < max_length:
        nominal = np.hstack((nominal, np.zeros(max_length - size, dtype=int)))
        exact = np.hstack((exact, np.zeros(max_length - size, dtype=float)))
        abundance = np.hstack((abundance,
                               np.zeros(max_length - size, dtype=float)))
    elif size > max_length:
        nominal = nominal[:max_length]
        exact = exact[:max_length]
        abundance = abundance[:max_length]
    return nominal, exact, abundance


def _add_missing_nominal(nominal: np.ndarray, exact: np.ndarray,
                         abundances: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    add missing nominal masses values (eg: Cl [35, 37] -> [35, 36, 37]).
    This makes more straightforward the process of computing isotopic
    envelopes for formulas.
    """
    if nominal.size == (nominal[-1] + 1 - nominal[0]):
        return nominal, exact, abundances
    else:
        new_nom = np.arange(nominal[0], nominal[-1] + 1)
        new_nom[~np.isin(new_nom, nominal)] = 0
        new_exact = np.zeros(new_nom.size, dtype=float)
        new_exact[nominal - nominal[0]] = exact
        new_abundances = np.zeros_like(new_nom, dtype=float)
        new_abundances[nominal - nominal[0]] = abundances
        return new_nom, new_exact, new_abundances


def _merge_same_nominal_mass(nominal_comb: np.ndarray, exact_comb: np.ndarray,
                             abundance_comb: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge isotopic combinations with the same nominal mass. aux function
    to get_isotopes_combinations.

    Returns
    -------
    nominal, exact, abundance: tuple
    """
    change_nominal_index = _get_nominal_change_index(nominal_comb)
    abundance_unique = list()
    mean_unique = list()
    
    for k in range(len(change_nominal_index) - 1):
        start = change_nominal_index[k]
        end = change_nominal_index[k + 1]
        abundance_sum = abundance_comb[start:end].sum()
        mean_mass = (abundance_comb[start:end] *
                     exact_comb[start:end]).sum() / abundance_sum
        mean_unique.append(mean_mass)
        abundance_unique.append(abundance_sum)
    nominal_unique = np.unique(nominal_comb)
    mean_unique = np.array(mean_unique)
    abundance_unique = np.array(abundance_unique)
    return nominal_unique, mean_unique, abundance_unique


def _get_nominal_change_index(nominal: np.ndarray):
    """
    finds index when nominal mass increases. Assumes sorted array.
    """
    change_index = np.where(np.diff(nominal) != 0)[0]
    change_index = np.hstack((0, 1 + change_index, nominal.size))
    return change_index


def _validate_abundances(abundances: np.ndarray, nominal: np.ndarray):
    is_all_non_negative = (abundances >= 0.0).all()
    is_normalized = np.isclose(abundances.sum(), 1.0)
    is_same_size = abundances.size == nominal.size
    if not (is_normalized and is_same_size and is_all_non_negative):
        msg = "abundances must have the same shape as the nominal mass" \
              "array and its sum must be equal to 1."
        raise ValueError(msg)


def _abundances_from_isotope(isotope: Isotope, n: int, max_length: int
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.zeros(max_length, dtype=int)
    exact = np.zeros(max_length)
    abundances = np.zeros(max_length)
    nominal[0] = isotope.a * n
    exact[0] = isotope.m * n
    abundances[0] = 1
    return nominal, exact, abundances


def _combine_element_abundances(nom1: np.ndarray, exact1: np.ndarray,
                                ab1: np.ndarray, nom2: np.ndarray,
                                exact2: np.ndarray, ab2: np.ndarray,
                                min_p: float = 1e-10
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes nominal mass, exact mass and abundances for two elements.
    All arrays must have the same size.

    Returns
    -------
    nominal: np.ndarray
        nominal mass of the possible combinations
    exact: np.ndarray
        exact mass of the possible combinations
    abundances: np.array
        abundances of the possible combinations

    """
    size = nom1.size
    nominal = np.zeros(size, dtype=int)
    exact = np.zeros(size, dtype=float)
    abundance = np.zeros(size, dtype=float)
    for k in range(size):
        p_k = (ab1[:k + 1] * ab2[k::-1]).sum()
        if p_k > min_p:
            nom_k = ((ab1[:k + 1] * nom1[:k + 1] * ab2[k::-1]) +
                     (ab1[:k + 1] * nom2[k::-1] * ab2[k::-1]))
            nominal[k] = np.round(nom_k.sum() / p_k)
            exact_k = ((ab1[:k + 1] * exact1[:k + 1] * ab2[k::-1]) +
                       (ab1[:k + 1] * exact2[k::-1] * ab2[k::-1]))
            exact[k] = exact_k.sum() / p_k
            abundance[k] = p_k
    return nominal, exact, abundance


def find_formula_abundances(d: Dict[Isotope, int], max_length: int,
                            abundance: Optional[Dict[str, np.ndarray]] = None,
                            min_p: float = 1e-10
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abundance is None:
        abundance = dict()

    res_ab = np.zeros(max_length, dtype=float)
    res_nom = np.zeros(max_length, dtype=int)
    res_ex = np.zeros(max_length, dtype=float)
    res_ab[0] = 1

    for isotope, coeff in d.items():
        isotope_symbol = isotope.get_symbol()
        isotope_abundance = abundance.get(isotope_symbol)
        tmp_nom, tmp_ex, tmp_ab = \
            _find_n_atoms_abundances(isotope, coeff, max_length,
                                     abundance=isotope_abundance)
        res_nom, res_ex, res_ab = \
            _combine_element_abundances(res_nom, res_ex, res_ab, tmp_nom,
                                        tmp_ex, tmp_ab, min_p=min_p)
    return res_nom, res_ex, res_ab


def _make_element_abundance_array(atom, n_min, n_max, length, abundance=None):
    n_rows = n_max - n_min + 1
    nominal_array = np.zeros((n_rows, length), dtype=int)
    exact_array = np.zeros((n_rows, length))
    abundance_array = np.zeros((n_rows, length))
    for n in range(n_min, n_max + 1):
        tmp_nom, tmp_ex, tmp_ab = \
            _find_n_atoms_abundances(atom, n, length, abundance=abundance)
        nominal_array[n - n_min] = tmp_nom
        exact_array[n - n_min] = tmp_ex
        abundance_array[n - n_min] = tmp_ab
    return nominal_array, exact_array, abundance_array


def _combine_array_abundances(nom1: np.ndarray, exact1: np.ndarray,
                              ab1: np.ndarray, nom2: np.ndarray,
                              exact2: np.ndarray, ab2: np.ndarray,
                              min_p: float = 1e-10
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes de nominal mass, exact mass and abundances for two elements.
    All arrays must have the same size.

    Returns
    -------
    nominal: numpy array
        nominal mass of the possible combinations
    exact: numpy array
        exact mass of the possible combinations
    abundances: numpy array
        abundances of the possible combinations

    """
    n_cols = nom1.shape[1]
    nominal = np.zeros(nom1.shape, dtype=int)
    abundance = np.zeros(nom1.shape, dtype=float)
    exact = np.zeros(nom1.shape, dtype=float)
    for k in range(n_cols):
        p_k = (ab1[:, :k + 1] * ab2[:, k::-1]).sum(axis=1)
        mask = p_k > min_p
        nom_k = ((ab1[:, :k + 1] * nom1[:, :k + 1] * ab2[:, k::-1]) +
                 (ab1[:, :k + 1] * nom2[:, k::-1] * ab2[:, k::-1]))
        nominal[mask, k] = np.round(nom_k[mask].sum(axis=1) / p_k[mask])
        exact_k = ((ab1[:, :k + 1] * exact1[:, :k + 1] * ab2[:, k::-1]) +
                   (ab1[:, :k + 1] * exact2[:, k::-1] * ab2[:, k::-1]))
        exact[mask, k] = exact_k[mask].sum(axis=1) / p_k[mask]
        abundance[mask, k] = p_k[mask]
    return nominal, exact, abundance


def make_coeff_abundances(bounds: List[Tuple[int, int]],
                          coefficients: np.ndarray,
                          isotopes: List[Isotope], length: int,
                          abundances: Optional[dict] = None,
                          min_p: float = 1e-10):
    if abundances is None:
        abundances = dict()

    n_rows = coefficients.shape[0]
    res_ab = np.zeros((n_rows, length))
    res_nom = np.zeros((n_rows, length), dtype=int)
    res_ex = np.zeros((n_rows, length))
    res_ab[:, 0] = 1

    # isotope_abundance_array = dict()
    for k, isotope, b in zip(range(len(isotopes)), isotopes, bounds):
        lower, upper = b
        symbol = isotope.get_symbol()
        tmp_abundance = abundances.get(symbol)
        tmp_nom, tmp_ex, tmp_ab = \
            _make_element_abundance_array(isotope, lower, upper, length,
                                          abundance=tmp_abundance)
        # lower corrects indices in cases when 0 is not the lower bound
        k_nom = tmp_nom[coefficients[:, k] - lower, :]
        k_ex = tmp_ex[coefficients[:, k] - lower, :]
        k_ab = tmp_ab[coefficients[:, k] - lower, :]
        res_nom, res_ex, res_ab = \
            _combine_array_abundances(res_nom, res_ex, res_ab, k_nom, k_ex,
                                      k_ab, min_p=min_p)
    return res_nom, res_ex, res_ab
