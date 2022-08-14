"""
Functions to find isotopic envelopes candidates in a list of m/z values.
"""


import numpy as np
from typing import List, Dict, Tuple
from .atoms import Element, PTABLE


# M is used for Molecular mass
# m for nominal mass
# p for abundances

class EnvelopeFinder(object):
    r"""
    Find isotopic envelopes candidates in a list of sorted m/z values.

    Attributes
    ----------
    elements : list of element symbols
        elements used to estimate m/z of isotopologues
    min_p : number between 0 and 1.
        The minimum abundance of the isotopes of each element to be used for m/z
        estimation.
    mz_tolerance : float
        tolerance used to extend the element based bounds
    max_charge : int
        max charge to search envelopes
    max_length : int
        max length of the envelopes

    Notes
    -----
    Using a list of elements, theoretical bounds are computed for each M1, M2,
    M3, etc... isotopologue. Then using these values and the `mz_tolerance` and
    the `max_charge`, the bounds are adjusted according to the following
    equations:

    .. math::

        mz_{k, min}= \frac{m_{k, min}{q} - mz_{tolerance}

        mz_{k, max}= \frac{m_{k, max}{q} + mz_{tolerance}

    where :math:`m_{k, min}` is the minimum theoretical value for the k-th
    isotopologue and q is the charge.

    The envelopes candidates found are determined based on m/z compatibility
    only. To reduce the number of candidates, the list of m/z values should be
    reduced by other means, such as correlation of the values.

    """

    def __init__(
        self,
        elements: List[str],
        mz_tolerance: float,
        max_charge: int = 3,
        max_length: int = 5,
        min_p: float = 0.01,
    ):
        self._elements = [PTABLE[x] for x in elements]
        self._tolerance = mz_tolerance
        self._max_charge = max_charge
        self._max_length = max_length
        self._bounds = _make_exact_mass_difference_bounds(self._elements, min_p)

    def find(
        self,
        mz: np.ndarray,
        mmi_index: int
    ) -> Dict[int, List[List[int]]]:
        """
        Finds isotopic envelope candidates starting from the minimum mass
        isotopologue (MMI).

        Parameters
        ----------
        mz : array
            sorted array of m/z values
        mmi_index : int
            index of the MMI

        Returns
        -------
        envelopes: dict
            a dictionary from charge values to a list of lists with indices of
            envelope candidates.

        """
        return _find_envelopes(
            mz,
            mmi_index,
            self._max_charge,
            self._bounds,
            self._max_length,
            self._tolerance,
        )


def _find_envelopes(
    mz: np.ndarray,
    mmi_index: int,
    max_charge: int,
    bounds: Dict[int, Tuple[float, float]],
    max_length: int,
    mz_tolerance: float,
) -> Dict[int, List[List[int]]]:
    """

    Finds isotopic envelope candidates using multiple charge states.

    Parameters
    ----------
    mz: array
        array of sorted m/z values
    mmi_index: int
        index of the first isotope in the envelope
    bounds: dict
        bounds obtained with _make_m_bounds
    max_length: int
        maximum length ot the isotope candidates
    max_charge: List[int]
        charge state of the isotopic envelope
    mz_tolerance: float

    Returns
    -------
    envelopes:
        List where each element is a list of indices with isotopic envelopes
        candidates.

    """
    if max_charge == 0:
        charge_list = [0]
    else:
        charge_list = list(range(1, abs(max_charge) + 1))

    completed_candidates = dict()

    for q in charge_list:
        candidates = [[mmi_index]]
        while candidates:
            candidate = candidates.pop()
            length = len(candidate)
            min_mz, max_mz = _get_next_mz_search_interval(
                mz[candidate], bounds, q, mz_tolerance)
            start, end = np.searchsorted(mz, [min_mz, max_mz])

            if (start < end) and (length < max_length):
                tmp = [candidate + [x] for x in range(start, end)]
                candidates.extend(tmp)
            else:
                q_candidates = completed_candidates.get(q)
                if q_candidates is None:
                    completed_candidates[q] = [candidate]
                else:
                    q_candidates.append(candidate)
    return completed_candidates


def _get_next_mz_search_interval(
    mz: np.ndarray,
    elements_mass_difference: Dict[int, Tuple[float, float]],
    charge: int,
    mz_tolerance: float,
) -> Tuple[float, float]:
    """
    Computes the valid m/z range for a k-th isotopologue using information from
    m/z values from previous isotopologues.

    Parameters
    ----------
    mz: sorted list
        List of previous found m/z values
    elements_mass_difference: dict
        bounds obtained with _make_m bounds
    charge: int
    mz_tolerance: float

    Returns
    -------
    min_mz: minimum mz value for the M + k isotopologue
    max_mz: maximum mz value for the M + K isotopologue

    """

    # If the charge is 0 (neutral mass) the results are the same as using
    # charge = 1. There is no difference between positive and negative
    # charges
    charge = max(1, abs(charge))
    length = len(mz)
    min_mz = mz[-1] + 2    # dummy values
    max_mz = mz[-1] - 2
    for dm, (min_dM, max_dM) in elements_mass_difference.items():
        i = length - dm
        if i >= 0:
            min_mz = min(min_mz, mz[i] + min_dM / charge)
            max_mz = max(max_mz, mz[i] + max_dM / charge)
    min_mz -= mz_tolerance
    max_mz += mz_tolerance
    return min_mz, max_mz


def _make_exact_mass_difference_bounds(
    elements: List[Element], min_p: float
) -> Dict[int, Tuple[float, float]]:
    """
    Computes possible mass differences obtaining from changing one isotope.

    Parameters
    ----------
    elements: list of Elements
    min_p: number between 0 and 1.
        Minimum abundance of the isotopes used.

    Returns
    -------
    bounds: dict
        mapping of possible nominal mass increments to exact mass increments,
        used by _get_k_bounds to estimate valid m/z ranges for isotopologues.

    """
    bounds = dict()
    for e in elements:
        m, M, p = e.get_abundances()
        for i in range(1, M.size):
            if p[i] > min_p:
                dm = m[i] - m[0]
                dM = M[i] - M[0]
                dM_list = bounds.get(dm)
                if dM_list is None:
                    bounds[dm] = [dM]
                else:
                    dM_list.append(dM)

    for dm in bounds:
        bounds[dm] = min(bounds[dm]), max(bounds[dm])
    return bounds
