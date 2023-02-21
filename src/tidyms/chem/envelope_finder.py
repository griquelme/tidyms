"""
Functions to find isotopic envelopes candidates in a list of m/z values.

"""


import bisect
import numpy as np
from typing import List, Dict, Set, Tuple
from .atoms import Element, PeriodicTable


# name conventions
# M is used for Molecular mass
# m for nominal mass
# p for abundances

class EnvelopeFinder(object):
    r"""
    Find isotopic envelopes candidates in a list of sorted m/z values.

    Attributes
    ----------
    tolerance : float
        tolerance used to extend the element based bounds
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
        max_length: int = 5,
        min_p: float = 0.01,
    ):
        """

        Parameters
        ----------
        elements : List[str]
            List of elements used to compute mass difference windows.
        mz_tolerance : float
            m/z tolerance used to match candidates.
        max_length : int, default=5
            Maximum envelope length to search.
        min_p : number between 0 and 1.
            The minimum abundance of the isotopes of each element to be used for m/z estimation.
        """

        elements = [PeriodicTable().get_element(x) for x in elements]
        self.tolerance = mz_tolerance
        self.max_length = max_length
        self.bounds = _make_exact_mass_difference_bounds(elements, min_p)

    def find(
        self, mz: np.ndarray, mmi_index: int, charge: int, valid_indices: Set[int]
    ) -> List[List[int]]:
        """
        Finds isotopic envelope candidates starting from the minimum mass
        isotopologue (MMI).

        Parameters
        ----------
        mz : array
            sorted array of m/z values
        mmi_index : int
            index of the MMI
        charge : int
            Absolute value of the charge state of the envelope
        valid_indices : array
            Indices of `mz` to search candidates.

        Returns
        -------
        envelopes: List[List[int]]
            List where each element is a list of indices in `mz` corresponding
            to an envelope candidate.

        """
        return _find_envelopes(
            mz, mmi_index, valid_indices, charge, self.max_length,
            self.tolerance, self.bounds
        )


def _find_envelopes(
    mz: np.ndarray,
    mmi_index: int,
    valid_indices: Set,
    charge: int,
    max_length: int,
    mz_tolerance: float,
    bounds: Dict[int, Tuple[float, float]],
) -> List[List[int]]:
    """

    Finds isotopic envelope candidates using multiple charge states.

    Parameters
    ----------
    mz: array
        array of sorted m/z values
    mmi_index: int
        index of the first isotope in the envelope
    charge: int
        Absolute value of the charge state of the isotopic envelope
    max_length: int
        maximum length ot the isotope candidates
    mz_tolerance: float
    bounds: dict
        bounds obtained with _make_m_bounds

    Returns
    -------
    envelopes:
        List where each element is a list of indices with isotopic envelopes
        candidates.

    """
    completed_candidates = list()
    candidates = [[mmi_index]]
    while candidates:
        candidate = candidates.pop()
        length = len(candidate)
        min_mz, max_mz = _get_next_mz_search_interval(
            mz[candidate], bounds, charge, mz_tolerance)
        start = bisect.bisect(mz, min_mz)
        end = bisect.bisect(mz, max_mz)
        new_elements = [x for x in range(start, end) if x in valid_indices]
        if new_elements and (length < max_length):
            tmp = [candidate + [x] for x in new_elements]
            candidates.extend(tmp)
        else:
            completed_candidates.append(candidate)
    completed_candidates = [x for x in completed_candidates if len(x) > 1]
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
