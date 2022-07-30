"""
Functions to find isotopic envelopes candidates in a list of m/z values.
"""


from .atoms import *
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Iterable


def _make_m_bounds(elements: List[Element],
                   min_abundance: float) -> Dict[int, Tuple[float, float]]:
    """
    Computes possible mass differences from changing one atom isotope.

    Parameters
    ----------
    elements: list of Elements
    min_abundance: number between 0 and 1.
        Minimum abundance of the isotopes used.

    Returns
    -------
    bounds: dict
        mapping of possible nominal mass increments to exact mass increments,
        used by _get_k_bounds to estimate valid m/z ranges for isotopologues.

    """
    bounds = dict()
    for e in elements:
        nom, ex, ab = e.get_abundances()
        ex = [x for k, x in enumerate(ex) if ab[k] > min_abundance]
        nom = [x for k, x in enumerate(nom) if ab[k] > min_abundance]
        nom -= nom[0]
        ex -= ex[0]
        for d_nom, d_ex in zip(nom[1:], ex[1:]):
            bounds.setdefault(d_nom, list())
            bounds[d_nom].append(d_ex)

    for d in bounds:
        bounds[d] = min(bounds[d]), max(bounds[d])
    return bounds


def _get_k_bounds(mz: Union[List[float], np.ndarray],
                  bounds: Dict[int, Tuple[float, float]],
                  n: int, charge: int, mz_tolerance: float
                  ) -> Tuple[float, float]:
    """
    Computes the valid m/z range for a k-th isotopologue using information from
    m/z values from previous isotopologues.

    Parameters
    ----------
    mz: sorted array or list
        List of previous found m/z values
    bounds: dict
        bounds obtained with _make_m bounds
    n: int
        relative mass increment to the first isotopologue found
    charge: int
    mz_tolerance: float

    Returns
    -------
    min_mz: minimum mz value for the M + k isotopologue
    max_mz: maximum mz value for the M + K isotopologue

    """
    if isinstance(mz, float):
        mz = np.array([mz])

    # If the charge is 0 (neutral mass) the results are the same as using
    # charge = 1. There is no difference between positive and negative
    # charges
    charge = max(1, abs(charge))
    min_list = list()
    max_list = list()

    # nominal mass difference between contiguous m/z values
    dm = [int(round((x - mz[0]) * charge)) for x in mz]

    # fill missing mz values using average. This starts to work badly when the
    # number of missing increases.
    missing_dm = [x for x in range(1, n) if x not in dm]
    if len(missing_dm) > 0:
        min_mz_fill, max_mz_fill = _get_k_bounds(mz, bounds, 1, charge,
                                                 mz_tolerance)
        dmz_fill = (min_mz_fill + max_mz_fill) / 2 - mz[0]
        missing_mz = [mz[0] + k * dmz_fill for k in missing_dm]
        mz = np.sort(np.hstack((mz, missing_mz)))
        dm = np.arange(mz.size)

    # the relative increment in nominal mass for each value in mz
    # dmj is the nominal mass increment of the j-th peak , dmk are the possible
    # mass increments obtained from the bounds
    for dmj_ind, dmj in enumerate(dm):

        if dmj > n:
            # we don't need to consider values greater than n
            break
        for dmk in bounds.keys():
            if (dmj + dmk) == n:
                min_list.append(mz[dmj_ind] + bounds[dmk][0] / charge)
                max_list.append(mz[dmj_ind] + bounds[dmk][1] / charge)

    min_mz = min(min_list) - mz_tolerance
    max_mz = max(max_list) + mz_tolerance
    return min_mz, max_mz


def _find_envelopes_aux(mz: np.ndarray, mz0_index: int,
                        bounds: Dict[int, Tuple[float, float]], n_isotopes: int,
                        max_missing: int, charge: int, mz_tolerance: float
                        ) -> List[List[int]]:
    """
    Finds isotopes candidates for a given charge state.

    Parameters
    ----------
    mz: array
        array with m/z values. Assumes that the values are sorted.
    mz0_index: int
        index of the first isotope in the envelope
    bounds: dict
        bounds obtained with _make_m_bounds
    n_isotopes: int
        maximum nominal increment to search
    max_missing: int
        maximum number of consecutive missing values
    charge: int
        charge state of the isotopic envelope
    mz_tolerance: float

    Returns
    -------
    envelopes: List[List[int]]
        List where each element is a list of indices with isotopic envelopes
        candidates.

    """

    # Isotopologues are searched by using information from the previous m/z
    # values. The M + 1 is considered a requirement to search the next
    # isotopologues. If this value is not found, the algorithm returns only
    # the first index

    envelopes = [[mz0_index]]
    n_missing = 0
    k = 1       # k-th isotope
    while (n_missing <= max_missing) and (k <= n_isotopes):
        new_envelopes = list()
        all_missing = True
        for e in envelopes:
            k_mz_bounds = _get_k_bounds(mz[e], bounds, k, charge, mz_tolerance)
            start, end = np.searchsorted(mz, k_mz_bounds)
            if start < end:
                new_index = np.arange(start, end)
                new_envelopes.extend([e + [k] for k in new_index])
                all_missing = False
            else:
                new_envelopes.append(e)
        envelopes = new_envelopes

        # the M + 1 isotope is found check
        if (k == 1) and (envelopes == [[mz0_index]]):
            break

        k += 1
        n_missing += all_missing
    envelopes = [x for x in envelopes if len(x) > 1]
    return envelopes


def _find_envelopes(mz: np.ndarray, monoisotopic_index: int,
                    charge_list: Optional[Union[int, Iterable[int]]],
                    bounds: Dict[int, Tuple[float, float]],
                    max_length: int, max_missing: int,
                    mz_tolerance: float
                    ) -> Dict[int, List[List[int]]]:
    """

    Finds isotopic envelope candidates using multiple charge states.

    Parameters
    ----------
    mz: array
        array of sorted m/z values
    monoisotopic_index: int
        index of the first isotope in the envelope
    bounds: dict
        bounds obtained with _make_m_bounds
    max_length: int
        maximum length ot the isotope candidates
    max_missing: int
        maximum number of missing isotopologues in the envelope.
    charge_list: List[int]
        charge state of the isotopic envelope
    mz_tolerance: float

    Returns
    -------
    envelopes:
        List where each element is a list of indices with isotopic envelopes
        candidates.

    """

    envelopes = dict()
    for q in charge_list:
        q_env = _find_envelopes_aux(mz, monoisotopic_index, bounds,
                                    max_length, max_missing, q,
                                    mz_tolerance)
        if len(q_env) > 0:
            envelopes[q] = q_env
    return envelopes


class EnvelopeFinder(object):
    r"""
    Find isotopic envelopes candidates in a list of sorted m/z values.

    Attributes
    ----------
    elements : list of element symbols
        elements used to estimate m/z of isotopologues
    min_abundance : number between 0 and 1.
        The minimum abundance of the isotopes of each element to be used for m/z
        estimation.
    mz_tolerance : float
        tolerance used to extend the element based bounds
    max_charge : int
        max charge to search envelopes
    max_length : int
        max length of the envelopes
    max_missing : int
        Maximum number missing isotopologues.

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
    def __init__(self, elements: List[str], mz_tolerance: float,
                 max_charge: int = 3, max_length: int = 5,
                 max_missing: int = 0, min_abundance: float = 0.01):
        self._elements = [PTABLE[x] for x in elements]
        self._tolerance = mz_tolerance
        self._max_charge = max_charge
        self._max_length = max_length
        self._max_missing = max_missing
        self._bounds = _make_m_bounds(self._elements, min_abundance)

    def find(self, mz: np.ndarray, monoisotopic_index: int):
        """
        Finds isotopologues candidates for an specific value in the list.

        Parameters
        ----------
        mz : sorted array of m/z values
        monoisotopic_index : index considered as M0 isotopologue

        Returns
        -------
        envelopes: dict
            a dictionary from charge values to a list of lists with possible
            envelope candidates.

        """
        charge = np.arange(1, self._max_charge + 1)
        envelopes = \
            _find_envelopes(mz, monoisotopic_index, charge,
                            self._bounds, self._max_length,
                            self._max_missing, self._tolerance)

        return envelopes
