"""
Functions to find isotopic envelopes candidates in a list of m/z values.

"""


import bisect
from typing import Tuple
from ..chem.atoms import Element, PeriodicTable
from ..lcms import Feature
from .annotation_data import AnnotationData, SimilarityCache
from collections.abc import Sequence

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
        elements: list[str],
        mz_tolerance: float,
        max_length: int = 5,
        min_p: float = 0.01,
        min_similarity: float = 0.9,
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
        min_similarity : float, default=0.9
            Minimum similarity to create candidates.

        """

        el_list = [PeriodicTable().get_element(x) for x in elements]
        self.tolerance = mz_tolerance
        self.max_length = max_length
        self.min_similarity = min_similarity
        self.bounds = _make_exact_mass_difference_bounds(el_list, min_p)

    def find(
        self,
        data: AnnotationData,
        mmi: Feature,
        charge: int,
    ) -> list[Sequence[Feature]]:
        """
        Finds isotopic envelope candidates starting from the minimum mass
        isotopologue (MMI).

        Parameters
        ----------
        data : AnnotationData
            List of features sorted by m/z.
        mmi : Feature
            Minimum Mass feature.
        non_annotated : set[Feature]
            Non annotated features
        charge : int
            Absolute value of the charge state of the isotopic envelope

        Returns
        -------
        envelopes: list[list[Feature]]
            List of isotopic envelope candidates.

        """
        return _find_envelopes(
            data.features,
            mmi,
            data.non_annotated,
            data.similarity_cache,
            charge,
            self.max_length,
            self.tolerance,
            self.min_similarity,
            self.bounds,
        )


def _find_envelopes(
    features: Sequence[Feature],
    mmi: Feature,
    non_annotated: set[Feature],
    cache: SimilarityCache,
    charge: int,
    max_length: int,
    mz_tolerance: float,
    min_similarity: float,
    bounds: dict[int, Tuple[float, float]],
) -> list[Sequence[Feature]]:
    """

    Finds isotopic envelope candidates using multiple charge states.

    Parameters
    ----------
    features: list[Feature]
        List of features sorted by m/z.
    mmi: Feature
        Minimum Mass feature.
    non_annotated: set[Feature]
        Non annotated features
    charge: int
        Absolute value of the charge state of the isotopic envelope
    max_length: int
        maximum length ot the isotope candidates
    mz_tolerance: float
    min_similarity : float, default=0.9
            Minimum similarity to create candidates.
    bounds: dict
        bounds obtained with _make_m_bounds

    Returns
    -------
    envelopes:
        List where each element is a list of indices with isotopic envelopes
        candidates.

    """
    completed_candidates = list()
    candidates = [[mmi]]
    while candidates:
        # remove and extend a candidate
        candidate = candidates.pop()

        # find features with compatible m/z and similarities
        min_mz, max_mz = _get_next_mz_search_interval(
            candidate, bounds, charge, mz_tolerance
        )
        start = bisect.bisect(features, min_mz)
        end = bisect.bisect(features, max_mz)
        new_features = list()
        for k in range(start, end):
            k_ft = features[k]
            is_similar = cache.get_similarity(mmi, k_ft) >= min_similarity
            is_non_annotated = k_ft in non_annotated
            if is_similar and is_non_annotated:
                new_features.append(k_ft)

        # extend candidates with compatible features
        length = len(candidate)
        if new_features and (length < max_length):
            tmp = [candidate + [x] for x in new_features]
            candidates.extend(tmp)
        else:
            completed_candidates.append(candidate)
    completed_candidates = [x for x in completed_candidates if len(x) > 1]
    return completed_candidates


def _get_next_mz_search_interval(
    envelope: Sequence[Feature],
    elements_mass_difference: dict[int, Tuple[float, float]],
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
    length = len(envelope)
    min_mz = envelope[-1].mz + 2  # dummy values
    max_mz = envelope[-1].mz - 2
    for dm, (min_dM, max_dM) in elements_mass_difference.items():
        i = length - dm
        if i >= 0:
            min_mz = min(min_mz, envelope[i].mz + min_dM / charge)
            max_mz = max(max_mz, envelope[i].mz + max_dM / charge)
    min_mz -= mz_tolerance
    max_mz += mz_tolerance
    return min_mz, max_mz


def _make_exact_mass_difference_bounds(
    elements: list[Element], min_p: float
) -> dict[int, Tuple[float, float]]:
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
