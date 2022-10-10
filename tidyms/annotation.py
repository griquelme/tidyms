import bisect
import numpy as np
import pandas as pd
from .lcms import LCRoi, Peak
from .chem import EnvelopeFinder, EnvelopeValidator,  MMIFinder
from .chem.atoms import EM, PeriodicTable
from . import _constants as c
from typing import Callable, Dict, List, Optional, Set, Tuple
from scipy.integrate import trapz


def create_annotator(
    bounds: Dict[str, Tuple[int, int]],
    max_mass: float,
    max_charge: int,
    max_length: int,
    min_M_tol: float,
    max_M_tol: float,
    p_tol: float,
    min_similarity: float,
    min_p: float,
) -> "_IsotopologueEnvelopeAnnotator":
    """
    Create an annotator object. Auxiliary function to _annotate

    Parameters
    ----------
    bounds : Dict
        A dictionary of expected elements to minimum and maximum formula coefficients.
    max_mass : float
        Maximum exact mass of the features.
    max_charge : int
        Maximum charge of the features. Use negative values for negative polarity.
    max_length : int
        Maximum length of the envelopes.
    min_M_tol : float
        Minimum mass tolerance used during search. isotopologues with abundance
        equal to 1 use this value. Isotopologues with abundance equal to 0 use
        `max_M_tol`. For values inbetween, a weighted tolerance is used based
        on the abundance.
    max_M_tol : float
    p_tol : float
        Abundance tolerance.
    min_similarity : float
        Minimum cosine similarity between a pair of features
    min_p : float
        Minimum abundance of isotopes to include in candidate search.

    Returns
    -------
    annotator: _IsotopologueEnvelopeAnnotator

    """
    # remove elements with only 1 stable isotope
    bounds = {k: bounds[k] for k in bounds if len(PeriodicTable().get_element(k).isotopes) > 1}

    min_overlap = 0.5
    bin_size = 100
    polarity = 1 if max_charge > 0 else -1
    max_charge = abs(max_charge)
    elements = list(bounds)
    mmi_finder = MMIFinder(
        bounds, max_mass, max_charge, max_length, bin_size, max_M_tol, p_tol
    )
    envelope_finder = EnvelopeFinder(elements, max_M_tol, max_length, min_p)
    envelope_validator = EnvelopeValidator(
        bounds,
        max_M=max_mass,
        max_length=max_length,
        min_M_tol=min_M_tol,
        max_M_tol=max_M_tol,
        p_tol=p_tol
    )
    similarity_checker = _SimilarityChecker(
        min_similarity, _feature_similarity_lc, min_overlap=min_overlap
    )
    annotator = _IsotopologueEnvelopeAnnotator(
        mmi_finder,
        envelope_finder,
        envelope_validator,
        similarity_checker,
        polarity,
    )
    return annotator


def _annotate(
    feature_table: pd.DataFrame,
    roi_list: List[LCRoi],
    annotator: "_IsotopologueEnvelopeAnnotator",
) -> None:
    """
    Annotates isotopologues in a sample.

    Annotations are added as three columns in the feature table:

    envelope_label
        An integer that groups isotopologues. ``-1`` is used for features that
        do not belong to any group.
    envelope_index
        An integer that labels the nominal mass relative to the MMI.  ``-1`` is
        used for features that do not belong to any group.
    charge
        Charge state of the envelope. ``-1`` is used for features that do not
        belong to any group.

    Parameters
    ----------
    feature_table : DataFrame
    roi_list : List[ROI]
    annotator : _IsotopologueEnvelopeAnnotator

    """
    annotator.load_data(feature_table, roi_list)
    mono_index = annotator.get_next_monoisotopologue_index()
    while mono_index > -1:
        annotator.annotate(mono_index)
        mono_index = annotator.get_next_monoisotopologue_index()
    annotator.add_annotations(feature_table)
    annotator.clear_data()


class _IsotopologueEnvelopeAnnotator:
    """
    Manages Isotopologue annotation in a sample.

    Attributes
    ----------
    mmi_finder : MMIFinder
    envelope_finder : EnvelopeFinder
    validator : EnvelopeValidator
    similarity_checker : _SimilarityChecker
    polarity: 1 or -1

    """

    def __init__(
        self,
        mmi_finder: MMIFinder,
        envelope_finder: EnvelopeFinder,
        validator: EnvelopeValidator,
        similarity_checker: "_SimilarityChecker",
        polarity: int,
    ):
        self.mmi_finder = mmi_finder
        self.envelope_finder = envelope_finder
        self.validator = validator
        self.polarity = polarity
        self.similarity_checker = similarity_checker

        # sample data
        self._roi_list: Optional[List[LCRoi]] = None
        self._mz_order: Optional[np.ndarray] = None
        self._mz: Optional[np.ndarray] = None
        self._area: Optional[np.ndarray] = None
        self._roi_index: Optional[np.ndarray] = None
        self._ft_index: Optional[np.ndarray] = None
        self._non_annotated: Optional[Set[int]] = None
        self._mono_candidates: Optional[List[int]] = None
        self._envelope_label: Optional[np.ndarray] = None
        self._envelope_charge: Optional[np.ndarray] = None
        self._label_counter = 0
        self._envelope_index: Optional[np.ndarray] = None

    def clear_data(self):
        """
        Deletes data from a sample.

        """
        self.similarity_checker.clear_data()
        self._roi_list = None
        self._mz = None
        self._mz_order = None
        self._area = None
        self._roi_index = None
        self._ft_index = None
        self._non_annotated = None
        self._mono_candidates = None
        self._envelope_label = None
        self._envelope_charge = None
        self._label_counter = 0
        self._envelope_index = None

    def load_data(self, feature_table: pd.DataFrame, roi_list: List[LCRoi]):
        """
        Load data from a sample.

        """
        mz = feature_table[c.MZ].to_numpy()
        mz_order = np.argsort(mz)
        self._mz = mz[mz_order]
        self._mz_order = mz_order
        self._area = feature_table[c.HEIGHT].to_numpy()[mz_order]
        self._roi_index = feature_table[c.ROI_INDEX].to_numpy()[mz_order]
        self._ft_index = feature_table[c.FT_INDEX].to_numpy()[mz_order]
        self._roi_list = roi_list
        self.similarity_checker.load_data(roi_list, self._roi_index, self._ft_index)
        self._non_annotated = set(range(self._mz.size))
        self._mono_candidates = list(np.argsort(self._area))
        self._envelope_label = -np.ones_like(self._roi_index)
        self._envelope_charge = -np.ones_like(self._roi_index)
        self._envelope_index = -np.ones_like(self._roi_index)

    def get_next_monoisotopologue_index(self) -> int:
        """
        Gets the next, non-annotated, monoisotopologue index. If all features
        are annotated, returns ``-1``.

        """
        next_mono_index = -1
        while self._mono_candidates:
            mono_index = self._mono_candidates.pop()
            if mono_index in self._non_annotated:
                next_mono_index = mono_index
                break
        return next_mono_index

    def _get_mmi_candidates(self, mono_index: int):
        mmi_candidates = self.mmi_finder.find(self._mz, self._area, mono_index)

        # check the similarity between the monoisotopologue and the mmi
        mmi_candidates = self.similarity_checker.filter_mmi_candidates(
            mmi_candidates, mono_index, self._non_annotated
        )
        return mmi_candidates

    def _get_envelope_candidates(self, mono_index: int, mmi: int, q: int):
        candidates = self.envelope_finder.find(self._mz, mmi, q,
                                               self._non_annotated)

        # remove candidates where the monoisotopologue is not present
        if mmi != mono_index:
            candidates = [x for x in candidates if mono_index in x]

        candidates = self.similarity_checker.filter_envelope_candidates(
            candidates)
        return candidates

    def annotate(self, mono_index: int):
        """
        Finds an annotation using a monoisotopologue candidate.

        If an annotation is found, the annotated features are removed. Else,
        only the monoisotopologue candidate is removed.

        """
        mmi_candidates = self._get_mmi_candidates(mono_index)
        validated_index = None
        validated_length = 1
        validated_charge = -1
        for mmi, q in mmi_candidates:
            candidates = self._get_envelope_candidates(mono_index, mmi, q)
            for candidate in candidates:
                candidate_length = len(candidate)
                if candidate_length > validated_length:
                    M, p = self._get_candidate_envelope(candidate, q)
                    candidate_validated_length = self.validator.validate(M, p)
                    if candidate_validated_length > validated_length:
                        validated_index = candidate[:candidate_validated_length]
                        validated_length = candidate_validated_length
                        validated_charge = q
        if validated_index is None:
            self._non_annotated.remove(mono_index)
        else:
            self._envelope_label[validated_index] = self._label_counter
            self._envelope_charge[validated_index] = validated_charge
            envelope_labels = np.arange(validated_length)
            self._envelope_index[validated_index] = envelope_labels
            self._label_counter += 1
            for x in validated_index:
                self._non_annotated.remove(x)

    def add_annotations(self, feature_table: pd.DataFrame):
        """
        Add annotations to the feature table.

        """
        original_order = np.argsort(self._mz_order)
        feature_table[c.ENVELOPE_INDEX] = self._envelope_index[original_order]
        feature_table[c.ENVELOPE_LABEL] = self._envelope_label[original_order]
        feature_table[c.CHARGE] = self._envelope_charge[original_order]

    def _get_candidate_envelope(
        self, candidate: List[int], charge: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes refined M and p values of a candidate.
        """
        p = _get_candidate_abundance(
            candidate, self._roi_list, self._roi_index, self._ft_index
        )
        M = _get_candidate_mass(candidate, self._mz, charge, self.polarity)
        return M, p


class _SimilarityChecker:
    """
    Stores and retrieves the similarity between features in a sample.

    Attributes
    ----------
    func: Callable
        A function to compute fhe similarity between a pair of features. Must
        have the following signature:

        .. code-block:: python

            def func(roi1, ft1, roi2, ft2) -> float:
                pass

        It must return a value between 0 and 1. 0 is associated with low
        similarity and 1 with high similarity.
    min_similarity : float
        Minimum similarity value used during feature filtering.

    """

    def __init__(self, min_similarity: float, similarity_func: Callable, **kwargs):
        """
        Constructor function.

        Parameters
        ----------
        func: Callable
            A function to compute fhe similarity between a pair of features. It
            Must have the following signature:

            .. code-block:: python

                def func(roi1, ft1, roi2, ft2) -> float:
                    pass

            It must return a value between 0 and 1. 0 is associated with low
            similarity and 1 with high similarity.
        min_similarity : float
            Minimum similarity value used during feature filtering.

        Other Parameters
        ----------------
        kwargs : dict
            Key-value parameters passed to `func`.

        """
        self._roi_list: Optional[List[LCRoi]] = None
        self._roi_index: Optional[np.ndarray] = None
        self._ft_index: Optional[np.ndarray] = None
        self._cache: Optional[Dict] = None
        self.func_params = kwargs
        self.min_similarity = min_similarity
        self.func = similarity_func

    def clear_data(self):
        """
        Clears data from a sample.

        """
        self._roi_list = None
        self._roi_index = None
        self._ft_index = None
        self._cache = None

    def load_data(
        self, roi_list: List[LCRoi], roi_index: np.ndarray, ft_index: np.ndarray
    ):
        """
        Loads data from a samples.
        """
        self._roi_list = roi_list
        self._ft_index = ft_index
        self._roi_index = roi_index
        self._cache = dict()

    def get_similarity(self, i: int, j: int):
        i_sim = self._cache.setdefault(i, dict())
        j_sim = self._cache.setdefault(j, dict())
        if j in i_sim:
            ij_similarity = i_sim[j]
        else:
            i_roi = self._roi_list[self._roi_index[i]]
            i_ft = i_roi.features[self._ft_index[i]]
            j_roi = self._roi_list[self._roi_index[j]]
            j_ft = j_roi.features[self._ft_index[j]]
            ij_similarity = self.func(i_roi, i_ft, j_roi, j_ft, **self.func_params)
            i_sim[j] = ij_similarity
            j_sim[i] = ij_similarity
        return ij_similarity

    def filter_mmi_candidates(
        self,
        candidates: List[Tuple[int, int]],
        mono_index: int,
        non_annotated: Set[int],
    ) -> List[Tuple[int, int]]:
        """
        Filter mmi candidates based on their similarity with the monoisotopologue.
        """
        candidates = [x for x in candidates if x[0] in non_annotated]
        valid_candidates = list()
        for candidate in candidates:
            ind, _ = candidate
            similarity = self.get_similarity(mono_index, ind)
            if similarity >= self.min_similarity:
                valid_candidates.append(candidate)
        return valid_candidates

    def filter_envelope_candidates(self, candidates: List[List[int]]):
        """
        Filter candidates based on their similarity with the MMI. The candidate
        ir cropped at the first position when an isotopologue has a similarity
        below the threshold. Candidates with length = 1 are removed.
        """
        valid_candidates = list()
        # filter by similarity
        for candidate in candidates:
            mmi = candidate[0]
            validated = [mmi]
            for i in candidate[1:]:
                i_sim = self.get_similarity(mmi, i)
                if i_sim >= self.min_similarity:
                    validated.append(i)
                else:
                    break
            if len(validated) > 1:
                valid_candidates.append(validated)
        valid_candidates = _remove_sub_candidates(valid_candidates)
        return valid_candidates


def _feature_similarity_lc(
    roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak, min_overlap: float
) -> float:
    """
    Feature similarity function used in LC-MS data.
    """
    start1 = roi1.scan[ft1.start]
    start2 = roi2.scan[ft2.start]
    if start1 > start2:
        roi1, roi2 = roi2, roi1
        ft1, ft2 = ft2, ft1
    overlap_ratio = _overlap_ratio(roi1, ft1, roi2, ft2)
    has_overlap = overlap_ratio > min_overlap
    if has_overlap:
        os1, oe1, os2, oe2 = _get_overlap_index(roi1, ft1, roi2, ft2)
        norm1 = np.linalg.norm(roi1.spint[ft1.start : ft1.end])
        norm2 = np.linalg.norm(roi2.spint[ft2.start : ft2.end])
        x1 = roi1.spint[os1:oe1] / norm1
        x2 = roi2.spint[os2:oe2] / norm2
        similarity = np.dot(x1, x2)
    else:
        similarity = 0.0
    return similarity


def _overlap_ratio(roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak) -> float:
    """
    Computes the overlap ratio, defined as the quotient between the overlap
    region and the extension of the longest feature.

    `ft1` must start before `ft2`

    Parameters
    ----------
    roi1 : LCRoi
    ft1 : Peak
    roi2 : LCRoi
    ft2 : Peak

    Returns
    -------
    overlap_ratio : float

    """
    start2 = roi2.scan[ft2.start]
    end1 = roi1.scan[ft1.end - 1]
    end2 = roi2.scan[ft2.end - 1]
    # start1 <= start2. end1 > start2 is a sufficient condition for overlap
    if end1 > start2:
        # the overlap ratio is the quotient between the length overlapped region
        # and the extension of the shortest feature.
        if end1 <= end2:
            start2_index_in1 = bisect.bisect_left(roi1.scan, start2)
            overlap_length = ft1.end - start2_index_in1
        else:
            overlap_length = ft2.end - ft2.start
        min_length = min(ft1.end - ft1.start, ft2.end - ft2.start)
        res = overlap_length / min_length
    else:
        res = 0.0
    return res


def _get_overlap_index(
    roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak
) -> Tuple[int, int, int, int]:
    """
    Computes the overlap indices for ft1 and ft2.

    `ft1` must start before `ft2`

    Parameters
    ----------
    roi1 : LCRoi
    ft1 : Peak
    roi2 : LCRoi
    ft2 : Peak

    Returns
    -------
    overlap_start1 : int
    overlap_end1 : int
    overlap_start2 : int
    overlap_end2 : int

    """
    end1 = roi1.scan[ft1.end - 1]
    end2 = roi2.scan[ft2.end - 1]
    start2 = roi2.scan[ft2.start]
    if end1 >= end2:
        overlap_start1 = bisect.bisect_left(roi1.scan, start2)
        overlap_end1 = bisect.bisect(roi1.scan, end2)
        overlap_start2 = ft2.start
        overlap_end2 = ft2.end
    else:
        overlap_start1 = bisect.bisect_left(roi1.scan, start2)
        overlap_end1 = ft1.end
        overlap_start2 = ft2.start
        overlap_end2 = bisect.bisect(roi2.scan, end1)
    return overlap_start1, overlap_end1, overlap_start2, overlap_end2


def _filter_mmi_candidates(
    mmi_candidates: List[Tuple[int, int]],
    mono_index: int,
    non_annotated: Set[int],
    similarity_cache: _SimilarityChecker,
    min_similarity: float,
) -> List[Tuple[int, int]]:
    mmi_candidates = [x for x in mmi_candidates if x[0] in non_annotated]
    valid_candidates = list()
    for candidate in mmi_candidates:
        ind, _ = candidate
        similarity = similarity_cache.get_similarity(mono_index, ind)
        if similarity >= min_similarity:
            valid_candidates.append(candidate)
    return valid_candidates


def _get_candidate_abundance(
    ind: List[int], roi_list: List[LCRoi], roi_index: np.ndarray, ft_index: np.ndarray
):
    scan_start = 0
    scan_end = 10000000000  # dummy value
    roi_ft_pairs = list()
    size = len(ind)
    for k in ind:
        roi = roi_list[roi_index[k]]
        ft = roi.features[ft_index[k]]
        scan_start = max(scan_start, roi.scan[ft.start])
        scan_end = min(scan_end, roi.scan[ft.end - 1])
        roi_ft_pairs.append((roi, ft))

    p = np.zeros(shape=size, dtype=float)
    if scan_start < scan_end:
        for k in range(size):
            roi, ft = roi_ft_pairs[k]
            start = bisect.bisect(roi.scan, scan_start)
            end = bisect.bisect(roi.scan, scan_end)
            p[k] = trapz(roi.spint[start:end], roi.time[start:end])
        p /= p.sum()
    return p


def _get_candidate_mass(ind: List[int], mz: np.ndarray, charge: int, polarity: int):
    return mz[ind] * charge - polarity * charge * EM


def _remove_sub_candidates(candidates: List[List[int]]) -> List[List[int]]:
    """
    Remove candidates that are subsets of other candidates. e.g [1, 2] is
    removed if [1, 2, 3] is present.

    aux function of `IsotopologueAnnotator.annotate`.

    """
    validated = list()
    while candidates:
        last = candidates.pop()
        last_set = set(last)
        is_contained = False
        for candidate in candidates:
            is_contained = last_set <= set(candidate)
        if not is_contained:
            validated.append(last)
    return validated
