import bisect
import numpy as np
import pandas as pd
from .lcms import LCRoi, Peak
from .chem import FormulaGenerator, IsotopeScorer, EnvelopeFinder, MMIFinder
from . import _constants as c
from typing import List, Set, Tuple


class IsotopologueAnnotator:

    def __init__(
        self,
        mmi_finder: MMIFinder,
        envelope_finder: EnvelopeFinder,
        envelope_validator: IsotopeScorer,
        min_similarity: float
    ):
        self.mmi_finder = mmi_finder
        self.envelope_finder = envelope_finder
        self.envelope_validator = envelope_validator
        self.min_similarity = min_similarity
        self.similarity_cache = None

        # sample data
        self._feature_table = None
        self._roi_list = None
        self._mz = None
        self._area = None
        self._roi_index = None
        self._ft_index = None
        self._non_annotated = None
        self._mono_candidates = None

    def clear_data(self):
        self._feature_table = None
        self._roi_list = None
        self._mz = None
        self._area = None
        self._roi_index = None
        self._ft_index = None
        self.similarity_cache = None
        self._non_annotated = None
        self._mono_candidates = None

    def load_data(self, feature_table: pd.DataFrame, roi_list: List[LCRoi]):
        self._feature_table = feature_table.sort_values(c.MZ)
        self._mz = self._feature_table[c.MZ].to_numpy()
        self._area = self._feature_table[c.AREA].to_numpy()
        self._roi_index = self._feature_table[c.ROI_INDEX].to_numpy()
        self._ft_index = self._feature_table[c.FT_INDEX].to_numpy()
        self._roi_list = roi_list
        self.similarity_cache = _SimilarityCache(
            self._roi_list, self._roi_index, self._ft_index
        )
        self._non_annotated = set(range(self._mz.size))
        self._mono_candidates = list(np.argsort(self._area))

    def annotate(self):
        if self._feature_table is None:
            # TODO: add a better message
            msg = "No sample set for annotation."
            raise ValueError(msg)

        # get nex non annotated monoisotopic index
        mono_index = -1
        while mono_index not in self._non_annotated:
            mono_index = self._mono_candidates.pop()

        # add monoisotopic is MMI candidates and merge with mmi_candidates

        # check if there are mmi candidates other than the monoisotopic index
        mmi_candidates = self.mmi_finder.find(self._mz, self._area, mono_index)
        mmi_candidates = _filter_mmi_candidates(
            mmi_candidates, mono_index, self._non_annotated,
            self.similarity_cache, self.min_similarity
        )
        for mmi_index, q in mmi_candidates:
            pass

        # for candidates, charge, find envelopes
        # envelope finder should take a charge parameter to find remove max
        # charge think about a smart validation strategy to reduce the number of
        # validations


class _SimilarityCache:
    """
    Stores and retrieves the cosine similarity for features in a sample.

    Attributes
    ----------

    """
    def __init__(
        self,
        roi_list: List[LCRoi],
        roi_index: np.ndarray,
        ft_index: np.ndarray,
    ):
        self.roi_list = roi_list
        self.roi_index = roi_index
        self.ft_index = ft_index
        self.similarity = dict()

    def get_similarity(self, i: int, j: int):
        i_sim = self.similarity.setdefault(i, dict())
        j_sim = self.similarity.setdefault(j, dict())
        if j in i_sim:
            ij_similarity = i_sim[j]
        else:
            i_roi = self.roi_list[self.roi_index[i]]
            i_ft = i_roi.features[self.ft_index[i]]
            j_roi = self.roi_list[self.roi_index[j]]
            j_ft = i_roi.features[self.ft_index[i]]
            min_overlap = 0.5
            ij_similarity = _feature_similarity(
                i_roi, i_ft, j_roi, j_ft, min_overlap)
            i_sim[j] = ij_similarity
            j_sim[i] = ij_similarity
        return ij_similarity


def _feature_similarity(
    roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak, min_overlap: float
) -> float:
    start1 = roi1.scan[ft1.start]
    start2 = roi2.scan[ft2.start]
    if start1 > start2:
        roi1, roi2 = roi2, roi1
        ft1, ft2 = ft2, ft1
    overlap_ratio = _overlap_ratio(roi1, ft1, roi2, ft2)
    has_overlap = overlap_ratio > min_overlap
    if has_overlap:
        os1, oe1, os2, oe2 = _get_overlap_index(roi1, ft1, roi2, ft2)
        norm1 = np.linalg.norm(roi1.spint[ft1.start:ft1.end])
        norm2 = np.linalg.norm(roi2.spint[ft2.start:ft2.end])
        x1 = roi1.spint[os1:oe1] / norm1
        x2 = roi2.spint[os2:oe2] / norm2
        similarity = np.dot(x1, x2)
    else:
        similarity = 0.0
    return similarity


def _overlap_ratio(
        roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak) -> float:
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
        # and the extension of the longest feature.
        if end1 <= end2:
            start2_index_in1 = bisect.bisect_left(roi1.scan, start2)
            overlap_length = ft1.end - start2_index_in1
        else:
            overlap_length = ft2.end - ft2.start
        max_length = min(ft1.end - ft1.start, ft2.end - ft2.start)
        res = overlap_length / max_length
    else:
        res = 0.0
    return res


def _get_overlap_index(
        roi1: LCRoi, ft1: Peak, roi2: LCRoi, ft2: Peak
) -> Tuple[int, int, int, int]:
    """
    Computes the overlap indices for ft1 and ft2

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
        similarity_cache: _SimilarityCache,
        min_similarity: float
) -> List[Tuple[int, int]]:
    mmi_candidates = [x for x in mmi_candidates if x[0] in non_annotated]
    valid_candidates = list()
    for candidate in mmi_candidates:
        ind, _ = candidate
        similarity = similarity_cache.get_similarity(mono_index, ind)
        if similarity >= min_similarity:
            valid_candidates.append(candidate)
    return valid_candidates
