from typing import Optional, Union
from ..lcms import Feature
from collections.abc import Sequence


class AnnotationData:
    """
    Feature data.

    Attributes
    ----------
    features : list[Feature]
        List of features sorted by m/z
    annotation : dict[Feature, int]
        Annotation of features
    similarity_cache : SimilarityCache
        Stores similarity between features.
    non_annotated : set[Feature]
        Non-annotated features.

    """

    def __init__(self, features: Sequence[Feature]):
        self.features = sorted(features)
        self.non_annotated = set(features)
        self._monoisotopologues = sorted(features, key=lambda x: x.area)
        self.similarity_cache = SimilarityCache()
        self._label_counter = 0

    def get_monoisotopologue(self) -> Optional[Feature]:
        """Gets the current non-annotated feature with the greatest area."""
        if self._monoisotopologues:
            return self._monoisotopologues[-1]
        else:
            return None

    def annotate(self, features: list[Feature], charge: int):
        """Labels a list of features as an isotopic envelope."""
        for k, ft in enumerate(features):
            ft.annotation.charge = charge
            ft.annotation.isotopologue_label = self._label_counter
            ft.annotation.isotopologue_index = k
        self._label_counter += 1

    def _flag_annotated(self, feature: Feature):
        """Flag features as annotated."""
        self.non_annotated.discard(feature)
        if self._monoisotopologues and (feature == self._monoisotopologues[-1]):
            self._monoisotopologues.pop()


class SimilarityCache:
    """Stores and retrieves the similarity between features in a sample."""

    def __init__(self):
        self._cache: dict[Feature, dict[Feature, float]] = dict()

    def get_similarity(self, ft1: Feature, ft2: Feature):
        ft1_sim = self._cache.setdefault(ft1, dict())
        ft2_sim = self._cache.setdefault(ft2, dict())
        if ft2 in ft1_sim:
            similarity = ft1_sim[ft2]
        else:
            similarity = ft1.compare(ft2)
            ft1_sim[ft2] = similarity
            ft2_sim[ft1] = similarity
        return similarity
