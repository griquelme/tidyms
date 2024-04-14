"""Storage tools for the annotation algorithm."""

from typing import Generic, Sequence

from ..core.models import AnnotableFeatureType


class AnnotationData(Generic[AnnotableFeatureType]):
    """
    Feature data.

    Attributes
    ----------
    features : list[Feature]
        List of features sorted by m/z
    similarity_cache : SimilarityCache
        Stores similarity between features.
    non_annotated : set[Feature]
        Non-annotated features.

    """

    def __init__(self, features: Sequence[AnnotableFeatureType]):
        self.features = sorted(features, key=lambda x: x.get("mz"))
        self.non_annotated = set(features)
        self._monoisotopologues = sorted(features, key=lambda x: x.get("height"))
        self.similarity_cache = SimilarityCache()
        self._label_counter = 0

    def get_monoisotopologue(self) -> AnnotableFeatureType | None:
        """Get the current non-annotated feature with the greatest area."""
        if self._monoisotopologues:
            mono = self._monoisotopologues[-1]
            while self._monoisotopologues and (mono not in self.non_annotated):
                self._monoisotopologues.pop()
                if self._monoisotopologues:
                    mono = self._monoisotopologues[-1]
                else:
                    mono = None
        else:
            mono = None
        return mono

    def annotate(self, features: Sequence[AnnotableFeatureType], charge: int):
        """Labels a list of features as an isotopic envelope."""
        if len(features) > 1:
            for k, ft in enumerate(features):
                ft.annotation.charge = charge
                ft.annotation.isotopologue_label = self._label_counter
                ft.annotation.isotopologue_index = k
                self._flag_annotated(ft)
            self._label_counter += 1
        else:
            self._flag_annotated(features[0])

    def _flag_annotated(self, feature: AnnotableFeatureType):
        """Flag features as annotated."""
        self.non_annotated.discard(feature)
        if self._monoisotopologues and (feature == self._monoisotopologues[-1]):
            self._monoisotopologues.pop()


class SimilarityCache(Generic[AnnotableFeatureType]):
    """Stores and retrieves the similarity between features in a sample."""

    def __init__(self):
        self._cache: dict[AnnotableFeatureType, dict[AnnotableFeatureType, float]] = dict()

    def get_similarity(self, ft1: AnnotableFeatureType, ft2: AnnotableFeatureType):
        """Get the similarity between a pair of features."""
        ft1_sim = self._cache.setdefault(ft1, dict())
        ft2_sim = self._cache.setdefault(ft2, dict())
        if ft2 in ft1_sim:
            similarity = ft1_sim[ft2]
        else:
            similarity = ft1.compare(ft2)
            ft1_sim[ft2] = similarity
            ft2_sim[ft1] = similarity
        return similarity
