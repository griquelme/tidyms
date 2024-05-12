"""Annotate function implementation."""

from typing import Sequence

import numpy as np
import pydantic

from ..chem import EnvelopeValidator
from ..chem.atoms import EM, PeriodicTable
from ..core.models import AnnotableFeature, AnnotableFeatureType
from .annotation_data import AnnotationData
from .envelope_finder import EnvelopeFinder
from .mmi_finder import MMIFinder


class AnnotatorParameters(pydantic.BaseModel):
    """Store isotopologue annotator parameters."""

    bounds: dict[str, tuple[int, int]] = dict()
    """A dictionary of expected elements to minimum and maximum formula coefficients. If ``None``, bounds are
    obtained using :func:`tidyms.chem.get_chnops_bound` with ``2000`` as parameter.
    """

    max_mass: pydantic.PositiveFloat = 2000.0
    """Maximum exact mass of the features"""

    max_charge: int = 3
    """Maximum charge of the features. Use negative values for negative polarity."""

    max_length: int = 10
    """Maximum length of the envelopes."""

    min_M_tol: pydantic.PositiveFloat = 0.005
    """Minimum mass tolerance used during search. isotopologues with abundance equal to 1 use this value.
    Isotopologues with abundance equal to 0 use `max_M_tol`. For values in between, a weighted tolerance
    is used based on the abundance.
    """

    max_M_tol: pydantic.PositiveFloat = 0.01
    """Maximum mass tolerance used during search"""

    p_tol: float = 0.05
    """Abundance tolerance"""

    min_similarity: float = pydantic.Field(default=0.9, ge=0.5, le=1.0)
    """Minimum cosine similarity between a pair of features"""

    min_p: float = pydantic.Field(default=0.005, ge=0.0, le=0.1)
    """Minimum abundance of isotopes to include in candidate search."""


def create_annotation_tools(config: AnnotatorParameters) -> tuple[MMIFinder, EnvelopeFinder, EnvelopeValidator]:
    """
    Create an annotator object. Auxiliary function to _annotate.

    Parameters
    ----------
    config: AnnotatorParameters

    Returns
    -------
    MMIFinder
    EnvelopeFinder
    EnvelopeValidator

    """
    # remove elements with only 1 stable isotope
    p_table = PeriodicTable()
    bounds = {k: config.bounds[k] for k in config.bounds if len(p_table.get_element(k).isotopes) > 1}

    bin_size = 100
    elements = list(bounds)
    mmi_finder: MMIFinder = MMIFinder(
        config.bounds,
        config.max_mass,
        config.max_charge,
        config.max_length,
        bin_size,
        config.max_M_tol,
        config.p_tol,
        config.min_similarity,
    )
    envelope_finder: EnvelopeFinder = EnvelopeFinder(
        elements, config.max_M_tol, config.max_length, config.min_p, config.min_similarity
    )
    envelope_validator = EnvelopeValidator(
        bounds,
        max_M=config.max_mass,
        max_length=config.max_length,
        min_M_tol=config.min_M_tol,
        max_M_tol=config.max_M_tol,
        p_tol=config.p_tol,
    )
    return mmi_finder, envelope_finder, envelope_validator


def annotate(
    feature_list: Sequence[AnnotableFeature],
    mmi_finder: MMIFinder,
    envelope_finder: EnvelopeFinder,
    envelope_validator: EnvelopeValidator,
) -> None:
    """
    Annotate isotopologues in a sample.

    Annotations are added to the `annotation` attribute of each feature.

    Parameters
    ----------
    feature_list : List[LCTrace]
        List of features obtained after feature extraction.
    mmi_finder : MMIFinder
    envelope_finder : EnvelopeFinder
    envelope_validator : EnvelopeValidator

    """
    data = AnnotationData(feature_list)
    monoisotopologue = data.get_monoisotopologue()
    polarity = mmi_finder.polarity
    while monoisotopologue is not None:
        mmi_candidates = mmi_finder.find(data)
        envelope, charge = find_best_envelope(
            data,
            monoisotopologue,
            polarity,
            mmi_candidates,
            envelope_finder,
            envelope_validator,
        )
        data.annotate(envelope, charge)
        monoisotopologue = data.get_monoisotopologue()


def find_best_envelope(
    data: AnnotationData[AnnotableFeatureType],
    monoisotopologue: AnnotableFeatureType,
    polarity: int,
    mmi_candidates: Sequence[tuple[AnnotableFeatureType, int]],
    envelope_finder: EnvelopeFinder,
    envelope_validator: EnvelopeValidator,
) -> tuple[Sequence[AnnotableFeatureType], int]:
    """Find the most fitting envelope candidate."""
    best_length = 1
    best_candidate = [monoisotopologue]
    best_charge = -1
    for mmi, charge in mmi_candidates:
        envelope_candidates = envelope_finder.find(data, mmi, charge)
        for candidate in envelope_candidates:
            validated_length = _validate_candidate(
                candidate,
                monoisotopologue,
                charge,
                polarity,
                best_length,
                envelope_validator,
            )
            if validated_length > best_length:
                best_length = validated_length
                best_candidate = candidate[:validated_length]
                best_charge = charge
    return best_candidate, best_charge


def _validate_candidate(
    candidate: Sequence[AnnotableFeatureType],
    monoisotopologue: AnnotableFeatureType,
    charge: int,
    polarity: int,
    min_length: int,
    validator: EnvelopeValidator,
) -> int:
    if len(candidate) <= min_length:
        return 0

    if monoisotopologue not in candidate:
        return 0

    envelope = candidate[0].compute_isotopic_envelope(candidate)
    em_correction = EM * charge * polarity
    M = np.array(envelope.mz) * charge - em_correction
    p = np.array(envelope.p)
    return validator.validate(M, p)
