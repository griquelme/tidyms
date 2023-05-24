import numpy as np
import pandas as pd
from typing import Sequence
from .annotation_data import AnnotationData
from .envelope_finder import EnvelopeFinder
from .mmi_finder import MMIFinder
from ..lcms import Feature
from ..chem import EnvelopeValidator
from ..chem.atoms import EM, PeriodicTable
from .. import _constants as c


def create_annotation_table(feature_list: list[Feature]) -> pd.DataFrame:
    d: dict[str, list[int]] = {
        c.ROI_INDEX: list(),
        c.FT_INDEX: list(),
        c.ENVELOPE_INDEX: list(),
        c.ENVELOPE_LABEL: list(),
        c.CHARGE: list(),
    }

    for ft in feature_list:
        annotation = ft.annotation
        d[c.CHARGE].append(annotation.charge)
        d[c.ENVELOPE_INDEX].append(annotation.isotopologue_index)
        d[c.ENVELOPE_LABEL].append(annotation.isotopologue_label)
        d[c.ROI_INDEX].append(ft.roi.id)
        d[c.FT_INDEX].append(ft.id)

    return pd.DataFrame(d)


def create_annotation_tools(
    bounds: dict[str, tuple[int, int]],
    max_mass: float,
    max_charge: int,
    max_length: int,
    min_M_tol: float,
    max_M_tol: float,
    p_tol: float,
    min_similarity: float,
    min_p: float,
) -> tuple[MMIFinder, EnvelopeFinder, EnvelopeValidator]:
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
        `max_M_tol`. For values in between, a weighted tolerance is used based
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
    annotator: _IsotopologueAnnotator

    """
    # remove elements with only 1 stable isotope
    p_table = PeriodicTable()
    bounds = {k: bounds[k] for k in bounds if len(p_table.get_element(k).isotopes) > 1}

    bin_size = 100
    elements = list(bounds)
    mmi_finder = MMIFinder(
        bounds,
        max_mass,
        max_charge,
        max_length,
        bin_size,
        max_M_tol,
        p_tol,
        min_similarity,
    )
    envelope_finder = EnvelopeFinder(elements, max_M_tol, max_length, min_p, min_similarity)
    envelope_validator = EnvelopeValidator(
        bounds,
        max_M=max_mass,
        max_length=max_length,
        min_M_tol=min_M_tol,
        max_M_tol=max_M_tol,
        p_tol=p_tol,
    )
    return mmi_finder, envelope_finder, envelope_validator


def annotate(
    feature_list: Sequence[Feature],
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
    data: AnnotationData,
    monoisotopologue: Feature,
    polarity: int,
    mmi_candidates: Sequence[tuple[Feature, int]],
    envelope_finder: EnvelopeFinder,
    envelope_validator: EnvelopeValidator,
) -> tuple[Sequence[Feature], int]:
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
    candidate: Sequence[Feature],
    monoisotopologue: Feature,
    charge: int,
    polarity: int,
    min_length: int,
    validator: EnvelopeValidator,
) -> int:
    if len(candidate) <= min_length:
        return 0

    if monoisotopologue not in candidate:
        return 0

    M, p = candidate[0].compute_isotopic_envelope(candidate)
    em_correction = EM * charge * polarity
    M = np.array(M) * charge - em_correction
    p = np.array(p)
    return validator.validate(M, p)
