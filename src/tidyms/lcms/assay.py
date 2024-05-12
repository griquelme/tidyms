"""Assay utilities for LC-MS data."""

import pathlib

from ..annotation.isotopologue_annotator import IsotopologueAnnotator
from ..core.assay import Assay, AssayConfiguration
from ..core.constants import MSInstrument, Polarity, SeparationMode
from ..core.processors import ProcessingPipeline
from . import processors
from .lcms import MZTrace, Peak


def create_lcms_assay(
    *,
    path: pathlib.Path | None = None,
    instrument: MSInstrument = MSInstrument.QTOF,
    separation: SeparationMode = SeparationMode.UPLC,
    polarity: Polarity = Polarity.NEGATIVE,
    annotate_isotopologues: bool = True,
) -> Assay[Peak, MZTrace]:
    """
    Create an assay instance for LC-MS data.

    Parameters
    ----------
    path : Path | None = None
        Path to an assay database. If ``None``, uses an in-memory database.
    instrument : MSinstrument, default=QTOF
        The instrument used to measure the assay samples.
    separation : SeparationMode, default=UPLC
        The analytical separation method used to measure the assay samples.
    polarity: Polarity, default=NEGATIVE
        The polarity in which the samples were measured.
    annotate_isotopologues: bool, default=True
        If set to ``True``, include an isotopologue annotation in the processing
        pipeline.

    Returns
    -------
    Assay:
        A new assay instance optimized.

    """
    sample_pipe = create_sample_pipeline(annotate_isotopologues)
    if separation not in [SeparationMode.HPLC, SeparationMode.UPLC]:
        raise ValueError(f"separation must be set either to HPLC or UPLC. Got {separation}.")

    config = AssayConfiguration(
        roi=MZTrace.__name__,
        feature=Peak.__name__,
        sample_pipeline=sample_pipe,
        path=path,
        polarity=polarity,
        separation=separation,
        instrument=instrument,
    )
    return Assay(config)


def create_sample_pipeline(annotate_isotopologues: bool = True) -> ProcessingPipeline:
    """Create a sample processing pipeline."""
    steps = [
        processors.LCTraceExtractor(id="roi_extractor"),
        processors.LCTraceBaselineEstimator(id="roi_baseline_estimator"),
        processors.PeakExtractor(id="peak_extractor"),
    ]
    if annotate_isotopologues:
        steps.append(IsotopologueAnnotator(id="isotopologue_annotator"))

    return ProcessingPipeline(id="sample_pipeline", processors=steps)
