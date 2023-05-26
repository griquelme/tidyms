"""
Core classes used by Tidyms.

See HERE for an explanation of the TidyMS architecture.

Annotation : Stores annotation data from a feature.
Assay : Manages data processing of complete datasets.
AssayData : Persists data from an Assay.
Feature : A region associated with a ROI that contains a chemical species.
MultipleSampleProcessor : Process data from multiple samples stored in an AssayData instance.
ProcessingPipeline : Apply several processing steps to data.
SingleSampleProcessor : Process a SampleData instance.
Roi : A Region of Interest extracted from raw data. Usually a subset of raw data.
Sample : Stores metadata from a measurement.
SampleData : Container class for a Sample and the ROIs detected.

"""

from .assay_data import AssayData
from .assay import (
    Assay,
    FeatureExtractor,
    ProcessingPipeline,
    Processor,
    SingleSampleProcessor,
    MultipleSampleProcessor,
)
from .base import Annotation, Feature, Roi, Sample, SampleData

__all__ = [
    "Annotation",
    "Assay",
    "AssayData",
    "Feature",
    "FeatureExtractor",
    "MultipleSampleProcessor",
    "ProcessingPipeline",
    "Processor",
    "Roi",
    "SingleSampleProcessor",
    "Sample",
    "SampleData",
]
