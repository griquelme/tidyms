"""
The Mass Spectrometry Kit for Metabolomics (MSKM)
=================================================

A package to work with Mass Spectrometry data from Metabolomics Experiments.

Provides
    1. The MSData object to analyze and process raw MS data.
    2. The DataContainer object to store metabolomics data sets.
    3. Pipeline and Processor objects to perform curation of data sets.
    4. Functions for feature detection and feature correspondence.
"""

from . import fileio
from . import data_container
from . import utils
from . import peaks
from . import filter
from . import lcms
from . import metabolomics
from .data_container import DataContainer
from .fileio import MSData
