"""
TidyMS
======

A package to work with Mass Spectrometry data from Metabolomics Experiments.

Provides
    1. The MSData object to analyze and process raw MS data.
    2. The DataContainer object to store metabolomics data sets.
    3. Pipeline and Processor objects to perform curation of data sets.
    4. Functions for feature detection and feature correspondence.
"""

from . import fileio
from . import container
from . import utils
from . import peaks
from . import filter
from . import lcms
from .metabolomics import *
from .container import DataContainer
from .fileio import MSData
from .lcms import Chromatogram, MSSpectrum

__version__ = "0.1.0"