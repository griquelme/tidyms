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

__version__ = "0.2.1."

from . import fileio
from . import container
from . import utils
from . import peaks
from . import filter
from . import lcms
from . import simulation
from . import _mzml
from .metabolomics import *
from .container import DataContainer
from .fileio import MSData
from .lcms import Chromatogram, MSSpectrum

if utils.SETTINGS["bokeh"]["apply_theme"]:
    from bokeh.themes import Theme as _Theme
    from bokeh.io import curdoc as _curdoc
    theme = utils.SETTINGS["bokeh"]["theme"]
    _curdoc().theme = _Theme(json=theme)

if utils.is_notebook():
    from bokeh.plotting import output_notebook as _output_notebook
    _output_notebook()
