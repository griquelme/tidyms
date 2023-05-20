"""
TidyMS
======

A package to work with Mass Spectrometry data from Metabolomics Experiments.

Provides
    1. The Assay object to process datasets from raw data.
    2. The MSData object to work with raw data.
    3. The DataContainer object to store metabolomics data sets.
    4. Pipeline and Processor objects to perform curation of data sets.

"""

from . import chem
from . import fileio
from . import container
from . import utils
from . import peaks
from . import filter
from . import lcms
from . import simulation
from . import raw_data_utils
from . import _mzml
from . import _build_data_matrix
from . import correspondence
from . import fill_missing
from . import consensus_annotation
from .container import DataContainer
from .fileio import MSData
from .lcms import Chromatogram, MSSpectrum
from .assay import LegacyAssay
from .raw_data_utils import *
from . import dartms
from .annotation import annotation

utils.create_tidyms_dir()
SETTINGS = utils.get_settings()

if SETTINGS["bokeh"]["apply_theme"]:
    from bokeh.themes import Theme as _Theme
    from bokeh.io import curdoc as _curdoc

    theme = SETTINGS["bokeh"]["theme"]
    _curdoc().theme = _Theme(json=theme)

if utils.is_notebook():
    from bokeh.plotting import output_notebook as _output_notebook

    _output_notebook()
