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

__version__ = "0.4.0"

from . import fileio
from . import container
from . import utils
from . import peaks
from . import filter
from . import lcms
from . import simulation
from . import raw_data_utils
from . import _mzml
from . import correspondence
from .container import DataContainer
from .fileio import MSData
from .lcms import Chromatogram, MSSpectrum
from .assay import Assay
from .raw_data_utils import *

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