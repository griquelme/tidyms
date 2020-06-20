
.. py:currentmodule:: tidyms

API reference
=============

Tools for working with raw data
-------------------------------

.. autosummary::
    :toctree: generated

    tidyms.MSData
    tidyms.Chromatogram
    tidyms.MSSpectrum
    tidyms.detect_features
    tidyms.feature_correspondence
    tidyms.make_data_container

Tools for working with processed data
-------------------------------------

.. autosummary::
    :toctree: generated

    tidyms.DataContainer
    tidyms.filter.Pipeline

List of available filters and processors
----------------------------------------

.. autosummary::
    :toctree: generated

    tidyms.filter.BatchCorrector
    tidyms.filter.BlankCorrector
    tidyms.filter.ClassRemover
    tidyms.filter.DRatioFilter
    tidyms.filter.PrevalenceFilter
    tidyms.filter.VariationFilter

Module reference
----------------

.. autosummary::
    :toctree: generated

    tidyms.container
    tidyms.fileio
    tidyms.filter
    tidyms.lcms
    tidyms.metabolomics
    tidyms.peaks
    tidyms.utils
