TidyMS: Tools for working with MS data in metabolomics
======================================================

TidyMS is a python library for processing Mass Spectrometry data. It aims to
provide easy to use tools to read, process and visualize MS data generated in
metabolomic studies.

Features
--------

TidyMS provides functionality to:

1. Read raw MS data in the mzML format
2. Spectrum and chromatogram creation.
3. Powerful and flexible peak picking functions optimized for chromatographic
   and spectral data.
4. Feature detection and feature correspondence in LC-MS data.
5. Reading processed data in a variety of formats (XCMS, MZMine2, ...)
5. Data matrix curation using widely accepted guidelines from the metabolomics
   community.
6. Interactive visualizations of raw and processed data using Bokeh, or
   publication quality plots using seaborn.

Installation
------------

The latest release can be installed from PyPI:

```
    pip install tidyms
```

Examples
--------

Jupyter notebooks with examples are available
[here](https://github.com/griquelme/tidyms-notebooks).

Documentation
-------------

The official documentation is available at 
[readthedocs](https://tidyms.readthedocs.io/en/latest/).


Citation
--------

If you find TidyMS useful, we would appreciate citations:

Riquelme, G.; Zabalegui, N.; Marchi, P.; Jones, C.M.; Monge, M.E. A Python-Based
Pipeline for Preprocessing LC–MS Data for Untargeted Metabolomics Workflows.
_Metabolites_ **2020**, 10, 416, doi:10.3390/metabo10100416.

