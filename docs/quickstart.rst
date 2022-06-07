.. _quickstart:

.. py:currentmodule:: tidyms

Quickstart
==========

TidyMS [1]_ is a Python package that provides tools to process and analyze
Mass Spectrometry (MS) data. Although suited for general use, it was designed
to be used with datasets from LC-HRMS metabolomics experiments. It uses
`Numpy <https://numpy.org/>`_, `Pandas <https://pandas.pydata.org/>`_ and
`scikit-learn <https://scikit-learn.org>`_ for data processing and analysis.
Some of the functionality that offers is:

*   read raw data in the mzML format using :class:`tidyms.MSData` class,
    optimized for speed and low memory usage.
*   Creation of chromatograms and accumulated spectra from raw data.
*   :term:`Feature detection<feature detection>` and
    :term:`feature correspondence` in metabolomics datasets using the
    :class:`tidyms.Assay` class.
*   Read processed data from other mass spectrometry processing software
    (XCMS, mzmine2, etc...).
*   A container object to manage metabolomics data.
*   :term:`Data curation<data curation>` of untargeted metabolomics data sets
    using widely accepted practices from the metabolomics community [2]_ [3]_
*   Interactive data visualization using `bokeh <https://bokeh.org/>`_, or
    publication quality plots using `seaborn <https://seaborn.pydata.org/>`_.

In the rest of this guide, you can find links for different use cases for the
TidyMS package. A basic knowledge of MS and metabolomics is assumed, but you can
look up in the :doc:`glossary` the concepts used in the guides.
Installation instructions are available :doc:`here<installation>`.

You can refer to the following guides to learn about specific topics:

*   :ref:`Working with raw data <working-with-raw-data>`
*   :ref:`Processing complete datasets from raw data <processing-datasets>`
*   :ref:`Curation of a metabolomics data matrix <data-curation>`
*   :ref:`Feature detection and extraction algorithms <peak-picking>`
*   :ref:`Feature correspondence algorithm <ft-correspondence>`
*   :ref:`Converting proprietary instrument-specific formats into mzML <mzml>`


References
----------

..  [1] Riquelme, G. *et al*, "A Python-Based Pipeline for Preprocessing LC–MS
    Data for Untargeted Metabolomics Workflows". Metabolites 2020, 10, 416.
    https://doi.org/10.3390/metabo10100416
    16, 1, (2015), Pages 104–117, https://doi.org/10.1093/bib/bbt080
..  [2] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography
    coupled to mass spectrometry", Nature Protocols volume 6, pages
    1060–1083 (2011).
..  [3] D Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies.", Metabolomics,
    2018;14(6):72. doi: 10.1007/s11306-018-1367-3