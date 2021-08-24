.. quickstart

.. py:currentmodule:: tidyms

Quickstart
==========

TidyMS [1]_ is a Python package that provides tools to process and analyze
Mass Spectrometry (MS) data. . Although suited for general use, it’s
functionality was mostly intended to be used with data from LC-MS metabolomics
experiments.It uses the `pyopenms <https://www.openms.de/>`_ package to read raw
data in the mzML format and `Numpy <https://numpy.org/>`_,
`Pandas <https://pandas.pydata.org/>`_ and
`scikit-learn <https://scikit-learn.org>`_ for data processing and analysis.
Some of the functionality that offers is:

*   creation of chromatograms and accumulated spectra from raw data.
*   :term:`Feature detection<feature detection>` and
    :term:`feature correspondence` in metabolomics datasets.
*   read processed data from other mass spectrometry processing software
    (XCMS, mzmine2, etc...).
*   A container object to manage metabolomics data.
*   :term:`Data curation<data curation>` of untargeted metabolomics data sets
    using widely accepted practices from the metabolomics community [4]_
*   Interactive data visualization using `bokeh <https://bokeh.org/>`_, or
    publication quality plots using `seaborn <https://seaborn.pydata.org/>`_.

In the rest of this guide, we will show different use cases for the TidyMS
package. A basic knowledge of MS and metabolomics is assumed, but you can look
up in the :doc:`glossary` the concepts used in this guide.
Installation instructions are available :doc:`here<installation>`.

Creating chromatograms
----------------------

A :class:`tidyms.Chromatogram` object can be created using a retention time
array and an intensity array.

.. code-block:: python

    import tidyms as ms
    import numpy as np

    # generate always the same plot
    np.random.seed(1234)
    # this creates two gaussian overlapping peaks simulating a chromatogram
    rt = np.arange(100)
    chrom_params = np.array([[50, 5, 10], [70, 5, 20]])
    spint = ms.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)
    # add a noise term
    spint += np.random.normal(size=spint.size, scale=0.1)
    chromatogram = ms.Chromatogram(rt, spint)

Using the :class:`tidyms.Chromatogram` object, we can perform common tasks like
plotting or peak picking:

.. code-block:: python

    chromatogram.plot()

.. bokeh-plot:: plots/chromatogram.py
    :source-position: none

The :meth:`tidyms.Chromatogram.find_peaks` method returns a list of descriptors
for each detected peak:

.. code-block:: python

    >>> chromatogram.find_peaks()
    [{'height': 10.051, 'area': 123.08, 'loc': 49.96  'width': 18, 'snr': 85.7},
     {'height': 19.90, 'area': 251.15, 'loc': 69.94, 'width': 19, 'snr': 169.7}]


Also, the detected peaks are stored in the :code:`peaks` attribute of the
chromatogram and can be visualized in a plot:

.. bokeh-plot:: plots/chromatogram-with-peaks.py
    :source-position: none


You can read about the peak detection process :doc:`here<peak-picking>`.

Creating mass spectra
---------------------

In a similar way, we can create a :class:`tidyms.MSSpectrum` object:

.. code-block:: python

    mz = np.linspace(400, 404, 1000)

    # creates three gaussian peaks simulating an isotopic envelope
    sp_params = np.array([[401, 0.01, 100], [402, 0.01, 15], [403, 0.01, 2]])
    spint = ms.utils.gaussian_mixture(mz, sp_params).sum(axis=0)
    np.random.seed(1234)
    spint += np.random.normal(size=spint.size, scale=0.1)
    spectrum = ms.MSSpectrum(mz, spint)
    spectrum.find_peaks()
    spectrum.plot()

.. bokeh-plot:: plots/ms-spectrum.py
    :source-position: none

Reading raw data
----------------

In the majority of cases, chromatograms and spectra are going to be created
from experimental data. Working with raw data can be done using the
:class:`tidyms.MSData` object, which reads files in the mzML
format and can be used to extract information from raw data. You can read
:doc:`this tutorial <mzml>` on how to convert experimental data from
proprietary, instrument-specific formats to mzML. The following code snippet
downloads an example data file that we are going to use to illustrate the
usage of the MSData object:

.. code-block:: python

    from ftplib import FTP
    import os

    study_path = "pub/databases/metabolights/studies/public/MTBLS1919"
    sample_path = os.path.join(study_path, "Applications/Centroid_data")
    filename = "NZ_20200227_041.mzML"
    ftp = FTP("ftp.ebi.ac.uk")
    ftp.login()
    ftp.cwd(sample_path)
    with open(filename, "wb") as fin:
        ftp.retrbinary("RETR " + filename, fin.write)
    ftp.close()

Total Ion Chromatograms (TIC) and Base Peak intensity (BPI) chromatograms can
be created using :meth:`tidyms.MSData.make_tic`, which returns a
:class:`tidyms.Chromatogram` object.

.. code-block:: python

    ms_data = ms.MSData(filename, ms_mode="centroid", instrument="qtof",
                        separation="uplc")
    tic = ms_data.make_tic(kind="tic")

Specifying the instrument type used and the separation technique on the
constructor, provides a reasonable set of default values for each method
according to the analytical platform being used.

Extracted ion chromatograms (EIC) are created with
:meth:`tidyms.MSData.make_chromatograms`, which accepts a list of m/z values and
return a list of :class:`tidyms.Chromatogram`:

.. code-block:: python

    mz_list = [205.09, 524.37, 188.07]
    chromatograms = ms_data.make_chromatograms(mz_list)

:meth:`tidyms.MSData.get_spectrum` takes a scan number and returns the spectrum
associated with the current scan in the data.

.. code-block:: python

    n_scan = 100
    sp = ms_data.get_spectrum(n_scan)

Usually, we want the average spectrum obtained from a series of scans. This can
be done with :meth:`tidyms.MSData.accumulate_spectra`.

.. code-block:: python

    start = 100
    end = 105
    sp = ms_data.accumulate_spectra(start, end)


Feature detection
-----------------

Feature detection is the first step performed in untargeted metabolomics to
build a :term:`data matrix` from raw data. In LC-MS based metabolomics, a
feature is usually defined as a chromatographic peak. Feature detection in LC-MS
is then the process of finding chromatographic peaks in a sample. In order to
perform feature detection, an implementation based on the centWave algorithm
[2]_ is used. This algorithm detects chromatographic peaks using **samples in
centroid mode** in two steps:

1.  Search Region Of Interest (ROI) in the whole experiment. A ROI is a
    time window in an experiment where a m/z trace is found. ROIs are built
    connecting m/z values across scans within a given m/z tolerance.
2.  Find chromatographic peaks on each ROI. A feature table is built
    using several descriptors associated with each peak: mean m/z, m/z
    standard deviation, mean rt, intensity, area and width.

Feature detection is available through the :func:`tidyms.detect_features`
function which returns a dictionary that maps sample names to a list of ROI
detected on each sample and a Pandas DataFrame where each row is a feature
and each column is a descriptor.

.. code-block:: python

    roi_dict, feature_data = ms.detect_features(filename)

:code:`filename` can be a string with the path to a mzML file, a list of
strings or a Path object. If a path to a directory is used, all mzML in the
directory are used.

:func:`tidyms.detect_features` can be customized and extended in several ways.
A detailed guide explaining how feature detection and customization are done can
be found :doc:`here<peak-picking>`


Feature correspondence
----------------------

**This algorithm wasn't thoroughly tested on untargeted data sets and should be
used with caution**

Before performing any kind of statistical comparison between samples, features
in the different samples must be matched. This process is known as feature
correspondence and can be quite complex due to differences in the m/z and
rt values obtained for the same species on different samples [3]_. We use a
cluster based approach to perform feature correspondence [1]_. An example
of feature correspondence can be found in the notebook :code:`Application 1`
that can be found in this
`Github repository <https://github.com/griquelme/tidyms-notebooks>`_.

Working with DataContainers
---------------------------

The functionality to work with metabolomics data in a data matrix form is
provided through the :py:class:`tidyms.DataContainer` object.
The DataContainer object organizes the data matrix, feature metadata and
sample metadata in three different DataFrames and manages several common tasks
such as computing metrics, normalization, plotting features and
:term:`data curation`.


Computing feature metrics
-------------------------

In order to evaluate the quality of the features, it's common to compute
metrics that show several characteristics of the features. This is done
through the metrics attribute, which has methods to compute metrics:

.. code-block:: python

    data = ms.fileio.load_dataset("reference-materials")
    # coefficient of variation
    cv = data.metrics.cv()

    # detection rate
    dr = data.metrics.detection_rate()

    # pca scores and loadings
    # pc_var is the variance of each PC
    # total_var is the total variance explained by the PCA model
    pca_scores, pca_loadings, pc_var, total_var= data.metrics.pca()


By default, the class of each sample is taken into account and metrics are
computed per class. Global metrics can be computed setting the parameter
:code:`intraclass` to False:

.. code-block:: python

    cv = data.metrics.cv(intraclass=False)
    dr = data.metrics.detection_rate(intraclass=False)

Plotting feature data
---------------------

Visualization of the data can be done in a similar way using the plot
attribute, which has methods to generate commonly used plots, for example a PCA
scores plot can be easily plotted:

.. code-block:: python

    data.plot.pca_scores()

.. bokeh-plot:: plots/pca-scores.py
    :source-position: none

The intensity of a feature, as function of the run order can also be plotted:

.. code-block:: python

    # search [M+H]+ from trp in the features
    mz = 205.097
    rt = 124
    # get a list of features compatible with the given m/z and rt
    ft_name = data.select_features(mz, rt)
    data.plot.feature(ft_name[0])

.. bokeh-plot:: plots/feature-plot.py
    :source-position: none

Data curation
-------------

In order to increase confidence in the results obtained during data analysis,
it's necessary to correct the bias in the data due to sample
preparation and also remove any features that cannot be measured in an
analytically robust way [4]_, [5]_. We call this process :term:`data curation`.
In TidyMS, the data curation is applied using :term:`filtration` and
:term:`correction` steps to obtain a robust data matrix. Before applying data
curation, it's recommended to define a :term:`mapping`. A mapping is a
dictionary that maps a sample type to a list of sample classes. You
can still use the Filters and Correctors without setting a mapping, but defining
one makes work easier, as it's used to set a default behaviour for the different
Filters and Correctors that are used for data curation. These default values are
set using the recommendations from Broadhurst *et al* [5]_. To define a mapping,
simply create a dictionary with sample types as keys and a list of sample
classes in your data and assign it to the mapping attribute of your data:

.. code-block:: python

    # available classes in the data set: "1", "2", "3", "4", "QC", "B", "Z"
    # 1, 2, 3, 4 are the study samples
    # Z and B are zero volume injection and process blank respectively
    # QC are pooled quality control samples.
    mapping = {"blank": ["Z", "B"],
               "sample": ["1", "2", "3", "4"],
               "qc": ["QC"]}
    data.mapping = mapping

Once the mapping is set, we can quickly perform data curation. In this example,
we apply a :term:`blank correction` and a :term:`prevalence filter` to our data.
All Filters and Correctors share the same process method, that accepts a
DataContainer and process it in place:

.. code-block:: python

    # subtract the blank contribution to samples. The blank contribution is
    # estimated using samples of type blank using the mapping.
    # using mode="mean" the mean of all blank samples is used to estimate
    # the blank contribution.
    bc = ms.filter.BlankCorrector(mode="mean")
    bc.process(data)
    # remove all features with a prevalence lower than 80 % in all classes.
    pf = ms.filter.PrevalenceFilter(lb=0.8)
    bc.process(data)

Refer to the :doc:`api` to see a list of available Filters and Correctors.
Often, we want to apply a series of filters and correctors to our data. This
can be done using the :class:`tidyms.filter.Pipeline` object,
which accepts a list of filters and correctors and applies them in order:

.. code-block:: python

    pipeline = ms.filter.Pipeline([bc, pf])
    pipeline.process(data)

The Pipeline object accepts Filters, Correctors and other Pipelines as elements
in the list. This allow us to create more sophisticated strategies for data
curation. For example, the implementation for :term:`batch correction` is
a Pipeline object that checks samples and features that cannot be corrected and
removes them before applying the correction.


Data Preprocessing
------------------

Before modelling the data, it's a common practice to normalize, scale or
transform the data matrix. All of these operations are available through the
:code:`preprocess` attribute of the DataContainer. The following code normalises
each sample to an unitary total intensity, and scales each feature to have a
zero mean and unitary variance:

.. code-block::

    data.preprocess.normalize("sum")
    data.preprocess.scale("autoscaling")

Finally, a DataContainer can be reset to the values that were used to create it
using the :py:meth:`tidyms.DataContainer.reset`

References
----------

..  [1] Riquelme, G. *et al*, "A Python-Based Pipeline for Preprocessing LC–MS
    Data for Untargeted Metabolomics Workflows". Metabolites 2020, 10, 416.
    https://doi.org/10.3390/metabo10100416
..  [2] Tautenhahn, R. *et al*, S. "Highly sensitive feature detection for high
    resolution LC/MS". BMC Bioinformatics 9, 504 (2008).
    https://doi.org/10.1186/1471-2105-9-504
..  [3] Smith, R., *et al*, "LC-MS alignment in theory and practice: a
    comprehensive algorithmic review", Briefings in Bioinformatics
    16, 1, (2015), Pages 104–117, https://doi.org/10.1093/bib/bbt080
..  [4] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography
    coupled to mass spectrometry", Nature Protocols volume 6, pages
    1060–1083 (2011).
..  [5] D Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies.", Metabolomics,
    2018;14(6):72. doi: 10.1007/s11306-018-1367-3