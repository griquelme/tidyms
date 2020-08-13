.. quickstart

.. py:currentmodule:: tidyms

Quickstart
==========

TidyMS is a Python package that provides tools to process and analyze
Mass Spectroscopy data from any data set, but it's functionality was mostly
thought to be used with data from metabolomics experiments. It uses the
`pyopenms <https://www.openms.de/>`_ package to read raw data in the mzML
format and `Numpy <https://numpy.org/>`_,
`Pandas <https://pandas.pydata.org/>`_ and
`scikit-learn <https://scikit-learn.org>`_ to process and analyze the data.
Some of the functionality that offers is:

*   creating chromatograms and accumulated spectra from raw data.
*   :term:`Feature detection<feature detection>` and
    :term:`feature correspondence` in metabolomics data sets.
*   Read processed data from other mass spectrometry processing software
    (XCMS, mzmine2, etc...).
*   A container object to manage metabolomics data.
*   :term:`Data curation<data curation>` of untargeted metabolomics data sets
    using widely accepted practices from the metabolomics community [4]_
*   Interactive data visualization using `bokeh <https://bokeh.org/>`_, or
    publication quality plots using `seaborn <https://seaborn.pydata.org/>`_.

In the rest of this guide, we show different use cases for the TidyMS package.
A basic knowledge of mass spectrometry and metabolomics is assumed, but you can
look at the :doc:`glossary` section to see the concepts used in this guide.
Installation instructions are available :doc:`here<installation>`.

Creating chromatograms
----------------------

A :class:`tidyms.Chromatogram` object can be created using a
retention time array and an intensity array.

.. code-block:: python

    import tidyms as ms
    import numpy as np

    # this creates two gaussian overlapping peaks simulating a chromatogram
    rt = np.arange(100)
    chrom_params = np.array([[50, 5, 10], [70, 5, 20]])
    spint = ms.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

    chromatogram = ms.Chromatogram(rt, spint)

Using the :class:`tidyms.Chromatogram` object, we can perform
common tasks like plotting or peak picking:

.. code-block:: python

    chromatogram.plot()

.. bokeh-plot:: plots/chromatogram.py
    :source-position: none

The :meth:`tidyms.Chromatogram.find_peaks` method detects peaks
and stores them as a list inside the peaks attribute. The method for peak
picking uses a CWT based algorithm with parameters optimized to be used on
chromatographic data. Using
:meth:`tidyms.Chromatogram.get_peak_params` we can obtain a
:py:class:`pandas.DataFrame` with peak information, or if we plot the
Chromatogram again, we can see the detected peaks.

.. code-block:: python

    chromatogram.find_peaks()
    chromatogram.plot()

.. bokeh-plot:: plots/chromatogram-with-peaks.py
    :source-position: none

Creating mass spectra
---------------------

In a similar way, we can create a :class:`tidyms.MSSpectrum`
object:

.. code-block:: python

    rt = np.linspace(400, 404, 1000)

    # creates three gaussian peaks simulating an isotopic envelope
    sp_params = np.array([[401, 0.01, 100], [402, 0.01, 15], [403, 0.01, 2]])
    spint = ms.utils.gaussian_mixture(rt, sp_params).sum(axis=0)

    spectrum = ms.MSSpectrum(rt, spint)
    spectrum.find_peaks()
    spectrum.plot()

.. bokeh-plot:: plots/ms-spectrum.py
    :source-position: none

Reading raw data
----------------

In the majority of cases, chromatograms and spectra are going to be created
from experimental data. Working with raw data can be done using the
:class:`tidyms.MSData` object, which reads data in the mzML
format. You can read :doc:`this tutorial <mzml>` on how to convert
experimental data from proprietary, instrument-specific formats to mzML. The
MSData object can create Total Ion Chromatograms (TIC), or Base Peak
intensity (BPI) chromatograms using the
:meth:`tidyms.MSData.make_tic` method, which returns a
:class:`tidyms.Chromatogram` object.

.. code-block:: python

    data_path = "exp_data.mzML"
    ms_data = ms.MSData(data_path,
                         ms_mode="centroid",
                         instrument="qtof",
                         separation="uplc")
    tic = ms_data.make_tic(mode="tic")

It's recommended to provide the instrument type used and the separation
technique on the constructor, as this provides a reasonable set of default
values for each method according to the analytical platform being used.
To create extracted ion chromatograms the
:meth:`tidyms.MSData.make_chromatograms` method accepts a list
or array of m/z values and return a list of
:class:`tidyms.Chromatogram`:

.. code-block:: python

    # mass of [M+H]+ of tryptophan and alanine
    mz_list = [205.0977, 89.0477]
    chromatograms = ms_data.make_chromatograms(mz_list)

To create a mass spectrum, the :meth:`tidyms.MSData.get_spectrum`
takes a scan number and returns a :class:`tidyms.MSSpectrum`.

.. code-block:: python

    n_scan = 100
    sp = ms_data.get_spectrum(n_scan)

If you want to create an average spectrum from a set of consecutive scans, the
:meth:`tidyms.MSData.accumulate_spectra` does the task.

.. code-block:: python

    start = 100
    end = 105
    sp = ms_data.accumulate_spectra(start, end)


Feature detection
-----------------

Feature detection is the first step performed in untargeted metabolomics to
convert raw data into a :term:`data matrix`. In LC-MS based metabolomics, a
feature is usually defined as a chromatographic peak. Feature detection in LC-MS
is then then process of finding chromatographic peaks in a sample. In order to
perform feature detection, an implementation of the centWave algorithm [1]_
is used. This algorithm detects chromatographic peaks using **samples in
centroid mode** in two steps:

1.  Region Of Interest (ROI) are searched in the whole experiment. A ROI is a
    time window in an experiment where a m/z trace is found. ROIs are built
    connecting m/z values across scans within a given tolerance.
2.  Chromatographic peaks are detected on each ROI. A feature table is built
    with mean m/z, m/z standard deviation, mean rt, peak intensity, peak area
    and peak width.

feature detection is available through the
:meth:`tidyms.MSData.detect_features` method which returns a
ROI list and the feature table.

.. code-block:: python

    roi, feature_data = ms_data.detect_features()

Default values for the method are set using the instrument and separation
attributes. Several parameters can be set, from peak filter criteria, to
estimators for each peak parameter. For a detailed explanation on each
parameter see the :doc:`api`.

To perform feature detection on several samples, you can use the
:py:func:`tidyms.detect_features` function.

.. code-block:: python

    import os
    # creates a list of path to each data file to analyze
    path = "data"
    file_list = [os.path.join(path, x) for x in os.listdir(path)]
    roi, feature_data = ms.detect_features(data_path, separation="uplc",
                                            instrument="qtof")

This creates a DataFrame where each row is a feature detected in a specific
sample.

Feature correspondence
----------------------

**This algorithm wasn't thoroughly tested on untargeted data sets and should be
used with caution**

Before performing any kind of statistical comparison between samples, features
in the different samples must be matched. This process is known as feature
correspondence and can be quite complex due to the difference in the m/z and
rt values obtained for the same species on different samples [2]_. We use a
cluster based approach to perform feature correspondence (you can read more
about the correspondence algorithm here).

.. code-block:: python

    mz_tolerance = 0.005
    rt_tolerance = 5
    cluster = ms.feature_correspondence(feature_data, mz_tolerance,
                                         rt_tolerance)

After performing feature correspondence, each feature is assigned to a cluster.
This Data can be converted to a data matrix, where each row is a sample and
each column is a feature. Working with data matrices is done with the
:class:`tidyms.DataContainer` object. To create a DataContainer
with the matched features, the function
:py:func:`tidyms.make_data_container` is used:

.. code-block:: python

    sample_metadata = pd.read_csv("sample_metadata.csv")
    data = ms.make_data_container(feature_data, cluster,
                                   sample_metadata)

To create a DataContainer object, in addition to the detected features and
the cluster information, metadata for each sample must be provided. The sample
metadata is a DataFrame where the index is the name of each sample and at the
very least a column named class with the class name of each sample must be
included. The DataContainer contains the data matrix, along with feature
metadata (e.g. feature m/z and  rt) computed using
:term:`feature descriptors<feature descriptor>` from each sample. In the function
documentation there is information on how each value is estimated. For users
wanting to create a custom data matrix or feature metadata, we recommend looking
at the following :py:class:`pandas.DataFrame` methods:
:py:meth:`pandas.DataFrame.groupby`, :py:meth:`pandas.DataFrame.pivot` and
:py:meth:`pandas.DataFrame.aggregate`.

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
    pca_scores, pca_loadings = data.metrics.pca()


By default, the class of each sample is taken into account and metrics are
computed per class. Global metrics can be computed setting the parameter
:code:`intraclass` to False:

.. code-block:: python

    cv = data.metrics.cv(intraclass=False)
    dr = data.metrics.detection_rate(intraclass=False)

Plotting feature data
---------------------

Visualization of the data can be done in a similar way using the plot
attribute, which has methods to generate common used plots, for example a PCA
scores plot can be easily plot:

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

In order to increase the confidence in the results obtained during analysis of
the data, it's necessary to correct the bias in the data due to sample
preparation and also remove any features that cannot be measured in an
analytically robust way [3]_, [4]_. We call this process :term:`data curation`.
In tidyms, the data curation is applied using :term:`filtration` and
:term:`correction` steps to obtain a robust data matrix. Before applying data
curation, it's recommended to define a :term:`mapping`. A mapping is a
dictionary that map a :term:`sample type` to a list of :term:`sample class`. You
can still use the Filters and Correctors without setting a mapping, but defining
one makes work easier, as it's used to set a default behaviour for the different
Filters and Correctors that are used for data curation. These default values are
set using the recommendations from Broadhurst *et al* [4]_. To define a mapping,
simply create a dictionary with sample types as keys and a list of sample
classes in your data and assign it to the mapping attribute of your data:

.. code-block:: python

    # available classes in the data set: "1", "2", "3", "4", "QC", "B", "Z"
    mapping = {"blank": ["Z", "B"],
               "sample": ["1", "2", "3", "4"],
               "qc": ["QC"]}
    data.mapping = mapping

Once the mapping is set, we can quickly perform data curation. In this example,
we are going to apply a :term:`blank correction` and a
:term:`prevalence filter` to our data. All Filters and Correctors share the
same process method, that accepts a DataContainer and process it in place:

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
See here to create custom Filters.

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

..  [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
    feature detection for high resolution LC/MS. BMC Bioinformatics 9,
    504 (2008). https://doi.org/10.1186/1471-2105-9-504
..  [2] Smith, R., Ventura, D., Prince, J.T., LC-MS alignment in theory and
    practice: a comprehensive algorithmic review, Briefings in Bioinformatics
    16, Issue 1, January 2015, Pages 104–117, https://doi.org/10.1093/bib/bbt080
..  [3] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography
    coupled to mass spectrometry", Nature Protocols volume 6, pages
    1060–1083 (2011).
..  [4] D Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies.", Metabolomics,
    2018;14(6):72. doi: 10.1007/s11306-018-1367-3