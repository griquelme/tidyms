.. _data-curation:

.. py:currentmodule:: tidyms

:orphan:

Working with DataContainers
---------------------------

The functionality to work with metabolomics data in a data matrix form is
provided through the :py:class:`tidyms.DataContainer` object.
The DataContainer object organizes the data matrix, feature metadata and
sample metadata into three different DataFrames and manages several common tasks
such as computing metrics, normalization, plotting features and
:term:`data curation`.

For these examples we will use a data matrix obtained from the data described in
[1]_. Briefly, it consists in several reference materials candidate for human
plasma, labelled as 1--4. This data set can be loaded into a DataContainer
with the following code:

.. code-block:: python

    data = ms.fileio.load_dataset("reference-materials")

The DataContainer stores the information of a data matrix in three different
DataFrames:

*   The sample metadata is stored in the `sample_metadata` attribute, and
    contains information related to each sample such as class, run order or
    analytical batch.
*   The `data_matrix` attribute stores the data matrix where each row is a
    sample and each feature is a column.
*   The `feature_metadata` attribute contains descriptors for each feature,
    such as m/z or retention time.

TidyMS contains functions to read processed data for several formats:

*   :func:`tidyms.fileio.read_progenesis` Reads csv files generated with
    ProgenesisQI into a DataContainer.
*   :func:`tidyms.fileio.read_mzmine` Reads csv files generated with MZMine2.
*   :func:`tidyms.fileio.read_xcms` Reads csv files generated with XCMS.

See this notebook for examples on how to use these functions.

In order to take advantage of the DataContainer functionality, it is highly
recommended to assign each sample class in the dataset to a sample type.
This allows to define a default behavior for several functions and is required
by several processing functions. The sample type is assigned through the
`mapping` attribute, which maps a sample type into a list of sample classes.
Valid keys for the mapping are ``"sample"`` for study samples, ``"qc"`` for
Quality control samples and ``"blank"`` for blank samples. Classes that
are not included in the mapping are ignored during data preprocessing:

.. code-block:: python

    # classes in the data set: "1", "2", "3", "4", "QC", "B", "Z"
    # "1", "2", "3" and "4" are study samples
    # "Z" and "B" are zero volume injection and process blank respectively
    # "QC" are pooled quality control samples.
    mapping = {
        "blank": ["Z", "B"],
        "sample": ["1", "2", "3", "4"],
        "qc": ["QC"]
    }
    data.mapping = mapping

Computing feature metrics
-------------------------

In order to evaluate the quality of the features, it's common to compute
metrics that show several characteristics of the features. This is done
through the `metrics` attribute, which has methods to compute several metrics:

*   :meth:`~tidyms.container.MetricMethods.correlation` computes the correlation
    coefficient of features with fields of the sample metadata.
*   :meth:`~tidyms.container.MetricMethods.cv` computes the coefficient of
    variation of each feature.
*   :meth:`~tidyms.container.MetricMethods.detection_rate` computes the
    coefficient of variation of each feature.
*   :meth:`~tidyms.container.MetricMethods.dratio` computes the D-ratio [3]_ of
    each feature.
*   :meth:`~tidyms.container.MetricMethods.pca` builds a PCA model from the
    data matrix.

.. code-block:: python

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
:code:`groupby` to ``None``:

.. code-block:: python

    cv = data.metrics.cv(groupby=None)
    dr = data.metrics.detection_rate(groupby=None)

Plotting feature data
---------------------

Visualization of the data can be done in a similar way using the plot attribute,
which has methods to generate commonly used plots. For example, a PCA score plot
can be easily plotted:

.. code-block:: python

    # ignore blank samples. SSS are system suitability samples
    # SCQC are system conditioning samples
    ignore = ["Z", "SV", "B", "SSS", "SCQC"]
    data.plot.pca_scores(scaling="autoscaling", ignore_classes=ignore)


.. raw:: html

    <iframe src="_static/pca-scores.html" height="650px" width="700px" style="border:none;"></iframe>

The intensity of a feature, as function of the run order can also be plotted:

.. code-block:: python

    # search [M+H]+ from trp in the features
    mz = 205.097
    rt = 124
    # get a list of features compatible with the given m/z and rt
    ft_name = data.select_features(mz, rt)
    data.plot.feature(ft_name[0])


.. raw:: html

    <iframe src="_static/feature-plot.html" height="450px" width="700px" style="border:none;"></iframe>

The :meth:`tidyms.DataContainer.set_plot_mode` method allows to switch between
visualizations using bokeh or seaborn.

Data curation
-------------

In order to increase confidence in the results obtained during data analysis,
it's necessary to correct the bias in the data due to sample
preparation and also remove any features that cannot be measured in an
analytically robust way [2]_, [3]_. We call this process :term:`data curation`.
In TidyMS, the data curation is applied using :term:`filtration` and
:term:`correction` steps to obtain a robust data matrix. Before applying data
curation, it's recommended to define a :term:`mapping` as described before. Once
the mapping is set, we can quickly perform data curation. In this example,
we apply a :term:`blank correction` and a :term:`prevalence filter` to our data.
All Filters and Correctors share the same process method, that accepts a
DataContainer and process it in place:

.. code-block:: python

    # subtract the blank contribution to samples. The blank contribution is
    # estimated using samples of type blank using the mapping.
    # using mode="mean" estimates the blank contribution as the mean in all
    # blank samples
    blank_corrector = ms.filter.BlankCorrector(mode="mean")
    blank_corrector.process(data)
    # remove all features with a prevalence lower than 80 % in all classes.
    prevalence_filter = ms.filter.PrevalenceFilter(lb=0.8)
    prevalence_filter.process(data)

Refer to the :doc:`api` to see a list of available Filters and Correctors.
Often, we want to apply a series of filters and correctors to our data. This
can be done using the :class:`tidyms.filter.Pipeline` object,
which accepts a list of filters and correctors and applies them in order:

.. code-block:: python

    pipeline = ms.filter.Pipeline([blank_corrector, prevalence_filter])
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

Storing processed data and exportation
--------------------------------------

DataContainers can be stored in a pickle object using the ``save`` method:

.. code-block:: python

    data.save("processed-data.pickle)


This allows to quickly store and load processed data. The pickle object can
be loaded using the :meth:`tidyms.fileio.read_pickle` method:

.. code-block:: python

    data = ms.fileio.read_pickle("processed-data.pickle")

The data can also be stored into several plain text formats using Pandas
functionality, for example :py:meth:`pandas.DataFrame.to_csv`:

.. code-block:: python

    data.data_matrix.to_csv("data-matrix.csv")

References
----------

..  [1] Riquelme, G. *et al*, "A Python-Based Pipeline for Preprocessing LC–MS
    Data for Untargeted Metabolomics Workflows". Metabolites 2020, 10, 416.
    https://doi.org/10.3390/metabo10100416
..  [2] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography
    coupled to mass spectrometry", Nature Protocols volume 6, pages
    1060–1083 (2011).
..  [3] D Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies.", Metabolomics,
    2018;14(6):72. doi: 10.1007/s11306-018-1367-3