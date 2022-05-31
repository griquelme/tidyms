.. _processing-datasets:

.. py:currentmodule:: tidyms

:orphan:

Processing datasets
===================

We describe here how to perform feature detection on a set of raw data files.
We work using the :class:`tidyms.Assay` class, which starting from a set of raw
data files enables the creation of a data matrix and the exploration of the
features detected in the data.

For this example we will use a dummy dataset that can be downloaded with the
following code:

.. code-block:: python

    import tidyms as ms

    ms.fileio.download_dataset("test-nist-raw-data", download_dir=".")
    # sample metadata for the set
    ms.fileio.download_tidyms_data(
        "reference-materials",
        ["sample.csv"],
        download_dir="."
    )


To create an :class:`~tidyms.Assay`, we need to provide the following
information:

1.  The path to the raw data files. If all files are inside a directory, the
    path to the directory can be used and all mzML files in the directory will
    be used. All files must be in centroid mode.
2.  An assay directory to store the processed data. All processed data is stored
    to disk to reduce the memory usage in the different processing stages.
3.  The sample metadata. The metadata can be provided as a csv file or as a
    Pandas DataFrame. See the docstring for the format.
4.  The ``separation`` method used in the study and the ``instrument`` used,
    which defines the defaults used in the different preprocessing stages.


.. code-block:: python

    sample_metadata_path = "reference-materials/sample.csv"
    data_path = "test-nist-raw-data"
    assay_path = "test-assay"

    assay = ms.Assay(
        data_path=data_path,
        assay_path=assay_path,
        sample_metadata=sample_metadata_path,
        separation="uplc",
        instrument="qtof"
    )

:class:`~tidyms.Assay` process raw MS data applying a pipeline-like workflow.
After each preprocessing step the results are stored in the assay directory to
reduce memory usage and to keep track of intermediate steps. The following
preprocessing steps are applied in order:

.. csv-table:: Preprocessing steps
   :file: preprocessing-steps.csv
   :widths: 30, 30, 70
   :header-rows: 1

We now describe how to perform and customize each one of the steps in detail.

Feature detection
-----------------

Feature detection is done through the :meth:`tidyms.Assay.detect_features`
method. In LC data, a ROI is an extracted chromatogram. The function
:func:`tidyms.raw_data_utils.make_roi` is used to build ROIs for each sample.
This function is described in :ref:`this guide <roi-creation>`.

.. code-block:: python

    # parameters adjusted to reduce the number of ROI created
    # to perform untargeted feature detection, remove the `targeted_mz` param
    mz_list = np.array(
        [118.0654, 144.0810, 146.0605, 181.0720, 188.0706,
         189.0738, 195.0875, 205.0969]
    )
    make_roi_params = {
        "tolerance": 0.015,
        "min_intensity": 5000,
        "targeted_mz": mz_list
    }
    assay.detect_features(verbose=False, **make_roi_params)

The ROIs found in each sample are stored in the assay directory and can be
retrieved using :meth:`tidyms.Assay.load_roi` or
:meth:`tidyms.Assay.load_roi_list` with the corresponding sample name:

.. code-block:: python

    sample_name = "NZ_20200226_039"
    roi_list = assay.load_roi_list(sample_name)

The results can be explored by analyzing each one of the ROI or by using the
plot methods of the Assay, accessible through the `plot` attribute. The
``plot.roi`` method plots a 2D view of the m/z vs Rt of each one of the detected
ROIs.

.. code-block:: python

    assay.plot.roi(sample_name)


.. raw:: html

    <iframe src="_static/roi-no-peaks.html" height="650px" width="700px" style="border:none;"></iframe>

Feature extraction
------------------

After feature detection, features are extracted from each ROI using using the
:meth:`tidyms.Assay.extract_features` method. The features detected in each ROI
are stored as a list in the ``features`` attribute of the corresponding ROI:

.. code-block:: python

    assay.extract_features(store_smoothed=True)

Once again, the ROI are stored in the assay directory. By default, the function
used for feature extraction is the ``extract_features`` method of each ROI. In
LC data, the :meth:`tidyms.lcms.LCRoi.extract_features` method is used and is
described in :ref:`this guide <feature-extraction>`.

If the ``plot.roi`` method is used after feature extraction, the peaks detected
in each ROI are highlighted.

.. raw:: html

    <iframe src="_static/roi-peaks.html" height="650px" width="700px" style="border:none;"></iframe>

Feature description
-------------------

Feature description consists in computing descriptors for each feature detected
(for LC data, the descriptors are the peak area, Rt, m/z, peak width,
among others). The descriptors are computed using
:meth:`tidyms.Assay.describe_features`:

.. code-block:: python

    assay.describe_features()

After this step, the descriptors for the features found in a sample are stored
in a Pandas DataFrame and can be retrieved using
:meth:`tidyms.Assay.load_features` method. Besides the descriptors, this
DataFrame contains two additional columns: `roi_index` and `ft_index`.
`roi_index` is used to indentify the ROI where the feature was detected, and can
be used to load the ROI using the ``load_roi`` method:

.. code-block:: python

    feature_df = assay.load_features(sample_name)
    # get the roi index for the feature in the third row
    k_row = 3
    k_roi_index = feature_df.at[k_row, "roi_index"]
    k_roi = assay.load_roi(sample_name, k_roi_index)

The `ft_index` value is used to identify the features in the `feature` attribute
of the ROI. :meth:`tidyms.Assay.describe_features` can be customized in the
same way as :meth:`tidyms.lcms.Roi.describe_features` (see
:ref:`this guide <feature-extraction>`).

Feature table construction
--------------------------

The feature table, which contains the descriptors from features in all samples
in a single DataFrame, is built using the
:meth:`tidyms.Assay.build_feature_table` method. The results are stored in the
`feature_table` attribute of the Assay. Two additional columns are included:
`sample_` contains the sample name where the feature was detected and `class_`
contains the corresponding class name:

.. code-block:: python

    assay.build_feature_table()
    assay.feature_table

Feature correspondence
----------------------

During feature correspondence features in different samples are grouped based on
their identity. Ina untargeted metabolomics, the identity is guessed based on the
similarity of the descriptors of each feature, listed in the feature table.
In TidyMS, a cluster-based approach is used to group features together. A
description of the algorithm used can be found :ref:`here <ADD-LINK>`.

Feature matching is done using the :meth:`tidyms.Assay.match_features` method:

.. code-block:: python

    assay.match_features()
    assay.feature_table

After this step, a new column called `label_` is added to the feature table.
`label_` groups features based on their identity. Features labelled with ``-1``
do not belong to any group.

Groups of features can be visualized using the ``plot.stacked_chromatogram``
method:

.. code-block:: python

    label = 6
    assay.plot.stacked_chromatogram(label)


.. raw:: html

    <iframe src="_static/stacked-chromatograms.html" height="450px" width="700px" style="border:none;"></iframe>

Data matrix creation
--------------------

Using the `sample_` and `label_` columns, the data matrix is built by pivoting
the feature table, as described in this
`link <https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping>`_.
The data matrix is created using the :meth:`tidyms.Assay.make_data_matrix`
method, which creates and stores the data matrix information inside a
:py:class:`tidyms.DataContainer`.

.. code-block:: python

    assay.make_data_matrix()
    data = assay.data_matrix

The information inside the DataContainer is stored in three different
DataFrames:

*   The sample metadata DataFrame provided during Assay creation is stored
    in the `sample_metadata` attribute.
*   The `data_matrix` attribute stores the data matrix created by pivoting the
    area of each feature in the feature table into a DataFrame where each column
    is a sample and each feature group is a column.
*   The `feature_metadata` contains another DataFrame that stores aggregated
    values of the descriptors in the feature table for each feature group.

Further data preprocessing using the :class:`~tidyms.DataContainer` object is
described in :ref:`this guide <data-curation>`.