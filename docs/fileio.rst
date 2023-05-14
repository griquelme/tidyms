.. _working-with-raw-data:

.. py:currentmodule:: tidyms

:orphan:

Working with raw data
=====================

TidyMS works with raw data in the mzML format using the :class:`~tidyms.MSData`
class. In this section we show commons operations on raw data. For file
conversion to the mzML format see :ref:`this guide <mzml>`

For the examples we will use an example mzML file that can be downloaded with
the following code:

.. code-block:: python

    import numpy as np
    import tidyms as ms

    filename = "NZ_20200227_039.mzML"
    dataset = "test-nist-raw-data"
    ms.fileio.download_tidyms_data(dataset, [filename], download_dir=".")


Raw data
--------

Raw MS data in the mzML format can be read through the :class:`~tidyms.MSData`
object.

.. code-block:: python

    ms_data = ms.MSData.create_MSData_instance(
        filename,
        ms_mode="centroid",
        instrument="qtof",
        separation="uplc"
    )

It is necessary to specify if the data is in centroid or profile mode using the
:code:`ms_mode` parameter, as some methods work in different ways for each
type of data. Specifying the :code:`instrument` and :code:`separation` is also
recommended, as these parameters set reasonable defaults in different functions
used.

:class:`~tidyms.MSData` is optimized for low memory usage and only loads the
required data into memory. A single MS spectrum can be loaded using
:meth:`~tidyms.MSData.get_spectrum` which returns a
:class:`~tidyms.lcms.MSSpectrum`.

.. code-block:: python

    index = 20
    sp = ms_data.get_spectrum(index)

The index used is the order in which the data was stored in the file. In the
same way, a stored chromatogram can be retrieved using
:meth:`~tidyms.MSData.get_chromatogram`. The total count of spectra and
chromatograms in the file can be obtained using
:meth:`tidyms.MSData.get_n_spectra` and
:meth:`tidyms.MSData.get_n_chromatograms` respectively. Iterating over all
the spectra in a file can be done using
:meth:`~tidyms.MSData.get_spectra_iterator`, which generates each one of the
spectra in the file and allows filtering by acquisition time or MS level.
Common operations with raw data are located in :mod:`tidyms.raw_data_utils`.


Working with Mass Spectra
-------------------------

:class:`~tidyms.MSSpectrum` stores the information from one scan. It is mostly
used as a data storage class in several data processing steps, but it also has
functionality to visualize the spectrum using the
:meth:`~tidyms.MSSpectrum.plot` method and to convert a profile data spectrum
into centroid mode using :meth:`tidyms.MSSpectrum.find_centroids`.

:func:`tidyms.raw_data_utils.accumulate_spectra` combines a series of scans in
a file into a single spectrum:

.. code-block:: python

    combined_sp = ms.accumulate_spectra(ms_data, start_time=110, end_time=115)

Chromatograms
-------------

Besides the chromatograms stored in a file, extracted chromatograms can be
created :func:`tidyms.raw_data_utils.make_chromatograms` which takes an array of
m/z and returns a list :class:`tidyms.Chromatogram` objects, each one associated
to one of the m/z values provided:

.. code-block:: python

    mz_list = np.array([189.0734, 205.0967, 188.071])
    chromatograms = ms.make_chromatograms(ms_data, mz_list)

A chromatogram can be visualized using ``plot`` method:

.. code-block:: python

    chrom = chromatograms[0]
    chrom.plot()

.. raw:: html

    <iframe src="_static/chromatogram.html" height="450px" width="700px" style="border:none;"></iframe>

Peaks in a chromatogram are detected using
:meth:`tidyms.lcms.LCRoi.extract_features`, which stores a list of
:class:`tidyms.lcms.Peak` objects in the `features` attribute of the
chromatogram. Plotting again the chromatogram shows the detected peaks:

.. code-block:: python

    chrom.extract_features()
    chrom.plot()

.. raw:: html

    <iframe src="_static/chromatogram-with-peaks.html" height="450px" width="700px" style="border:none;"></iframe>

Peak descriptors can be obtained using
:meth:`tidyms.lcms.Roi.describe_features`:

.. code-block:: python

    >>> chrom.describe_features()
    [{'height': 16572.38, 'area': 108529.94, 'rt': 125.73, 'width': 14.06,
      'snr': 385.44, 'mz': None, 'mz_std': None}]

A detailed description of the algorithm used for peak picking can be found
:ref:`here <peak-picking>`. These methods are also used to create a data matrix from
a dataset. See :ref:`here <processing-datasets>` a tutorial on how to work with
complete datasets to extract a data matrix.
