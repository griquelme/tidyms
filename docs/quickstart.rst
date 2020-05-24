.. quickstart

.. py:currentmodule:: ms_feature_validation

Quickstart
==========

The mskm is a Python package to process and analyze Mass Spectroscopy data from
any data set, but it's functionality was mostly thought to be used in data
from metabolomics experiments. It uses the pyopenms package to read raw data in
the mzML format and Pandas to work with Numpy and Pandas. Some of the
functionality that it offers is:

*   creating chromatograms and accumulated spectrum from raw data.
*   Feature detection and feature correspondence in whole data sets.
*   Read processed data from other mass spectroscopy processing software (XCMS,
    mzmine2, etc...)
*   A container object for metabolomics data
*   Correction and filtration untargeted metabolomics data sets using widely
    accepted practices from the metabolomics community (agregar ref)
*   Interactive plots using the bokeh package, or publication quality plots
    using seaborn.

Take a look at the :doc:`glossary` section to see the concepts used in the different
parts of the documentation.

Creating chromatograms
----------------------

A :class:`ms_feature_validation.Chromatogram` can be created using a retention
time array and an intensity array.

.. code-block:: python

    import ms_feature_validation as mfv
    import numpy as np
    rt = np.arange(100)

    # creates two gaussian overlapping peaks simulating a chromatogram
    chrom_params = np.array([[50, 5, 10], [70, 5, 20]])
    spint = mfv.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

    chromatogram = mfv.Chromatogram(rt, spint)

Using the :class:`ms_feature_validation.Chromatogram` object, we can perform
common tasks like plotting or peak picking:

.. code-block:: python

    chromatogram.plot()

.. bokeh-plot:: plots/chromatogram.py
    :source-position: none

The :meth:`ms_feature_validation.Chromatogram.find_peaks` detects peaks and
stores them as a list inside the peaks attribute. The method for peak picking
uses a CWT based algorithm with parameters optimized to be used on
chromatographic data. Using
:meth:`ms_feature_validation.Chromatogram.get_peak_params` we can obtain a
pandas DataFrame with peak information, or if we plot the Chromatogram again,
we can see the detected peaks.

.. code-block:: python

    chromatogram.find_peaks()
    chromatogram.plot()

.. bokeh-plot:: plots/chromatogram-with-peaks.py
    :source-position: none

Creating mass spectrum
----------------------

In the same way that a Chromatogram, we can create
:class:`ms_feature_validation.MSSpectrum objects`:

.. code-block:: python

    rt = np.linspace(400, 404, 1000)

    # creates three gaussian peaks simulating an isotopic envelope
    sp_params = np.array([[401, 0.01, 100], [402, 0.01, 15], [403, 0.01, 2]])
    spint = mfv.utils.gaussian_mixture(rt, sp_params).sum(axis=0)

    spectrum = mfv.MSSpectrum(rt, spint)
    spectrum.find_peaks()
    spectrum.plot()

.. bokeh-plot:: plots/ms-spectrum.py
    :source-position: none

Reading raw data
----------------

In the majority of cases, chromatograms and spectrum are going to be created
from experimental data. Working with raw data can be done using the
:class:`ms_feature_validation.MSData` object, which reads data in the mzML
format. You can read this tutorial on how to convert experimental data from
propietary, instrument-specific formats to mzML. The MSData object can create
Total Ion Chromatograms (TIC), or Base Peak intensity (BPI) chromatograms using
the :meth:`ms_feature_validation.MSData.make_tic` method, which returns a
:class:`ms_feature_validation.Chromatogram` object.

.. code-block:: python

    data_path = "exp_data.mzML"
    ms_data = mfv.MSData(data_path,
                         ms_mode="centroid",
                         instrument="qtof",
                         separation="uplc")
    tic = ms_data.make_tic(mode="tic")

It's recommended to provide the instrument used and the separation technique on
the constructor, as this provides a reasonable set of default values for each
method, according to the analytical platform being used.
To create extracted ion chromatograms the
:meth:`ms_feature_validation.MSData.make_chromatograms` method accepts a list
or array of m/z values and return a list of
:class:`ms_feature_validation.Chromatogram`:

.. code-block:: python

    # mass of [M+H]+ of tryptophan and alanine
    mz_list = [205.0977, 89.0477]
    chromatograms = ms_data.make_chromatograms(mz_list)

To create a mass spectrum, the :meth:`ms_feature_validation.MSData.get_spectrum`
takes a scan number and returns a :class:`ms_feature_validation.MSSpectrum`.

.. code-block:: python

    n_scan = 100
    sp = ms_data.get_spectrum(n_scan)

If you want to create an average spectrum from a set of consecutive scans, the
:meth:`ms_feature_validation.MSData.accumulate_spectra` does the task.

.. code-block:: python

    start = 100
    end = 105
    sp = ms_data.accumulate_spectra(start, end)


Feature detection
-----------------
