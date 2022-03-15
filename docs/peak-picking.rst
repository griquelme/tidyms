.. quickstart

.. py:currentmodule:: tidyms

Feature detection
=================

This guide describes the process of feature detection, which is the first step
done in the analysis of any metabolomics dataset. We start by giving a general
idea of feature detection and then we give an in depth description of each
step.

Feature detection can be defined as the process of detecting interesting
characteristics in data to solve a specific problem. In LC-MS datasets, features
are usually defined as chromatographic peaks, and can be described by retention
time, m/z and peak area, among other descriptors. In TidyMS, feature detection is
done by using an approach similar to the one described by Tautenhahn *et al* in
[1]_, but with some modifications. Loosely, the algorithm can be described
in two steps:

1.  Build regions of interest (ROI) from raw MS data.
2.  Detect chromatographic peaks in each ROI.

For the examples described in this tutorial we are going to use an example file
that can be downloaded from Metabolights using the following code:

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

The complete process of feature detection can be easily made using
:py:func:`tidyms.detect_features`, which performs feature detection on several
samples and returns a dictionary of sample names to a list of ROI created
from each sample and a Pandas DataFrame with the descriptors (see the table
below for a list of the descriptors computed by default) associated to all
the features detected in each ROI:

.. code-block:: python

    import tidyms as ms

    roi_dict, ft_table = ms.detect_features(filename, instrument="qtof",
        separation="uplc")

The :code:`instrument` and :code:`separation` parameters are used to provide
reasonable defaults for the different parameters used according to the MS
instrument and analytical separation method used.

.. csv-table:: Feature descriptors computed by default
   :file: descriptors.csv
   :widths: 30, 70
   :header-rows: 1

:py:func:`tidyms.detect_features` is a rather complex function accepting
several different parameters. In order to explain each one of them, in this
tutorial we are going to break down this function into parts to understand each
parameter in detail.


ROI creation
------------

ROI are similar to chromatograms but with two differences: information related
to the m/z value used in each scan is included and the traces are defined only
where m/z values were detected.

..  plot:: plots/roi-definition.py
    :caption:   A ROI is defined by three arrays storing information related to
                m/z, retention time and intensity.

In the first step of feature detection, ROIs are built from raw MS data in
centroid mode using :py:meth:`tidyms.fileio.MSData.make_roi`, which implements
the strategy described in [1]_. ROIs are created and extended connecting close
m/z values across successive scans using the following method:

1.  m/z values in the first scan are used to initialize a list of ROI.
2.  m/z values from the next scan are used to extend ROIs in the ROI list if
    they are closer than :code:`tolerance` to the mean m/z of a ROI. m/z values
    that don't match any ROI are used to create new ROIs and are appended to the
    ROI list.
3.  If more than one m/z value is within the tolerance threshold, m/z and
    intensity values are computed according to the :code:`multiple_match`
    strategy. Two strategies are available: merge multiple peaks into an
    average peak or use only the closest peak to extend the ROI and create
    new ROIs with the others.
4.  If a ROI can't be extended with any m/z value from the new scan, it is
    extended using NaNs.
5.  If there are more than :code:`max_missing` consecutive NaN in a ROI, then
    the ROI is flagged as completed. If the maximum intensity of a completed ROI
    is greater than :code:`min_intensity` and the number of points is greater
    than :code:`min_length`, then the ROI is flagged as valid. Otherwise, the ROI is
    discarded.
6.  Repeat from step 2 until no more new scans are available.

The following example shows how ROI can be created from raw data:

.. code-block:: python

    ms_data = ms.MSData(filename, ms_mode="centroid", instrument="qtof",
                        separation="uplc")
    roi_list = ms_data.make_roi()

The parameters associated with ROI creation can be tuned in the feature
detection function through the :code:`roi_params` parameter, that passes a
dictionary of parameters used in :py:meth:`tidyms.fileio.MSData.make_roi`.

After finding the ROIs in a sample, the next step is detecting the
chromatographic peaks on each ROI.


Peak detection
--------------

In the first release of TidyMS, peak picking worked using a modified version of
the CWT algorithm, described in [2]_. In chromatographic data, and in particular
in untargeted datasets, optimizing the parameters to cover the majority of peaks
present in the data can be a tricky process. Some of the problems that may
appear while using the CWT algorithm are:

1.  sometimes when a lot of peak overlap occurs, peaks are missing. This is
    because peaks are identified as local maximum in the ridge lines from the
    wavelet transform. If the widths selected don't have the adequate
    resolution, this local maximum may not be found. Also, it is possible to
    have more than one local maximum in a given ridgeline, which causes to
    select one of them using ad hoc rules.
2.  The Ricker wavelet is the most used wavelet to detect peaks, as it has been
    demonstrated to work very with gaussian peaks. In LC-MS data, is common to
    find peaks with a certain degree of asymmetry (eg. peak tailing). Using the
    Ricker wavelet in these cases, results in a wrong estimation of the peak
    extension, which in turn results in bad estimates for the peak area.
3.  The interaction between the parameters in the CWT algorithm is rather
    complex, and sometimes it is not very clear how they affect the peak picking
    process. The user must have a clear knowledge of the wavelet transform to
    interpret parameters such as the SNR. Also there are a lot of specific
    parameters to tune the detection of the ridgelines.

These reasons motivated us to replace the CWT peak picking function. The
new peak picking function uses the thoroughly tested function
:py:func:`scipy.signal.find_peaks`. We focused on keeping the function simple
and easy to extend. In this section we describe how peak picking works with
arbitrary signals, examples of usage and how peak detection is used inside
:py:func:`tidyms.detect_features`.

Peak detection usually involves detecting the peak apex, but in order to compute
peak descriptors such as area or width, the peak start and end must also be
found. The region defined between the peak start and end is the peak extension.
We decoupled the tasks of detecting peaks and computing peak descriptors.
:py:func:`tidyms.peaks.detect_peaks` builds a list of
:py:class:`tidyms.peaks.Peak` objects that store the peak start, apex and end.
This is done in five steps:

1.  The noise level in the signal is estimated.
2.  Using the noise level estimation, each point in the signal is classified as
    either baseline or signal. Baseline points are interpolated to build a
    baseline.
3.  Peaks apex are detected using :py:func:`scipy.signal.find_peaks`. Peaks
    with a prominence lower than three times the noise level or in regions
    classified as baseline are removed.
4.  For each peak its extension is determined by finding the closest baseline
    point to its left and right.
5.  If there are overlapping peaks (i.e. overlapping peak extensions),
    the extension is fixed by defining a boundary between the peaks as
    the minimum value between the apex of the two peaks.

..  plot:: plots/peak-definition.py
    :caption: Peak start, apex and end.


Steps 1 and 2 are described in detail at the end of this guide. We start first
by showing how this function is used. Besides the signal, noise estimation and
baseline estimation, `find_peaks_params` pass parameters to the underlying
peak picking function :py:func:`scipy.signal.find_peaks`. In general, it is not
necessary to change this parameter, since peak filtering is managed at a later
stage.

.. plot:: plots/peak-detection-example.py
    :include-source: true
    :caption: Peak detection example with two gaussian peaks

After obtaining a list of peaks, the next step is computing descriptors for each
peak. This is done with the function
:py:func:`tidyms.peaks.get_peak_descriptors`. This function uses the list of
peaks and the values of x, y, noise and baseline and computes peak location,
area, height, width and SNR.

.. code-block:: python

    >>> peaks
    [Peak(start=15, loc=25, end=37), Peak(start=43, loc=50, end=61)]
    >>> peaks, descriptors = get_peak_descriptors(x, y, noise, baseline, peaks)
    >>> descriptors
    [{'height': 29.50, 'area': 222.03, 'loc': 25.08, 'width': 12, 'snr': 48.98},
    {'height': 60.29, 'area': 297.97, 'loc': 50.12, 'width': 8, 'snr': 100.09}]

:code:`descriptors` can be used to compute arbitrary peak descriptors:

.. code-block:: python

    # custom descriptors must have the following prototype, even if some
    # parameters aren't used.
    # peak is a single Peak
    >>> def symmetry(x, y, noise, baseline, peak):
    ...     # we are defining the symmetry as the quotient between the left
    ...     # and right peak prominences
    ...     left_prominence = x[peak.apex] - x[peak.start]
    ...     right_prominence = x[peak.apex] - x[peak.end - 1]
    ...     return left_prominence / right_prominence
    >>> custom_descriptors = {"symmetry": symmetry}
    >>> peaks, descriptors = get_peak_descriptors(
    ...        x, y, noise, baseline, peaks, descriptors=custom_descriptors)
    >>> descriptors
    [{'height': 29.50, 'area': 222.03, 'loc': 25.08, 'width': 12,
      'snr': 48.98, 'symmetry': 1.00}, {'height': 60.29, 'area': 297.97,
      'loc': 50.12, 'width': 8, 'snr': 100.09, 'symmetry': 1.01}]

Finally, :code:`filters` can be used to filter peaks according to a specific
range for each descriptor. This parameter takes a dictionary of descriptor
names to a tuple of minimum and maximum values. If a descriptor has values
outside this range, the peak is removed. For example, we can filter peaks with a
signal-to-noise ratio lower than 50 in the following way:

.. code-block:: python

    >>> filters = {"snr": (50, None)}
    >>> peaks, descriptors = get_peak_descriptors(
    ...        x, y, noise, baseline, peaks, filters=filters)
    >>> peaks
    [Peak(start=43, loc=50, end=61)]


The task of peak detection and computing peak descriptors is combined in
:py:meth:`tidyms.lcms.Chromatogram.find_peaks` to analyze chromatographic data.
Going back to the example data used in the previous section, the following code
can be used to detect peaks:

.. code-block:: python

    >>> roi = roi_list[70]      # select a ROI from the list
    >>> roi.fill_nan()      # fill missing values in the ROI
    >>> peak_descriptors = roi.find_peaks()
    >>> peak_descriptors
    [{'height': 79848.54, 'area': 310080.0, 'loc': 336.12, 'width': 11.32,
      'snr': 851.49},
      {'height': 1723.73, 'area': 8130.52, 'loc': 355.17, 'width': 9.26,
       'snr': 18.38}]

The complete workflow can be described as follows:

1.  Estimate the noise level in the chromatogram using
    :py:func:`tidyms.peaks.estimate_noise`. The parameter :code:`noise_params`
    pass parameters to this function.
2.  Optionally, smooth the chromatogram using a gaussian filter.
    :code:`smoothing_strength` specifies the standard deviation of the gaussian
    kernel used to perform the smoothing.
3.  Estimate the baseline using :py:func:`tidyms.peaks.estimate_baseline`.
    :code:`baseline_params` pass parameters to this function.
4.  Detect peaks in the chromatogram using :py:func:`tidyms.peaks.detect_peaks`.
5.  Compute peak descriptors and filter peaks using
    :py:func:`tidyms.peaks.get_peak_estimators`. :code:`estimators` and
    :code:`filters` are used in the same way as described above.

These parameters are used also by :py:func:`tidyms.detect_features` to customize
the peak picking process.


Noise estimation
----------------

To estimate the noise and baseline, the discrete signal :math:`x[n]` is modelled
as three additive components:

.. math::
    x[n] = s[n] + b[n] + e[n]

:math:`s` is the peak component, which is deterministic, non negative and small
except regions where peaks are present. The baseline :math:`b` is a
smooth slow changing function. The noise term :math:`e[n]` is assumed to be
independent and identically distributed (iid) samples from a gaussian
distribution :math:`e[n] \sim N(0, \sigma)`.

If we consider the second finite difference of :math:`x[n]`, :math:`y[n]`:

.. math::
    y[n] = x[n] - 2 x[n-1] + x[n-2]

As :math:`b` is a slow changing function we can ignore its contribution. We
expect that the contribution from :math:`s` in the peak region is greater than
the noise contribution, but if we ignore higher values of :math:`y` we can focus
on regions where :math:`s` is small we can say that most of the variation in
:math:`y` is due to the noise:

.. math::
    y[n] \approx (e[n] - 2 e[n-1] + e[n-2])

Within this approximation, we can say that :math:`y[n] \sim N(0, 2\sigma)`.
The noise estimation tries to exploit this fact, estimating the noise from
the standard deviation of the second difference of :math:`x`. The algorithm
used can be summarized in the following steps:

1.  Compute the second difference of :math:`x`, :math:`y`.
2.  Set :math:`p=90`, the percentile of the data to evaluate.
3.  compute :math:`y_{p}` the p-th percentile of the absolute value of
    :math:`y`.
4.  Compute the mean :math:`\overline{y}` and standard deviation
    :math:`S_{y}` of :math:`y` restricted to elements with an absolute
    value lower than :math:`y_{p}`. This removes the contribution of :math:`s`.
5.  If :math:`|\overline{y}| \leq S_{y}` or :math:`p \leq 20` then the noise
    level is :math:`\sigma = 0.5 S_{y}`. Else decrease :math:`p` by 10 and go
    back to step 3.

If the contribution from :math:`s` is not completely removed, the noise
estimation will be biased. Despite this, this method gives a good enough
approximation of the noise level that can be used to remove noisy peaks.

Baseline  estimation
--------------------

Baseline estimation is done with the following approach: first, every point
in :math:`x` is classified as signal if a peak can potentially be found in the
region or as or as baseline otherwise. Then, the baseline is estimated for the
whole signal by interpolating baseline points.

The main task of baseline estimation is then to perform this classification
process. To do this, all local extrema in the signal are searched (including
first and last points). Then, we take all closed intervals defined between
consecutive local maxima and minima (or viceversa) and try to evaluate if there
is a significant contribution to the signal coming from :math:`s` in each
interval. If :math:`j` and :math:`k` are the indices defining one such interval,
then the sum of :math:`x` in the interval is:

.. math::
    \sum_{i=j}^{k}x[i] = \sum_{i=j}^{k} s[i] + b[i] + e[i]

If :math:`l = k - j` is the length of the interval, and assuming that
:math:`b` is constant in the interval we can write:

.. math::
    \sum_{i=j}^{k} x[i] - x[j] = \sum_{i=j}^{k} s[i] - s[j] +
    \sum_{i=j}^{k} e[i] -e[j]

.. math::
    a = \sum_{i=j}^{k} x[i] - x[j] = \sum_{i=j}^{k} s[i] - s[j] + e_{sum}

Where :math:`e_{sum} \sim N(0, \sqrt{2l}\sigma)` (we know :math:`\sigma` from
the noise estimation). We can get an idea of the contribution of :math:`s` by
using the value of :math:`a` as follows: If the signal term is contributing to
:math:`a`, then the probability of obtaining a value greater than :math:`a`
from noise is going to be small. This can be computed in the following way:

.. math::
    P(|e_{sum}| > |a|)= \textrm{erfc} \left (\frac{|a|}{2\sqrt{l}\sigma}
    \right )

An interval is classified as baseline if this probability is greater than 0.05.

..  plot:: plots/peak_detection_baseline_example.py
    :caption: Peak detection and baseline estimation in noisy signals.

References
----------

..  [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive feature
    detection for high resolution LC/MS. BMC Bioinformatics 9, 504 (2008).
    https://doi.org/10.1186/1471-2105-9-504
..  [2] Pan Du, Warren A. Kibbe, Simon M. Lin, Improved peak detection in mass
    spectrum by incorporating continuous wavelet transform-based pattern
    matching, Bioinformatics, Volume 22, Issue 17, 1 September 2006,
    Pages 2059–2065, https://doi.org/10.1093/bioinformatics/btl355
