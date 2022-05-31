.. _mzml:

.. py:currentmodule:: tidyms

Converting raw data to mzML format
==================================

We recommend using `msconvert
<http://proteowizard.sourceforge.net/download.html>`_ to convert raw data
generated from the different instruments to mzML format. Files can be converted
from a GUI or from the command line. To convert all the files with names ending
in :code:`.RAW` inside a directory from the command line the following command
can be used:

.. code-block:: bat

    msconvert *.RAW -o my_output_dir

If you are using a Waters instrument with lockspray correction, the
:code:`scanEvent` filter can be used to remove the signal from the lockspray.

.. code-block:: bat

    msconvert *.RAW --filter "msEvent 1" -o my_output_dir

To perform feature detection, data must be provided in centroid format. This
can be done using the :code:`peakPicking` filter option:

.. code-block:: bat

    msconvert data.RAW --filter "peakPicking cwt snr=1 peakSpace=0.01"

A :code:`snr=1` is recommended as noisy peaks will be removed during feature
detection anyway. :code:`peakSpacing` should be chosen according to the
instrument used. For QTOF instruments a value of 0.01 is recommended, but
for higher resolution instruments, such as orbitrap or FT-ICR, lower values
may be used.
