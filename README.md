Data Curation in Mass Spectrometry
==================================

The proposed workflow is to read the processed data into a data container
object, which stores three types of data:


- Data Matrix: Areas of detected Features. Each column is a feature (pair rt, mz)
- and each row is a sample. Features Definitions: Information about features
  (mz, rt, etc…).
- Sample Information: Information about samples (class, run order, etc…)

These data container objects are processed through a series of Filters and
Correctors:

- A Filter removes Features or samples according to diverse criteria
(prevalence, type of sample, etc…).
- Correctors transform Data (blank correction, inter batch correction, etc…).

Filters and correctors can be grouped together using a Pipeline object which
concatenates each filter and reports features or samples removed in each step
or metrics associated with a correction step (eg: diminution of CV after
performing interbatch correction).

What is done right now
----------------------

- Data container object
- Generic Filter and corrector objects
- File IO:
    - Reading Processed data into Data Containers objects
    - Config file from a YAML file.
- Pipeline object
- Correctors (Blank Corrector)
- Filters (Prevalence Filter, Variation Filter)
- Tests with simulated data to check the correct implementation of each filter.

TODO
----

- Read several common formats into Data Container (Progenesis,
xcms, mzmine, etc…).
- Create chromatograms and pseudospectra from raw mzML data.
- Review intensity filter, peak shape filter, peak area filter.
- Add flag attribute to data_container.
- Refactor evaluate isotopic profile code.
- Test filter on real data sets and compare with manually curated data.

Filter details
--------------

- **Prevalence filter:** Removes features with a great number of zero or NA 
values across samples. Prevalence is defined  as the fraction of samples  with
detected values for each feature. Features with values outside an interval
defined as [lb, ub] (lower bound, upper bound. Default: [0.5, 1]) are removed.
Prevalence can be calculated globally or per class and selected classes can be
ignored (blanks, QA, etc...).
