.. definitions

Definitions
===========

Here is a list of some of the most used concepts in the MSKM.

Data Matrix :
    A matrix of feature values where each row is a sample or observation and
    each column is a feature.
Feature :
    A measurable property of a phenomenon being observed. In LC-MS a feature is
    usually represented as a chromatographic peak.
Feature descriptor :
    A series of characteristics of a feature. In the case of a chromatographic
    peak, a feature can be described as a combination of retention time, m/z and
    area.
Sample descriptor :
    A characteristic of a sample. Can be the sample type, class, run order,
    analytical batch.
Mapping :
    A dictionary that maps the sample type to sample classes The available
    sample types are: study sample, quality control, blank, system suitability.
Sample Class :
    The category of the sample. Can be related to the study (e.g: healthy,
    disease) or to the experiment design (quality control, blank, etc...).
Sample type :
    The type of sample used in the experiment. Sample types can be: study
    sample, quality control, blank, system suitability.
Data curation :
    The process of reducing the bias introduced in the measurements during
    sample preparation and data acquisition. Also, the filtration of samples
    that cannot be measured in an analytically robust way.
Carryover :
    A measurement artifact in LC-MS. Occurs when signals from one sample are
    detected in the next sample (signals are “carried over”).
Feature detection :
    The process of finding a feature in a data set. Once a feature is detected
    it can be extracted into a feature descriptor. In LC-MS the feature
    detection procedure involves the detection of chromatographic peaks and
    extraction into rt, m/z and area information.
Feature correspondence :
    The process of match features extracted in different samples. In LC-MS, due
    to difference in m/z and rt in different samples, this is a critical step
    before performing any kind of statistical comparison. (LC-MS alignment in
    theory and practice: a comprehensive algorithmic review)
System suitability check :
    The analysis of a series of samples to assess the performance of an
    analytical platform.
Quality control sample :
    Samples applied to demonstrate analytical accuracy, precision, and
    repeatability after data processing and can be converted to metrics
    describing data quality.
Analytical batch :
    Complete
Run order:
    Temporal order in which the different samples were analyzed.
