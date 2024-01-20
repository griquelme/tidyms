.. definitions

Definitions
===========

Here is a list of the concepts used in TidyMS.

.. glossary::

    batch correction
        A correction step applied to reduce the time dependent variation in the
        metabolite signals due to instrumental response changes, carryover,
        or metabolite degradation, among others.

    blank correction
        A correction applied on study samples to remove the contribution to
        the signal coming from sample preparation. This process consist in
        measuring a set of blank samples and using them to estimate the
        sample preparation contribution to the signal.

    carryover
        A measurement artifact in LC-MS. Occurs when signals from one sample are
        detected in the next sample (signals are “carried over”).

    correction
        A data curation step where the data matrix is transformed to correct
        the data.

    data curation
        The process of reducing the bias introduced in the measurements during
        sample preparation and data acquisition. Also, the filtration of samples
        that cannot be measured in an analytically robust way.

    data matrix
        A matrix of feature values where each row is a sample or observation and
        each column is a feature.

    feature
        In machine learning a feature is a measurable property of an observed
        phenomenon. In TidyMS, a feature is a region from a row with associated
        with the signal of compound. For example, In LC-MS, a feature is usually
        represented as a chromatographic peak.

    feature correspondence
        The process of match features extracted in different samples.

    feature descriptor
        A series of characteristics of a feature. In the case of a
        chromatographic peak, feature descriptors can be peak area, retention
        time, mean m/z among, others.

    feature detection
        The process of finding a feature in a data set. Once a feature is
        detected it can be extracted into a feature descriptor. In LC-MS the
        feature detection procedure involves the detection of chromatographic
        peaks and extraction into rt, m/z and area information.

    feature table
        The table obtained after feature extraction, where each row is a
        feature detected in a sample and each column is a descriptor.

    filtration
        A data curation step where samples or features are removed according
        to an specific criteria.

    mapping
        A dictionary that maps the sample type to sample classes The available
        sample types are: study sample, quality control, blank, system
        suitability.

    normalization
        An operation on the data matrix to adjust the sample values. Common
        normalization methods use different norms, such as the euclidean
        norm, Manhattan norm or maximum norm.

    prevalence filter
        A filter applied on a data matrix to remove features that are detected
        in a low number of samples.

    quality control sample
        Samples applied to demonstrate analytical accuracy, precision, and
        repeatability after data processing and can be converted to metrics
        describing data quality.

    ROI
        A region extracted from raw data where features may be found.

    run order
        Temporal order in which the different samples were analyzed.

    sample class
        The category of the sample. Can be related to the study (e.g: healthy,
        disease) or to the experiment design (quality control, blank, etc...).

    sample descriptor
        A characteristic of a sample. Can be the sample type, class, run order,
        analytical batch.

    sample type
        The type of sample used in the experiment. Sample types can be: study
        sample, quality control, blank, system suitability.

    scaling
        An operation on the data matrix to change the distribution of features.

    system suitability check
        The analysis of a series of samples to assess the performance of an
        analytical platform.
