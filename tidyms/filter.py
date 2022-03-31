"""
Tools to filter and correct DataContainers.

A Filter removes features or samples from a DataContainer according to some
criteria. Correctors perform transformations on the data matrix. Each filter and
corrector has a default behaviour based on recommendations by widely accepted by
the metabolomics community. Operations on DataContainers are made in place. To
generate corrections and filters, the default corrections are generated using
information in the DataContainer mapping.

For example, The BlankCorrector generates and estimation of the blank
contribution to the signal using sample classes that are mapped from
the "blank" sample type. See each Processor object for detailed information.
Processor and Pipeline objects are used in a similar way:

    1.  Create a filter or Pipeline instance.
    2.  Use the process method on a DataContainer to process your data.

Objects
-------
Processor : Abstract object used to create custom Filters and correctors.
DuplicateMerger : Merge duplicate samples.
ClassRemover : Remove samples with a given class name.
BlankCorrector : corrects features contribution originated from sample prep.
BatchCorrector : Removes time dependent bias in features
PrevalenceFilter : Remove features with low detection rate.
VariationFilter : Remove features with high coefficient of variation.
DRatioFilter : Removes features using the D-Ratio.
Pipeline : Combines Processors to apply them simultaneously.

Exceptions
----------
MissingMappingInformation : Error raised when there are no sample classes \
                            assigned to sample type.
MissingValueError : Error raised when the data matrix has missing values.

"""


from .container import DataContainer
from . import container
from ._names import *
from ._filter_functions import *
from ._batch_corrector import *
from . import validation
import os.path
from warnings import warn
from typing import Optional, List, Union, Callable
Number = Union[float, int]

# TODO: replace Processor with Filter and Corrector objects.
# TODO : refactor Processor using the following func prototype:
#   func(dc, **self.params)


class Reporter(object):
    """
    Abstract class with methods to report metrics.

    Attributes
    ----------
    metrics: dict
        stores number of features, number of samples and mean coefficient of
        variation before and after processing.
    name: str
    """
    def __init__(self, name: Optional[str] = None):
        self.metrics = dict()
        self.name = name
        self.results = None

    def _record_metrics(self, dc: DataContainer, status: str):
        """
        record metrics of a DataContainer.

        Parameters
        ----------
        dc: DataContainer
        status: {"before", "after"}
        """
        metrics = dict()
        metrics["cv"] = (dc.metrics.cv(groupby="class", fill_value=0)
                         .median()
                         .median())
        n_samples, n_features = dc.data_matrix.shape
        metrics["features"] = n_features
        metrics["samples"] = n_samples
        if status in ["before", "after"]:
            self.metrics[status] = metrics
        else:
            msg = "status must be before or after."
            raise ValueError(msg)

    def _make_results(self):
        """
        Computes variation in the number of features, samples and CV.
        """
        if ("before" in self.metrics) and ("after" in self.metrics):
            removed_features = (self.metrics["before"]["features"]
                                - self.metrics["after"]["features"])
            removed_samples = (self.metrics["before"]["samples"]
                               - self.metrics["after"]["samples"])
            cv_reduction = (self.metrics["before"]["cv"]
                            - self.metrics["after"]["cv"]) * 100
            results = dict()
            results["features"] = removed_features
            results["samples"] = removed_samples
            results["cv"] = cv_reduction
            self.results = results

    def report(self):
        if self.results:
            msg = "Applying {}: {} features removed, " \
                  "{} samples removed, " \
                  "Mean CV reduced by {:.2f} %."
            msg = msg.format(self.name, self.results["features"],
                             self.results["samples"], self.results["cv"])
            print(msg)


class Processor(Reporter):
    """
    Abstract class to process DataContainer Objects. This class is intended to
    be subclassed to generate specific filters. Filter implementation is done
    overwriting the func method.

    Attributes
    ----------
    mode: {"filter", "flag", "transform"}
        filter removes feature/samples from a DataContainer. flag selects
        features/samples to be inspected manually. Transform applies a
        transformation on the DataContainer.
    axis: {"samples", "features"}, optional
        Axis to process. Only necessary when using mode "filter" or "flag".
    verbose: bool
    params: dict
        parameter used by the filter function
    _default_process: str
        default sample type used to apply filter
    _default_correct: str
        default sample type to be corrected.
    _requirements: dict
        dictionary with the same keys as the obtained from the diagnose method
        from a DataContainer. If any value is different compared to the values
        from diagnose an error is raised.

    Methods
    -------
    process(data) : Applies a filter/correction to a DataContainer
    """
    def __init__(self, mode: str, axis: Optional[str] = None,
                 verbose: bool = False, default_process: Optional[str] = None,
                 default_correct: Optional[str] = None,
                 requirements: Optional[dict] = None):
        super(Processor, self).__init__()
        self.mode = mode
        self.axis = axis
        self.verbose = verbose
        self.remove = list()
        self.params = dict()
        self._default_process = default_process
        self._default_correct = default_correct
        if requirements is None:
            self._requirements = dict()
        else:
            self._requirements = requirements

    def func(self, func):
        raise NotImplementedError

    def check_requirements(self, dc: DataContainer):
        dc_status = dc.diagnose()

        # check process and corrector classes in the processor against values in
        # the DataContainer
        class_types = ["process_classes", "corrector_classes"]
        for class_type in class_types:
            if class_type in self.params:
                class_list = self.params[class_type]
                if class_type == "process_classes":
                    default = self._default_process
                else:
                    default = self._default_correct
                has_mapping = dc_status[default]
                if (class_list is None) and (not has_mapping):
                    msg = "no classes where assigned to {} sample type in " \
                        "the sampling mapping"
                    msg = msg.format(default)
                    raise MissingMappingInformation(msg)
                elif not dc.is_valid_class_name(class_list):
                    msg = "classes listed in {} aren't present " \
                          "in the DataContainer"
                    msg = msg.format(class_type)
                    raise container.ClassNameError(msg)

        for requirement in self._requirements:
            if self._requirements[requirement] != dc_status[requirement]:
                raise _requirements_error[requirement]

    def set_default_sample_types(self, dc: DataContainer):

        try:
            name = "process_classes"
            if self.params[name] is None:
                self.params[name] = dc.mapping[self._default_process]
        except KeyError:
            pass

        try:
            name = "corrector_classes"
            if self.params[name] is None:
                self.params[name] = dc.mapping[self._default_correct]
        except KeyError:
            pass

    def process(self, dc):
        self._record_metrics(dc, "before")
        self.set_default_sample_types(dc)
        self.check_requirements(dc)
        if self.mode == "filter":
            self.remove = self.func(dc)
            dc.remove(self.remove, self.axis)
        elif self.mode == "flag":
            self.remove = self.func(dc)
        elif self.mode == "transform":
            self.func(dc)
        else:
            msg = "mode must be `filter`, `transform` or `flag`"
            raise ValueError(msg)
        self._record_metrics(dc, "after")
        self._make_results()
        if self.verbose:
            self.report()


class Pipeline(Reporter):
    """
    Combines Filters and Correctors and applies them simultaneously.

    Attributes
    ----------
    processors: list[Processors]
        A list of processors to apply. Can also contain another Pipeline.
    verbose: bool
        If True prints a message each time a Processor is applied.
    """
    def __init__(self, processors: list, verbose: bool = False):
        _validate_pipeline(processors)
        super(Pipeline, self).__init__()
        self.processors = list(processors)
        self.verbose = verbose

    def process(self, dc):
        self._record_metrics(dc, "before")
        for x in self.processors:
            if self.verbose:
                x.verbose = True
            x.process(dc)
        self._record_metrics(dc, "after")


# FILTERS maintains a dictionary of filter names to filter instances. This can
# be used to create filters or pipelines from dictionaries of filter names and
# parameters. decorating a new filter with the register function add the new
# filter to FILTERS

FILTERS = dict()


def register(f):
    """
    register to available filters

    Parameters
    ----------
    f : Filter or Corrector

    Returns
    -------
    f
    """
    FILTERS[f.__name__] = f
    return f


@register
class DuplicateMerger(Processor):
    """
    Merge sample replicates.
    """
    # TODO: add merger parameter, classes to merge.
    def __init__(self, process_classes: Optional[List[str]] = None):
        super(DuplicateMerger, self).__init__(axis=None, mode="transform")
        self.params["process_classes"] = process_classes

    def func(self, dc: DataContainer):
        if self.params["process_classes"] is None:
            self.params["process_classes"] = dc.mapping[_study_sample_type]
        dc.data_matrix = average_replicates(dc.data_matrix, dc.id,
                                            dc.classes, **self.params)
        dc.sample_metadata = (dc.sample_metadata
                              .loc[dc.data_matrix.index, :])


@register
class ClassRemover(Processor):
    """
    Remove samples from the specified classes.

    Parameters
    ----------
    classes: list
        List of classes to remove.

    """
    def __init__(self, classes: List[str]):
        requirements = {"empty": False, "missing": False}
        super(ClassRemover, self).__init__(axis="samples", mode="filter",
                                           requirements=requirements)
        self.name = "Class Remover"
        if classes is None:
            self.params["classes"] = list()
        else:
            self.params["classes"] = classes

    def func(self, dc):
        remove_samples = dc.classes.isin(self.params["classes"])
        remove_samples = remove_samples[remove_samples].index
        # remove classes from mapping
        for sample_type, sample_classes in dc.mapping.items():
            if sample_classes is not None:
                diff = set(sample_classes)
                diff = diff.difference(self.params["classes"])
                if not diff:
                    diff = None
                else:
                    diff = list(diff)
            else:
                diff = None
            dc.mapping[sample_type] = diff
        return remove_samples


@register
class BlankCorrector(Processor):
    r"""
    Corrects systematic bias due to sample preparation.

    Parameters
    ----------
    corrector_classes : list[str], optional
        Classes used to generate the blank correction. If None, uses the
        value from blank in the DataContainer mapping attribute.
    process_classes :  list[str], optional
        Classes to be corrected. If None, uses the value from sample in the
        DataContainer mapping attribute.
    factor : float
        factor used to convert values to zero (see notes)
    mode : {"mean", "max", "lod", "loq"} or callable.
        Function used to generate the blank correction. If `mode` is mean,
        the correction is generated as the mean of all blank samples. If
        max, the correction is generated as the maximum value for each
        feature in all blank samples. If `mode` is lod, the correction is
        the mean plus three times the standard deviation of the blanks. If
        `mode` is loq, the correction is the mean plus ten times the standard
        deviation.
    process_blanks : bool
        If True applies blank correction to blanks also.
    verbose : bool
        Shows a message with information after the correction has been
        applied.

    Notes
    -----
    Blank correction is applied for each feature in the following way:

    .. math::

        X_{corrected} = 0 \textrm{ if } X < factor * mode(X_{blank}) \\
        X_{corrected} = X - mode(X_{blank}) \textrm{ else}

    """
    def __init__(self, corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 mode: Union[str, Callable] = "lod", factor: float = 1,
                 robust: bool = True,
                 process_blanks: bool = True, verbose=False):
        """
        Constructor for the BlankCorrector.

        """
        requirements = {"empty": False, "missing": False,
                        _blank_sample_type: True}
        super(BlankCorrector, self).__init__(axis=None, mode="transform",
                                             verbose=verbose,
                                             requirements=requirements)
        self.name = "Blank Corrector"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["mode"] = mode
        self.params["factor"] = factor
        self.params["process_blanks"] = process_blanks
        self.params["robust"] = robust
        self._default_process = _study_sample_type
        self._default_correct = _blank_sample_type
        validation.validate_blank_corrector_params(self.params)

    def func(self, dc):
        dc.data_matrix = \
            correct_blanks(dc.data_matrix, dc.classes, **self.params)


@register
class PrevalenceFilter(Processor):
    """
    Remove Features detected in a low number of samples.

    Parameters
    ----------
    process_classes : List[str], optional
        Classes used to compute prevalence. If None, classes are obtained
        from sample classes in the DataContainer mapping.
    lb : Number between 0 and 1
        Lower bound of acceptance.
    ub : Number between 0 and 1
        Upper bound of acceptance. Must be greater or equal than `lb`.
    threshold : non negative number
        Minimum intensity to consider a feature as detected.
    intraclass : bool
        Whether to evaluate a global prevalence or a per class prevalence.
        If intraclass is True, the detection rate is computed for each class,
        and the prevalence is defined as the minimum value for the classes
        analyzed. If intraclass is False, the prevalence is computed as the
        detection rate for all the samples that belong to the `process_classes`.
    verbose : bool
        Shows a message with information after the correction has been
        applied.

    Notes
    -----
    The prevalence is computed using the detection rate, that is, the fraction
    of samples where a feature was detected. A feature is considered detected
    if its value is above a threshold. The `mode` parameter controls how the
    prevalence is computed.

    """
    def __init__(self, process_classes: Optional[List[str]] = None,
                 lb: Number = 0.5, ub: Number = 1,
                 intraclass: bool = True, verbose: bool = False,
                 threshold: Number = 0):
        """
        Constructor of the PrevalenceFilter.
        """
        requirements = {"empty": False, "missing": False}
        super(PrevalenceFilter, self).__init__(axis="features", mode="filter",
                                               verbose=verbose,
                                               requirements=requirements)
        self.name = "Prevalence Filter"
        self.params["process_classes"] = process_classes
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["intraclass"] = intraclass
        self.params["threshold"] = threshold
        self._default_correct = _study_sample_type
        self._default_process = _study_sample_type
        validation.validate_prevalence_filter_params(self.params)

    def func(self, dc):
        if self.params["intraclass"]:
            groupby = "class"
        else:
            groupby = None
        dr = dc.metrics.detection_rate(groupby=groupby,
                                       threshold=self.params["threshold"])
        dr = dr.loc[self.params["process_classes"], :]
        lb = self.params["lb"]
        ub = self.params["ub"]
        return get_outside_bounds_index(dr, lb, ub)


@register
class DRatioFilter(Processor):
    r"""
    Remove Features with low biological information.

    To use this filter the qc sample type and the study sample type must been
    specified in the DataContainer mapping.

    Parameters
    ----------
    lb: number between 0 and 1
        Lower bound of acceptance
    ub: number between 0 and 1
        Upper bound of acceptance.
    robust: bool
        If True uses the MAD to compute the d-ratio. Else uses the standard
        deviation.
    verbose : bool
        Shows a message with information after the correction has been
        applied.

    Notes
    -----
    D-Ratio is a metric defined in [1]_ as the quotient between the technical
    and the biological variation of a feature:

    .. math::

        D-Ratio = \frac{\sigma_{technical}}
            {\sqrt{\sigma_{technical}^{2} + \sigma_{biological}^{2}}}

    The technical variation is estimated as the dispersion from the QC samples,
    while the total variation (technical and biological) is estimated from
    the study samples. Lower D-Ratio values suggest features that are measured
    in a robust way. A maximum acceptance value of 0.5 is suggested.

    References
    ----------
    .. [1] D.Broadhurst *et al*, "Guidelines and considerations for the use
        of system suitability and quality control samples in mass spectrometry
        assays applied in untargeted clinical metabolomic studies",
        Metabolomics (2018) 14:72.

    """

    def __init__(self, lb=0, ub=0.5, robust=False,
                 verbose=False):
        """
        Constructor of the DRatioFilter.
        """
        requirements = {"empty": False, "missing": False, _qc_sample_type: True,
                        _study_sample_type: True}
        super(DRatioFilter, self).__init__(axis="features", mode="filter",
                                           verbose=verbose,
                                           requirements=requirements)
        self.name = "D-Ratio Filter"
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["robust"] = robust
        validation.validate_dratio_filter_params(self.params)

    def func(self, dc: DataContainer):
        lb = self.params["lb"]
        ub = self.params["ub"]
        dratio = dc.metrics.dratio(robust=self.params["robust"])
        return get_outside_bounds_index(dratio, lb, ub)


@register
class VariationFilter(Processor):
    """
    Remove features with low reproducibility.

    The reproducibility of the features is evaluated using the Relative standard
    deviation of each feature in samples of a specific class or classes. By
    default, the QC samples are analyzed.

    Parameters
    ----------
    lb : number between 0 and 1
        Lower bound of acceptance
    ub : number between 0 and 1
        Upper bound of acceptance. Must be greater than `lb`.
    process_classes: List[str], optional
        Classes used to evaluate the coefficient of variation. If None,
        list of classes is taken from the qc sample type from the
        DataContainer mapping attribute.
    robust: bool
        If false uses the mean and standard deviation to compute the cv.
        Else, the cv is estimated using the MAD and the median of the
        feature, assuming a normal distribution.
    intraclass: bool
        If True, the cv is computed for each class in `process_classes` and
        then the maximum value is compared against `lb` and `ub`. Else
        a global cv is computed for all classes in `process_classes`.
    verbose: bool
        If True, prints a message

    """
    def __init__(self, lb=0, ub=0.25, process_classes=None, robust=False,
                 intraclass=True, verbose=False):
        """
        Constructor of the VariationFilter.
        """
        requirements = {"empty": False, "missing": False}
        super(VariationFilter, self).__init__(axis="features", mode="filter",
                                              verbose=verbose,
                                              requirements=requirements)
        self.name = "Variation Filter"
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["process_classes"] = process_classes
        self.params["robust"] = robust
        self.params["intraclass"] = intraclass
        self._default_process = _qc_sample_type
        validation.validate_variation_filter_params(self.params)

    def func(self, dc: DataContainer):
        lb = self.params["lb"]
        ub = self.params["ub"]
        if self.params["intraclass"]:
            groupby = "class"
        else:
            groupby = None
        variation = dc.metrics.cv(groupby=groupby, robust=self.params["robust"])
        if self.params["intraclass"]:
            variation = variation.loc[self.params["process_classes"], :]

        return get_outside_bounds_index(variation, lb, ub)


@register
class DilutionFilter(Processor):
    """
    Filter features based on the correlation with a dilution factor.

    In order to use this filter, the dilution column must be specified in the
    sample_metadata of the DataContainer. Also, the QCs used for the
    analysis must be specified under the `dqc` key in the DataContainer mapping.

    Parameters
    ----------
    min_corr : number between 0 and 1
        Lower bound for the correlation coefficient.
    plim : number between 0 and 1
        p-value limit for the Jarque-Bera test. Used only when mode is `ols`.
    mode : {"ols", "spearman"}
        `ols` computes the ordinary least squares linear regression.The r
        squared from the fit and the p-value from the Jarque-Bera test are used
        to evaluate the linearity of the signal with the dilution. Features with
        correlation values lower than `min_corr` or p-values lower than `plim`
        are removed. `spearman` compares the correlation threshold with the
        spearman rank correlation coefficient.
    verbose: bool
        If True, prints a message

    Notes
    -----
    Correlation with the dilution is a measure of the linearity of the response
    of the feature in the experimental conditions [2].

    References
    ----------
    ..  [2] Lewis MR, *et al*, Development and Application of Ultra-Performance
        Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic
        Phenotyping. Anal Chem. (2016), 88(18):9004-13.
        doi: 10.1021/acs.analchem.6b01481

    """
    def __init__(self, min_corr: float = 0.8, plim: float = 0.1,
                 mode: str = "ols", verbose: bool = False):
        """
        Constructor of the DilutionFilter.
        """
        requirements = {"empty": False, "missing": False,
                        _dilution_qc_type: True}
        super(DilutionFilter, self).__init__(axis="features", mode="filter",
                                             verbose=verbose,
                                             requirements=requirements)
        self.name = "Dilution Filter"
        self.params["min_corr"] = min_corr
        self.params["plim"] = plim
        self.params["mode"] = mode
        self._default_process = _qc_sample_type
        validation.validate_dilution_filter_params(self.params)

    def func(self, dc: DataContainer):
        mode = self.params["mode"]
        min_corr = self.params["min_corr"]
        plim = self.params["plim"]
        corr = dc.metrics.correlation("dilution", mode=mode,
                                      classes=dc.mapping[_dilution_qc_type])
        if mode == "ols":
            r2_ind = get_outside_bounds_index(corr.loc["r2", :], min_corr, 1)
            jb_ind = get_outside_bounds_index(corr.loc["JB", :], plim, 1)
            rm_ind = r2_ind.union(jb_ind)
        else:
            rm_ind = get_outside_bounds_index(corr, min_corr, 1)
        return rm_ind


class _TemplateValidationFilter(Processor):
    """
    Checks that process samples in each batch are surrounded by corrector
    samples. Samples that do not meet te requirements are removed.
    The minimum number of corrector samples necessary to perform LOESS based
    correction is also tested. Analytical batches where the number of corrector
    samples is lower than four are removed. This filter is used as a first step
    during batch correction and should not be called directly.

    Parameters
    ----------
    process_classes: list[str], optional
        list of classes used as corrector samples
    corrector_classes: list[str], optional
        list of classes used as process samples
    verbose: bool

    """
    def __init__(self, process_classes: Optional[List[str]] = None,
                 corrector_classes: Optional[List[str]] = None,
                 verbose: bool = False):

        requirements = {"empty": False, "missing": False, _sample_order: True,
                        _sample_batch: True}
        (super(_TemplateValidationFilter, self)
         .__init__(axis="samples", mode="filter", verbose=verbose,
                   requirements=requirements))
        self.name = "Batch Template Check"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self._default_process = _study_sample_type
        self._default_correct = _qc_sample_type
        validation.validate_batch_corrector_params(self.params)

    def func(self, dc: DataContainer):
        # TODO: create a function in _filter_functions with this code

        dc.sort(_sample_order, "samples")

        block_type, block_number = \
            make_sample_blocks(dc.classes, self.params["corrector_classes"],
                               self.params["process_classes"])
        qc_block_mask = block_type == 0
        sample_block_mask = block_type == 1
        sample_names = block_number[sample_block_mask].index

        # check the minimum number of qc samples in a batch
        n_qc = dc.classes[qc_block_mask].groupby(dc.batch).count()
        min_n_qc = 4    # the minimum number of QC samples to use LOESS
        valid_batch = n_qc >= min_n_qc
        valid_batch = valid_batch[valid_batch].index
        invalid_samples = ~dc.batch.isin(valid_batch)
        invalid_samples = invalid_samples[invalid_samples].index

        # min and max run order per batch
        min_qc_order = dc.order[qc_block_mask].groupby(dc.batch).min()
        max_qc_order = dc.order[qc_block_mask].groupby(dc.batch).max()

        sample_order = dc.order.loc[sample_names].groupby(dc.batch)
        for batch_number, batch_order in sample_order:
            rm_mask = ((batch_order < min_qc_order[batch_number]) |
                       (batch_order > max_qc_order[batch_number]))
            rm_sample = rm_mask[rm_mask].index
            invalid_samples = invalid_samples.union(rm_sample)
        return invalid_samples


class _BatchCorrectorPrevalenceFilter(Processor):
    """
    Checks the prevalence of the features in the corrector samples.

    Prevalence in the corrector samples is checked using the concept of
    corrector blocks: A corrector block is a set of consecutive samples of
    the corrector class. a feature is detected in a block if it was detected
    in at least one sample in the block. For example:

    start                   Middle                  End
    B1          B2          B3          B4          B5
    CCC PPPP    C   PPPP    C   PPPP    C   PPPP    CCC

    B1, B2, etc... are corrector blocks. C is a corrector sample and P is a
    process class. The prevalence is tested on each block by comparing against
    a threshold value. The feature is detected in a block if at least one sample
    is above the threshold. Then the fraction of the blocks where the feature
    was detected is the prevalence. A feature is removed if:

    1. The prevalence if lower than the `min_qc_dr` parameter.
    2. The feature is not detected in the start or end block.


    Parameters
    ----------
    min_qc_dr: float
        minimum fraction of QC blocks where the feature was detected.
        This value is corrected in a way such that the minimum number of QC
        samples is greater than 4.
    threshold: float
        Minimum intensity to consider a feature detected
    process_classes: list[str], optional
        list of classes used as corrector samples
    corrector_classes: list[str], optional
        list of classes used as process samples
    verbose: bool
    """

    def __init__(self, min_qc_dr: float = 0.9, verbose: bool = False,
                 process_classes: Optional[List[str]] = None,
                 corrector_classes: Optional[List[str]] = None,
                 threshold: float = 0.0):

        requirements = {"empty": False, "missing": False, _sample_order: True,
                        _sample_batch: True}
        (super(_BatchCorrectorPrevalenceFilter, self)
         .__init__(axis="features", mode="filter", verbose=verbose,
                   requirements=requirements))
        self.name = "Batch Prevalence Checker"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["min_qc_dr"] = min_qc_dr
        self.params["threshold"] = threshold
        self._default_process = _study_sample_type
        self._default_correct = _qc_sample_type
        validation.validate_batch_corrector_params(self.params)

    def func(self, dc: DataContainer):
        ps = self.params["process_classes"]
        ps = [x for x in ps if x not in self.params["corrector_classes"]]
        self.params["process_classes"] = ps
        res = check_qc_prevalence(dc.data_matrix, dc.batch,
                                  dc.classes, self.params["corrector_classes"],
                                  self.params["process_classes"],
                                  threshold=self.params["threshold"],
                                  min_qc_dr=self.params["min_qc_dr"])
        return res


class _BatchCorrectorProcessor(Processor):
    """
    Corrects instrumental drift between in a batch. Part of the batch
    corrector pipeline
    """
    def __init__(self, corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 frac: Optional[float] = None,
                 interpolator: str = "splines",
                 n_qc: Optional[int] = None,
                 method: str = "multiplicative",
                 process_qc: bool = True,
                 verbose: bool = False):
        requirements = {"empty": False, "missing": False, _sample_order: True,
                        _sample_batch: True}
        super(_BatchCorrectorProcessor, self).__init__(axis=None,
                                                       mode="transform",
                                                       verbose=verbose,
                                                       requirements=requirements
                                                       )
        self.name = "Batch Corrector"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["interpolator"] = interpolator
        self.params["n_qc"] = n_qc
        self.params["frac"] = frac
        self.params["method"] = method
        self.params["process_qc"] = process_qc
        self._default_process = _study_sample_type
        self._default_correct = _qc_sample_type
        validation.validate_batch_corrector_params(self.params)

    def func(self, dc: DataContainer):
        dc.data_matrix = interbatch_correction(dc.data_matrix, dc.order,
                                               dc.batch, dc.classes,
                                               **self.params)


@register
class BatchCorrector(Pipeline):
    r"""
    Correct time dependant systematic bias along samples due to variation in
    instrumental response.

    Parameters
    ----------
    min_qc_dr : float
        minimum fraction of QC where the feature was detected. See the notes for
        an explanation of how this value is computed.
    first_n_qc : int, optional
        The number of first QC samples used to estimate the expected
        value for each feature in the QC. If None uses all QC samples in a
        batch. See notes for an explanation of its use.
    threshold : float
        Minimum value to consider a feature detected. Used to compute the
        detection rate of each feature.
    frac : float, optional
        frac parameter of the LOESS model. If None, the best value for each
        feature is estimated using LOOCV.
    interpolator : {"splines", "linear"}
        Interpolator used to estimate the correction for each sample.
    method : {"additive", "multiplicative"}
        Method used to model the variation in samples.
    corrector_classes : list[str], optional
        list of classes used to generate the correction. If None uses
        QC sample types from the mapping.
    process_classes : list[str], optional
        list of classes used to correct. If None uses sample sample types from
        the mapping.
    verbose : bool
        If True a message is shown after processing the data matrix.

    Notes
    -----
    The correction is applied as described by Broadhurst in [1]. Using QC
    samples, a correction is generated for each feature in the following way:
    The signal of a feature is modeled as three additive components: a expected
    value :math:`m_{jk}`, a systematic bias :math:`f_{k}` and error term
    :math:`\epsilon`:

    .. math::
       m_{jk} = \bar{m_{k}} + f_{k}(t_{j}) + \epsilon

    Where :math:`m_{jk}` is the element in the j-th row and k-th column of the
    data matrix. 
    
    First, :math:`\bar{m_{k}}` is subtracted to the detected values and then
    :math:`f_{k}` is estimated using Locally weighted scatter plot smoothing
    (LOESS). The optimal fraction of samples for each feature is obtained using
    Leave One Out Cross Validation (LOOCV).

    In order to apply this correction, several checks needs to be made. First,
    the QC template is checked and samples that cannot be corrected are removed.
    A study sample is valid if it is surrounded by QC samples.
    This is a necessary step because the correction for the study samples is
    built using interpolation. It's recommended to have three QC samples at the
    beginning and at the end of each batch. See [1] for recommendations on
    analytical batches templates.

    After checking the QC template, each feature is checked to see if the
    minimum number of QC samples necessary to perform LOESS are available. This
    step is done grouping samples of the same type into QC blocks: a QC block is
    a set of consecutive QC samples. A feature is detected in a block if it was
    detected in at least one sample in the block. For example, in an analytical
    batch:

    +---------------+---------------+---------------+
    | Run order     | Sample type   | Block number  |
    +===============+===============+===============+
    | 1, 2, 3       | Q, Q, Q       |   1 (start)   |
    +---------------+---------------+---------------+
    | 4, 5, 6, 7    | S, S, S, S    |   2 (middle)  |
    +---------------+---------------+---------------+
    | 8             | Q             |   3           |
    +---------------+---------------+---------------+
    | 9, 10, 11, 12 | S, S, S, S    |   4           |
    +---------------+---------------+---------------+
    | 13            | Q             |   5           |
    +---------------+---------------+---------------+
    | 13, 14, 15, 16| S, S, S, S    |   6           |
    +---------------+---------------+---------------+
    | 17            | Q             |   7           |
    +---------------+---------------+---------------+
    | 18, 19, 20, 21| S, S, S, S    |   8           |
    +---------------+---------------+---------------+
    | 22, 23, 24    | Q, Q, Q       |   9 (end)     |
    +---------------+---------------+---------------+

    Detection is evaluated on each block, comparing samples against a
    `threshold` value. Then the fraction of the blocks where the feature was
    detected is the detection rate in the QC samples. A feature is removed if:

    1.  The prevalence if lower than the `min_qc_dr` parameter (this parameter
        is corrected in a way such that the minimum number of QC samples must be
        always greater or equal than 4).
    2.  The feature is not detected in the start or end block.

    After these two checks, the remaining samples and features are suitable
    for LOESS batch correction. A final consideration is how to estimate the
    :math:`\bar{m_{k}}` for each feature. This value is usually computed as the
    mean or median of the QC values in a batch, but if the temporal bias becomes
    stronger as more samples are analyzed, a better estimation of
    :math:`\bar{m_{k}}` can be obtained using the average of the first samples
    analyzed in a batch. To this end, the `n_qc` parameter controls how many
    QC samples are used to estimate the expected values in the QC samples.

    References
    ----------
    .. [1] D Broadhurst *et al*, "Guidelines and considerations for the use of
        system suitability and quality control samples in mass spectrometry
        assays applied in untargeted clinical metabolomic studies.",
        Metabolomics, 2018;14(6):72. doi: 10.1007/s11306-018-1367-3

    """
    def __init__(self,
                 min_qc_dr: float = 0.9,
                 first_n_qc: Optional[int] = None,
                 threshold: float = 0, frac: Optional[float] = None,
                 interpolator: str = "splines",
                 method: str = "multiplicative",
                 corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 verbose: bool = False):

        deprecation_msg = \
            "{} is deprecated and is going to be removed in a future release. " \
            "To perform batch correction use the method " \
            " `preprocess.correct_batches` from DataContainer"
        warn(deprecation_msg.format(self.__class__), DeprecationWarning,
             stacklevel=2)

        checker = \
            _TemplateValidationFilter(process_classes=process_classes,
                                      corrector_classes=corrector_classes,
                                      verbose=verbose)
        prevalence = \
            _BatchCorrectorPrevalenceFilter(min_qc_dr=min_qc_dr,
                                            verbose=verbose,
                                            process_classes=process_classes,
                                            corrector_classes=corrector_classes,
                                            threshold=threshold)
        corrector = \
            _BatchCorrectorProcessor(corrector_classes=corrector_classes,
                                     process_classes=process_classes, frac=frac,
                                     interpolator=interpolator, verbose=verbose,
                                     n_qc=first_n_qc, method=method)
        pipeline = [checker, prevalence, corrector]
        super(BatchCorrector, self).__init__(pipeline, verbose=verbose)
        self.name = "Batch Corrector"


def pipeline_from_list(param_list: list, verbose=False):
    processors = list()
    for d in param_list:
        processors.append(filter_from_dictionary(d))
    pipeline = Pipeline(processors, verbose)
    return pipeline


def filter_from_dictionary(d):
    filt = None
    for name, params in d.items():
        filt = FILTERS[name](**params)
    return filt


def sample_name_from_path(path):
    file_name = os.path.split(path)[1]
    sample_name = os.path.splitext(file_name)[0]
    return sample_name


def _validate_pipeline(t):
    if not isinstance(t, (list, tuple)):
        t = [t]
    for filt in t:
        if not isinstance(filt, (Processor, Pipeline)):
            msg = ("elements of the Pipeline must be",
                   "instances of Filter, DataCorrector or another Pipeline.")
            raise TypeError(msg)


class MissingMappingInformation(ValueError):
    """error raised when an empty sample type is used from a mapping"""
    pass


class MissingValueError(ValueError):
    """error raise when a DataContainer's data matrix has missing values"""
    pass


_requirements_error = {"empty": container.EmptyDataContainerError,
                       "missing": MissingValueError,
                       _qc_sample_type: MissingMappingInformation,
                       _blank_sample_type: MissingMappingInformation,
                       _sample_batch: container.BatchInformationError,
                       _sample_order: container.RunOrderError}
