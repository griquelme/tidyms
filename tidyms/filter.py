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


from .data_container import DataContainer
from . import data_container
from ._names import *
from ._filter_functions import *
from . import validation
import yaml
import os.path
from typing import Optional, List, Union, Callable
Number = Union[float, int]

# TODO: replace Processor with Filter and Corrector objects.


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
        metrics["cv"] = dc.metrics.cv(intraclass=False).mean()
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

        # check process and corrector classes in the processor agains values in
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
                    raise data_container.ClassNameError(msg)

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
        If True prints a message each time a Proessor is applied.
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
            self.params["process_classes"] = dc.mapping[_sample_type]
        dc.data_matrix = average_replicates(dc.data_matrix, dc.id,
                                            dc.classes, **self.params)
        dc.sample_metadata = (dc.sample_metadata
                              .loc[dc.data_matrix.index, :])


@register
class ClassRemover(Processor):
    """
    Remove samples from the specified classes.
    """
    def __init__(self, classes: List[str]):
        super(ClassRemover, self).__init__(axis="samples", mode="filter")
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
            dc.mapping[sample_type] = diff
        return remove_samples


@register
class BlankCorrector(Processor):
    """
    Corrects systematic bias due to sample preparation.
    """
    def __init__(self, corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 mode: Union[str, Callable] = "lod", factor: float = 1,
                 process_blanks: bool = True, verbose=False):
        """
        Constructor for the BlankCorrector.

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
        mode : {"mean", "max", "lod", "loq"}, function.
            Function used to generate the blank correction. If `mode` is mean,
            the correction is generated as the mean of all blank samples. If
            max, the correction is generated as the maximum value for each
            feature in all blank samples. If `mode` is lod, the correction is
            the mean plus three standard deviations. If `mode` is loq, the
            correction is the mean plus ten times the standard deviation.
        process_blanks : bool
            If True applies blank correction to blanks also.
        verbose : bool
            Shows a message with information after the correction has been
            applied.

        Notes
        -----
        Blank correction is applied for each feature in the following way:

        .. math::

            X_{corrected} = 0 if X < factor * mode(X_{blank})

        .. math::

            X_{corrected} = X - mode(X_{blank}) else

        """
        super(BlankCorrector, self).__init__(axis=None, mode="transform",
                                             verbose=verbose,
                                             requirements={"empty": False,
                                                           "missing": False})
        self.name = "Blank Corrector"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["mode"] = mode
        self.params["factor"] = factor
        self.params["process_blanks"] = process_blanks
        self._default_process = _sample_type
        self._default_correct = _blank_type

        validation.validate(self.params, validation.blankCorrectorValidator)

    def func(self, dc):
        dc.data_matrix = correct_blanks(dc.data_matrix, dc.classes,
                                        **self.params)


@register
class PrevalenceFilter(Processor):
    """
    Remove Features detected in a low number of samples.

    The prevalence is defined as the fraction of samples where a given feature
    has been detected.
    """
    def __init__(self, process_classes: Optional[List[str]] = None,
                 lb: Number = 0.5, ub: Number = 1,
                 intraclass: bool = True, verbose: bool = False,
                 threshold: Number = 0):
        """
        Constructor of the PrevalenceFilter.

        Parameters
        ----------
        process_classes : List[str], optional
            Classes used to compute prevalence. If None, classes are obtained
            from sample classes in the DataContainer mapping.
        lb : float
            Lower bound of prevalence.
        ub : float
            Upper bound of prevalence.
        threshold : float
            Minimum intensity to consider a feature as detected.
        intraclass : bool
            Whether to evaluate a global prevalence or a per class prevalence.
            If intraclass is True, prevalence is computed for each class and
            features in which the prevalence is outside the selected bounds for
            all classes are removed.
        verbose : bool
            Shows a message with information after the correction has been
            applied.
        """
        super(PrevalenceFilter, self).__init__(axis="features", mode="filter",
                                               verbose=verbose,
                                               requirements={"empty": False,
                                                             "missing": False})
        self.name = "Prevalence Filter"
        self.params["process_classes"] = process_classes
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["intraclass"] = intraclass
        self.params["threshold"] = threshold
        self._default_correct = _sample_type
        self._default_process = _sample_type
        validation.validate(self.params, validation.prevalenceFilterValidator)

    def func(self, dc):
        dr = dc.metrics.detection_rate(intraclass=self.params["intraclass"],
                                       threshold=self.params["threshold"])
        dr = dr.loc[self.params["process_classes"], :]
        lb = self.params["lb"]
        ub = self.params["ub"]
        return get_outside_bounds_index(dr, lb, ub)


@register
class DRatioFilter(Processor):
    r"""
    Remove Features with low biological information.

    D-Ratio is a metric defined in [1]_ as the quotient between the technical
    and the biological variation of a feature:

    .. math::

        D-Ratio = \frac{\sigma_{technical}}
            {\sqrt{\sigma_{technical}^{2} + \sigma_{biological}^{2}}}

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
        Constructor of the DRatioFilter. To use this filter the qc sample type
        and the study sample type must been specified in the DataContainer
        mapping.

        Parameters
        ----------
        lb: float
            Lower bound of D-ratio. Should be zero
        ub: float
            Upper bound of D-ratio. Usually 50% or lower, the lower the better.
        robust: bool
        if True uses the MAD to compute the d-ratio. Else uses the standard
        deviation.
        verbose : bool
            Shows a message with information after the correction has been
            applied.
        """
        (super(DRatioFilter, self)
         .__init__(axis="features", mode="filter", verbose=verbose,
                   requirements={"empty": False, "missing": False,
                                 _qc_type: True, _sample_type: True})
         )
        self.name = "D-Ratio Filter"
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["robust"] = robust
        validation.validate(self.params, validation.dRatioFilterValidator)

    def func(self, dc: DataContainer):
        lb = self.params["lb"]
        ub = self.params["ub"]
        dratio = dc.metrics.dratio(robust=self.params["robust"])
        return get_outside_bounds_index(dratio, lb, ub)


@register
class VariationFilter(Processor):
    """
    Remove features with low reproducibility.
    """
    def __init__(self, lb=0, ub=0.25, process_classes=None, robust=False,
                 intraclass=True, verbose=False):
        """
        Constructor of the VariationFilter.

        Parameters
        ----------
        lb : float
            Lower bound for the coefficient of variation. Must be a positive
            number between zero and one and lower or equal than `ub`.
        ub : float
            Upper bound for the coefficient of variation. Must be a positive
            number between zero and one and greater or equal than `lb`.
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
        (super(VariationFilter, self)
         .__init__(axis="features", mode="filter", verbose=verbose,
                   requirements={"empty": False, "missing": False})
         )
        self.name = "Variation Filter"
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["process_classes"] = process_classes
        self.params["robust"] = robust
        self.params["intraclass"] = intraclass
        self._default_process = _qc_type
        validation.validate(self.params, validation.variationFilterValidator)

    def func(self, dc: DataContainer):
        lb = self.params["lb"]
        ub = self.params["ub"]
        variation = dc.metrics.cv(intraclass=self.params["intraclass"],
                                  robust=self.params["robust"])
        if self.params["intraclass"]:
            variation = variation.loc[self.params["process_classes"], :]

        return get_outside_bounds_index(variation, lb, ub)


class _BatchDesignChecker(Processor):
    """
    Checks that process samples in each batch are surrounded by corrector
    samples. Batches that do not meet te requirements are removed.
    This filter is used as a first step in batch correction and should not be
    called directly.

    Parameters
    ----------
    n_min: int
        Minimum number of QC samples per batch
    process_classes: list[str], optional
        list of classes used as corrector samples
    corrector_classes: list[str], optional
        list of classes used as process samples
    verbose: bool

    Notes
    -----
    For a batch to be valid the samples at the start and at the end of each
    batch needs to be a corrector sample. For example, if q is a corrector
    sample and s is a process sample, then a valid batch has the following
    structure:

        qqssssqssssqsssssqsssssqq

    On the other side, an invalid batch is for example a batch with the
    following structure:

        sssssqsssssqsssssq
    """
    def __init__(self, n_min: int = 4, verbose: bool = False,
                 process_classes: Optional[List[str]] = None,
                 corrector_classes: Optional[List[str]] = None):

        req = {"empty": False, "missing": False, "order": True, "batch": True}
        (super(_BatchDesignChecker, self)
         .__init__(axis="samples", mode="filter", verbose=verbose,
                   requirements=req))
        self.name = "Batch Scheme Checker"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["n_min"] = n_min
        self._default_process = _sample_type
        self._default_correct = _qc_type
        validation.validate(self.params, validation.batchCorrectorValidator)

    def func(self, dc: DataContainer):

        dc.sort(_sample_order, "samples")

        def invalid_batch_aux(x):
            n_min = self.params["n_min"]
            qc_classes = self.params["corrector_classes"]
            return x.isin(qc_classes).sum() < n_min

        ps = self.params["process_classes"]
        ps = [x for x in ps if x not in self.params["corrector_classes"]]
        self.params["process_classes"] = ps
        low_qc_batch = (dc.classes
                        .groupby(dc.batch)
                        .apply(invalid_batch_aux))
        low_qc_batch = low_qc_batch[low_qc_batch].index

        min_corr_order = batch_ext(dc.order, dc.batch, dc.classes,
                                   self.params["corrector_classes"], "min")
        max_corr_order = batch_ext(dc.order, dc.batch, dc.classes,
                                   self.params["corrector_classes"], "max")
        min_proc_order = batch_ext(dc.order, dc.batch, dc.classes,
                                   self.params["process_classes"], "min")
        max_proc_order = batch_ext(dc.order, dc.batch, dc.classes,
                                   self.params["process_classes"], "max")

        invalid_batches = ((min_corr_order > min_proc_order)
                           | (max_corr_order < max_proc_order))
        invalid_batches = invalid_batches[invalid_batches].index
        invalid_batches = invalid_batches.union(low_qc_batch)
        invalid_samples = dc.batch[dc.batch.isin(invalid_batches)].index
        return invalid_samples


class _BatchPrevalenceChecker(Processor):
    """
    Check prevalence of Corrector samples. To be used as a part of the batch
    correction pipeline.

    Prevalence of the corrector samples is checked in a similar way as the
    one described in BatchSchemeChecker.
    Start blocK: features must be detected in at least one sample on the
    starting block.
    middle block: features must be detected in at least n_min - 2 samples
    end block: features must be detected in at least one sample on the
    ending block.

    Parameters
    ----------
    n_min: int
        Minimum number of QC samples per batch
    threshold: float
        Minimum intensity to consider a sample detected
    process_classes: list[str], optional
        list of classes used as corrector samples
    corrector_classes: list[str], optional
        list of classes used as process samples
    verbose: bool
    """

    def __init__(self, n_min: int = 4, verbose: bool = False,
                 process_classes: Optional[List[str]] = None,
                 corrector_classes: Optional[List[str]] = None,
                 threshold: float = 0.0):

        req = {"empty": False, "missing": False, "order": True, "batch": True}
        (super(_BatchPrevalenceChecker, self)
         .__init__(axis="features", mode="filter", verbose=verbose,
                   requirements=req))
        self.name = "Batch Prevalence Checker"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["n_min"] = n_min
        self.params["threshold"] = threshold
        self._default_process = _sample_type
        self._default_correct = _qc_type
        validation.validate(self.params, validation.batchCorrectorValidator)

    def func(self, dc: DataContainer):
        ps = self.params["process_classes"]
        ps = [x for x in ps if x not in self.params["corrector_classes"]]
        self.params["process_classes"] = ps
        res = check_qc_prevalence(dc.data_matrix, dc.order, dc.batch,
                                  dc.classes, self.params["corrector_classes"],
                                  self.params["process_classes"],
                                  threshold=self.params["threshold"],
                                  min_n_qc=self.params["n_min"])
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
                 process_qc: bool = True,
                 verbose: bool = False):

        super(_BatchCorrectorProcessor, self).__init__(axis=None,
                                                       mode="transform",
                                                       verbose=verbose,
                                                       requirements={
                                                          "empty": False,
                                                          "missing": False,
                                                          "order": True,
                                                          "batch": True}
                                                       )
        self.name = "Batch Corrector"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["interpolator"] = interpolator
        self.params["n_qc"] = n_qc
        self.params["frac"] = frac
        self.params["process_qc"] = process_qc
        self._default_process = _sample_type
        self._default_correct = _qc_type
        validation.validate(self.params, validation.batchCorrectorValidator)

    def func(self, dc: DataContainer):
        dc.data_matrix = \
            interbatch_correction(dc.data_matrix, dc.order, dc.batch,
                                  dc.classes, **self.params)


@register
class BatchCorrector(Pipeline):
    r"""
    Correct systematic bias along samples due to variation in instrumental
    response

    Notes
    -----
    The correction is applied as described by Broadhurst in [1]. Using QC
    samples a correction is generated for each feature in the following way:
    The signal of a Quality control can be described in terms of three
    components: a mean value, a systematic bias f and error.

    .. math::
       m_{i} = \bar{m_{i}} + f(t) + \epsilon

    f(t) is estimated after mean subtraction using Locally weighted scatterplot
    smoothing (LOESS). The optimal fraction of samples for each local
    regression is found using LOOCV.

    Before applying batch correction, the QC template is checked and samples
    that cannot be corrected are removed. A study sample is valid if it is
    surrounded by a quality control sample. This is necessary because to
    estimate the correction an interpolation step is performed. After checking
    the QC template, each feature is checked to see if the minimum number of
    QC samples necessary to perform LOESS are available. This is done counting
    the number of QC samples above a `threhold`. A minimum of four QC features
    is necessary to perform LOESS, but in order to estimate the `frac` parameter
    with LOOCV, we suggest a minimum of six QC samples.
    To estimate the mean during batch correction, all of the QC samples can be
    used or only a subset. It can be argued that during the first QC samples the
    mean value is going to be closer to the real value because there's a small
    contribution from carryover. Taking this into account, the `n_qc` parameter
    controls the number of first n QC samples used to estimate the mean of the
    batch.

    References
    ----------
    .. [1] D Broadhurst *et al*, "Guidelines and considerations for the use of
        system suitability and quality control samples in mass spectrometry
        assays applied in untargeted clinical metabolomic studies.",
        Metabolomics, 2018;14(6):72. doi: 10.1007/s11306-018-1367-3

    """
    def __init__(self, corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 n_min: int = 6, frac: Optional[float] = None,
                 interpolator: str = "splines", threshold: float = 0,
                 verbose: bool = False,
                 n_qc: Optional[int] = None):
        """
        BatchCorrector Constructor.

        Parameters
        ----------
        corrector_classes : list[str], optional
            list of classes used to generate the correction. If None uses
            QC sample types from the mapping.
        process_classes : list[str], optional
            list of classes used to correct. If None uses
            sample sample types from the mapping.
        n_min : int
            Minimum number of QC samples in the batch to apply the correction.
            Features with a lower number of QC samples are removed.
        frac : float, optional
            frac parameter of the LOESS model. If None, the best value for each
            feature is estimated using LOOCV.
        interpolator : {"splines", "linear"}
            Interpolator used to estimate the correction for each sample.
        threshold : float
            Minimum value to consider a QC detected. Used to count the number
            of QC samples available for each feature.
        verbose : bool
            If True a message is shown after processing the data matrix.
        n_qc : int, optional
            The number of first QC samples used to estimate the expected
            value for each feature in the QC. If None uses all samples.

        """
        checker = \
            _BatchDesignChecker(n_min=n_min, verbose=verbose,
                                process_classes=process_classes,
                                corrector_classes=corrector_classes)
        prevalence = \
            _BatchPrevalenceChecker(n_min=n_min, verbose=verbose,
                                    process_classes=process_classes,
                                    corrector_classes=corrector_classes,
                                    threshold=threshold)
        corrector = \
            _BatchCorrectorProcessor(corrector_classes=corrector_classes,
                                     process_classes=process_classes, frac=frac,
                                     interpolator=interpolator, verbose=verbose,
                                     n_qc=n_qc)
        pipeline = [checker, prevalence, corrector]
        super(BatchCorrector, self).__init__(pipeline, verbose=verbose)
        self.name = "Batch Corrector"


def read_config(path):
    with open(path) as fin:
        config = yaml.load(fin, Loader=yaml.UnsafeLoader)
    return config


def pipeline_from_list(param_list: list, verbose=False):
    procs = list()
    for d in param_list:
        procs.append(filter_from_dictionary(d))
    pipeline = Pipeline(procs, verbose)
    return pipeline


def pipeline_from_yaml(path):
    d = read_config(path)
    filters_list = d["Pipeline"]
    pipeline = pipeline_from_list(filters_list)
    return pipeline


def filter_from_dictionary(d):
    filt = None
    for name, params in d.items():
        filt = FILTERS[name](**params)
    return filt


def sample_name_from_path(path):
    fname = os.path.split(path)[1]
    sample_name = os.path.splitext(fname)[0]
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


_requirements_error = {"empty": data_container.EmptyDataContainerError,
                       "missing": MissingValueError,
                       _qc_type: MissingMappingInformation,
                       _blank_type: MissingMappingInformation,
                       "batch": data_container.BatchInformationError,
                       "order": data_container.RunOrderError}


# TODO : refactor Processor using the following func prototype:
#   func(dc, **self.params)
