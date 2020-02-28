"""
Filter objects to curate data
"""


from .data_container import DataContainer
from ._names import *
from . import filter_functions
from . import utils
from . import fileio
import yaml
import os.path
import pandas as pd
from typing import Optional, List, Union, Callable
Number = Union[float, int]


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
    """
    def __init__(self, mode: str, axis: Optional[str] = None,
                 verbose: bool = False, default_process=None,
                 default_correct=None):
        super(Processor, self).__init__()
        self.mode = mode
        self.axis = axis
        self.verbose = verbose
        self.remove = list()
        self.params = dict()
        self._default_process = default_process
        self._default_correct = default_correct

    def func(self, func):
        raise NotImplementedError

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
    Applies a series of Filters and Correctors to a DataContainer

    Attributes
    ----------
    processors: list[Processors]
        A list of processors to apply
    verbose: bool
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
class DuplicateAverager(Processor):
    """
    A filter that averages duplicates
    """
    def __init__(self, process_classes: Optional[List[str]] = None):
        super(DuplicateAverager, self).__init__(axis=None, mode="transform")
        self.params["process_classes"] = process_classes

    def func(self, dc: DataContainer):
        if self.params["process_classes"] is None:
            self.params["process_classes"] = dc.mapping[_sample_type]
        dc.data_matrix = \
            filter_functions.average_replicates(dc.data_matrix, dc.id,
                                                dc.classes, **self.params)
        dc.sample_metadata = (dc.sample_metadata
                              .loc[dc.data_matrix.index, :])


@register
class ClassRemover(Processor):
    """
    A filter to remove samples of a given class
    """
    def __init__(self, classes: List[str]):
        super(ClassRemover, self).__init__(axis="samples", mode="filter")
        if classes is None:
            self.params["classes"] = list()
        else:
            self.params["classes"] = classes

    def func(self, dc):
        remove_samples = dc.classes.isin(self.params["classes"])
        remove_samples = remove_samples[remove_samples].index
        return remove_samples


@register
class BlankCorrector(Processor):
    """
    Corrects values using blank information
    """
    def __init__(self, corrector_classes: Optional[List[str]] = None,
                 process_classes: Optional[List[str]] = None,
                 mode: Union[str, Callable] = "lod", verbose=False):
        """
        Correct sample values using blank samples.

        Parameters
        ----------
        corrector_classes: list[str], optional
            Classes used to generate the blank correction. If None, uses blank
            samples defined on the DataContainer mapping attribute.
        process_classes:  list[str], optional
            Classes to be corrected. If None, uses all classes listed on the
            DataContainer mapping attribute.
        mode: {"mean", "max", "lod", "loq"}, function.
            Function used to generate the blank correction.
        verbose: bool
        """
        super(BlankCorrector, self).__init__(axis=None, mode="transform",
                                             verbose=verbose)
        self.name = "Blank Corrector"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["mode"] = mode
        self._default_process = _sample_type
        self._default_correct = _blank_type

    def func(self, dc):
        self.set_default_sample_types(dc)
        # TODO: view if there are side effects from modifying params
        dc.data_matrix = filter_functions.correct_blanks(dc.data_matrix,
                                                         dc.classes,
                                                         **self.params)


@register
class BatchCorrector(Processor):
    """
    Corrects instrumental drift between in a batch.
    """
    def __init__(self, corrector_classes=None, process_classes=None,
                 interpolator="splines", verbose=False, **kwargs):
        super(BatchCorrector, self).__init__(axis=None, mode="transform",
                                             verbose=verbose)
        self.name = "Batch Corrector"

        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["interpolator"] = interpolator
        self.params = {**self.params, **kwargs}
        self._default_process = _sample_type
        self._default_correct = _qc_type
        # TODO: chequear la linea de kwargs

    def func(self, dc: DataContainer):
        self.set_default_sample_types(dc)
        print(self.params)
        dc.data_matrix = filter_functions.batch_correction(dc.data_matrix,
                                                           dc.order,
                                                           dc.batch,
                                                           dc.classes,
                                                           **self.params)


@register
class PrevalenceFilter(Processor):
    """
    Remove Features with a low number of occurrences. The prevalence is defined
    as the fraction of samples where the signal is observed.

    Parameters
    ----------
    process_classes: List[str], optional
        Classes used to compute prevalence. If None, classes are obtained from
        sample classes in the DataContainer mapping.
    lb: float
        Lower bound of prevalence.
    ub: float
        Upper bound of prevalence.
    threshold: float
        Minimum intensity to consider a feature as detected.
    intraclass: bool
        Whether to evaluate a global prevalence or a per class prevalence.
        If intraclass is True, prevalence is computed for each class and
        features in which the prevalence is outside the selected bounds for all
        classes are removed.

    """
    def __init__(self, process_classes: Optional[List[str]] = None,
                 lb: Number = 0.5, ub: Number = 1,
                 intraclass: bool = True, verbose: bool = False,
                 threshold: Number = 0):
        super(PrevalenceFilter, self).__init__(axis="features", mode="filter",
                                               verbose=verbose)
        self.name = "Prevalence Filter"
        self.params["process_classes"] = process_classes
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["intraclass"] = intraclass
        self.params["threshold"] = threshold
        self._default_correct = _sample_type
        self._default_process = _sample_type

    def func(self, dc):
        self.set_default_sample_types(dc)
        dr = dc.metrics.detection_rate(intraclass=self.params["intraclass"],
                                       threshold=self.params["threshold"])
        lb = self.params["lb"]
        ub = self.params["ub"]
        return filter_functions.get_outside_bounds_index(dr, lb, ub)


@register
class VariationFilter(Processor):
    """
    Remove samples with high coefficient of variation.
    """
    def __init__(self, lb=0, ub=0.25, process_classes=None, robust=False,
                 intraclass=True, verbose=False):
        super(VariationFilter, self).__init__(axis="features", mode="filter",
                                              verbose=verbose)
        self.name = "Variation Filter"
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["process_classes"] = process_classes
        self.params["robust"] = robust
        self.params["intraclass"] = intraclass
        self._default_process = _qc_type

    def func(self, dc: DataContainer):
        self.set_default_sample_types(dc)
        lb = self.params["lb"]
        ub = self.params["ub"]
        variation = dc.metrics.cv(intraclass=self.params["intraclass"],
                                  robust=self.params["robust"])
        variation = variation.loc[self.params["process_classes"], :]
        return filter_functions.get_outside_bounds_index(variation, lb, ub)


# @register
# class ChromatogramMaker(Processor):
#     """
#     Computes chromatograms for all samples using available raw data.
#     """
#     def __init__(self, tolerance=0.005, cluster_tolerance=0.0002,
#                  verbose=False):
#         super(ChromatogramMaker, self).__init__("add")
#         self.name = "Chromatogram Maker"
#         self.params["tolerance"] = tolerance
#         self.params["cluster_tolerance"] = cluster_tolerance
#         self.verbose = verbose
#
#     def func(self, dc):
#         dc.cluster_mz(self.params["cluster_tolerance"])
#         mean_mz_cluster = utils.mean_cluster_value(dc.get_mz(),
#                                                    dc.get_mz_cluster())
#         chromatogram_dict = dict()
#         samples = dc.get_available_samples()
#         for k, sample in enumerate(samples):
#             if self.verbose:
#                 msg = "Computing EICs for {} ({}/{})"
#                 msg = msg.format(sample, k + 1, len(samples))
#                 print(msg)
#             reader = fileio.reader(sample)
#             c = fileio.chromatogram(reader,
#                                     mean_mz_cluster,
#                                     tolerance=self.params["tolerance"])
#             sample_name = sample_name_from_path(sample)
#             chromatogram_dict[sample_name] = c
#         dc.chromatograms = chromatogram_dict


class BatchSchemeChecker(Processor):
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
        super(BatchSchemeChecker, self).__init__(axis="samples",
                                                 mode="filter",
                                                 verbose=verbose)
        self.name = "Batch Scheme Checker"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["n_min"] = n_min
        self._default_process = _sample_type
        self._default_correct = _qc_type

    def func(self, dc):

        def invalid_batch_aux(x):
            n_min = self.params["n_min"]
            qc_classes = self.params["corrector_classes"]
            return x.isin(qc_classes).sum() < n_min

        self.set_default_sample_types(dc)
        low_qc_batch = (dc.classes
                        .groupby(dc.batch)
                        .apply(invalid_batch_aux))
        low_qc_batch = low_qc_batch[low_qc_batch].index

        min_corr_order = _get_ext_per_batch(dc.order, dc.batch, dc.classes,
                                            self.params["corrector_classes"],
                                            "min")
        max_corr_order = _get_ext_per_batch(dc.order, dc.batch, dc.classes,
                                            self.params["corrector_classes"],
                                            "max")
        min_proc_order = _get_ext_per_batch(dc.order, dc.batch, dc.classes,
                                            self.params["process_classes"],
                                            "min")
        max_proc_order = _get_ext_per_batch(dc.order, dc.batch, dc.classes,
                                            self.params["process_classes"],
                                            "max")

        invalid_batches = ((min_corr_order > min_proc_order)
                           | (max_corr_order < max_proc_order))
        invalid_batches = invalid_batches[invalid_batches].index
        invalid_batches = invalid_batches.union(low_qc_batch)
        invalid_samples = dc.batch[dc.batch.isin(invalid_batches)].index
        return invalid_samples


class BatchPrevalenceChecker(Processor):
    """
    Check prevalence of Corrector samples. To be used as a part of the batch
    correction pipeline.

    Prevalence of the corrector samples is checked in a similiar way as the
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
        super(BatchPrevalenceChecker, self).__init__(axis="features",
                                                     mode="filter",
                                                     verbose=verbose)
        self.name = "Batch Prevalence Checker"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["n_min"] = n_min
        self.params["threshold"] = threshold
        self._default_process = _sample_type
        self._default_correct = _qc_type

    def func(self, dc: DataContainer):
        self.set_default_sample_types(dc)
        return check_qc_prevalence(dc.data_matrix, dc.order, dc.batch,
                                   dc.classes, self.params["corrector_classes"],
                                   self.params["process_classes"],
                                   threshold=self.params["threshold"],
                                   min_n_qc=self.params["n_min"])


def merge_data_containers(dcs):
    """
    Applies concat on `data_matrix`, `sample_information` and
    `feature_definitions`.
    Parameters
    ----------
    dcs : Iterable[DataContainer]

    Returns
    -------
    merged_dc : DataContainer
    """
    dms = [x.data_matrix for x in dcs]
    sis = [x.sample_metadata for x in dcs]
    fds = [x.feature_metadata for x in dcs]
    dm = pd.concat(dms)
    si = pd.concat(sis)
    fd = pd.concat(fds)
    return DataContainer(dm, fd, si)


def data_container_from_excel(excel_file):
    data_matrix = pd.read_excel(excel_file,
                                sheet_name="data_matrix",
                                index_col="sample")
    sample_information = pd.read_excel(excel_file,
                                       sheet_name="sample_information",
                                       index_col="sample")
    feature_definitions = pd.read_excel(excel_file,
                                        sheet_name="feature_definitions",
                                        index_col="feature")
    data_container = DataContainer(data_matrix,
                                   feature_definitions,
                                   sample_information)
    return data_container


def read_config(path):
    with open(path) as fin:
        config = yaml.load(fin, Loader=yaml.UnsafeLoader)
    return config


def pipeline_from_list(l, verbose=False):
    procs = list()
    for d in l:
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


def _get_ext_per_batch(order: pd.Series, batch: pd.Series, classes: pd.Series,
                       class_list: List[str], ext: str):
    """
    get minimum/maximum order of samples of classes in class_list. Auxiliar
    function to be used with BatchChecker / FeatureCheckerBatchCorrection
    Parameters
    ----------
    order: pandas.Series
        run order
    batch: pandas.Series
        batch number
    classes: pandas.Series
        sample classes
    class_list: list[str]
        classes to be considererd
    ext: {"min", "max"}
        Search for the min/max order in each batch.

    Returns
    -------
    pd.Series with the corresponding min/max order with batch as index.
    """
    func = {"min": lambda x: x.min(), "max": lambda x: x.max()}
    func = func[ext]

    ext_order = (order
                 .groupby([classes, batch])
                 .apply(func)
                 .reset_index()
                 .groupby(classes.name)
                 .filter(lambda x: x.name in class_list)
                 .groupby(batch.name)
                 .apply(func)[order.name])

    # ext_order = (order[mask]
    #              .groupby([classes[mask], batch[mask]])
    #              .pipe(func)
    #              .reset_index()
    #              .pivot(index=classes.name, columns=batch.name,
    #                     values=order.name)
    #              .min()     # min used to convert to pandas Series.
    #              .astype(int))
    return ext_order


def check_qc_prevalence(data_matrix, order, batch, classes, qc_classes,
                        sample_classes,
                        threshold=0, min_n_qc=4):
    min_qc_order = _get_ext_per_batch(order, batch, classes, qc_classes, "min")
    min_sample_order = _get_ext_per_batch(order, batch, classes,
                                          sample_classes, "min")
    max_qc_order = _get_ext_per_batch(order, batch, classes, qc_classes, "max")
    max_sample_order = _get_ext_per_batch(order, batch, classes,
                                          sample_classes, "max")
    batches = batch[classes.isin(qc_classes)].unique()
    valid_features = data_matrix.columns

    for k_batch in batches:
        # feature check is done for each batch in three parts:
        # | start block | middle block      | end block |
        #   q   q         ssss q ssss q ssss  q  q
        #  where q is a qc sample and s is a biological sample
        # in the start block, a feature is valid if is detected
        # in at least one sample of the block
        # in the middle block, a feature is valid if the number
        # of qc samples where the feature was detected is greater
        # than the total number of qc samples in the block minus the
        # n_missing parameter
        # in the end block the same strategy applied in the start
        # block is used.
        # A feature is considered valid only if is valid in the totallity
        # of the batches.

        # start block check
        start_block_qc_samples = (order[(order >= min_qc_order[k_batch])
                                        & (order < min_sample_order[k_batch])
                                        & classes.isin(qc_classes)]
                                  .index)
        start_block_valid_features = \
            (data_matrix.loc[start_block_qc_samples] > threshold).any()
        start_block_valid_features = \
            start_block_valid_features[start_block_valid_features].index
        valid_features = valid_features.intersection(start_block_valid_features)

        # middle block check
        middle_block_qc_samples = (order[(order > min_sample_order[k_batch])
                                         & (order < max_sample_order[k_batch])
                                         & classes.isin(qc_classes)]
                                   .index)
        middle_block_valid_features = ((data_matrix
                                        .loc[middle_block_qc_samples] > threshold)
                                       .sum() >= min_n_qc)
        middle_block_valid_features = \
            middle_block_valid_features[middle_block_valid_features].index

        valid_features = valid_features.intersection(
            middle_block_valid_features)

        print(valid_features)
        # end block check
        end_block_qc_samples = (order[(order > max_sample_order[k_batch])
                                      & (order <= max_qc_order[k_batch])
                                      & classes.isin(qc_classes)]
                                .index)
        end_block_valid_features = \
            (data_matrix.loc[end_block_qc_samples] > threshold).any()
        end_block_valid_features = end_block_valid_features[
            end_block_valid_features].index
        valid_features = valid_features.intersection(end_block_valid_features)

    invalid_features = data_matrix.columns.difference(valid_features)
    return invalid_features
