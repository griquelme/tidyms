"""
Implementations of DataContainer, Processor and Pipeline Classes for automatic
correction and validation of LC-MS metabolomics data.
"""


from . import filter
from . import utils
from . import fileio
from . import peaks
import pandas as pd
import numpy as np
import yaml
import os.path


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


class DataContainer(object):
    """
    A container class for Metabolomics Data.

    Attributes
    ---------
    data_matrix : pandas.DataFrame.
                  DataFrame with sample names on indices and features on the
                  columns.
    sample_information : pandas.DataFrame.
                         DataFrame with sample names as indices. class is a
                         required column.
    feature_definitions : pandas.DataFrame.
                          DataFrame with features names as indices. mz and rt
                          are required columns.
    data_path : str.
        Path to raw data directory.
    """

    def __init__(self, data_matrix_df, feature_definitions_df,
                 sample_information_df, data_path=None):
        self.data_matrix = data_matrix_df
        self.feature_definitions = feature_definitions_df
        self.sample_information = sample_information_df
        self.data_path = data_path

    def get_raw_path(self):
        available_files = list()
        for sample in self.sample_information.index:
            sample_file = os.path.join(self.data_path, sample + ".mzML")
            if os.path.isfile(sample_file):
                available_files.append(sample_file)
        return available_files

    def is_valid_class_name(self, class_names):
        valid_classes = np.isin(class_names, self.get_classes().unique())
        return np.all(valid_classes)

    def remove(self, remove, axis):
        """
        Remove selected features / samples

        Parameters
        ----------
        remove : list[str]
                   Feature/Samples names to remove.
        axis : str
               "features", "samples". axis to remove from
        """
        if axis == "features":
            self.data_matrix.drop(columns=remove, inplace=True)
            self.feature_definitions.drop(index=remove, inplace=True)
        elif axis == "samples":
            self.data_matrix.drop(index=remove, inplace=True)
            self.sample_information.drop(index=remove, inplace=True)

    def select(self, selection, axis):
        """
        Return a selection of the DataContainer
        Parameters
        ----------
        selection : list[str]
                    features/samples to select
        axis : str
               "features" or "samples".
        Returns
        -------
        data_selection : DataContainer
                         DataContainer with selection.
        """
        if axis == "samples":
            dm_selection = self.data_matrix.loc[selection, :]
            si_selection = self.sample_information.loc[selection, :]
            fd_selection = self.feature_definitions
        elif axis == "features":
            dm_selection = self.data_matrix.loc[:, selection]
            si_selection = self.sample_information
            fd_selection = self.feature_definitions.loc[selection, :]
        else:
            raise ValueError("axis should be `columns` or `features`")

        data_selection = DataContainer(dm_selection, fd_selection, si_selection)
        return data_selection

    def get_classes(self):
        return self.sample_information["class"]

    def get_id(self):
        return self.sample_information["id"]

    def get_batches(self):
        try:
            return self.sample_information["batch"]
        except KeyError:
            raise BatchInformationError("No batch information available.")

    def get_run_order(self):
        try:
            return self.sample_information["order"]
        except KeyError:
            raise RunOrderError("No run order information available")

    def get_n_features(self):
        return self.data_matrix.shape[1]

    def get_mean_cv(self, classes=None):
        if classes is None:
            return filter.cv(self.data_matrix).mean()
        elif isinstance(classes, str):
            classes = [classes]
        classes_mask = self.get_classes().isin(classes)
        return filter.cv(self.data_matrix[classes_mask]).mean()

    def get_mz(self):
        return self.feature_definitions["mz"]

    def get_rt(self):
        return self.feature_definitions["rt"]

    def cluster_mz(self, tolerance=0.0002):
        """
        Groups features with similar mz to reduce the number of calculated EICs.

        Parameters
        ----------
        tolerance : float
        """
        self.feature_definitions["mz_cluster"] = peaks.cluster(self.get_mz(),
                                                               tolerance)
    def get_mz_cluster(self):
        return self.feature_definitions["mz_cluster"]


class Reporter(object):
    """
    Abstract class with methods to report metrics.
    """
    def __init__(self):
        self.metrics = dict()
        self.params = dict()
        self.name = None

    def _record_metrics(self, dc, name):
        metrics = dict()
        metrics["cv"] = dc.get_mean_cv(self.params.get("process_classes"))
        metrics["features"] = dc.get_n_features()
        self.metrics[name] = metrics

    def report(self):
        removed_features = (self.metrics["before"]["features"]
                            - self.metrics["after"]["features"])
        cv_reduction = (self.metrics["before"]["cv"]
                        - self.metrics["after"]["cv"]) * 100
        msg = "Applying {}: {} features removed. Mean CV reduced by {:.2f} %."
        print(msg.format(self.name, removed_features, cv_reduction))


class Processor(Reporter):
    """
    Abstract class to process DataContainer Objects.
    """
    def __init__(self, mode=None, axis=None, verbose=None):
        super(Processor, self).__init__()
        self.mode = mode
        self.axis = axis
        self.verbose = verbose

    def func(self, func):
        raise NotImplementedError

    def process(self, dc):
        self._record_metrics(dc, "before")
        if self.mode == "filter":
            remove = self.func(dc)
            dc.remove(remove, self.axis)
        elif self.mode == "add":
            self.func(dc)
        elif self.mode == "correction":
            self.func(dc)
        self._record_metrics(dc, "after")
        if self.verbose:
            self.report()


class Pipeline(Reporter):
    """
    Applies a series of Filters and Correctors to a DataContainer
    """
    def __init__(self, processors, verbose=False):
        _validate_pipeline(processors)
        super(Pipeline, self).__init__()
        self.processors = list(processors)
        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        for processor in self.processors:
            processor.verbose = value

    def process(self, dc):
        self._record_metrics(dc, "before")
        for x in self.processors:
            x.process(dc)
        self._record_metrics(dc, "after")


# Filters and Correctors implementation


@register
class DuplicateAverager(Processor):
    """
    A filter that averages duplicates
    """
    def __init__(self, process_classes):
        super(DuplicateAverager, self).__init__(axis=None, mode="correction")
        self.params["process_classes"] = process_classes

    def func(self, dc):
        dc.data_matrix = filter.replicate_averager(dc.data_matrix,
                                                   dc.get_id(),
                                                   dc.get_classes(),
                                                   **self.params)
        dc.sample_information = (dc.sample_information
                                     .loc[dc.data_matrix.index,:])


@register
class ClassRemover(Processor):
    """
    A filter to remove samples of a given class
    """
    def __init__(self, classes=None):
        super(ClassRemover, self).__init__(axis="samples", mode="filter")
        self.param["classes"] = classes

    def func(self, dc):
        remove_samples = dc.get_classes().isin(self.param["classes"])
        return remove_samples.index


@register
class BlankCorrector(Processor):
    """
    Corrects values using blank information
    """
    def __init__(self, corrector_classes=None, process_classes=None,
                 mode="mean", blank_relation=3, verbose=False):
        super(BlankCorrector, self).__init__(axis=None, mode="correction",
                                             verbose=verbose)
        self.name = "Blank Correction"
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["mode"] = mode
        self.params["blank_relation"] = blank_relation

    def func(self, dc):
        dc.data_matrix = filter.blank_correction(dc.data_matrix,
                                                 dc.get_classes(),
                                                 **self.params)


@register
class BatchCorrector(Processor):
    """
    Corrects instrumental drift between in a batch.
    """
    def __init__(self, corrector_classes=None, process_classes=None,
                 mode="splines", verbose=False, **kwargs):
        super(BatchCorrector, self).__init__(axis=None, mode="correction",
                                             verbose=verbose)
        self.params["corrector_classes"] = corrector_classes
        self.params["process_classes"] = process_classes
        self.params["mode"] = mode
        self.params = {**self.params, **kwargs}
        # TODO: chequear la linea de kwargs

    def func(self, dc):
        dc.data_matrix = filter.batch_correction(dc.data_matrix,
                                                 dc.get_run_order(),
                                                 dc.get_classes(),
                                                 **self.params)


@register
class PrevalenceFilter(Processor):
    """
    Remove Features with a low number of occurrences.
    """
    def __init__(self, process_classes=None, lb=0.5, ub=1,
                 intraclass=True, verbose=False):
        super(PrevalenceFilter, self).__init__(axis="features", mode="filter",
                                               verbose=verbose)
        self.name = "Prevalence Filter"
        self.params["process_classes"] = process_classes
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["intraclass"] = intraclass

    def func(self, dc):
        return filter.prevalence_filter(dc.data_matrix, dc.get_classes(),
                                        **self.params)


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

    def func(self, dc):
        return filter.variation_filter(dc.data_matrix, dc.get_classes(),
                                       **self.params)


@register
class ChromatogramMaker(Processor):
    """
    Computes chromatograms for all samples using available raw data.
    """
    def __init__(self, tolerance=0.005, cluster_tolerance=0.0002,
                 verbose=False):
        super(ChromatogramMaker, self).__init__("add")
        self.name = "Chromatogram Maker"
        self.params["tolerance"] = tolerance
        self.params["cluster_tolerance"] = cluster_tolerance
        self.verbose = verbose

    def func(self, dc):
        dc.cluster_mz(self.params["cluster_tolerance"])
        mean_mz_cluster = peaks.mean_cluster_value(dc.get_mz(),
                                                   dc.get_mz_cluster())
        chromatogram_dict = dict()
        samples = dc.get_raw_path()
        for k, sample in enumerate(samples):
            reader = fileio.read(sample)
            c = fileio.chromatogram(reader,
                                    mean_mz_cluster,
                                    tolerance=self.params["tolerance"])
            sample_name = sample_name_from_path(sample)
            chromatogram_dict[sample_name] = c
            if self.verbose:
                msg = "Computing EICs for {} ({}/{})".format(sample, k + 1,
                                                             len(samples))
                print(msg)
        dc.chromatograms = chromatogram_dict


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
    sis = [x.sample_information for x in dcs]
    fds = [x.feature_definitions for x in dcs]
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
    _validate_data_container(data_container)
    return data_container


def read_progenesis(path):
    """
    Read a progenesis file into a DataContainer

    Parameters
    ----------
    path : path to an Progenesis csv output

    Returns
    -------
    dc = DataContainer
    """
    df = pd.read_csv(path, skiprows=2, index_col="Compound")
    df_header = pd.read_csv(path, nrows=2)
    df_header = df_header.fillna(axis=1, method="ffill")
    norm_index = df_header.columns.get_loc("Normalised abundance") - 1
    raw_index = df_header.columns.get_loc("Raw abundance") - 1
    ft_def = df.iloc[:, 0:norm_index]
    data = df.iloc[:, raw_index:(2 * raw_index - norm_index)].T
    sample_info = df_header.iloc[:, raw_index:(2 * raw_index - norm_index)].T
    sample_info.set_index(sample_info.iloc[:, 1], inplace=True)
    sample_info.drop(labels=[1],  axis=1, inplace=True)

    # rename sample info
    sample_info.index.rename("sample", inplace=True)
    sample_info.rename({sample_info.columns[0]: "class"},
                       axis="columns", inplace=True)
    # rename data matrix
    data.index = sample_info.index
    data.columns.rename("feature", inplace=True)
    # rename features def
    ft_def.index.rename("feature", inplace=True)
    ft_def.rename({"m/z": "mz", "Retention time (min)": "rt"},
                  axis="columns",
                  inplace=True)
    dc = DataContainer(data, ft_def, sample_info)
    _validate_data_container(dc)
    return dc


def _validate_pipeline(t):
    if not isinstance(t, (list, tuple)):
        t = [t]
    for filt in t:
        if not isinstance(filt, (Processor, Pipeline)):
            msg = ("elements of the Pipeline must be",
                   "instances of Filter, DataCorrector or another Pipeline.")
            raise TypeError(msg)


def _validate_data_container(data_container):

    # feature names and sample names check
    sample_index_equal = (data_container.data_matrix.index
                          .equals(data_container.sample_information.index))
    if not sample_index_equal:
        msg = "sample_information data_matrix indices should be equal"
        raise ValueError(msg)
    features_equal = (data_container.data_matrix.columns
                      .equals(data_container.feature_definitions.index))
    if not features_equal:
        msg = ("sample_information columns and",
               "feature_definitions should be equal")
        raise ValueError(msg)
    # rt, mz and class information check
    if "mz" not in data_container.feature_definitions.columns:
        raise KeyError("mz values are required for all features")
    if "rt" not in data_container.feature_definitions.columns:
        raise KeyError("rt values are required for all features")
    if "class" not in data_container.sample_information.columns:
        raise KeyError("class information is required for all samples")
    # rt, mz and class and data matrix check:
    if (data_container.feature_definitions["mz"] < 0).any():
        raise ValueError("mz values should be greater than zero")
    if (data_container.feature_definitions["rt"] < 0).any():
        raise ValueError("mz values should be greater than zero")
    if (data_container.data_matrix < 0).any().any():
        msg = "Raw values in data_matrix should be greater than zero"
        raise ValueError(msg)
    # TODO: pensar si siempre seria valido imputar NaN en datamatrix como 0.
    # TODO: separar esta funcion en tres funciones distintas


def read_config(path):
    with open(path) as fin:
        config = yaml.load(fin, Loader=yaml.UnsafeLoader)
    return config


def pipeline_from_list(l):
    pipeline = Pipeline()
    for d in l:
        pipeline.proccesors.append(filter_from_dictionary(d))
    return pipeline


def filter_from_dictionary(d):
    filter = None
    for name, params in d.items():
        filter = FILTERS[name](**params)
    return filter


def sample_name_from_path(path):
    fname = os.path.split(path)[1]
    sample_name = os.path.splitext(fname)[0]
    return sample_name


class BatchInformationError(KeyError):
    """
    Error class when there is no batch information
    """
    pass


class RunOrderError(KeyError):
    """
    Error class raised when there is no run order information
    """
    pass


class InvalidClassName(ValueError):
    """
    Error class raised when using invalid class names
    """


def get_function_parameters(only=None, exclude=None, ignore='self'):
    """Returns a dictionary of the calling functions
       parameter names and values.

       The optional arguments can be used to filter the result:

           only           use this to only return parameters
                          from this list of names.

           exclude        use this to return every parameter
                          *except* those included in this list
                          of names.

           ignore         use this inside methods to ignore
                          the calling object's name. For
                          convenience, it ignores 'self'
                          by default.

    """
    import inspect
    args, varargs, varkw, defaults = \
        inspect.getargvalues(inspect.stack()[1][0])
    if only is None:
        only = args[:]
        if varkw:
            only.extend(defaults[varkw].keys())
            defaults.update(defaults[varkw])
    if exclude is None:
        exclude = []
    exclude.append(ignore)
    return dict([(attrname, defaults[attrname])
        for attrname in only if attrname not in exclude])
# TODO: posible acortamiento de nombres: data_matrix: data,
#  sample_information: sample, feature_definitions: features.
