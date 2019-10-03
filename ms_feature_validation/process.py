"""
Implementations of DataContainer, Processor and Pipeline Classes for automatic
correction and validation of LC-MS metabolomics data.
"""


from . import filter_functions
from . import utils
from . import validation
import numpy as np


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
        validation.validate_data_container(data_matrix_df,
                                           feature_definitions_df,
                                           sample_information_df,
                                           data_path)
        self.data_matrix = data_matrix_df
        self.feature_definitions = feature_definitions_df
        self.sample_information = sample_information_df
        if data_path is not None:
            path_mapping = utils.sample_to_path(data_matrix_df.index, data_path)
            self.sample_information["raw path"] = \
                self.sample_information.index.map(path_mapping)

    def get_available_samples(self):
        return self.sample_information["raw path"].dropna()

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

    def set_raw_path(self, path):
        mapper = utils.sample_to_path(self.sample_information, path)
        self.sample_information["raw path"] = \
            self.sample_information.index.map(mapper)

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
            return filter_functions.cv(self.data_matrix).mean()
        elif isinstance(classes, str):
            classes = [classes]
        classes_mask = self.get_classes().isin(classes)
        return filter_functions.cv(self.data_matrix[classes_mask]).mean()

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
        self.feature_definitions["mz_cluster"] = utils.cluster(self.get_mz(),
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
        self.remove = list()

    def func(self, func):
        raise NotImplementedError

    def process(self, dc):
        self._record_metrics(dc, "before")
        if self.mode == "filter":
            self.remove = self.func(dc)
            dc.remove(self.remove, self.axis)
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
        self.verbose = verbose

    def process(self, dc):
        self._record_metrics(dc, "before")
        for x in self.processors:
            if self.verbose:
                x.verbose = True
            x.process(dc)
        self._record_metrics(dc, "after")


# Filters and Correctors implementation

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


def _validate_pipeline(t):
    if not isinstance(t, (list, tuple)):
        t = [t]
    for filt in t:
        if not isinstance(filt, (Processor, Pipeline)):
            msg = ("elements of the Pipeline must be",
                   "instances of Filter, DataCorrector or another Pipeline.")
            raise TypeError(msg)

# TODO: posible acortamiento de nombres: data_matrix: data,
#  sample_information: sample, feature_definitions: features.
