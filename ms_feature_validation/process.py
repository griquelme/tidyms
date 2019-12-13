# -*- coding: utf-8 -*-
"""
Objects used for automatic curation and validation of LC-MS metabolomics data.

Complete with examples.    
"""


from . import filter_functions
from . import utils
from . import validation
import numpy as np
import pandas as pd


# variables used to name sample information columns
_raw_path = "raw path"
_sample_class = "class"
_sample_id = "id"
_sample_batch = "batch"
_sample_order = "order"
SAMPLE_TYPES = ["sample", "qc", "blank", "suitability", "zero"]

class DataContainer(object):
    """
    A container class for Metabolomics Data.
    
    Consists of three Pandas DataFrames with features values, feature metadata
    and sammple metadata. Index are shared for features and samples
    respectively.
    
    Contains functions to remove samples or features.

    Attributes
    ---------
    data_matrix : pd.DataFrame.
                  Feature values for each measured sample. Each row is a
                  sample and each column is a feature.                  
    sample_information : pd.DataFrame.
                         Metadata for each sample. class is a required column.
    feature_definitions : pd.DataFrame.
                          DataFrame with features names as indices. mz and rt
                          are required columns.
    data_path : str.
        Path to raw data directory.
    mapping : dict.
        maps a sample types to sample classes.
    """

    def __init__(self, data_matrix_df, feature_definitions_df,
                 sample_information_df, data_path=None, mapping=None):
        
        """
        Creates a DataContainer from feature values, features metadata and
        sample metadata.
        
        Atributes
        ---------
        data_matrix_df : pd.DataFrame.
            Feature values for each measured sample. Each row is a sample and
            each column is a feature.                  
        sample_information_df : pd.DataFrame.
            Metadata for each sample. class is a required column.
        feature_definitions_df : pd.DataFrame.
            DataFrame with features names as indices. mz and rt are required
            columns.
        data_path : str.
            path to raw Data. Files must have the same name as each sample.
        mapping : dict or None
            if dict, set each sample class to sample type.
        """
        validation.validate_data_container(data_matrix_df,
                                           feature_definitions_df,
                                           sample_information_df,
                                           data_path)
        self.data_matrix = data_matrix_df
        self.feature_definitions = feature_definitions_df
        self.sample_information = sample_information_df
        self.data_path = data_path
        self.mapping = mapping

    
    @property
    def data_path(self):
        """str : directory where raw data is stored."""
        return self._data_path
    
    @data_path.setter
    def data_path(self, path):
        """
        sets raw data path, search for available samples and adds them to
        sample information.
        """
        if path is not None:
            path_mapping = utils.sample_to_path(self.data_matrix.index, path)
            self.sample_information[_raw_path] = \
                self.sample_information.index.map(path_mapping)
            self._data_path = path
        else:
            self._data_path = None
            
    
    @property
    def mapping(self):
        """
        dict : Set the sample type of a sample_classes. keys must be one of
        the following: {'qc', 'blank', 'zero', 'sample', 'suitability'}
        """
        return self._mapping
    
    @mapping.setter
    def mapping(self, mapping):
        
        
        if (mapping is None) or (not mapping):
            self._mapping = {"qc": None, "blank": None, "sample": None,
                             "suitability": None, "zero": None}
        else:
            # check valid sample types
            if not set(SAMPLE_TYPES).issuperset(set(mapping)):
                msg = "keys should be one of the following: "
                msg += str(SAMPLE_TYPES).strip("[]") + "."
                raise ValueError(msg)
            for k in SAMPLE_TYPES:
                c = mapping.setdefault(k)
                if (not self.is_valid_class_name(c)) and (c is not None):
                    msg = "one of the following is an invalid class:"
                    msg += str(c).strip("[]") + "."
                    raise ValueError(msg)
            self._mapping = mapping

    
    @property
    def id(self):
        """pd.Series[str] : name id of each sample."""
        return self.sample_information[_sample_id]

    @id.setter
    def id(self, id):
        self.sample_information[_sample_id] = id
        
    
    @property
    def classes(self):
        """pd.Series[str] : class of each sample."""
        return self.sample_information[_sample_class]
    
    @classes.setter
    def classes(self, classes):
        self.sample_information[_sample_class] = classes
    
    @property
    def batch(self):
        """pd.Series[str] or pd.Series[int]. Batch identification"""
        try:
            return self.sample_information[_sample_batch]
        except KeyError:
            raise BatchInformationError("No batch information available.")
            
    @batch.setter
    def batch(self, batch):
        self.sample_information[_sample_batch] = batch
    
    @property
    def order(self):
        """pd.Series[int] : order of analysis of samples"""
        try:
            return self.sample_information[_sample_order]
        except KeyError:
            raise RunOrderError("No run order information available")
    
    @order.setter
    def order(self, order):
        self.sample_information[_sample_order] = order    

    def get_available_samples(self):
        """
        Returns the absolute path for each raw data file present in
        data_path.
        
        Returns
        -------
        available_samples : pd.Series
            Pandas series with absolute path for each available file.
        """
        available_samples = self.sample_information[_raw_path].dropna()
        return available_samples

    def is_valid_class_name(self, class_name):
        """
        Check if at least one sample class is`class_name`.
        
        Atributes
        ---------
        class_name : str or Iterable[str]
        
        Returns
        -------
        is_valid : bool
        """
        valid_classes = np.isin(class_name, self.classes.unique())
        is_valid = np.all(valid_classes)
        return is_valid

    def remove(self, remove, axis):
        """
        Remove selected features / samples

        Parameters
        ----------
        remove : list[str]
                   Feature / Sample names to remove.
        axis : str
               "features", "samples". axis to remove from
        """
        if axis == "features":
            self.data_matrix.drop(columns=remove, inplace=True)
            self.feature_definitions.drop(index=remove, inplace=True)
        elif axis == "samples":
            self.data_matrix.drop(index=remove, inplace=True)
            self.sample_information.drop(index=remove, inplace=True)
        else:
            raise ValueError("axis should be `columns` or `features`")
    
    def diagnose(self):
        """
        Check if DataContainer has information to perform several correction
        types
        
        Returns
        -------
        rep : dict
        """
        
        rep = dict()
        rep["empty"] = self.data_matrix.empty
        rep["missing"] = self.data_matrix.isna().any().any()
        rep["qc"] = bool(self.mapping["qc"])
        rep["blank"] = bool(self.mapping["blank"])
        try:
            rep["order"] = self.order
        except RunOrderError:
            rep["order"] = False
        
        try:
            rep["batch"] = self.batch
        except BatchInformationError:
            rep["batch"] = False
        return rep

    def get_mean_cv(self, classes=None):
        if classes is None:
            return filter_functions.cv(self.data_matrix).mean()
        elif isinstance(classes, str):
            classes = [classes]
        classes_mask = self.get_classes().isin(classes)
        return filter_functions.cv(self.data_matrix[classes_mask]).mean()
    
#    def _remove_empty_features(self):
#        remove_index = (self.data_matrix.sum() == 0).columns
#        self.remove(remove_index, "features")
# this should be moved to a MSDataContainer object
#    def get_mz(self):
#        return self.feature_definitions["mz"]
#
#    def get_rt(self):
#        return self.feature_definitions["rt"]
#
#    def cluster_mz(self, tolerance=0.0002):
#        """
#        Groups features with similar mz to reduce the number of calculated
#        EICs.
#
#        Parameters
#        ----------
#        tolerance : float
#        """
#        self.feature_definitions["mz_cluster"] = utils.cluster(self.get_mz(),
#                                                               tolerance)
#
#    def get_mz_cluster(self):
#        return self.feature_definitions["mz_cluster"]


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


class BatchInformationError(Exception):
    """
    Error class when there is no batch information
    """
    pass


class RunOrderError(Exception):
    """
    Error class raised when there is no run order information
    """
    pass


class InvalidClassName(Exception):
    """
    Error class raised when using invalid class names
    """
    pass

class EmptyDataContainerError(Exception):
    """
    Error class raised when remove leaves an empty DataContainer.
    """
    pass

def _validate_pipeline(t):
    if not isinstance(t, (list, tuple)):
        t = [t]
    for filt in t:
        if not isinstance(filt, (Processor, Pipeline)):
            msg = ("elements of the Pipeline must be",
                   "instances of Filter, DataCorrector or another Pipeline.")
            raise TypeError(msg)

# TODO: posible acortamiento de nombres: data_matrix: data,
# TODO:  sample_information: sample, feature_definitions: features.
# TODO: documentar todos los metodos
# TODO: agregar tests para el modulo
# TODO: agregar una funcion de validacion luego de aplicar correccion (chequear
# la igualdad de columnas y filas)
# TODO: documentacion para Reporter, Processor y Pipeline.
# TODO: crear subclasses para DataContainer de RMN y MS (agregar DI, LC)
# TODO: repensar la forma en que se miden metricas a lo largo de la aplicacion
            # de los distintos filtros