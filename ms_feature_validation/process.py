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
        self._sample_mask = data_matrix_df.index
        self._feature_mask = data_matrix_df.columns
        self._original_data = data_matrix_df.copy()
        self.metrics = Metrics(self)

    
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
    def data_matrix(self):
        return self._data_matrix.loc[self._sample_mask, self._feature_mask]
    
    @data_matrix.setter
    def data_matrix(self, value):
        self._data_matrix = value
        
    @property
    def feature_definitions(self):
        return self._feature_definitions.loc[self._feature_mask, :]
    
    @feature_definitions.setter
    def feature_definitions(self, value):
        self._feature_definitions = value
    
    @property
    def sample_information(self):
        return self._sample_information.loc[self._sample_mask, :]
    
    @sample_information.setter
    def sample_information(self, value):
        self._sample_information = value
               
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
    def id(self, value):
        self.sample_information[_sample_id] = value
        
    
    @property
    def classes(self):
        """pd.Series[str] : class of each sample."""
        return self._sample_information[_sample_class]
    
    @classes.setter
    def classes(self, value):
        self._sample_information[_sample_class] = value
    
    @property
    def batch(self):
        """pd.Series[str] or pd.Series[int]. Batch identification"""
        try:
            return self._sample_information[_sample_batch]
        except KeyError:
            raise BatchInformationError("No batch information available.")
            
    @batch.setter
    def batch(self, value):
        self._sample_information[_sample_batch] = value
    
    @property
    def order(self):
        """pd.Series[int] : order of analysis of samples"""
        try:
            return self._sample_information[_sample_order]
        except KeyError:
            raise RunOrderError("No run order information available")
    
    @order.setter
    def order(self, value):
        self._sample_information[_sample_order] = value

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
        
        if not self._is_valid(remove, axis):
            msg = "Some samples/features aren't in the DataContainer"
            raise ValueError(msg)
        
        if axis == "features":
            # self.data_matrix.drop(columns=remove, inplace=True)
            # self.feature_definitions.drop(index=remove, inplace=True)
            self._feature_mask = self._feature_mask.difference(remove)
        elif axis == "samples":
            # self.data_matrix.drop(index=remove, inplace=True)
            # self.sample_information.drop(index=remove, inplace=True)
            self._sample_mask = self._sample_mask.difference(remove)
        else:
            msg = "axis should be `columns` or `features`"
            raise ValueError(msg)
        
    def _is_valid(self, index, axis):
        """
        Check if all samples/features are present in the DataContainer.
        
        Parameters
        ----------
        index: list[str]
            Features / Samples name to check.
        axis: {"sample", "feature"}
            axis to check.
        """
        ind = pd.Index(index)
        if axis == "features":
            return ind.isin(self.data_matrix.columns).all()
        elif axis == "samples":
            return ind.isin(self.data_matrix.index).all()
        else:
            msg = "axis must be `features` or `samples`."
            raise ValueError(msg)
            
    
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
    
    def reset(self):
        """
        Reset applied filters/corrections.
        """
        self._sample_mask = self._original_data.index
        self._feature_mask = self._original_data.columns
        self.data_matrix = self._original_data
    
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

class Metrics:
    """
    Functions to get metrics from DataContainer
    """
    
    def __init__(self, data):
        self.data = data
    
    def cv(self, mode="intraclass", robust=False):
        """
        Coefficient of variation.
        
        Parameters
        ----------
        mode: {"intraclass", "global"}
            if "intraclass", computes the coefficient of variation for each
            class. if "global", computes the mean coefficient of variation
            for all sample classes.
        robust: bool
            If True, computes the relative MAD. Else, computes the Coefficient
            of variation.
        """
        if robust:
            cv_func = utils.rmad
        else:
            cv_func = utils.cv
        
        if mode == "intraclass":
            result = (self.data.data_matrix
                      .groupby(self.data.classes)
                      .apply(cv_func))
        elif mode == "global":
            sample_class = self.data.mapping[_sample_class]
            is_sample_class = self.data.classes.isin(sample_class)
            result = cv_func(self.data.data_matrix[is_sample_class])
        else:
            msg = "`mode` must be intraclass or global"
            raise ValueError(msg)
        return result
    
    def dratio(self, robust=False):
        """
        Computes the D-Ratio using sample variation and quality control
        variaton [1].
        
        Parameters
        ----------
        qc_df : pd.DataFrame
            DataFrame with quality control samples
        samples_df : pd.DataFrame
            DataFrame with biological samples
        Returns
        -------
        dr : pd.Series:
            D-Ratio for each feature
        References
        ----------
        .. [1] D.Broadhurst *et al*, "Guidelines and considerations for the use
        of system suitability and quality control samples in mass spectrometry
        assays applied in untargeted clinical metabolomic studies",
        Metabolomics (2018) 14:72.
        """
        if robust:
            cv_func = utils.rmad
        else:
            cv_func = utils.cv
            
        sample_class = self.data.mapping["sample"]
        is_sample_class = self.data.classes.isin(sample_class)
        qc_class = sample_class = self.data.mapping["qc"]
        is_qc_class = self.data.classes.isin(qc_class)
        sample_variation = cv_func(self.data.data_matrix[is_sample_class],
                                   robust = robust)
        qc_variation = cv_func(self.data.data_matrix[is_qc_class],
                               robust=robust)
        dr = qc_variation / sample_variation
        dr = dr.fillna(np.inf)
        return dr
    
    def detection_rate(self, mode="intraclass", threshold=0):
        """
        Computes the fraction of samples with intensity above a threshold
        
        Parameters
        ----------
        mode: {"intraclass", "global"}
            if intraclass, computes the detection rate for each class, if
            global computes the mean detection rate
        threshold: float
            Minimum value to consider a feature detected
        """
        dr_func = lambda x: x[x > threshold].count() / x.count()
        if mode == "intraclass":
            results = (self.data.data_matrix
                       .groupby(self.data.classes)
                       .apply(dr_func))
        elif mode == "global":
            sample_class = self.data.mapping["sample"]
            is_sample_class = self.data.classes.isin(sample_class)
            results = self.data.data_matrix[is_sample_class].apply()
        else:
            msg = "`mode` must be intraclass or global"
            raise ValueError(msg)
        return results
    
    # TODO: implement PCA function that return scores and loadings


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
# TODO: set sample type
# TODO: generic Filter Object