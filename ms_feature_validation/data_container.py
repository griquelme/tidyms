# -*- coding: utf-8 -*-
"""
Objects used for automatic curation and validation of LC-MS metabolomics data.

TODO: add examples.
"""


from . import utils
from . import validation
from ._names import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from typing import List, Optional, Iterable


class DataContainer(object):
    """
    A container class for Metabolomics Data.
    
    Consists of three Pandas DataFrames with features values, feature metadata
    and sample metadata. Index are shared for features and samples
    respectively.
    
    Contains functions to remove samples or features.

    Attributes
    ---------
    data_matrix : pd.DataFrame.
                  Feature values for each measured sample. Each row is a
                  sample and each column is a feature.                  
    sample_metadata : pd.DataFrame.
                         Metadata for each sample. class is a required column.
    feature_metadata : pd.DataFrame.
                          DataFrame with features names as indices. mz and rt
                          are required columns.
    data_path : str.
        Path to raw data directory.
    mapping : dict[str, list[str]].
        maps a sample types to sample classes. valid samples types are `qc`,
        `blank`, `sample` or`suitability`. values are list of sample classes.
        Mapping is used by Filter objects to select which samples are going
        to be used to perform corrections.
    """

    def __init__(self, data_matrix: pd.DataFrame,
                 feature_metadata: pd.DataFrame,
                 sample_metadata: pd.DataFrame,
                 data_path: Optional[str] = None,
                 mapping: Optional[dict] = None):
        
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
        validation.validate_data_container(data_matrix,
                                           feature_metadata,
                                           sample_metadata,
                                           data_path)
        self.data_matrix = data_matrix
        self.feature_metadata = feature_metadata
        self.sample_metadata = sample_metadata
        self.data_path = data_path
        self.mapping = mapping
        self._sample_mask = data_matrix.index
        self._feature_mask = data_matrix.columns
        self._original_data = data_matrix.copy()
        self.metrics = _Metrics(self)

    @property
    def data_path(self) -> pd.DataFrame:
        """str : directory where raw data is stored."""
        return self._data_path
    
    @data_path.setter
    def data_path(self, path: str):
        """
        sets raw data path, search for available samples and adds them to
        sample information.
        """
        if path is not None:
            path_mapping = utils.sample_to_path(self.data_matrix.index, path)
            self.sample_metadata[_raw_path] = \
                self.sample_metadata.index.map(path_mapping)
            self._data_path = path
        else:
            self._data_path = None
    
    @property
    def data_matrix(self) -> pd.DataFrame:
        return self._data_matrix.loc[self._sample_mask, self._feature_mask]
    
    @data_matrix.setter
    def data_matrix(self, value: pd.DataFrame):
        self._data_matrix = value
        
    @property
    def feature_metadata(self) -> pd.DataFrame:
        return self._feature_definitions.loc[self._feature_mask, :]
    
    @feature_metadata.setter
    def feature_metadata(self, value: pd.DataFrame):
        self._feature_definitions = value
    
    @property
    def sample_metadata(self) -> pd.DataFrame:
        return self._sample_information.loc[self._sample_mask, :]
    
    @sample_metadata.setter
    def sample_metadata(self, value: pd.DataFrame):
        self._sample_information = value
               
    @property
    def mapping(self):
        """
        dict : Set the sample type of a sample_classes. keys must be one of
        the following: {'qc', 'blank', 'zero', 'sample', 'suitability'}
        """
        return self._mapping
    
    @mapping.setter
    def mapping(self, mapping: dict):
        self._mapping = _make_empty_mapping()
        if mapping is not None:
            valid_samples = self.classes.unique()
            _validate_mapping(mapping, valid_samples)
            self._mapping.update(mapping)

    @property
    def id(self) -> pd.Series:
        """pd.Series[str] : name id of each sample."""
        return self.sample_metadata[_sample_id]

    @id.setter
    def id(self, value: pd.Series):
        self.sample_metadata[_sample_id] = value
        
    @property
    def classes(self) -> pd.Series:
        """pd.Series[str] : class of each sample."""
        return self._sample_information[_sample_class]
    
    @classes.setter
    def classes(self, value: pd.Series):
        self._sample_information[_sample_class] = value
    
    @property
    def batch(self) -> pd.Series:
        """pd.Series[str] or pd.Series[int]. Batch identification"""
        try:
            return self._sample_information[_sample_batch]
        except KeyError:
            raise BatchInformationError("No batch information available.")
            
    @batch.setter
    def batch(self, value: pd.Series):
        self._sample_information[_sample_batch] = value
    
    @property
    def order(self) -> pd.Series:
        """pd.Series[int] : order of analysis of samples"""
        try:
            return self._sample_information[_sample_order]
        except KeyError:
            raise RunOrderError("No run order information available")
    
    @order.setter
    def order(self, value: pd.Series):
        self._sample_information[_sample_order] = value

    def get_available_samples(self) -> pd.Series:
        """
        Returns the absolute path for each raw data file present in
        data_path.
        
        Returns
        -------
        available_samples : pd.Series
            Pandas series with absolute path for each available file.
        """
        available_samples = self.sample_metadata[_raw_path].dropna()
        return available_samples

    def is_valid_class_name(self, test_class: str) -> bool:
        """
        Check if at least one sample class is`class_name`.
        
        Parameters
        ----------
        test_class : str
        
        Returns
        -------
        is_valid : bool
        """
        return test_class in self.classes.unique()

    def remove(self, remove: Iterable[str], axis: str):
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
        
    def _is_valid(self, index: Iterable[str], axis: str) -> bool:
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
            
    def diagnose(self) -> dict:
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
            rep["order"] = self.order.any()
        except RunOrderError:
            rep["order"] = False
        
        try:
            rep["batch"] = self.batch.any()
        except BatchInformationError:
            rep["batch"] = False
        return rep
   
    def reset(self):
        """
        Reset applied filters/corrections.
        """
        self._sample_mask = self._original_data.index
        self._feature_mask = self._original_data.columns
        self.data_matrix = self._original_data

    def get_process_classes(self) -> List[str]:
        """
        return all classes assigned in the sample type mapping.

        Returns
        -------
        process_classes: List[str]
        """
        process_classes = list()
        for classes in self.mapping.values():
            if classes is not None:
                process_classes += classes
        return process_classes
    

class _Metrics:
    """
    Functions to get metrics from a DataContainer
    """
    
    def __init__(self, data):
        self.__data = data
    
    def cv(self, intraclass=True, robust=False):
        """
        Coefficient of variation.
        
        Parameters
        ----------
        intraclass: True
            if True computes the coefficient of variation for each
            class. Else computes the mean coefficient of variation
            for all sample classes.
        robust: bool
            If True, computes the relative MAD. Else, computes the Coefficient
            of variation.

        Returns
        -------
        result: pd.Series or pd.DataFrame
        """
        if robust:
            cv_func = utils.rmad
        else:
            cv_func = utils.cv
        
        if intraclass:
            result = (self.__data.data_matrix
                      .groupby(self.__data.classes)
                      .apply(cv_func))
        else:
            sample_class = self.__data.mapping[_sample_type]
            is_sample_class = self.__data.classes.isin(sample_class)
            result = cv_func(self.__data.data_matrix[is_sample_class])
        return result
    
    def dratio(self, robust=False):
        """
        Computes the D-Ratio using sample variation and quality control
        variaton [1].
        
        Parameters
        ----------
        robust: bool
            If True, uses the relative MAD to compute the D-ratio. Else, uses t
            he Coefficient of variation.

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
            
        sample_class = self.__data.mapping["sample"]
        is_sample_class = self.__data.classes.isin(sample_class)
        qc_class = self.__data.mapping["qc"]
        is_qc_class = self.__data.classes.isin(qc_class)
        sample_variation = cv_func(self.__data.data_matrix[is_sample_class])
        qc_variation = cv_func(self.__data.data_matrix[is_qc_class])
        dr = qc_variation / sample_variation
        dr = dr.fillna(np.inf)
        return dr
    
    def detection_rate(self, intraclass=True, threshold=0):
        """
        Computes the fraction of samples with intensity above a threshold
        
        Parameters
        ----------
        intraclass: bool
            if True, computes the detection rate for each class, else
            computes the mean detection rate
        threshold: float
            Minimum value to consider a feature detected
        """
        def dr_func(x):
            """Auxiliar function to compute the detection rate."""
            return x[x > threshold].count() / x.count()

        if intraclass:
            results = (self.__data.data_matrix
                       .groupby(self.__data.classes)
                       .apply(dr_func))
        else:
            sample_class = self.__data.mapping["sample"]
            is_sample_class = self.__data.classes.isin(sample_class)
            results = self.__data.data_matrix[is_sample_class].apply()
        return results
    
    def pca(self, n_components=2):
        """
        Computes PCA score, loadings and variance of each component.
        
        Parameters
        ----------
        n_components: int
            Number of Principal components to compute.
        
        Returns
        -------
        scores: np.array
        loadings: np.array
        variance: np.array
            Explained variance for each component.
        """
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(self.__data.data_matrix)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        variance = pca.explained_variance_
        pc_str = ["PC" + str(x) for x in range(1, n_components + 1)]
        scores = pd.DataFrame(data=scores,
                              index=self.__data.data_matrix.index,
                              columns=pc_str)
        loadings = pd.DataFrame(data=loadings,
                                index=self.__data.data_matrix.columns,
                                columns=pc_str)
        variance = pd.Series(data=variance, index=pc_str)
        return scores, loadings, variance


class _Plotter:
    """
    Functions to plot data from a DataContainer.
    The methods return a bokeh figure object.
    """
    def __init__(self, data):
        self._data_container = data
        self.data = None
        self.chromographic_data = None
        self.ms_data = None
    
    def pca_scores(self, x=1, y=2, fig=None, **kwargs):
        """
        plots PCA scores
        
        Parameters
        ----------
        x: int
            Principal component number to plot along X axis.
        y: int
            Principal component number to plot along Y axis.
        fig: bokeh.plotting.figure, optional
            Figure used to plot. If None returns a new figure
        kwargs: optional arguments to pass into figure
        
        Returns
        -------
        If no figure is specified returns a new figure. Else returns None.
        """
        return_fig = False
        if fig is None:
            return_fig = True
            fig = figure(**kwargs)
        x_name = "PC" + str(x)
        y_name = "PC" + str(y)
        n = max(x, y)
        scores, _, _ = self._data_container.metrics.pca(n_components=n)
        scores = ColumnDataSource(scores)
        scores.add(self._data_container.classes)
        classes = self._data_container.classes.unique()
        fig.scatter(data=scores, x=x_name, y=y_name,
                    color=factor_cmap('class', 'Category10_3', classes))
        if return_fig:
            return fig


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


def _validate_mapping(mapping, valid_samples):
    for sample_type, classes in mapping.items():
        if sample_type not in SAMPLE_TYPES:
            msg = "{} is not a valid sample type".format(sample_type)
            raise ValueError(msg)
        else:
            for c in classes:
                if c not in valid_samples:
                    msg = "{} is not a valid sample class".format(c)
                    raise ValueError(msg)


def _make_empty_mapping():
    empty_mapping = {x: None for x in SAMPLE_TYPES}
    return empty_mapping

# TODO: subclass DataContainer into  NMRDataContainer and MSDataContainer
#  (agregar DI, LC)
# TODO: implement a PCA function to avoid importing sklearn
