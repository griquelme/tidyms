# -*- coding: utf-8 -*-
"""
The object used to store metabolomics data

Objects
-------

- DataContainer: Stores metabolomics data.

Exceptions
----------

- BatchInformationError
- RunOrderError
- ClassNameError
- EmptyDataContainerError

Usage
-----

DataContainers can be created in two different ways other than using the
constructor:

- Using the functions in the fileio module to read data processed with a third
  party software (XCMS, MZMine2, etc...)
- Performing Feature correspondence algorithm on features detected from raw
  data (not implemented yet...)
"""

from . import utils
from . import validation
from . import fileio
from ._names import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Optional, Iterable, Union, BinaryIO, TextIO
import bokeh.plotting
import pickle
from bokeh.palettes import Spectral
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.models import LabelSet


# TODO: remove data_path attribute. check with webapp example.
# TODO: add order_from_csv file method.
# TODO: maybe its a good idea to combine export methods into and ExportMethods
#       object
# TODO: export datacontainer to metaboanalyst format.


class DataContainer(object):
    """
    A container object to store Metabolomics Data.

    The data is separated in three attributes: data_matrix, sample_metadata and
    feature_metadata. Each one is a pandas DataFrame. DataContainers can be
    created, apart from using the constructor, importing data in common formats
    (such as: XCMS, MZMmine2, Progenesis, etc..) static methods.

    Attributes
    ----------
    data_matrix : DataFrame.
        feature values for each sample. Data is organized in a "tidy" way:
        each row is an observation, each column is a feature. dtype must
        be float and all values should be non negative, but NANs are fine.
    sample_metadata : DataFrame.
        Metadata associated to each sample (eg: sample class). Has the same
        index as the data_matrix. "class" (standing for sample class) is a
        required column.
    feature_metadata : DataFrame.
        Metadata associated to each feature (eg: mass to charge ratio (mz),
        retention time (rt), etc...). The index is equal to the `data_matrix`
        column. "mz" and "rt" are required columns.
    mapping : dictionary of sample types to a list of sample classes.
        Maps sample types to sample classes. valid samples types are `qc`,
        `blank`, `sample` or `suitability`. values are list of sample classes.
        Mapping is used by Processor objects to define a default behaviour. For
        example, when using a BlankCorrector, the blank contribution to each
        feature is estimated using the sample classes that are values of the
        `blank` sample type.
    metrics : methods to compute common feature metrics.
    plot : methods to plot features.
    preprocess : methods to perform common preprocessing tasks.
    data_path
    id
    batch
    order

    Methods
    -------
    remove(remove, axis) : Remove samples/features from the DataContainer.
    reset(reset_mapping=True) : Reset the DataContainer, ie: recover removed
    samples/features, transformed values.
    is_valid_class_name(value) : checks if a class is present in the
    DataContainer
    diagnose() : creates a dictionary with information about the status of the
    DataContainer. Used by Processor objects as a validity check.
    select_features(mz, rt, mz_tol=0.01, rt_tol=5) : Search features within
    a m/z and rt tolerance.
    set_default_order() : Assigns a default run order of the samples assuming
    that the data matrix is sorted by run order already.
    sort(field, axis) : sort features/samples using metadata information.
    save(filename) : save the DataContainer as a pickle.

    See Also
    --------
    from_progenesis
    from_pickle
    MetricMethods
    PlotMethods
    PreprocessMethods
    """

    def __init__(self, data_matrix: pd.DataFrame,
                 feature_metadata: pd.DataFrame,
                 sample_metadata: pd.DataFrame,
                 data_path: Optional[str] = None,
                 mapping: Optional[dict] = None):
        
        """
        See help(DataContainer) for more details
        
        Parameters
        ----------
        data_matrix : pandas.DataFrame.
            Feature values for each measured sample. Each row is a sample and
            each column is a feature.                  
        sample_metadata : pandas.DataFrame.
            Metadata for each sample. class is a required column.
        feature_metadata : pandas.DataFrame.
            DataFrame with features names as indices. mz and rt are required
            columns.
        data_path : str.
            path to raw Data. Files must have the same name as each sample.
        mapping : dict or None
            if dict, set each sample class to sample type.
        """
        validation.validate_data_container(data_matrix, feature_metadata,
                                           sample_metadata, data_path)
        self.data_matrix = data_matrix
        self.feature_metadata = feature_metadata
        self.sample_metadata = sample_metadata
        self._sample_mask = data_matrix.index
        self._feature_mask = data_matrix.columns
        self._original_data = data_matrix.copy()
        self.data_path = data_path
        self.mapping = mapping
        self.id = data_matrix.index

        # adding methods
        self.metrics = MetricMethods(self)
        self.plot = PlotMethods(self)
        self.preprocess = PreprocessMethods(self)

    @property
    def data_path(self) -> pd.DataFrame:
        """str : path where raw data is stored."""
        return self._data_path
    
    @data_path.setter
    def data_path(self, path: str):
        """
        sets raw data path, search for available samples and adds them to
        sample information.
        """
        if path is not None:
            path_mapping = utils.sample_to_path(self.data_matrix.index, path)
            self._sample_metadata.loc[self._sample_mask, _raw_path] = \
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
        return self._feature_metadata.loc[self._feature_mask, :]
    
    @feature_metadata.setter
    def feature_metadata(self, value: pd.DataFrame):
        self._feature_metadata = value
    
    @property
    def sample_metadata(self) -> pd.DataFrame:
        return self._sample_metadata.loc[self._sample_mask, :]
    
    @sample_metadata.setter
    def sample_metadata(self, value: pd.DataFrame):
        self._sample_metadata = value
               
    @property
    def mapping(self):
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
        return self._sample_metadata.loc[self._sample_mask, _sample_id]

    @id.setter
    def id(self, value: pd.Series):
        self._sample_metadata.loc[self._sample_mask, _sample_id] = value
        
    @property
    def classes(self) -> pd.Series:
        """pd.Series[str] : class of each sample."""
        return self._sample_metadata.loc[self._sample_mask, _sample_class]
    
    @classes.setter
    def classes(self, value: pd.Series):
        self._sample_metadata.loc[self._sample_mask, _sample_class] = value
    
    @property
    def batch(self) -> pd.Series:
        """pd.Series[str] or pd.Series[int]. Analytical batch number"""
        try:
            return self._sample_metadata.loc[self._sample_mask, _sample_batch]
        except KeyError:
            raise BatchInformationError("No batch information available.")
            
    @batch.setter
    def batch(self, value: pd.Series):
        try:
            _validate_batch_order(value, self.order)
        except RunOrderError:
            pass
        self._sample_metadata.loc[self._sample_mask, _sample_batch] = value
    
    @property
    def order(self) -> pd.Series:
        """pd.Series[int] : Run order in which samples were analyzed."""
        try:
            return self._sample_metadata.loc[self._sample_mask, _sample_order]
        except KeyError:
            raise RunOrderError("No run order information available")
    
    @order.setter
    def order(self, value: pd.Series):
        self._sample_metadata.loc[self._sample_mask, _sample_order] = value

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

    def is_valid_class_name(self, test_class: Union[str, List[str]]) -> bool:
        """
        Check if at least one sample class is`class_name`.
        
        Parameters
        ----------
        test_class : str or list[str]
            classes to search in the DataContainer.
        Returns
        -------
        is_valid : bool
        """
        valid_classes = self.classes.unique()
        if isinstance(test_class, str):
            return test_class in valid_classes
        else:
            for c in test_class:
                if not (c in valid_classes):
                    return False
            return True

    def remove(self, remove: Iterable[str], axis: str):
        """
        Remove selected features / samples

        Parameters
        ----------
        remove : Iterable[str]
            List of sample/feature names to remove.
        axis : {"features", "samples"}
        """
        
        if not self._is_valid(remove, axis):
            msg = "Some samples/features aren't in the DataContainer"
            raise ValueError(msg)
        
        if axis == "features":
            self._feature_mask = self._feature_mask.difference(remove)
        elif axis == "samples":
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
            List of feature/sample names to check.
        axis: {"samples", "features"}
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
        diagnostic : dict
            Each value is a bool indicating the status. `empty` is True if the
            size in at least one dimension of the data matrix is zero; "missing"
            is True if there are NANs in the data matrix;  "order" is True
            if there is run order information for the samples; "batch" is True
            if there is batch number information associated to the samples.
        """
        
        diagnostic = dict()
        diagnostic["empty"] = self.data_matrix.empty
        diagnostic["missing"] = self.data_matrix.isna().any().any()
        diagnostic[_qc_type] = bool(self.mapping[_qc_type])
        diagnostic[_blank_type] = bool(self.mapping[_blank_type])
        diagnostic[_sample_type] = bool(self.mapping[_sample_type])
        try:
            diagnostic["order"] = self.order.any()
        except RunOrderError:
            diagnostic["order"] = False
        
        try:
            diagnostic["batch"] = self.batch.any()
        except BatchInformationError:
            diagnostic["batch"] = False
        return diagnostic
   
    def reset(self, reset_mapping: bool = True):
        """
        Reloads the original data matrix.

        Parameters
        ----------
        reset_mapping: bool
            If True, clears sample classes from the mapping.
        """
        self._sample_mask = self._original_data.index
        self._feature_mask = self._original_data.columns
        self.data_matrix = self._original_data
        if reset_mapping:
            self.mapping = None

    def get_mapped_classes(self) -> List[str]:
        """
        return all classes assigned in the sample type mapping.

        Returns
        -------
        process_classes: list of mapped classes.
        """
        process_classes = list()
        for classes in self.mapping.values():
            if classes is not None:
                process_classes += classes
        return process_classes

    def select_features(self, mzq: float, rtq: float, mz_tol: float = 0.01,
                        rt_tol: float = 5) -> pd.Index:
        """
        Find feature names within the defined mass-to-charge and retention time
        tolerance.

        Parameters
        ----------
        mzq: positive number
            Mass-to-charge value to search
        rtq: positive number
            Retention time value to search
        mz_tol: positive number
            Mass-to-charge tolerance used in the search.
        rt_tol: positive number
            Retention time tolerance used in the search.

        Returns
        -------
        Index
        """
        mz_match = (self.feature_metadata["mz"] - mzq).abs() < mz_tol
        rt_match = (self.feature_metadata["rt"] - rtq).abs() < rt_tol
        mz_match_ft = mz_match[mz_match].index
        rt_match_ft = rt_match[rt_match].index
        result = mz_match_ft.intersection(rt_match_ft)
        return result

    def set_default_order(self):
        """
        set the order of the samples, assuming that de data is already sorted.
        """
        order_data = np.arange(1, self.sample_metadata.shape[0] + 1)
        ind = self.data_matrix.index
        order = pd.Series(data=order_data, index=ind, dtype=int)
        batch = pd.Series(data=1, index=ind, dtype=int)
        self.order = order
        self.batch = batch

    def sort(self, field: str, axis: str):
        """
        Sort samples/features using metadata values.

        Parameters
        ----------
        field: str
            field to sort by. Must be a column of `sample_metadata` or
            `feature_metadata`.
        axis: {"samples", "features"}
        """
        if axis == "samples":
            sorted_index = self.sample_metadata.sort_values(field).index
            self.sample_metadata = self.sample_metadata.loc[sorted_index, :]
            self.data_matrix = self.data_matrix.loc[sorted_index, :]
        elif axis == "features":
            sorted_index = self.feature_metadata.sort_values(field).index
            self.feature_metadata = self.feature_metadata.loc[sorted_index, :]
            self.data_matrix = self.data_matrix.loc[:, sorted_index]
        else:
            msg = "axis must be `samples` or `features`"
            raise ValueError(msg)

    def save(self, filename: str) -> None:
        """
        Save DataContainer into a pickle

        Parameters
        ----------
        filename: str
            name used to save the file.
        """
        with open(filename, "wb") as fin:
            pickle.dump(self, fin)

    def to_csv(self, filename: str) -> None:
        """
        Save the DataContainer into a csv file.

        Parameters
        ----------
        filename: str
        """
        df = pd.concat([self.feature_metadata.T, self.data_matrix], axis=0)
        df = pd.concat([self.sample_metadata, df], axis=1)
        df.to_csv(filename)

    @staticmethod
    def from_progenesis(path: Union[str, TextIO]):
        """
        Read a progenesis file into a DataContainer

        Parameters
        ----------
        path : str or file
            path to an Progenesis csv output or file object

        Returns
        -------
        dc = DataContainer
        """
        return fileio.read_progenesis(path)

    @staticmethod
    def from_pickle(path: Union[str, BinaryIO]):
        """
        read a DataContainer stored as a pickle

        Parameters
        ----------
        path: str or file
            path to read DataContainer

        Returns
        -------
        DataContainer
        """
        return fileio.read_pickle(path)


class MetricMethods:
    """
    Methods to compute feature metrics from a DataContainer

    Methods
    -------
    cv(intraclass=True, robust=False): Computes the coefficient of variation for
    each feature.
    dratio(robust=False, sample_classes=None, qc_classes=None): Computes the
    D-Ratio of features, a metric used to compare technical to biological
    variation.
    detection_rate(intraclass=True, threshold=0): Computes the ratio of samples
    where a features was detected.
    pca(n_components=2, normalization=None, scaling=None): Computes the PCA
    scores, loadings and  PC variance.

    """
    
    def __init__(self, data: DataContainer):
        self.__data = data
    
    def cv(self, intraclass=True, robust=False):
        """
        Computes the Coefficient of variation defined as the ratio between the
        standard deviation and the mean of each feature.
        
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
            cv_func = utils.robust_cv
        else:
            cv_func = utils.cv
        
        if intraclass:
            result = (self.__data.data_matrix
                      .groupby(self.__data.classes)
                      .apply(cv_func))
        # TODO: check where this is used, maybe its to compute the cv for the
        #  whole data_matrix here...
        else:
            if self.__data.mapping[_sample_type] is None:
                result = cv_func(self.__data.data_matrix)
            else:
                sample_class = self.__data.mapping[_sample_type]
                is_sample_class = self.__data.classes.isin(sample_class)
                result = cv_func(self.__data.data_matrix[is_sample_class])
        return result
    
    def dratio(self, robust=False,
               sample_classes: Optional[List[str]] = None,
               qc_classes: Optional[List[str]] = None) -> pd.Series:
        """
        Computes the D-Ratio using sample variation and quality control
        variation [1]. This metric is useful to compare technical to biological
        variation.
        
        Parameters
        ----------
        sample_classes: list[str], optional
            classes used to estimate biological variation. If None, uses
            values from sample_type in sample mapping
        qc_classes: list[str], optional
            classes used to estimate technical variation. If None, uses
            values from qc_type in sample mapping
        robust: bool
            If True, uses MAD to compute the D-ratio. Else, uses standard
            deviation.

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
            cv_func = utils.mad
        else:
            cv_func = utils.sd

        if sample_classes is None:
            sample_classes = self.__data.mapping[_sample_type]

        if qc_classes is None:
            qc_classes = self.__data.mapping[_qc_type]

        is_sample_class = self.__data.classes.isin(sample_classes)
        is_qc_class = self.__data.classes.isin(qc_classes)
        sample_variation = cv_func(self.__data.data_matrix[is_sample_class])
        qc_variation = cv_func(self.__data.data_matrix[is_qc_class])
        dratio = qc_variation / sample_variation
        dratio = dratio.fillna(np.inf)
        return dratio
    
    def detection_rate(self, intraclass=True, threshold=0):
        """
        Computes the fraction of samples where a feature was detected.
        
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
            # TODO: same as CV. check if its better to return a global DR.
            sample_class = self.__data.mapping["sample"]
            is_sample_class = self.__data.classes.isin(sample_class)
            results = self.__data.data_matrix[is_sample_class].apply()
        return results
    
    def pca(self, n_components: Optional[int] = 2,
            normalization: Optional[str] = None,
            scaling: Optional[str] = None):
        """
        Computes PCA score, loadings and  PC variance of each component.
        
        Parameters
        ----------
        n_components: int
            Number of Principal components to compute.
        scaling: {`autoscaling`, `rescaling`, `pareto`}, optional
            scaling method.
        normalization: {`sum`, `max`, `euclidean`}, optional
            normalizing method
        
        Returns
        -------
        scores: np.array
        loadings: np.array
        variance: np.array
            Explained variance for each component.
        total_variance: float
            Total variance of the scaled data.
        """
        data = self.__data.data_matrix

        if normalization:
            data = utils.normalize(data, normalization)

        if scaling:
            data = utils.scale(data, scaling)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data)
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
        total_variance = data.var().sum()
        return scores, loadings, variance, total_variance


class PlotMethods:
    """
    Methods to plot data from a DataContainer. Generates Bokeh Figures.

    Methods
    -------
    pca_scores()
    pca_loadings()
    feature()
    """
    def __init__(self, data: DataContainer):
        self._data_container = data

    def pca_scores(self, x_pc: int = 1, y_pc: int = 2, hue: str = "class",
                   show_order: bool = False, scaling: Optional[str] = None,
                   normalization: Optional[str] = None, draw: bool = True,
                   fig_params: Optional[dict] = None,
                   scatter_params: Optional[dict] = None
                   ) -> bokeh.plotting.Figure:
        """
        plots PCA scores
        
        Parameters
        ----------
        x_pc: int
            Principal component number to plot along X axis.
        y_pc: int
            Principal component number to plot along Y axis.
        hue: {"class", "type", "batch"}
            How to color samples. "class" color points according to sample
            class, "type" color points according to the sample type
            assigned in the mapping and "batch" uses batch information. Samples
            classes without a mapping are not shown in the plot
        show_order: bool
            add a label with the run order.
        scaling: {`autoscaling`, `rescaling`, `pareto`}, optional
            scaling method.
        normalization: {`sum`, `max`, `euclidean`}, optional
            normalization method
        draw: bool
            If True calls bokeh.plotting.show on fig.
        fig_params: dict, optional
            Optional parameters to pass to bokeh figure
        scatter_params: dict, optional
            Optional parameters to pass to bokeh scatter plot.
        
        Returns
        -------
        bokeh.plotting.Figure.
        """
        if fig_params is None:
            fig_params = dict()

        if scatter_params is None:
            scatter_params = dict()

        tooltips = [("class", "@class"), ("order", "@order"),
                    ("batch", "@batch"), ("id", "@id")]
        fig = bokeh.plotting.figure(tooltips=tooltips, **fig_params)

        x_name = "PC" + str(x_pc)
        y_name = "PC" + str(y_pc)
        n_comps = max(x_pc, y_pc)
        score, _, variance, total_var = \
            self._data_container.metrics.pca(n_components=n_comps,
                                             normalization=normalization,
                                             scaling=scaling)
        score = score.join(self._data_container.sample_metadata)

        if hue == "type":
            rev_map = _reverse_mapping(self._data_container.mapping)
            score["type"] = score["class"].apply(lambda x: rev_map.get(x))
            score = score[~pd.isna(score["type"])]
        elif hue == "batch":
            score["batch"] = score["batch"].astype(str)

        # setup the colors
        unique_values = score[hue].unique().astype(str)
        score = ColumnDataSource(score)
        cmap = Spectral[11]
        palette = cmap * (int(unique_values.size / len(cmap)) + 1)
        palette = palette[:unique_values.size]
        # TODO: Category10_3 should be in a parameter file

        fig.scatter(source=score, x=x_name, y=y_name,
                    color=factor_cmap(hue, palette, unique_values),
                    legend_group=hue, **scatter_params)

        x_label = x_name + " ({:.1f} %)"
        x_label = x_label.format(variance[x_pc - 1] * 100 / total_var)
        y_label = y_name + " ({:.1f} %)"
        y_label = y_label.format(variance[y_pc - 1] * 100 / total_var)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        if show_order:
            labels = LabelSet(x=x_name, y=y_name, text='order', level='glyph',
                              x_offset=3, y_offset=3, source=score,
                              render_mode='canvas', text_font_size="8pt")
            fig.add_layout(labels)

        if draw:
            bokeh.plotting.show(fig)
        return fig

    def pca_loadings(self, x_pc=1, y_pc=2, scaling: Optional[str] = None,
                     normalization: Optional[str] = None, draw: bool = True,
                     fig_params: Optional[dict] = None,
                     scatter_params: Optional[dict] = None
                     ) -> bokeh.plotting.Figure:
        """
        plots PCA loadings.

        Parameters
        ----------
        x_pc: int
            Principal component number to plot along X axis.
        y_pc: int
            Principal component number to plot along Y axis.
        scaling: {`autoscaling`, `rescaling`, `pareto`}, optional
            scaling method.
        normalization: {`sum`, `max`, `euclidean`}, optional
            normalizing method
        draw: bool
            If True, calls bokeh.plotting.show on figure
        fig_params: dict, optional
            Optional parameters to pass into bokeh figure
        scatter_params: dict, optional
            Optional parameters to pass into bokeh scatter plot.


        Returns
        -------
        bokeh.plotting.Figure.
        """
        if fig_params is None:
            fig_params = dict()

        if scatter_params is None:
            scatter_params = dict()

        tooltips = [("id", "@feature"), ("m/z", "@mz"),
                    ("rt", "@rt"), ("charge", "@charge")]
        fig = bokeh.plotting.figure(tooltips=tooltips, **fig_params)

        x_name = "PC" + str(x_pc)
        y_name = "PC" + str(y_pc)
        n_comps = max(x_pc, y_pc)
        _, loadings, variance, total_var = \
            self._data_container.metrics.pca(n_components=n_comps,
                                             normalization=normalization,
                                             scaling=scaling)
        loadings = loadings.join(self._data_container.feature_metadata)
        loadings = ColumnDataSource(loadings)

        fig.scatter(source=loadings, x=x_name, y=y_name, **scatter_params)

        # set axis label names with % variance
        x_label = x_name + " ({:.1f} %)"
        x_label = x_label.format(variance[x_pc - 1] * 100 / total_var)
        y_label = y_name + " ({:.1f} %)"
        y_label = y_label.format(variance[y_pc - 1] * 100 / total_var)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        if draw:
            bokeh.plotting.show(fig)
        return fig

    def feature(self, ft: str, color_by: str = "class", draw: bool = True,
                fig_params: Optional[dict] = None,
                scatter_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        plots a feature intensity as a function of the run order.

        Parameters
        ----------
        ft: str
            Feature to plot. Index of feature in `feature_metadata`
        color_by: {"class", "type"}
        draw: bool
            If True calls bokeh.plotting.show on figure.
        fig_params: dict
            key-value parameters to pass to bokeh figure
        scatter_params: dict
            key-value parameters to pass to bokeh circle

        Returns
        -------
        bokeh.plotting.Figure
        """

        if fig_params is None:
            fig_params = dict()
        if scatter_params is None:
            scatter_params = dict()

        source = (self._data_container.sample_metadata
                  .join(self._data_container.data_matrix[ft]))

        if color_by == "type":
            rev_map = _reverse_mapping(self._data_container.mapping)
            source["type"] = (self._data_container.classes
                              .apply(lambda x: rev_map.get(x)))
            source = source[~source["type"].isna()]

        # setup the colors
        unique_values = source[color_by].unique().astype(str)
        cmap = Spectral[11]
        palette = cmap * (int(unique_values.size / len(cmap)) + 1)
        palette = palette[:unique_values.size]

        source = ColumnDataSource(source)

        tooltips = [("class", "@class"), ("order", "@order"),
                    ("batch", "@batch"), ("id", "@id")]
        fig = bokeh.plotting.figure(tooltips=tooltips, **fig_params)
        cmap_factor = factor_cmap(color_by, palette, unique_values)
        fig.scatter(source=source, x="order", y=ft, color=cmap_factor,
                    legend_group=color_by, **scatter_params)
        if draw:
            bokeh.plotting.show(fig)
        return fig


class PreprocessMethods:
    """
    Common Preprocessing operations.

    Methods
    -------
    normalize(method, inplace=True): Adjust sample values.
    scale(method, inplace=True): Adjust feature distribution values.
    transform(method, inplace=True): element-wise transformations of data.
    """

    def __init__(self, dc: DataContainer):
        self.__data = dc

    def normalize(self, method: str, inplace: bool = True,
                  feature: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Normalize samples.

        Parameters
        ----------
        method: {"sum", "max", "euclidean"}
            Normalization method. `sum` normalizes using the sum along each row,
            `max` normalizes using the maximum of each row. `euclidean`
            normalizes using the euclidean norm of the row.
        inplace: bool
            if True modifies in place the DataContainer. Else, returns a
            normalized data matrix.
        feature: str, optional
            Feature used for normalization in `feature` mode.

        Returns
        -------
        normalized: pandas.DataFrame
        """
        normalized = utils.normalize(self.__data.data_matrix, method,
                                     feature=feature)
        if inplace:
            self.__data.data_matrix = normalized
        else:
            return normalized

    def scale(self, method: str,
              inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        scales features using different methods.

        Parameters
        ----------
        method: {"autoscaling", "rescaling", "pareto"}
            Scaling method. `autoscaling` performs mean centering scaling of
            features to unitary variance. `rescaling` scales data to a 0-1
            range. `pareto` performs mean centering and scaling using the
            square root of the standard deviation
        inplace: bool
            if True modifies in place the DataContainer. Else, returns a
            normalized data matrix.

        Returns
        -------
        scaled: pandas.DataFrame
        """
        scaled = utils.scale(self.__data.data_matrix, method)
        if inplace:
            self.__data.data_matrix = scaled
        else:
            return scaled

    def transform(self, method: str,
                  inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        Perform element-wise data transformations.

        Parameters
        ----------
        method: {"log", "power"}
            transform method. `log` applies the base 10 logarithm on the data.
            `power`
        inplace: bool
            if True modifies in place the DataContainer. Else, returns a
            normalized data matrix.

        Returns
        -------
        transformed: pandas.DataFrame
        """
        transformed = utils.transform(self.__data.data_matrix, method)
        if inplace:
            self.__data.data_matrix = transformed
        else:
            return transformed


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


class ClassNameError(Exception):
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
# TODO: move _validate_mapping to validation module


def _make_empty_mapping():
    empty_mapping = {x: None for x in SAMPLE_TYPES}
    return empty_mapping


def _validate_batch_order(batch: pd.Series, order: pd.Series):
    if batch.dtype != int:
        msg = "batch must be of integer dtype"
        raise BatchInformationError(msg)
    if order.dtype != int:
        msg = "order must be of integer dtype"
        raise RunOrderError(msg)

    grouped = order.groupby(batch)
    for _, batch_order in grouped:
        if not np.array_equal(batch_order, batch_order.unique()):
            msg = "order value must be unique for each batch"
            raise RunOrderError(msg)


def _reverse_mapping(mapping):
    rev_map = dict()
    for k, v in mapping.items():
        if v is not None:
            for class_value in v:
                rev_map[class_value] = k
    return rev_map


# TODO: subclass DataContainer into  NMRDataContainer and MSDataContainer
#  (add DI, LC)
# TODO: implement a PCA function to avoid importing sklearn
