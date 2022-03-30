"""
Objects used to store and manage metabolomics data

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
from . import _batch_corrector
from ._names import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Optional, Iterable, Union, BinaryIO, TextIO
import bokeh.plotting
import pickle
from bokeh.palettes import Category10
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.models import LabelSet
import seaborn as sns


class DataContainer(object):
    """
    Container object that stores processed metabolomics data.

    The data is separated in three attributes: data_matrix, sample_metadata and
    feature_metadata. Each one is a pandas DataFrame. DataContainers can be
    created, apart from using the constructor, importing data in common formats
    (such as: XCMS, MZMine2, Progenesis, etc..) static methods.

    Attributes
    ----------
    data_matrix : DataFrame.
        feature values for each sample. Data is organized in a "tidy" way:
        each row is an observation, each column is a feature. dtype must
        be float and all values should be non-negative, but NANs are fine.
    sample_metadata : DataFrame.
        Metadata associated to each sample (eg: sample class). Has the same
        index as the data_matrix. `class` (standing for sample class) is a
        required column. Analytical batch and run order information can be
        included under the `batch` and `order` columns. Both must be integer
        numbers, and the run order must be unique for each sample. If the
        run order is specified in a per-batch fashion, the values will be
        converted to a unique value.
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
                 mapping: Optional[dict] = None,
                 plot_mode: str = "bokeh"):
        
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
        mapping : dict or None
            if dict, set each sample class to sample type.
        plot_mode : {"seaborn", "bokeh"}
            The package used to generate plots with the plot methods

        """
        validation.validate_data_container(data_matrix, feature_metadata,
                                           sample_metadata)

        # sort columns and indices of each DataFrame
        data_matrix = data_matrix.sort_index()
        data_matrix = data_matrix.reindex(sorted(data_matrix.columns), axis=1)
        sample_metadata = sample_metadata.sort_index()
        sample_metadata = sample_metadata.reindex(
            sorted(sample_metadata.columns), axis=1)
        feature_metadata = feature_metadata.sort_index()
        feature_metadata = feature_metadata.reindex(sorted(
            feature_metadata.columns), axis=1)

        # check and convert order and batch information
        try:
            order = sample_metadata.pop(_sample_order)
            try:
                batch = sample_metadata.pop(_sample_batch)
            except KeyError:
                batch = pd.Series(data=np.ones_like(order.values),
                                  index=order.index)
            order = _convert_to_interbatch_order(order, batch)
            sample_metadata[_sample_order] = order
            sample_metadata[_sample_batch] = batch
        except KeyError:
            pass

        # values are copied to prevent that modifications on the original
        # objects affect the DataContainer attributes
        self.data_matrix = data_matrix.copy()
        self.feature_metadata = feature_metadata.copy()
        self.sample_metadata = sample_metadata.copy()
        self._sample_mask = data_matrix.index.copy()
        self._feature_mask = data_matrix.columns.copy()
        self.mapping = mapping
        self.id = data_matrix.index
        self.plot = None

        # copy back up data for resetting
        self._original_data_matrix = self.data_matrix.copy()
        self._original_sample_metadata = self.sample_metadata.copy()
        self._original_feature_metadata = self.feature_metadata.copy()

        # adding methods
        self.metrics = MetricMethods(self)
        self.preprocess = PreprocessMethods(self)
        self.set_plot_mode(plot_mode)
    
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
        """pd.Series[int]. Analytical batch number"""
        try:
            return self._sample_metadata.loc[self._sample_mask, _sample_batch]
        except KeyError:
            raise BatchInformationError("No batch information available.")
            
    @batch.setter
    def batch(self, value: pd.Series):

        self._sample_metadata.loc[self._sample_mask,
                                  _sample_batch] = value.astype(int)
    
    @property
    def order(self) -> pd.Series:
        """
        pd.Series[int] : Run order in which samples were analyzed. It must be
        an unique integer for each sample.
        """
        try:
            return self._sample_metadata.loc[self._sample_mask, _sample_order]
        except KeyError:
            raise RunOrderError("No run order information available")
    
    @order.setter
    def order(self, value: pd.Series):
        if utils.is_unique(value):
            self._sample_metadata.loc[self._sample_mask,
                                      _sample_order] = value.astype(int)
        else:
            msg = "order values must be unique"
            raise ValueError(msg)

    @property
    def dilution(self) -> pd.Series:
        try:
            return self._sample_metadata.loc[self._sample_mask,
                                             _sample_dilution]
        except KeyError:
            msg = "No dilution information available."
            raise DilutionInformationError(msg)

    @dilution.setter
    def dilution(self, value):
        self._sample_metadata.loc[self._sample_mask, _sample_dilution] = value

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
        diagnostic[_qc_sample_type] = bool(self.mapping[_qc_sample_type])
        diagnostic[_blank_sample_type] = bool(self.mapping[_blank_sample_type])
        diagnostic[_study_sample_type] = bool(self.mapping[_study_sample_type])
        diagnostic[_dilution_qc_type] = bool(self.mapping[_dilution_qc_type])
        try:
            diagnostic[_sample_order] = self.order.any()
        except RunOrderError:
            diagnostic[_sample_order] = False
        
        try:
            diagnostic[_sample_batch] = self.batch.any()
        except BatchInformationError:
            diagnostic[_sample_batch] = False
        return diagnostic
   
    def reset(self, reset_mapping: bool = True):
        """
        Reloads the original data matrix.

        Parameters
        ----------
        reset_mapping: bool
            If True, clears sample classes from the mapping.
        """
        self._sample_mask = self._original_data_matrix.index
        self._feature_mask = self._original_data_matrix.columns
        self.data_matrix = self._original_data_matrix
        self.sample_metadata = self._original_sample_metadata
        self.feature_metadata = self._original_feature_metadata
        if reset_mapping:
            self.mapping = None

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
        Sort samples/features in place using metadata values.

        Parameters
        ----------
        field: str
            field to sort by. Must be a column of `sample_metadata` or
            `feature_metadata`.
        axis: {"samples", "features"}
        """
        if axis == "samples":
            tmp = self._sample_metadata.sort_values(field).index
            self._sample_mask = tmp.intersection(self._sample_mask)

            # self.sample_metadata = self.sample_metadata.loc[sorted_index, :]
            # self.data_matrix = self.data_matrix.loc[sorted_index, :]
        elif axis == "features":
            tmp = self.feature_metadata.sort_values(field).index
            self._feature_mask = tmp.intersection(self._feature_mask)
            # self.feature_metadata = self.feature_metadata.loc[sorted_index, :]
            # self.data_matrix = self.data_matrix.loc[:, sorted_index]
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

    def set_plot_mode(self, mode: str):
        """
        Set the library used to generate plots.

        Parameters
        ----------
        mode: {"bokeh", "seaborn"}

        """
        if mode == "bokeh":
            self.plot = BokehPlotMethods(self)
        elif mode == "seaborn":
            self.plot = SeabornPlotMethods(self)
        else:
            msg = "plot mode must be `seaborn` or `bokeh`"
            raise ValueError(msg)

    def add_order_from_csv(self, path: Union[str, TextIO],
                           interbatch_order: bool = True) -> None:
        """
        adds sample order and sample batch using information from a csv file.
        A column with the name `sample`  with the same values as the index of
        the DataContainer sample_metadata must be provided.
        order information is taken from a column with name `order` and the same
        is done with batch information. order data must be positive integers
        and each batch must have unique values. Each batch must be identified
        with a positive integer.

        Parameters
        ----------
        path : str
            path to the file with order data. Data format is inferred from the
            file extension.
        interbatch_order : bool
            If True converts the order value to a unique value for the whole
            DataContainer. This makes plotting the data as a function of order
            easier.

        """
        df = pd.read_csv(path, index_col="sample")
        order = df[_sample_order].astype(int)
        batch = df[_sample_batch].astype(int)

        if interbatch_order:
            order = _convert_to_interbatch_order(order, batch)
        self.order = order
        self.batch = batch

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
    cv: Computes the coefficient of variation for each feature.
    dratio: Computes the D-Ratio of features, a metric used to compare technical
    to biological variation.
    detection_rate: Computes the ratio of samples where a features was detected.
    pca: Computes the PCA scores, loadings and  PC variance.

    """
    
    def __init__(self, data: DataContainer):
        self.__data = data
    
    def cv(self, groupby: Union[str, List[str], None] = "class",
           robust: bool = False, fill_value: float = np.inf):
        """
        Computes the Coefficient of variation for each feature.

        The coefficient of variation is the quotient between the standard
        deviation and the mean of a feature.
        
        Parameters
        ----------
        groupby: str, List[str] or None
            If groupby isa column or a list of columns of sample metadata, the
            values of CV are computed on a per group basis. If None, the CV
            is computed for all samples in the data.
        robust: bool
            If True, computes the relative MAD. Else, computes the Coefficient
            of variation.
        fill_value: float
            Value used to replace NaN. By default, NaNs are replaced by np.inf.

        Returns
        -------
        pd.Series or pd.DataFrame

        """
        if robust:
            cv_func = utils.robust_cv
        else:
            cv_func = utils.cv

        if groupby is not None:
            if isinstance(groupby, str):
                by = self.__data.sample_metadata[groupby]
            else:
                by = [self.__data.sample_metadata[x] for x in groupby]
            result = (self.__data.data_matrix.groupby(by)
                      .apply(cv_func, fill_value=fill_value))
        else:
            result = cv_func(self.__data.data_matrix, fill_value=fill_value)
        return result

    def dratio(self, robust=False) -> pd.Series:
        """
        Computes the ratio between the sample variation and the quality control
        variation.

        The D-Ratio is useful to compare technical to biological variation and
        non informative features.
        
        Parameters
        ----------
        robust: bool
            If True, uses MAD to compute the D-ratio. Else, uses standard
            deviation.

        Returns
        -------
        dr : pd.Series:
            D-Ratio for each feature

        """
        if self.__data.mapping[_study_sample_type] is None:
            msg = "Study sample classes not specified in the sample mapping"
            raise ValueError(msg)
        sample_classes = self.__data.mapping[_study_sample_type]

        if self.__data.mapping[_qc_sample_type] is None:
            msg = "QC class not specified in the sample mapping"
            raise ValueError(msg)
        qc_classes = self.__data.mapping[_qc_sample_type]

        is_sample_class = self.__data.classes.isin(sample_classes)
        is_qc_class = self.__data.classes.isin(qc_classes)
        sample_data = self.__data.data_matrix[is_sample_class]
        qc_data = self.__data.data_matrix[is_qc_class]
        # NaN are filled to inf to make filtration easier.
        dratio = utils.sd_ratio(qc_data, sample_data, robust=robust,
                                fill_value=np.inf)
        return dratio
    
    def detection_rate(self, groupby: Union[str, List[str], None] = "class",
                       threshold=0):
        """
        Computes the fraction of samples where a feature was detected.
        
        Parameters
        ----------
        groupby : str, List[str] or None
            If groupby isa column or a list of columns of sample metadata, the
            values of CV are computed on a per group basis. If None, the CV
            is computed for all samples in the data.
        threshold : float
            Minimum value to consider a feature detected

        """
        if groupby is not None:
            if isinstance(groupby, str):
                by = self.__data.sample_metadata[groupby]
            else:
                by = [self.__data.sample_metadata[x] for x in groupby]
            result = (self.__data.data_matrix.groupby(by)
                      .apply(utils.detection_rate, threshold=threshold))
        else:
            result = (self.__data.data_matrix
                      .apply(utils.detection_rate, threshold=threshold))
        return result

    def pca(self, n_components: Optional[int] = 2,
            normalization: Optional[str] = None,
            scaling: Optional[str] = None,
            ignore_classes: Optional[List[str]] = None):
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
        ignore_classes : list[str], optional
            classes in the data to ignore to build the PCA model.
        
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
        if ignore_classes:
            class_mask = ~self.__data.classes.isin(ignore_classes)
            ind = data.index[class_mask]
            data = data[class_mask]
        else:
            ind = data.index

        if normalization is not None:
            data = utils.normalize(data, normalization)

        if scaling is not None:
            data = utils.scale(data, scaling)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data)
        n_components = pca.n_components_    # effective number of pc
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        variance = pca.explained_variance_
        pc_str = ["PC" + str(x) for x in range(1, n_components + 1)]
        scores = pd.DataFrame(data=scores, index=ind, columns=pc_str)
        loadings = pd.DataFrame(data=loadings,
                                index=self.__data.data_matrix.columns,
                                columns=pc_str)
        variance = pd.Series(data=variance, index=pc_str)
        total_variance = data.var().sum()
        return scores, loadings, variance, total_variance

    def correlation(self, field: str, mode: str = "ols",
                    classes: Optional[List[str]] = None):
        """
        Correlates features with sample metadata properties.

        Parameters
        ----------
        field : str
            A column of `sample_metadata`. Must have a numeric dtype.
        mode: {"ols", "spearman"}
            `ols` computes the ordinary least squares linear regression.
            Computes the Pearson r squared, p-value for the Jarque-Bera test and
            the Durwin-Watson statistic for each feature. `spearman` computes
            the spearman rank correlation coefficient for each feature
        classes: List[str], optional
            Compute the correlation on the selected classes only. If None,
            computes the correlation on all samples.

        Returns
        -------
        pandas.Series or pandas.DataFrame

        """
        if mode not in ["ols", "spearman"]:
            msg = "Valid modes are `ols` or `spearman`"
            raise ValueError(msg)

        if classes is None:
            x = self.__data.sample_metadata[field]
            df = self.__data.data_matrix
        else:
            mask = self.__data.classes.isin(classes)
            x = self.__data.sample_metadata.loc[mask, field]
            df = self.__data.data_matrix[mask]
        correlation_aux = lambda y: utils.metadata_correlation(y, x, mode)
        res = df.apply(correlation_aux, result_type="expand")
        return res


class BokehPlotMethods:  # pragma: no cover
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

    def pca_scores(self, x_pc: int = 1, y_pc: int = 2, hue: str = _sample_class,
                   ignore_classes: Optional[List[str]] = None,
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
        ignore_classes : list[str], optional
            classes in the data to ignore to build the PCA model.
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
        default_fig_params = {"aspect_ratio": 1}
        if fig_params is None:
            fig_params = default_fig_params
        else:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params

        default_scatter_params = {"size": 6}
        if scatter_params is None:
            scatter_params = default_scatter_params
        else:
            default_scatter_params.update(scatter_params)
            scatter_params = default_scatter_params

        tooltips = [(_sample_class, "@{}".format(_sample_class)),
                    (_sample_order, "@{}".format(_sample_order)),
                    (_sample_batch, "@{}".format(_sample_batch)),
                    (_sample_id, "@{}".format(_sample_id))]
        fig = bokeh.plotting.figure(tooltips=tooltips, **fig_params)

        x_name = "PC" + str(x_pc)
        y_name = "PC" + str(y_pc)
        n_comps = max(x_pc, y_pc)
        score, _, variance, total_var = \
            self._data_container.metrics.pca(n_components=n_comps,
                                             ignore_classes=ignore_classes,
                                             normalization=normalization,
                                             scaling=scaling)
        score = score.join(self._data_container.sample_metadata)

        if hue == _sample_type:
            rev_map = _reverse_mapping(self._data_container.mapping)
            score[_sample_type] = (score[_sample_class]
                                   .apply(lambda x: rev_map.get(x)))
            score = score[~pd.isna(score[_sample_type])]
        elif hue == _sample_batch:
            score[_sample_batch] = score[_sample_batch].astype(str)

        # setup the colors
        unique_values = score[hue].unique().astype(str)
        score = ColumnDataSource(score)
        cmap = Category10[10]
        palette = cmap * (int(unique_values.size / len(cmap)) + 1)
        palette = palette[:unique_values.size]
        # TODO: Category10_3 should be in a parameter file

        fig.scatter(source=score, x=x_name, y=y_name,
                    color=factor_cmap(hue, palette, unique_values),
                    legend_group=hue, **scatter_params)

        #  figure appearance
        x_label = x_name + " ({:.1f} %)"
        x_label = x_label.format(variance[x_pc - 1] * 100 / total_var)
        y_label = y_name + " ({:.1f} %)"
        y_label = y_label.format(variance[y_pc - 1] * 100 / total_var)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.xaxis.axis_label_text_font_style = "bold"

        if show_order:
            labels = LabelSet(x=x_name, y=y_name, text=_sample_order,
                              level="glyph", x_offset=3, y_offset=3,
                              source=score, render_mode="canvas",
                              text_font_size="8pt")
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
        default_fig_params = {"aspect_ratio": 1}
        if fig_params is None:
            fig_params = default_fig_params
        else:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params

        if scatter_params is None:
            scatter_params = dict()

        tooltips = [("feature", "@feature"), ("m/z", "@mz"),
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
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.xaxis.axis_label_text_font_style = "bold"

        if draw:
            bokeh.plotting.show(fig)
        return fig

    def feature(self, ft: str, hue: str = _sample_class,
                ignore_classes: Optional[List[str]] = None,
                draw: bool = True,
                fig_params: Optional[dict] = None,
                scatter_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        plots a feature intensity as a function of the run order.

        Parameters
        ----------
        ft: str
            Feature to plot. Index of feature in `feature_metadata`
        hue: {"class", "type"}
        ignore_classes : list[str], optional
            exclude samples from the listed classes in the plot
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

        default_fig_params = {"aspect_ratio": 1.5}
        if fig_params is None:
            fig_params = default_fig_params
        else:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params

        if scatter_params is None:
            scatter_params = dict()

        if ignore_classes is None:
            ignore_classes = list()

        source = (self._data_container.sample_metadata
                  .join(self._data_container.data_matrix[ft]))

        ignore_samples = source[_sample_class].isin(ignore_classes)
        source = source[~ignore_samples]

        if hue == _sample_type:
            rev_map = _reverse_mapping(self._data_container.mapping)
            source[_sample_type] = (source[_sample_class]
                                    .apply(lambda x: rev_map.get(x)))
            source = source[~source[_sample_type].isna()]
        elif hue == _sample_batch:
            source[_sample_batch] = source[_sample_batch].astype(str)

        # setup the colors
        unique_values = source[hue].unique().astype(str)
        cmap = Category10[10]
        palette = cmap * (int(unique_values.size / len(cmap)) + 1)
        palette = palette[:unique_values.size]

        source = ColumnDataSource(source)

        tooltips = [(_sample_class, "@{}".format(_sample_class)),
                    (_sample_order, "@{}".format(_sample_order)),
                    (_sample_batch, "@{}".format(_sample_batch)),
                    (_sample_id, "@{}".format(_sample_id))]
        fig = bokeh.plotting.figure(tooltips=tooltips, **fig_params)
        cmap_factor = factor_cmap(hue, palette, unique_values)
        fig.scatter(source=source, x=_sample_order, y=ft, color=cmap_factor,
                    legend_group=hue, **scatter_params)

        fig.xaxis.axis_label = "Run order"
        fig.yaxis.axis_label = "{} intensity [au]".format(ft)
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.yaxis.formatter.precision = 2
        fig.xaxis.formatter.precision = 2
        fig.xaxis.axis_label_text_font_style = "bold"

        if draw:
            bokeh.plotting.show(fig)
        return fig


class SeabornPlotMethods(object):   # pragma: no cover
    """
    Methods to plot feature data from a DataContainer using Matplotlib/Seaborn.
    """

    def __init__(self, data: DataContainer):
        self._data = data

    def pca_scores(self, x_pc: int = 1, y_pc: int = 2, hue: str = _sample_class,
                   ignore_classes: Optional[List[str]] = None,
                   show_order: bool = False,
                   scaling: Optional[str] = None,
                   normalization: Optional[str] = None,
                   relplot_params: Optional[dict] = None):
        """
        plots PCA scores using seaborn relplot function.

        Parameters
        ----------
        x_pc : int
            Principal component number to plot along X axis.
        y_pc : int
            Principal component number to plot along Y axis.
        hue : {"class", "type", "batch"}
            How to color samples. "class" color points according to sample
            class, "type" color points according to the sample type
            assigned in the mapping and "batch" uses batch information. Samples
            classes without a mapping are not shown in the plot
        ignore_classes : list[str], optional
            classes in the data to ignore to build the PCA model.
        show_order: bool
            add a label with the run order.
        scaling : {`autoscaling`, `rescaling`, `pareto`}, optional
            scaling method.
        normalization : {`sum`, `max`, `euclidean`}, optional
            normalization method
        relplot_params : dict, optional
            key-values to pass to relplot function.

        Returns
        -------
        seaborn.FacetGrid

        """

        x_name = "PC" + str(x_pc)
        y_name = "PC" + str(y_pc)
        n_comps = max(x_pc, y_pc)
        score, _, variance, total_var = \
            self._data.metrics.pca(n_components=n_comps, scaling=scaling,
                                   normalization=normalization,
                                   ignore_classes=ignore_classes)
        score = score.join(self._data.sample_metadata)

        tmp_params = {"x": x_name, "y": y_name, "hue": hue, "kind": "scatter"}
        if relplot_params is None:
            relplot_params = tmp_params
        else:
            relplot_params.update(tmp_params)

        if hue == _sample_type:
            rev_map = _reverse_mapping(self._data.mapping)
            score[_sample_type] = (score[_sample_class]
                                   .apply(lambda s: rev_map.get(s)))
            score = score[~pd.isna(score[_sample_type])]
        elif hue == _sample_batch:
            score[_sample_batch] = score[_sample_batch].astype(str)
        elif hue == _sample_class:
            score[_sample_class] = self._data.classes

        g = sns.relplot(data=score, **relplot_params)

        if show_order:
            for ind in score.index:
                x = score.loc[ind, x_name] * 1.01
                y = score.loc[ind, y_name] * 1.01
                t = str(self._data.order[ind].astype(int))
                g.ax.annotate(t, (x, y))

        # set x and y label
        x_var = variance[x_name] * 100 / total_var
        y_var = variance[y_name] * 100 / total_var
        x_label = "{} ({:.0f} %)".format(x_name, x_var)
        y_label = "{} ({:.0f} %)".format(y_name, y_var)
        g.ax.set_xlabel(x_label)
        g.ax.set_ylabel(y_label)
        return g

    def pca_loadings(self, x_pc: int = 1, y_pc: int = 2,
                     ignore_classes: Optional[List[str]] = None,
                     scaling: Optional[str] = None,
                     normalization: Optional[str] = None,
                     relplot_params: Optional[dict] = None):
        """
        plots PCA scores using seaborn relplot function.

        Parameters
        ----------
        x_pc : int
            Principal component number to plot along X axis.
        y_pc : int
            Principal component number to plot along Y axis.
        ignore_classes : list[str], optional
            classes in the data to ignore to build the PCA model.
        scaling : {`autoscaling`, `rescaling`, `pareto`}, optional
            scaling method.
        normalization : {`sum`, `max`, `euclidean`}, optional
            normalization method
        relplot_params : dict, optional
            key-values to pass to relplot function.

        Returns
        -------
        seaborn.FacetGrid

        """

        x_name = "PC" + str(x_pc)
        y_name = "PC" + str(y_pc)
        n_comps = max(x_pc, y_pc)
        _, loadings, variance, total_var = \
            self._data.metrics.pca(n_components=n_comps, scaling=scaling,
                                   normalization=normalization,
                                   ignore_classes=ignore_classes)

        tmp_params = {"x": x_name, "y": y_name, "kind": "scatter"}
        if relplot_params is None:
            relplot_params = tmp_params
        else:
            relplot_params.update(tmp_params)

        g = sns.relplot(data=loadings, **relplot_params)

        # set x and y label
        x_var = variance[x_name] * 100 / total_var
        y_var = variance[y_name] * 100 / total_var
        x_label = "{} ({:.0f} %)".format(x_name, x_var)
        y_label = "{} ({:.0f} %)".format(y_name, y_var)
        g.ax.set_xlabel(x_label)
        g.ax.set_ylabel(y_label)
        return g

    def feature(self, ft: str, hue: str = _sample_class,
                ignore_classes: Optional[List[str]] = None,
                relplot_params: Optional[dict] = None):

        tmp_params = {"x": _sample_order, "y": ft, "hue": hue,
                      "kind": "scatter"}
        if relplot_params is None:
            relplot_params = tmp_params
        else:
            relplot_params.update(tmp_params)

        if ignore_classes is None:
            ignore_classes = list()

        df = (self._data.sample_metadata.join(self._data.data_matrix[ft]))
        ignore_samples = df[_sample_class].isin(ignore_classes)
        df = df[~ignore_samples]

        if hue == _sample_type:
            rev_map = _reverse_mapping(self._data.mapping)
            df[_sample_type] = (df[_sample_class]
                                .apply(lambda x: rev_map.get(x)))
            df = df[~df[_sample_type].isna()]
        elif hue == _sample_batch:
            df[_sample_batch] = df[_sample_batch].astype(str)
        elif hue == _sample_class:
            df[_sample_class] = df[_sample_class].astype(str)

        g = sns.relplot(data=df, **relplot_params)
        return g


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

    def correct_batches(
        self,
        min_qc_dr: float = 0.9,
        first_n_qc: Optional[int] = None,
        threshold: float = 0.0,
        frac: Optional[float] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False
    ):
        r"""
        Correct time dependant systematic bias along samples due to variation in
        instrumental response.

        Parameters
        ----------
        min_qc_dr : float
            minimum fraction of QC where a feature was detected. See the notes
            for an explanation of how this value is computed.
        first_n_qc : int, optional
            The number of first QC samples used to estimate the expected
            value for each feature in the QC. If None uses all QC samples in a
            batch. See notes for an explanation of its use.
        threshold : float
            Minimum value to consider a feature detected. Used to compute the
            detection rate of each feature in the QC samples. Only features in
            QC samples above this value are used to compute the correction
            factor.
        frac : float, optional
            frac parameter of the LOESS model. If None, the best value for each
            feature is estimated using cross validation.
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool
            If True displays a progress bar.

        Notes
        -----
        The correction is applied as follows:

        1.  Split the data matrix using the batch number.
        2.  For each feature in a batch compute an intra-batch correction that
            removes time-dependent variations.
        3.  Once the features where corrected in all batches, apply an
            inter-batch where the mean across different batches is corrected.

        A detailed explanation of the correction algorithm can be found here.

        """
        mapping = self.__data.mapping

        if mapping[_qc_sample_type] is None:
            msg = "QC samples not defined in sample mapping"
            raise ValueError(msg)

        if mapping[_study_sample_type] is None:
            msg = "Study samples not defined in sample mapping"
            raise ValueError(msg)

        sample_metadata = self.__data.sample_metadata

        if _sample_order not in sample_metadata:
            msg = "Run order information not available"
            raise RunOrderError(msg)

        if _sample_batch not in sample_metadata:
            msg = "Batch information not available"
            raise BatchInformationError(msg)

        sample_class = mapping[_study_sample_type]
        qc_class = mapping[_qc_sample_type]

        rm_samples = _batch_corrector.find_invalid_samples(
            sample_metadata, sample_class, qc_class
        )
        self.__data.remove(rm_samples, "samples")

        # TODO: this should be removed after fixing the Datacontainer
        #   00implementation
        sample_metadata = self.__data.sample_metadata
        data_matrix = self.__data.data_matrix

        rm_features = _batch_corrector.find_invalid_features(
            data_matrix,
            sample_metadata,
            sample_class,
            qc_class,
            threshold,
            min_qc_dr
        )
        self.__data.remove(rm_features, "features")
        if verbose:
            msg = "{} samples and {} features were removed"
            print(msg.format(rm_samples.size, rm_features.size))

        sample_metadata = self.__data.sample_metadata
        data_matrix = self.__data.data_matrix

        corrected = _batch_corrector.correct_batches(
            data_matrix,
            sample_metadata,
            sample_class,
            qc_class,
            threshold=threshold,
            frac=frac,
            first_n=first_n_qc,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.__data.data_matrix = corrected

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


class DilutionInformationError(Exception):
    """
    Error class raised when no dilution factor information has been provided.
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


def _convert_to_interbatch_order(order: pd.Series,
                                 batch: pd.Series) -> pd.Series:
    """
    Convert the order values from a per-batch order to a interbatch order.

    Parameters
    ----------
    order: pandas.Series
        order and batch must share the same index, size and be of dtype int.
    batch: pandas.Series

    Returns
    -------
    interbatch_order: pandas.Series

    Raises
    ------
    ValueError: if the order values are already unique.

    Examples
    --------
    >>>order = pd.Series([1, 2, 3, 1, 2, 3])
    >>>batch = pd.Series([1, 1, 1, 2, 2, 2])
    >>>_convert_to_interbatch_order(order, batch)
    pd.Series([1, 2, 3, 4, 5, 6])
    """

    if order.unique().size == order.size:
        return order

    # find a value to add to each batch to make unique and sorted order values
    max_order = order.groupby(batch).max()
    add_to_order = np.roll(max_order, 1)
    add_to_order[0] = 0
    add_to_order = add_to_order.cumsum()
    add_to_order = pd.Series(data=add_to_order, index=max_order.index)
    add_to_order = batch.map(add_to_order)
    interbatch_order = order + add_to_order
    return interbatch_order


def _make_empty_mapping():
    empty_mapping = {x: None for x in SAMPLE_TYPES}
    return empty_mapping


def _reverse_mapping(mapping):
    rev_map = dict()
    for k, v in mapping.items():
        if v is not None:
            for class_value in v:
                rev_map[class_value] = k
    return rev_map
