from . import filter
import pandas as pd
import numpy as np
import yaml


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
    """

    def __init__(self, data_matrix_df, feature_definitions_df,
                 sample_information_df):
        self.data_matrix = data_matrix_df
        self.feature_definitions = feature_definitions_df
        self.sample_information = sample_information_df

    def is_valid_class_name(self, class_names):
        valid_classes = np.isin(class_names, self.sample_information["class"])
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
        return self.sample_information["class"].unique()

    def group_by(self, name):
        """
        groups data_matrix Using class information.
        Returns
        -------
        grouped_data_matrix : DataFrameGroupBy
        """
        for group_name, group in self.sample_information.groupby(name):
            yield DataContainer(self.data_matrix.loc[group.index],
                                self.feature_definitions,
                                self.sample_information.loc[group.index])

    def select_classes(self, classes):
        """
        select classes from data_matrix

        Parameters
        ----------
        classes: Iterable[str]

        Returns
        -------
        data_matrix_selection : DataFrame
        """

        if isinstance(classes, str):
            classes = [classes]

        if self.is_valid_class_name(classes):
            selection = self.sample_information["class"].isin(classes)
            return self.data_matrix[selection]
        else:
            raise ValueError("Invalid class names")

    def apply_correction(self, mapper, corrector, **kwargs):
        """
        applies corrector to data_matrix.

        Parameters
        ----------
        mapper: dict[str: list[str]]
                dictionary to generate a mapper
        corrector: function.
                   function used to correct data.
        """
        for corrector_index, to_correct_index in self.generate_mapper(mapper):
            corrected = corrector(self.data_matrix.loc[to_correct_index],
                                  self.data_matrix.loc[corrector_index],
                                  **kwargs)
            self.data_matrix.loc[to_correct_index] = corrected

    def generate_mapper(self, mapper):
        """
        yields a tuple of indexes corresponding to classes in key and values
        of mapper.

        Parameters
        ----------
        mapper: dict[str: list[str]]
                dictionary  whose keys are classes used to generate a
                correction and the values are classes to be corrected.

        Yields
        ------
        corrector_dc, to_correct_dc
        """
        if isinstance(mapper, dict):
            for k, v in mapper.items():
                k = [k] if isinstance(k, str) else list(k)
                v = [v] if isinstance(v, str) else list(v)
                corrector_dc = self.select_classes(k)
                to_correct_dc = self.select_classes(v)
                yield corrector_dc.index, to_correct_dc.index
        elif isinstance(mapper, (list, tuple)):
            for k in mapper:
                yield self.select_classes([k])
        elif mapper is None:
            yield self.data_matrix.index
        # TODO: este metodo necesita un nombre y documentacion mas clara.


class Filter(object):
    """
    Selects or removes samples or features from a DataContainer.
    """

    def __init__(self, axis=None, mode="remove"):
        self.axis = axis
        self.mode = mode
        self.mapper = None
        self.filter = None
        self.params = dict()

    def fit(self, data_container):
        pass

    def transform(self, data_container):
        if self.filter is None:
            self.fit(data_container)
        if self.mode == "select":
            return data_container.select(self.filter, self.axis)
        elif self.mode == "remove":
            data_container.remove(self.filter, self.axis)


class Corrector(object):
    """
    Corrects data_matrix in a DataContainer.
    """
    def __init__(self, mapper=None):
        self.mapper = mapper
        self.corrector = None
        self.params = dict()

    def transform(self, data_container):
        data_container.apply_correction(self.mapper,
                                        self.corrector,
                                        **self.params)


@register
class ClassSelector(Filter):
    """
    A filter to select samples of a given class
    """
    def __init__(self, classes=None):
        super(ClassSelector, self).__init__(axis="samples", mode="select")
        self.param["classes"] = classes

    def fit(self, data_container):
        self.filter = data_container.select_classes(**self.params).index


@register
class BlankCorrector(Corrector):
    """
    Corrects values using blank information
    """
    def __init__(self, mapper=None, mode="mean",
                 blank_relation=3):
        super(BlankCorrector, self).__init__(mapper=mapper)
        self.params["mode"] = mode
        self.params["blank_relation"] = blank_relation
        self.corrector = filter.blank_correction

@register
class BatchCorrector(Corrector):
    """
    Corrects instrumental drift between batches.
    """
    def __init__(self, mapper=None, mode="splines"):
        super(BatchCorrector, self).__init__(mapper=mapper)
        self.params["mode"] = mode
        self.corrector = filter.batch_correction

    def transform(self, data_container):
        sample_classes = list(self.mapper.values())[0]
        corrector_classes = list(self.mapper.keys())[0]
        scaling_factor = data_container.select_classes(corrector_classes).mean()
        # Prefilters and LOESS
        steps = [PrevalenceFilter(include_classes=corrector_classes, lb=1),
                 VariationFilter(include_classes=corrector_classes, ub=0.5),
                 IntraBatchCorrector(mode=self.params["mode"],
                                     mapper=self.mapper)]
        pipe = Pipeline(*steps)
        batches = list()
        for batch in data_container.group_by("batch"):
            pipe.transform(batch)
            batches.append(batch)
        corrected = merge_data_containers(batches)
        data_container.data_matrix =corrected.data_matrix
        data_container.sample_information = corrected.sample_information
        data_container.feature_definitions = corrected.feature_definitions

        # Prevalence post LOESS on QC
        data_container.data_matrix = (data_container.data_matrix
                                      * scaling_factor).dropna(axis=1)
        prev_filter = PrevalenceFilter(include_classes=corrector_classes, lb=1)
        prev_filter.transform(data_container)


class IntraBatchCorrector(Corrector):
    """
    Corrects instrumental drift between in a batch.
    """
    def __init__(self, mapper=None, mode="splines"):
        super(IntraBatchCorrector, self).__init__(mapper=mapper)
        self.params["mode"] = mode
        self.corrector = filter.batch_correction

    def transform(self, dc):
        for ref_index, sample_index in dc.generate_mapper(self.mapper):
            ref_order = dc.sample_information["order"].loc[ref_index]
            ref = dc.data_matrix.loc[ref_index]
            sample_order = dc.sample_information["order"].loc[sample_index]
            sample = dc.data_matrix.loc[sample_index]
            correction = self.corrector(ref_order, ref, sample_order,
                                        sample, **self.params)
            dc.data_matrix.loc[sample_index] = correction



@register
class PrevalenceFilter(Filter):
    """
    Performs a prevalence filter on selected classes
    """
    def __init__(self, include_classes=None, lb=0.5, ub=1, intraclass=True):
        super(PrevalenceFilter, self).__init__(axis="features", mode="remove")
        self.mapper = include_classes
        self.intraclass = intraclass
        self.params["lb"] = lb
        self.params["ub"] = ub

    def fit(self, data_container):
        func = filter.prevalence_filter
        if self.intraclass:
            self.filter = pd.Index([])
            for groups in data_container.generate_mapper(self.mapper):
                self.filter = self.filter.union(func(groups, **self.params))
        else:
            self.filter = func(data_container.select_classes(self.mapper),
                               **self.params)


@register
class VariationFilter(Filter):
    """
    Remove samples with high variation coefficient.
    """
    def __init__(self, lb=0, ub=0.25, include_classes=None,
                 robust=False, intraclass=True):
        super(VariationFilter, self).__init__(axis="features", mode="remove")
        self.params["lb"] = lb
        self.params["ub"] = ub
        self.params["robust"] = robust
        self.mapper = include_classes
        self.intraclass = intraclass

    def fit(self, data_container):
        func = filter.variation_filter
        if self.intraclass:
            self.filter = pd.Index([])
            for groups in data_container.generate_mapper(self.mapper):
                self.filter = self.filter.union(func(groups, **self.params))
        else:
            self.filter = func(data_container.select_classes(self.mapper),
                               **self.params)


class Pipeline(list):
    """
    Applies a series of Filters and Correctors to a DataContainer
    """
    def __init__(self, *args):
        _validate_pipeline(args)
        super(Pipeline, self).__init__(args)

    def transform(self, data_container):
        for x in self:
            x.transform(data_container)


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
    path : path to an Excel file

    Returns
    -------
    dc = DataContainer
    """



def _validate_pipeline(t):
    if len(t) == 1 and isinstance(t[0], tuple):
        t = t[0]
    for filt in t:
        if not isinstance(filt, (Filter, Corrector, Pipeline)):
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
        pipeline.append(filter_from_dictionary(d))
    return pipeline

def filter_from_dictionary(d):
    for name, params in d.items():
        filter = FILTERS[name](**params)
    return filter

# TODO: posible acortamiento de nombres: data_matrix: data,
#  sample_information: sample, feature_definitions: features.
