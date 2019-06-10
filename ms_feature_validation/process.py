from . import filter
import pandas as pd
import numpy as np


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
                 DataFrame with sample names on indices and features on the columns.
    sample_information : pandas.DataFrame.
                        DataFrame with sample names as indices. class is a required column
    feature_definitions : pandas.DataFrame.
                         DataFrame with features names as indices. mz and rt are required columns.
    """

    def __init__(self, data_matrix_df, feature_definitions_df, sample_information_df):
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
            data_matrix_selection = self.data_matrix.loc[selection, :]
            sample_information_selection = self.sample_information.loc[selection, :]
            feature_definitions_selection = self.feature_definitions
        elif axis == "features":
            data_matrix_selection = self.data_matrix.loc[:, selection]
            sample_information_selection = self.sample_information
            feature_definitions_selection = self.feature_definitions.loc[selection, :]

        data_selection = DataContainer(data_matrix_selection,
                                       feature_definitions_selection,
                                       sample_information_selection)
        return data_selection

    def get_classes(self):
        return self.sample_information["class"].unique()

    def group_by_class(self):
        """
        groups data_matrix Using class information.
        Returns
        -------
        grouped_data_matrix : DataFrameGroupBy
        """
        return self.data_matrix.groupby(self.sample_information["class"])

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
        if self.is_valid_class_name(classes):
            selection = self.sample_information["class"].isin(classes)
            return self.data_matrix[selection]
        else:
            raise ValueError("Invalid class names")


class Filter(object):
    """
    Selects or removes samples or features from a DataContainer.
    """

    def __init__(self, axis=None, mode="remove"):
        self.axis = axis
        self.mode = mode
        self.filter = None

    def fit(self, data_container):
        self.filter = lambda x: None

    def transform(self, data_container):
        if self.filter is None:
            self.fit(data_container)
        if self.mode == "select":
            return data_container.select(self.filter, self.axis)
        elif self.mode == "remove":
            data_container.remove(self.filter, self.axis)


class Corrector(object):
    """
    Corrects examples in a DataContainer.
    """
    def __init__(self, axis=None):
        self.axis = axis
        self.corrector = lambda x: None

    def transform(self, data_container):
        self.corrector(data_container)


@register
class ClassSelector(Filter):
    """
    A filter to select samples of a given class
    """
    def __init__(self, classes=None):
        super(ClassSelector, self).__init__(axis="samples", mode="select")
        self.classes = classes

    def fit(self, data_container):
        self.filter = data_container.select_classes(self.classes).index


@register
class BlankCorrector(Corrector):
    """
    Corrects values using blank information
    """
    def __init__(self, blank_classes=None, correction_type="mean", blank_relation=3):
        super(BlankCorrector, self).__init__(axis=None)
        self.blankClasses = blank_classes
        self.correctionType = correction_type
        self.blankRelation = blank_relation
        self.corrector = filter.blank_correction

    def transform(self, data_container):
        filter.blank_correction(data_container, self.blankClasses, self.correctionType, self.blankRelation)


@register
class PrevalenceFilter(Filter):
    """
    Performs a prevalence filter on selected classes
    """
    def __init__(self, include_classes=None, lb=0.5, ub=1):
        # TODO : add all classes when include_classes is None
        super(PrevalenceFilter, self).__init__(axis="features")
        self.include_classes = include_classes
        self.lb = lb
        self.ub = ub

    def fit(self, data_container):
        self.filter = filter.prevalence_filter(data_container, self.include_classes, self.lb, self.ub)


@register
class VariationFilter(Filter):
    """
    Remove samples with high variation coefficient.
    """
    def __init__(self, lb=0, ub=0.25, include_classes=None, robust=False):
        super(VariationFilter, self).__init__(axis="features")
        self.lb = lb
        self.ub = ub
        self.robust = robust
        self.include_classes = include_classes

    def fit(self, data_container):
        self.filter = filter.variation_filter(data_container, self.include_classes, self.lb, self.ub, self.robust)


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


def data_container_from_excel(excel_file):
    data_matrix = pd.read_excel(excel_file, sheet_name="data_matrix", index_col="sample")
    sample_information = pd.read_excel(excel_file, sheet_name="sample_information", index_col="sample")
    feature_definitions = pd.read_excel(excel_file, sheet_name="feature_definitions", index_col="feature")
    data_container = DataContainer(data_matrix, feature_definitions, sample_information)
    _validate_data_container(data_container)
    return data_container


def _validate_pipeline(t):
    for filt in t:
        if not isinstance(filt, (Filter, Corrector)):
            raise TypeError("elements of the Pipeline must be instances of Filter or DataCorrector")


def _validate_data_container(data_container):

    # feature names and sample names check
    if not data_container.data_matrix.index.equals(data_container.sample_information.index):
        raise ValueError("sample_information data_matrix indices should be equal")
    if not data_container.data_matrix.columns.equals(data_container.feature_definitions.index):
        raise  ValueError("sample_information columns and feature_definitions should be equal")
    # rt, mz and class information check
    if not "mz" in data_container.feature_definitions.columns:
        raise KeyError("mz values are required for all features")
    if not "rt" in data_container.feature_definitions.columns:
        raise KeyError("rt values are required for all features")
    if not "class" in data_container.sample_information.columns:
        raise KeyError("class information is required for all samples")
    # rt, mz and class and data matrix check:
    if (data_container.feature_definitions["mz"] < 0).any():
        raise ValueError("mz values should be greater than zero")
    if (data_container.feature_definitions["rt"] < 0).any():
        raise ValueError("mz values should be greater than zero")
    if (data_container.data_matrix < 0).any().any():
        raise ValueError("Raw values in data_matrix should be greater than zero")
    # TODO: pensar si siempre seria valido imputar NaN en datamatrix como 0.






# TODO: leer filtros desde un archivo yaml a diccionario. Hacer test para todos los filtros.
# TODO: Hacer un mapper para correccion de blancos multiples (idem correccion interbatch).
# snippet para leer un archivo yaml
# import yaml
# with open(yaml_path) as fin:
#   config = yaml.load(fin, Loader=yaml.UnsafeLoader)
