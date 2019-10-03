"""
Filter objects to curate data
"""

import pandas as pd
from .process import Processor
from .process import DataContainer
from .process import Pipeline
from . import filter_functions
from . import utils
from . import fileio
from . import validation
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


@register
class DuplicateAverager(Processor):
    """
    A filter that averages duplicates
    """
    def __init__(self, process_classes):
        super(DuplicateAverager, self).__init__(axis=None, mode="correction")
        self.params["process_classes"] = process_classes

    def func(self, dc):
        dc.data_matrix = filter_functions.replicate_averager(dc.data_matrix,
                                                             dc.get_id(),
                                                             dc.get_classes(),
                                                             **self.params)
        dc.sample_information = (dc.sample_information
                                     .loc[dc.data_matrix.index, :])


@register
class ClassRemover(Processor):
    """
    A filter to remove samples of a given class
    """
    def __init__(self, classes=None):
        super(ClassRemover, self).__init__(axis="samples", mode="filter")
        self.params["classes"] = classes

    def func(self, dc):
        remove_samples = dc.get_classes().isin(self.params["classes"])
        remove_samples = remove_samples[remove_samples]
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
        dc.data_matrix = filter_functions.blank_correction(dc.data_matrix,
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
        dc.data_matrix = filter_functions.batch_correction(dc.data_matrix,
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
        return filter_functions.prevalence_filter(dc.data_matrix,
                                                  dc.get_classes(),
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
        return filter_functions.variation_filter(dc.data_matrix,
                                                 dc.get_classes(),
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
        mean_mz_cluster = utils.mean_cluster_value(dc.get_mz(),
                                                   dc.get_mz_cluster())
        chromatogram_dict = dict()
        samples = dc.get_available_samples()
        for k, sample in enumerate(samples):
            if self.verbose:
                msg = "Computing EICs for {} ({}/{})"
                msg = msg.format(sample, k + 1, len(samples))
                print(msg)
            reader = fileio.read(sample)
            c = fileio.chromatogram(reader,
                                    mean_mz_cluster,
                                    tolerance=self.params["tolerance"])
            sample_name = sample_name_from_path(sample)
            chromatogram_dict[sample_name] = c
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
    sample_info = df_header.iloc[:,
                  (raw_index+1):(2 * raw_index - norm_index + 1)].T
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
    ft_def["rt"] = ft_def["rt"] * 60
    validation.validate_data_container(data, ft_def, sample_info, None)
    dc = DataContainer(data, ft_def, sample_info)

    return dc


def read_config(path):
    with open(path) as fin:
        config = yaml.load(fin, Loader=yaml.UnsafeLoader)
    return config


def pipeline_from_list(l):
    pipeline = Pipeline()
    for d in l:
        pipeline.proccesors.append(filter_from_dictionary(d))
    return pipeline


def pipeline_from_yaml(path):
    d = read_config(path)
    filters_list = d["Pipeline"]
    pipeline = pipeline_from_list(filters_list)
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
