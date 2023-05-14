"""
Functionality to process DART-MS datasets

Objects
-------

* DartMSAssay: Stores raw and processed DART-MS data.


Usage
-----

Predefined workflow:

* prefab_DARTMS_dataProcessing_pipeline

Semi-automated parameter optimization:

* compare_parameters_for_function

Spot detection: 

* create_assay_from_chronogramFiles

Data import: 

* create_assay_from_chronogramFiles

Data manipulation:

* select_top_n_spectra
* correct_MZ_shift_across_samples
* calculate_consensus_spectra_for_samples
* bracket_consensus_spectrum_samples
* build_data_matrix
* batch_correction
* blank_subtraction

Annotation

* annotate_features
* annotate_with_compounds

Import/Export

* save_self_to_dill_file
* read_from_dill_file
* export_data_matrix
* export_for_R
* write_bracketing_results_to_featureML
* generate_feature_raw_plot

Statistics

* restrict_to_high_quality_features__found_in_replicates
* restrict_to_high_quality_features__minimum_intensity_filter
* print_results_overview
* plot_RSDs_per_group
* calc_volcano_plots
* calc_2D_Embedding
* generate_feature_abundance_plot

"""


#####################################################################################################
####################################################################################################
##
# File for many functions concerning the processing of a DART-MS experiment.
#

__tidyMSdartmsVersion__ = "0.9.0"


#####################################################################################################
####################################################################################################
##
# Imports
#

from . import fileio, _constants
from . import assay as Assay
from .chem.formula import Formula

import numpy as np

# import numba
import pandas as pd
import plotnine as p9
import math
import scipy

import tqdm
import natsort
import os
from pathlib import Path
import datetime
import functools
import bs4  ## beautifulsoup4 for writing mzML files
import random
import dill
from copy import deepcopy
import logging
import contextlib
import datetime
import csv
import tempfile
import traceback
import time
from collections import OrderedDict
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import umap


#####################################################################################################
####################################################################################################
##
# Convenience functions
#


## Object to record processing times
## useage with RecordExecutionTime():
##     code...
class RecordExecutionTime(object):
    def __init__(self, name=None):
        self.name = name
        self.start = None

    def __enter__(self):
        self.startTime = time.perf_counter()

    def __exit__(self, *args):
        duration = time.perf_counter() - self.startTime
        unit = "seconds"
        if duration > 60 * 60:
            duration = duration / 60 / 60
            unit = "hours"
        elif duration > 60:
            duration = duration / 60
            unit = "minutes"
        elif duration > 0:
            pass
        elif duration > 1 / 1e3:
            duration = duration * 1e3
            unit = "milli-seconds"
        elif duration > 1 / 1e6:
            duration = duration * 1e6
        logging.info(".. took %.1f %s to execute" % (duration, unit))


def _add_row_to_pandas_creation_dictionary(dict=None, **kwargs):
    """Generate or extend a dictonary object with list objects that can later be converted to a pandas dataframe.
    Warning: no sanity checks will be performed

    Parameters
    ----------
    dict : dict, optional
        The dictionary to be extended, by default None
    **kwargs : dict
        The entries to be added to the dictionary


    Returns
    -------
    dict
        The extended dictionary
    """

    if dict is None:
        dict = {}

    if "dict" in kwargs:
        del kwargs["dict"]

    for kwarg in kwargs:
        if kwarg not in dict:
            dict[kwarg] = []
        dict[kwarg].append(kwargs[kwarg])

    return dict


global _average_and_std


# @numba.jit(nopython=True)
def _average_and_std(values, weights=None):
    """
    Return the weighted average and standard deviation.

    Parameters
    ----------
    values : numpy.ndarray
        The values to average and standard deviation.
    weights : numpy.ndarray, optional
        The weights, by default equal weights are used.

    Returns
    -------
    float, float
        The weighted average and standard deviation.
    """
    if weights is None:
        weights = np.ones((values.shape[0]))

    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


global _relative_standard_deviation


# @numba.jit(nopython=True)
def _relative_standard_deviation(values, weights=None):
    """
    Calculated the relative standard deviation for vals, optionally using the weights

    Parameters
    ----------
    vals : list of numbers or numpy numeric array
        the values of which the RSD is seeked
    weights : list of numbers or numpy numeric array, optional
        weights of the individual values. Defaults to None, in which case an equal weight will be used for each value.

    Returns
    -------
    numeric
        calculated relative standard deviation
    """
    if weights == None:
        weights = np.ones((values.shape[0]))

    avg, std = _average_and_std(values, weights)
    return std / avg


def _mz_deviationPPM_between(a, b):
    """
    Calculate the difference between the two mz values a and b in ppm relative to b

    Parameters
    ----------
    a : numeric
        the first mz value
    b : numeric
        the second mz value

    Returns
    -------
    numeric
        the difference in ppm
    """
    return (a - b) / b * 1e6


def _find_feature_by_mz(features, mz, max_deviation_ppm=None):
    """
    Search for the feature (in feature) that is closest to a given mz value (mz) and has a maximum deviation of (max_deviation_ppm)

    Parameters
    ----------
    features : list of numerics or numpy array of numeric values
        all features to be used for the search
    mz : numeric
        the mz value to be searched for
    max_deviation_ppm : int, optional
        the maximum allowed deviation between the searched mz and potential hits. Defaults to None, which does not restrict the difference.

    Returns
    -------
    index, mz_deviation
        the index of the found feature closest to the reference mz value and the deviation in ppm
    """
    mzmax = 1e6
    mzmin = 0
    if max_deviation_ppm is not None:
        mzmax = mz * (1.0 + max_deviation_ppm / 1e6)
        mzmin = mz * (1.0 - max_deviation_ppm / 1e6)
    ind = np.argmin(np.abs(features[:, 1] - mz))

    if max_deviation_ppm is None or (mzmin <= features[ind, 1] <= mzmax):
        return ind, (features[ind, 1] - mz) / mz * 1e6

    return None, None


def cohen_d(d1, d2):
    """
    Calculate cohen's d value for effect size

    Parameters
    ----------
    d1 : list or numpy array of numerics
        numeric values of group 1
    d2 : list or numpy array of numerics
        numeric values of group 2

    Returns
    -------
    numeric
        cohen's d value for the two groups
    """
    # copied from https://machinelearningmastery.com/effect-size-measures-in-python/ and modified
    # calculate the size of samples
    n1, n2 = d1.shape[0], d2.shape[0]
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


#####################################################################################################
####################################################################################################
##
# MSData import filters
#


def import_filter_mz_range(msData, min_mz, max_mz):
    """
    Filter mz values within a certain range before importing the mzML data

    Parameters
    ----------
    msData : MSData object of tidyms
        the loaded mzML raw data to be filtered
    min_mz : numeric
        the lower mz value to be used
    max_mz : numeric
        the higher mz value to be used

    Returns
    -------
    MSData
        the altered or changed MSData object
    """
    for spectrumi, spectrum in msData.get_spectra_iterator():
        use = np.logical_and(spectrum.mz >= min_mz, spectrum.mz <= max_mz)

        spectrum.mz = spectrum.mz[use]
        spectrum.spint = spectrum.spint[use]

    return msData


def import_filter_artifact_removal(msData, artifacts):
    """
    Filter artefacts before importing the mzML data

    Parameters
    ----------
    msData : MSData object of tidyms
        the loaded mzML raw data to be filtered
    artifacts : list of (mz_min and mz_max) tuples
        a variable number of artifacts to be removed from the dataset. Each eantry must be a tuple of a minimum and maximum mz value describing the artifacts

    Returns
    -------
    MSData
        the altered or changed MSData object
    """
    for spectrumi, spectrum in msData.get_spectra_iterator():
        use = spectrum.mz >= 0

        for artifact in artifacts:
            use = np.logical_and(use, np.logical_and(spectrum.mz >= artifact[0], spectrum.mz <= artifact[1]))

        spectrum.mz = spectrum.mz[use]
        spectrum.spint = spectrum.spint[use]

    return msData


def import_filter_remove_signals_below_intensity(msData, minimum_signal_intensity):
    """
    Filter signals below a minimum intensity threshold

    Parameters
    ----------
    msData : MSData object of tidyms
        the loaded mzML raw data to be filtered
    minimum_signal_intensity : numeric
        the minimum intensity value for signals to be used

    Returns
    -------
    MSData
        the altered or changed MSData object
    """
    for spectrumi, spectrum in msData.get_spectra_iterator():
        use = spectrum.spint >= minimum_signal_intensity

        spectrum.mz = spectrum.mz[use]
        spectrum.spint = spectrum.spint[use]

    return msData


#####################################################################################################
####################################################################################################
##
# Feature quality test function
#


def cluster_quality_check_function__peak_form(sample, msDataObj, spectrumIDs, time, mz, intensity, cluster, min_correlation_for_cutoff=0.5):
    """
    A function to check the detected feature clusters for certain attributes.
    This particular function checks if the distribution form somehow resembles a spot (approximated by a normal distribution).
    Any cluster not resembling such a form will be removed in a subsequent step (by setting the cluster ids to -1)

    Parameters
    ----------
    sample : string)
        the name of the sample
    msDataObj : MSData object of tidyms
        he MSData object in which this feature was detected
    spectrumIDs : list of ids
        the spectrum id of each found signal in a cluster
    time : list of numeric
        the chronogram time of each found signal in a cluster
    mz : list of numeric
        the mz value of each found signal in a cluster
    intensity : list of numeric
        the intensity value of each found signal in a cluster
    cluster : list of integer
        the cluster ids each signal was assigned to
    min_correlation_for_cutoff : float, optional
        the minimum Pearson correlation cutoff for the spot shape form comparison [-1 to 1]. Defaults to 0.5.

    Returns
    -------
    list of integer
        the new cluster ids each signal is assigned to. clusters to be removed are set to -1
    """
    removed = 0
    clustInds = np.unique(cluster)
    refTimes = np.array([spectrum.time for k, spectrum in msDataObj.get_spectra_iterator()])
    refEIC = scipy.stats.norm.pdf(refTimes, loc=np.mean(refTimes), scale=(np.max(refTimes) - np.min(refTimes)) / 6)
    corrs = []

    for clusti, clustID in enumerate(clustInds):
        if clustID >= 0:
            mzs = mz[cluster == clustID]
            if mzs.shape[0] > 0:
                ints = intensity[cluster == clustID]
                times = time[cluster == clustID]

                eic = np.zeros_like(refTimes)
                for i in range(ints.shape[0]):
                    eic[np.argwhere(times[i] == refTimes)] += ints[i]

                corr = np.corrcoef(refEIC, eic)[1, 0]
                corrs.append(corr)

                if corr < min_correlation_for_cutoff:
                    cluster[cluster == clustID] = -1
                    removed = removed + 1

    if False:
        temp = pd.DataFrame({"correlations": corrs})
        p = (
            p9.ggplot(data=temp, mapping=p9.aes(x="correlations"))
            + p9.geom_histogram(binwidth=0.1)
            + p9.geom_vline(xintercept=min_correlation_for_cutoff)
            + p9.ggtitle("correlations in sample '%s' removed %d corrs %d" % (sample, removed, len(corrs)))
        )
        print(p)

    return cluster


def cluster_quality_check_function__ppmDeviationCheck(sample, msDataObj, spectrumIDs, time, mz, intensity, cluster, max_weighted_ppm_deviation=15):
    """
    A function to check the detected feature clusters for certain attributes.
    This particular function checks if the clusters are within a certain ppm devaition. Any cluster exceeding this deviation will be removed in a subsequent step (by setting the cluster ids to -1)

    Parameters
    ----------
    sample : string
        he name of the sample
    msDataObj : MSData object of tidyms
        the MSData object in which this feature was detected
    spectrumIDs : list of ids
        the spectrum id of each found signal in a cluster
    time : list of numeric
        the chronogram time of each found signal in a cluster
    mz : list of numeric
        the mz value of each found signal in a cluster
    intensity : list of numeric
        the intensity value of each found signal in a cluster
    cluster : list of integer
        the cluster ids each signal was assigned to
    max_weighted_ppm_deviation : float, optional
        the maximum allowed ppm deviation for all signals within a cluster.

    Returns
    -------
    list of integer
        the new cluster ids each signal is assigned to. clusters to be removed are set to -1
    """
    removed = 0
    clustInds = np.unique(cluster)

    for clusti, clustID in enumerate(clustInds):
        if clustID >= 0:
            mzs = mz[cluster == clustID]
            if mzs.shape[0] > 0:
                ints = intensity[cluster == clustID]

                mmzW = np.average(mzs, weights=intensity[cluster == clustID])
                ppmsW = (mzs - mmzW) / mmzW * 1e6
                stdppmW = np.sqrt(np.cov(ppmsW, aweights=ints))

                if stdppmW > max_weighted_ppm_deviation:
                    cluster[cluster == clustID] = -1
                    removed = removed + 1

    return cluster


global _refine_clustering_for_mz_list


# @numba.jit(nopython=True)
def _refine_clustering_for_mz_list(
    sample,
    mzs,
    intensities,
    spectrumID,
    clusts,
    closest_signal_max_deviation_ppm=None,
    max_mz_deviation_ppm=None,
    max_ppm_std_deviation_in_cluster=None,
):
    """
    Function to refine a crude clustering of signals based on their mz values. This step is performed separately for each crude cluster (i.e., all signals put into the same cluster based on their cluster ids)
    The idea of the refinement process resembles a hierarichal clustering.
    For each cluster, all signals will be treated considered in an iteration. Then, those two signals with the closest mz values are put into a new cluster and new signals not assigned to new clusters are added.
    However, once a new mz value with a too high mz deviation were to be added, the adding step is aborted, the previous new cluster is closed and a new cluster is started. This process is repeated until
    no further signals remain in for the currently inspected cluster

    Parameters
    ----------
    sample : string
        the name of the sample to be processed
    mzs : list of numeric
        the mz values of the signals or features
    intensities : list of numeric
        the intensity values of the signals or features
    spectrumID : list of ids
        the spectrum ids of the signals or features
    clusts : list of ids
        the cluster ids of the signals or features to which they were assigned
    closest_signal_max_deviation_ppm : int, optional
        the search window for adjacent signals. Defaults to 20.
    max_mz_deviation_ppm : _type_, optional
        the maximum mz devation above which a cluster is automatically split into subcluster. Defaults to None.

    Returns
    -------
    list of ids
        the new cluster ids for each signal or feature
    """
    clustInds = np.unique(clusts)
    nextClust = 0
    newClusts = np.zeros_like(clusts) - 1

    debugPrint_ = False

    for i, clust in enumerate(clustInds):
        n = np.sum(clusts == i)

        if debugPrint_:
            print("current cluster is ", i, clust)

        pos = np.argwhere(clusts == clust)[:, 0]
        mzs_ = mzs[pos]
        intensities_ = intensities[pos]

        ## suggestion by Gabriel: sort array to improve speed of algorithm
        ord_ = np.argsort(mzs_)
        mzs_ = mzs_[ord_]
        intensities_ = intensities_[ord_]
        diffs_ = mzs_[1:] - mzs_[: mzs_.shape[0] - 1]
        maxmz_ = np.amax(mzs_) * 2
        usedSignals_ = 0

        if debugPrint_:
            print("  size of signals in cluster", mzs_.shape[0])

        ## test signals until no more are available
        while mzs_.shape[0] - usedSignals_ >= 2:
            ## find seed to start, two closest mz values
            minInd = np.argmin(diffs_)
            start, end = minInd, minInd

            ## calculate extending variance and mean
            newMean, newStd = mzs_[start], 0
            curMean, curStd = newMean, newStd

            if debugPrint_:
                print("  selected mz as seed", curMean, diffs_[minInd])

            ## extend similar mz values until difference gets too large
            run = True
            while run:
                left = start > 0
                right = end < mzs_.shape[0] - 1

                newStart = start
                newEnd = end

                closeCluster = False
                if not left and not right:
                    if debugPrint_:
                        print("     - closing cluster without extension to left/right")
                    closeCluster = True

                elif (left and right and abs(curMean - mzs_[start - 1]) <= abs(mzs_[end + 1] - curMean)) or (left and not right):
                    newStart = start - 1
                    addedPPMDev = abs(mzs_[start] - mzs_[newStart]) / curMean * 1e6
                    newTotalPPMDev = abs(mzs_[end] - mzs_[newStart]) / curMean * 1e6
                    if debugPrint_:
                        print("     - extending cluster to the left with new values", addedPPMDev, newTotalPPMDev)

                elif (left and right and abs(mzs_[end + 1] - curMean) < abs(curMean - mzs_[start - 1])) or (not left and right):
                    newEnd = end + 1
                    addedPPMDev = abs(mzs_[newEnd] - mzs_[end]) / curMean * 1e6
                    newTotalPPMDev = abs(mzs_[newEnd] - mzs_[start]) / curMean * 1e6
                    if debugPrint_:
                        print("     - extending cluster to the right with new values", addedPPMDev, newTotalPPMDev)

                else:
                    raise NotImplementedError("Unknwon branch, aborting")

                if max_ppm_std_deviation_in_cluster is not None:
                    newMean, newStd = _average_and_std(mzs_[newStart : newEnd + 1], weights=intensities_[newStart : newEnd + 1])
                if debugPrint_:
                    print("     - new mean is", newMean, newStd, newStd / newMean * 1e6, "old was", curMean, curStd, curStd / curMean * 1e6)

                if (
                    closeCluster
                    or (closest_signal_max_deviation_ppm is not None and addedPPMDev > closest_signal_max_deviation_ppm)
                    or (max_mz_deviation_ppm is not None and newTotalPPMDev > max_mz_deviation_ppm)
                    or (max_ppm_std_deviation_in_cluster is not None and newStd / newMean * 1e6 > max_ppm_std_deviation_in_cluster)
                ):
                    if debugPrint_:
                        print("    --> closing cluster as ", nextClust, "using", start, end, "which are", mzs_[start : end + 1])

                    for i in range(start, end + 1):
                        newClusts[pos[ord_[i]]] = nextClust

                    for i in range(start, end):
                        diffs_[i] = maxmz_
                    if start > 0:
                        diffs_[start - 1] = maxmz_
                    if end < diffs_.shape[0]:
                        diffs_[end] = maxmz_

                    usedSignals_ += end - start + 1

                    nextClust += 1
                    run = False

                else:
                    start = newStart
                    end = newEnd

                    curMean = newMean
                    curStd = newStd

        ## Closing last cluster no longer necessary since this is done already at the beginning
        # if mzs_.shape[0] - usedSignals_ > 0:
        #    if debugPrint_:
        #        print("    --> closing last cluster as ", nextClust, "which are", mzs_)
        #
        #    ## Not necessary any more, is set per default earlier
        #    #newClusts[pos[ord_]] = nextClust
        #    #nextClust += 1

    return newClusts


#####################################################################################################
####################################################################################################
##
# Data handling assay
#


class Parameters:
    def __init__(self, name, comment=None, args=None, kwargs=None):
        if type(name) != str:
            raise ValueError("name of Parameters must be a str")
        if comment is not None and type(comment) != str:
            raise ValueError("comment of Parameters must be a str")
        if args is not None and type(args) != list:
            raise ValueError("args of Parameters must be a list")
        if kwargs is not None and type(kwargs) != dict:
            raise ValueError("kwargs of Parameters must be dict")

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        self.name = name
        self.comment = comment
        self.args = args
        self.kwargs = kwargs


def compare_parameters_for_function(
    function_to_optimize,
    parameter_values,
    spotFile,
    dartMSFiles,
    referenceFeatures,
    referenceFeatures_allowedPPMDev,
    qualityTestFunction=None,
    add_execution_time=True,
):
    succeeded = {}
    failed = {}

    print()
    print()
    print("##############################################################################")
    print("##############################################################################")
    print()
    print("Starting parameter comparison")
    print()
    print()

    for parami, param in enumerate(parameter_values):
        _startTime = time.time()
        print("##############################################################################")
        print("Setname: %s" % (param.name))
        print("Comment: %s" % (param.comment))

        try:
            dartMSAssay = function_to_optimize(*param.args, **(param.kwargs | {"spotFile": spotFile, "files": dartMSFiles}))
            result = None
            if qualityTestFunction is None:
                result = dartMSAssay.get_summary_of_results(
                    reference_features=referenceFeatures, reference_features_allowed_deviationPPM=referenceFeatures_allowedPPMDev
                )
            else:
                result = qualityTestFunction(dartMSAssay)

            res = OrderedDict()
            res["_parameterSet"] = param.name
            res["_parameterSetComment"] = param.comment
            res["_executionTime"] = time.time() - _startTime
            res = res | result

            succeeded = _add_row_to_pandas_creation_dictionary(succeeded, **res)

        except Exception as ex:
            print("***************")
            print("Exception occurred")
            print(traceback.print_exc())

            res = OrderedDict()
            res["_parameterSet"] = param.name
            res["_parameterSetComment"] = param.comment
            res["_executionTime"] = time.time() - _startTime
            res = res | {"exception": traceback.format_exc()}

            failed = _add_row_to_pandas_creation_dictionary(failed, **res)

        print()
        print()

    print("##############################################################################")
    print()

    print("Parameter comparison finished")
    print("   .. took %.1f minutes" % ((time.time() - _startTime) % 60.0))
    print("   .. %d sets finished, %d sets failed" % (len(succeeded), len(failed)))

    print("##############################################################################")
    print("##############################################################################")
    print()
    print()

    return succeeded, failed


## Generic pipeline for data processing and semi-automated parameter optimization
## Each parameter of the function is translated to a parameter of the respective method in the pipeline
## Different parameter combinations can easily be tested. The keys of the dictionary must be the names of the parameters
## and the values the parameter values.
def prefab_DARTMS_dataProcessing_pipeline(
    spotFile,
    files,
    ms_mode="centroid",
    instrument="qtof",
    fileNameChangeFunction=None,
    create_assay_from_chronogramFiles__import_filters=None,
    select_top_n_spectra__top_n_spectra=None,
    correct_mz_shift__referenceMZs=None,
    correct_mz_shift__max_mz_deviation_absolute=0.1,
    correct_mz_shift__max_deviationPPM_to_use_for_correction=200,
    correct_mz_shift__correctby="mzDeviationPPM",
    correct_mz_shift__selection_criteria="mostabundant",
    correct_mz_shift__correct_on_level="file",
    calculate_consensus_spectra_for_samples__min_difference_ppm=100,
    calculate_consensus_spectra_for_samples__min_signals_per_cluster=5,
    calculate_consensus_spectra_for_samples__minimum_intensity_for_signals=2.5e2,
    calculate_consensus_spectra_for_samples__cluster_quality_check_functions=None,
    normalize_to_internal_standard__perform=False,
    normalize_to_internal_standard__internal_standard_mzs=None,
    normalize_to_internal_standard__multiplication_factor=1,
    bracket_consensus_spectrum_samples__max_ppm_deviation=25,
    annotate_features__useGroups=None,
    annotate_features_remove_other_ions=False,
    build_data_matrix__originalData_mz_deviation_multiplier_PPM=30,
    build_data_matrix__aggregation_fun="average",
    results_file=None,
    dill_file=None,
):
    """
    Entire DART-MS workflow

    Function processes samples (parameter file) and their spots (spotFile) using
    1. Top-N spectra selection (optional)
    2. m/z shift correction
    3. Consensus spectra generation
    4. Internal standard normalizatoin
    5. Bracketing of features across samples
    6. Generation of data matrix
    7. Annotation of common adducts and isotopologs

    The results of the processing are returned in form of a new DartMSAssay object and can optionally also be saved
    to a dill file.

    For a detailed explanation of the parameters <parameter_name> please see the respective function function_name.
    Each parameter name has the form <function_name>__<parameter_name>.

    Note: the spotFile must already exist

    Parameters
    ----------
    spotFile : string
        The path to the spotFile
    files : list of str
        The path to the raw-data
    dill_file : optional, str
        the path to the dill file for storing the results

    Returns
    -------
        (DartMSAssay object of the processing)
    """
    if create_assay_from_chronogramFiles__import_filters is None:
        create_assay_from_chronogramFiles__import_filters = []
    if fileNameChangeFunction is None:
        fileNameChangeFunction = lambda x: x
    if correct_mz_shift__referenceMZs is None:
        correct_mz_shift__referenceMZs = [149.02671]
    if calculate_consensus_spectra_for_samples__cluster_quality_check_functions is None:
        calculate_consensus_spectra_for_samples__cluster_quality_check_functions = list(
            [
                functools.partial(
                    cluster_quality_check_function__ppmDeviationCheck,
                    max_weighted_ppm_deviation=15,
                ),
                functools.partial(
                    cluster_quality_check_function__peak_form,
                    min_correlation_for_cutoff=0.25,
                ),
            ],
        )
    if normalize_to_internal_standard__internal_standard_mzs is None:
        normalize_to_internal_standard__internal_standard_mzs = (316.29, 316.325)
    if annotate_features__useGroups is None:
        annotate_features__useGroups = []

    ## Process samples if the last results cannot be loaded
    ## Import chronograms and separate them into spots.
    with RecordExecutionTime():
        logging.info("Importing and separating chronograms")
        dartMSAssay = DartMSAssay.create_assay_from_chronogramFiles(
            "Semi-automated parameter optimization",
            files,
            ms_mode=ms_mode,
            instrument=instrument,
            spot_file=spotFile,
            centroid_profileMode=True,
            fileNameChangeFunction=fileNameChangeFunction,
            intensity_threshold_spot_extraction=0,
            import_filters=create_assay_from_chronogramFiles__import_filters,
        )
    logging.info("")

    ## Select only top-n spectra
    with RecordExecutionTime():
        logging.info("Selecting only top-n abundant scans")
        dartMSAssay.select_top_n_spectra(n=select_top_n_spectra__top_n_spectra)
    logging.info("")

    ## Account for and correct mz values of individual spots by reference mz values
    with RecordExecutionTime():
        logging.info("Correcting for mz shifts between samples (not chronograms) with reference mz values")
        dartMSAssay.correct_MZ_shift_across_samples(
            correct_mz_shift__referenceMZs,
            max_mz_deviation_absolute=correct_mz_shift__max_mz_deviation_absolute,
            max_deviationPPM_to_use_for_correction=correct_mz_shift__max_deviationPPM_to_use_for_correction,
            correctby=correct_mz_shift__correctby,
            selection_criteria=correct_mz_shift__selection_criteria,
            correct_on_level=correct_mz_shift__correct_on_level,
            plot=False,
        )
    logging.info("")

    ## Calculate consensus spectra for each spot file to reduce following processing time
    with RecordExecutionTime():
        logging.info("Collapsing multiple spectra of samples to consensus spectra for each sample")
        dartMSAssay.calculate_consensus_spectra_for_samples(
            min_difference_ppm=calculate_consensus_spectra_for_samples__min_difference_ppm,
            min_signals_per_cluster=calculate_consensus_spectra_for_samples__min_signals_per_cluster,
            minimum_intensity_for_signals=calculate_consensus_spectra_for_samples__minimum_intensity_for_signals,
            cluster_quality_check_functions=calculate_consensus_spectra_for_samples__cluster_quality_check_functions,
            aggregation_function="average",
            exportAsFeatureML=False,
        )
    logging.info("")

    ## Normalize to internal standard
    if normalize_to_internal_standard__perform:
        with RecordExecutionTime():
            logging.info("Normalize to internal standard")
            dartMSAssay.normalize_to_internal_standard(
                normalize_to_internal_standard__internal_standard_mzs,
                multiplication_factor=normalize_to_internal_standard__multiplication_factor,
                plot=False,
            )
        logging.info("")

    ## Bracket/group mz features across samples of the experiment
    with RecordExecutionTime():
        logging.info("Bracketing features across samples")
        dartMSAssay.bracket_consensus_spectrum_samples(
            max_ppm_deviation=bracket_consensus_spectrum_samples__max_ppm_deviation, show_diagnostic_plots=False
        )
        logging.info("   .. bracketed to %d features" % (len(dartMSAssay.features)))
    logging.info("")

    ## Generate data matrix
    with RecordExecutionTime():
        logging.info("Generating matrix")
        dartMSAssay.build_data_matrix(
            on="originalData",
            originalData_mz_deviation_multiplier_PPM=build_data_matrix__originalData_mz_deviation_multiplier_PPM,
            aggregation_fun=build_data_matrix__aggregation_fun,
        )
        dartMSAssay.annotate_features(useGroups=annotate_features__useGroups, remove_other_ions=annotate_features_remove_other_ions, plot=False)
    logging.info("")

    ## Save results to file for another user-check
    with RecordExecutionTime():
        if dill_file is not None:
            dartMSAssay.save_self_to_dill_file(dill_file)
        if results_file is not None:
            dartMSAssay.write_bracketing_results_to_featureML(featureMLlocation=results_file, featureMLStartRT=0, featureMLEndRT=1400)

    return dartMSAssay


#####################################################################################################
####################################################################################################
##
# Data handling assay
#


class DartMSAssay:
    """
    An object inspired by tidyms' Assay class that encapsulates a DARTMS experiment.
    """

    #####################################################################################################
    # Init
    #
    def __init__(self, name="Generic"):
        """
        Constructor for a new DartMSAssay object

        Parameters
        ----------
        name : str, optional
            name of the experiment. Defaults to "Generic".
        """
        self.name = name

        self.assay = None
        self.dat = None
        self.features = None
        self.featureAnnotations = None
        self.samples = None
        self.groups = None
        self.batches = None

        self.processingHistory = []

    def clone_DartMSAssay(self):
        """
        Clones the DartMSAssay object (deepcopy)

        Returns
        -------
        DartMSAssay
            the cloned, new DartMSAssay object
        """
        _c = deepcopy(self)
        _c.add_data_processing_step("Clone", "This object is a clone.")

        return _c

    #####################################################################################################
    # Data handling
    #
    def set_data(self, dat, features, featureAnnotations, samples, groups, batches):
        """
        Sets the data for a DartMSAssay object

        Parameters
        ----------
        dat : numpy.ndarray of [n, m]
            the feature table of the experiment
        features : list of mz values
            the features' information (mz values)
        featureAnnotations : list of dictionaries
            the features' derived information (sister ions, etc.)
        samples : list of string
            the names of the samples in the experiment
        groups : list of str
            the group names the samples in the experiment are assigned to
        batches : list of int
            the batch ids the samples in the experiment are assigned to
        """
        self.dat = dat
        self.features = features
        self.featureAnnotations = featureAnnotations
        self.samples = samples
        self.groups = groups
        self.batches = batches

    def get_data_matrix_and_meta(
        self,
        keep_features=None,
        remove_features=None,
        keep_samples=None,
        remove_samples=None,
        keep_groups=None,
        remove_groups=None,
        keep_batches=None,
        remove_batches=None,
        copy=False,
    ):
        """
        Returns the data matrix as well as feature and samples informatoin available in the DartMSAssay object.
        Features and samples/groups/batches can be included or excluded before the export.

        Parameters
        ----------
        keep_features : list of indices, optional
            features to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_features : list of indices, optional
            features to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        keep_samples : list of strings, optional
            samples to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_samples : list of strings, optional
            samples to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        keep_groups : list of strings, optional
            groups to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_groups : list of strings, optional
            groups to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        keep_batches : list of strings, optional
            batches to be inlcuded in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_batches : list of integers, optional
            batches to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        copy : bool, optional
            indicator if the object should be cloned before export. This will be done automatically if any inclusion or restriction is provided. Defaults to False.

        Raises
        ======
        RuntimeError
            an exception is raised when the necessary data is not available

        Returns
        -------
        (numpy data matrix [samples x features], list of feature properties, list of feature annotaitons, list of sample names, list of assigned group names, list of assigned batches
            data matrix and meta-data
        """
        if self.dat is None:
            raise RuntimeError("Data matrix not set/generated")

        if (
            copy
            or keep_features is not None
            or remove_features is not None
            or keep_samples is not None
            or remove_samples is not None
            or keep_groups is not None
            or remove_groups is not None
            or keep_batches is not None
            or remove_batches is not None
        ):
            _dat = np.copy(self.dat)
            _features = deepcopy(self.features)
            _featureAnnotations = deepcopy(self.featureAnnotations)
            _samples = deepcopy(self.samples)
            _groups = deepcopy(self.groups)
            _batches = deepcopy(self.batches)

            if keep_features is not None:
                keeps = [i for i in range(_dat.shape[1]) if i in keep_features]

                _dat = _dat[:, keeps]
                _features = [_features[i] for i in keeps]
                _featureAnnotations = [_featureAnnotations[i] for i in keeps]

            if remove_features is not None:
                keeps = [i for i in range(_dat.shape[1]) if i not in remove_features]

                _dat = _dat[:, keeps]
                _features = [_features[i] for i in keeps]
                _featureAnnotations = [_featureAnnotations[i] for i in keeps]

            if keep_samples is not None:
                keeps = [i for i in range(len(_samples)) if _samples[i] in keep_samples]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            if remove_samples is not None:
                keeps = [i for i in range(len(_samples)) if _samples[i] not in remove_samples]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            if keep_groups is not None:
                keeps = [i for i in range(len(_groups)) if _groups[i] in keep_groups]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            if remove_groups is not None:
                keeps = [i for i in range(len(_groups)) if _groups[i] not in remove_groups]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            if keep_batches is not None:
                keeps = [i for i in range(len(_batches)) if _batches[i] in keep_batches]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            if remove_batches is not None:
                keeps = [i for i in range(len(_batches)) if _batches[i] not in remove_batches]

                _dat = _dat[keeps, :]
                _samples = [_samples[i] for i in keeps]
                _groups = [_groups[i] for i in keeps]
                _batches = [_batches[i] for i in keeps]

            return _dat, _features, _featureAnnotations, _samples, _groups, _batches

        return self.dat, self.features, self.featureAnnotations, self.samples, self.groups, self.batches

    def export_data_matrix(self, to_file, separator="\t", quotechar='"'):
        """
         Export the data matrix to a tsv file

         Parameters
         ----------
        to_file : string
             the file to save the results to
        """
        with open(to_file, "w") as fout:
            tsvWriter = csv.writer(fout, delimiter=separator, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
            headers = ["mzmean", "mzmin", "mzmax", "annotations"] + self.samples
            headers = [h.replace(separator, "--SEP--") for h in headers]
            tsvWriter.writerow(headers)

            for rowi in range(self.dat.shape[1]):
                row = [str(self.features[rowi][1]), str(self.features[rowi][0]), str(self.features[rowi][2]), str(self.featureAnnotations[rowi])]
                row.extend((str(f) if not np.isnan(f) else "" for f in self.dat[:, rowi]))
                row = [r.replace(separator, "--SEP--") for r in row]
                tsvWriter.writerow(row)

    def export_annotations(self, to_file):
        with open(to_file, "w") as fout:
            fout.write(json.dumps(self.featureAnnotations, indent=2))
            fout.write("\n")

    def export_for_R(self, to_file):
        """
        Export the results to a tsv file and generate R code to import it

        Parameters
        ----------
        to_file : string
            the location of the file (without extension, '.tsv' will be added automatically)
        """
        self.export_data_matrix(to_file + ".tsv", separator="\t", quotechar='"')
        self.export_annotations(to_file + ".json")
        with open(to_file + "_import.R", "w") as fout:
            fout.write("## Optional package installs")
            fout.write("\n")
            fout.write("")
            fout.write("\n")
            fout.write("## (execute if necessary)")
            fout.write("\n")
            fout.write("## install.packages('rjson')")
            fout.write("\n")
            fout.write("")
            fout.write("\n")
            fout.write("## Load required libraries")
            fout.write("\n")
            fout.write("library('rjson')")
            fout.write("\n")
            fout.write("")
            fout.write("\n")
            fout.write("## Import data matrix")
            fout.write("\n")
            fout.write("## ")
            fout.write("\n")
            fout.write("## The following code imports the generated data matrix.")
            fout.write("\n")
            fout.write("## Please note that when the file is moved or R is executed in a different")
            fout.write("\n")
            fout.write("##      working directory, the path to the tsv file needs to be adapted accordingly")
            fout.write("\n")
            fout.write("##")
            fout.write("\n")
            fout.write("data = read.table('%s', header = TRUE, sep = '\\t', stringsAsFactors = FALSE)" % (to_file + ".tsv"))
            fout.write("\n")
            fout.write("annotations = fromJSON(paste(readLines('%s'), collapse=''))" % (to_file + ".json"))
            fout.write("\n")
            fout.write("metaData = data[, 1:4]")
            fout.write("\n")
            fout.write("data = data[,-(1:4)]")
            fout.write("\n")
            fout.write("samples = c(%s)" % (", ".join("'%s'" % sample for sample in self.samples)))
            fout.write("\n")
            fout.write("groups = c(%s)" % (", ".join("'%s'" % group for group in self.groups)))
            fout.write("\n")
            fout.write("batches = c(%s)" % (", ".join("%s" % batche for batche in self.batches)))
            fout.write("\n")
            fout.write("")
            fout.write("\n")
            fout.write("print(sprintf('Data imported, there are %d features and %d samples in the data matrix', nrow(data), ncol(data)))")
            fout.write("\n")
            fout.write("")
            fout.write("\n")

            print("load the data into R with the command 'source('%s')'" % (to_file + "_import.R").replace("\\", "/"))
            print("## absolute path version: 'source('%s')'" % str(Path(to_file + "_import.R").absolute()).replace("\\", "/"))

    #####################################################################################################
    # Processing history
    #

    def add_data_processing_step(self, step_identifier_text, log_text, processing_data=None):
        """
        Adds a data processing step to the log of the DartMSAssay object

        Parameters
        ----------
        step_identifier_text : string
            name of the data processing step
        log_text : string
            description of the data processing step
        processing_data : dict, optional
            further inforamtion (e.g., parameters) of the data processing step. Defaults to None.
        """
        self.processingHistory.append(
            {"step_identifier": step_identifier_text, "log_text": log_text, "processing_data": processing_data, "at": str(datetime.datetime.now())}
        )

    #####################################################################################################
    # IO
    #

    def save_self_to_dill_file(self, dill_file):
        """
        Save the DartMSAssay object to a file

        Parameters
        ----------
        dill_file : string
            the *.dill file to save the DartMSAssay to
        """
        self.add_data_processing_step("export", "exported assay to dill file", {"file": dill_file})
        with open(dill_file, "wb") as fout:
            dill.dump(
                {
                    "name": self.name,
                    "assay": self.assay,
                    "dat": self.dat,
                    "features": self.features,
                    "featureAnnotations": self.featureAnnotations,
                    "samples": self.samples,
                    "groups": self.groups,
                    "batches": self.batches,
                    "processingHistory": self.processingHistory,
                },
                fout,
            )

    @staticmethod
    def read_from_dill_file(dill_file):
        """
        Load a DartMSAssay from a file

        Parameters
        ----------
        dill_file : string
            the *.dill file to load the DartMSAssay from

        Returns
        -------
        DartMSAssay
            the loaded DartMSAssay object
        """
        with open(dill_file, "rb") as fin:
            di = dill.load(fin)

            dartMSAssay = DartMSAssay(di["name"])

            dartMSAssay.assay = di["assay"]
            dartMSAssay.dat = di["dat"]
            dartMSAssay.features = di["features"]
            dartMSAssay.featureAnnotations = di["featureAnnotations"]
            dartMSAssay.samples = di["samples"]
            dartMSAssay.groups = di["groups"]
            dartMSAssay.batches = di["batches"]

            dartMSAssay.processingHistory = di["processingHistory"]

            dartMSAssay.add_data_processing_step("imported", "imported from dill file", {"file": dill_file})

            return dartMSAssay

    #####################################################################################################
    # Subset results
    #

    def subset_features(self, keep_features_with_indices=None, remove_features_with_indices=None):
        """
        Subset the detected features and include or exclude them

        Parameters
        ----------
        keep_features : list of indices, optional
            features to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_features : list of indices, optional
            features to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.

        Raises
        ======
        RuntimeError
            if no features have been detected, this exception will be raised
        """
        if self.dat is None:
            raise RuntimeError("Cannot subset DartMSAssay until a data matrix has been generated")

        self.add_data_processing_step(
            "subset features",
            "subsetting features",
            {"keep_features_with_indices": keep_features_with_indices, "remove_features_with_indices": remove_features_with_indices},
        )
        if keep_features_with_indices is not None:
            self.dat = self.dat[:, keep_features_with_indices]
            if self.features is not None:
                self.features = [self.features[i] for i in keep_features_with_indices]
            if self.featureAnnotations is not None:
                self.featureAnnotations = [self.featureAnnotations[i] for i in keep_features_with_indices]

        if remove_features_with_indices is not None:
            keeps = [i for i in range(self.dat.shape[1]) if i not in remove_features_with_indices]
            self.dat = self.dat[:, keeps]
            if self.features is not None:
                self.features = [self.features[i] for i in keeps]
            if self.featureAnnotations is not None:
                self.featureAnnotations = [self.featureAnnotations[i] for i in keeps]

    def subset_samples(self, keep_samples=None, keep_groups=None, keep_batches=None, remove_samples=None, remove_groups=None, remove_batches=None):
        """
        Subset certain samples, groups or batches in the DartMSAssay object

        Parameters
        ----------
        keep_samples : list of strings, optional
            samples to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_samples : list of strings, optional
            samples to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        keep_groups : list of strings, optional
            groups to be included in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_groups : list of strings, optional
            groups to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.
        keep_batches : list of strings, optional
            batches to be inlcuded in the export. To export all, the parameter must be set to None. Defaults to None.
        remove_batches : list of integers, optional
            batches to be removed before exporting. To remove none, the parameter must be set to None. Defaults to None.

        Raises
        ======
        RuntimeError
            if no features have been detected, this exception will be raised
        """
        if self.samples is None or self.groups is None or self.batches is None:
            raise RuntimeError("Unknonw samples/group/batches in DartMSAssay")

        self.add_data_processing_step(
            "subset samples/groups/batches",
            "subset samples/groups/batches",
            {
                "keep_samples": keep_samples,
                "remove_samples": remove_samples,
                "keep_groups": keep_groups,
                "remove_groups": remove_groups,
                "keep_batches": keep_batches,
                "remove_batches": remove_batches,
            },
        )
        if keep_samples is not None:
            keeps = [i for i in range(len(self.samples)) if self.samples[i] in keep_samples]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

        if keep_groups is not None:
            keeps = [i for i in range(len(self.groups)) if self.groups[i] in keep_groups]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

        if keep_batches is not None:
            keeps = [i for i in range(len(self.batches)) if self.batches[i] in keep_batches]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

        if remove_samples is not None:
            keeps = [i for i in range(len(self.samples)) if self.samples[i] not in remove_samples]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

        if remove_groups is not None:
            keeps = [i for i in range(len(self.groups)) if self.groups[i] not in remove_groups]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

        if remove_batches is not None:
            keeps = [i for i in range(len(self.batches)) if self.batches[i] not in remove_batches]

            self.dat = self.dat[keeps, :]
            self.samples = [self.samples[i] for i in keeps]
            self.groups = [self.groups[i] for i in keeps]
            self.batches = [self.batches[i] for i in keeps]

    #####################################################################################################
    # FeatureML functions
    #

    def write_bracketing_results_to_featureML(self, featureMLlocation="./results.featureML", featureMLStartRT=0, featureMLEndRT=1400):
        """
        export the bracketed results to a featureML file for easy visualization in TOPPView

        Parameters
        ----------
        featureMLlocation : str, optional
            path to the featureML file. Defaults to "./results.featureML".
        featureMLStartRT : int, optional
            the earliest chronogram time. Defaults to 0.
        featureMLEndRT : int, optional
            the latest chronogram time. Defaults to 1400.
        """
        self.add_data_processing_step("export bracketed results to featureML", "export bracketed results to featureML", {"file": featureMLlocation})
        with open(featureMLlocation, "w") as fout:
            bracRes = [b[3] for b in self.features]
            ns = len(bracRes)
            fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
            fout.write(
                '  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
            )
            fout.write('    <dataProcessing completion_time="%s">\n' % datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            fout.write('      <software name="tidyms" version="%s" />\n' % (__tidyMSdartmsVersion__))
            fout.write('      <software name="tidyms.write_bracketing_results_to_featureML" version="%s" />\n' % (__tidyMSdartmsVersion__))
            fout.write("    </dataProcessing>\n")
            fout.write('    <featureList count="%d">\n' % (ns))

            for j in tqdm.tqdm(range(ns), desc="exporting featureML"):
                fout.write('<feature id="%s">\n' % j)
                fout.write('  <position dim="0">%f</position>\n' % ((featureMLStartRT + featureMLEndRT) / 2))
                fout.write('  <position dim="1">%f</position>\n' % (bracRes[j]["meanMZ"]))
                fout.write("  <intensity>1</intensity>\n")
                fout.write('  <quality dim="0">0</quality>\n')
                fout.write('  <quality dim="1">0</quality>\n')
                fout.write("  <overallquality>%d</overallquality>\n" % (bracRes[j]["overallquality"]))
                fout.write("  <charge>1</charge>\n")
                fout.write('  <convexhull nr="0">\n')
                fout.write('    <pt x="%f" y="%f" />\n' % (featureMLStartRT, bracRes[j]["meanMZ"] * (1.0 - bracRes[j]["mzDevPPM"] / 1e6)))
                fout.write('    <pt x="%f" y="%f" />\n' % (featureMLStartRT, bracRes[j]["meanMZ"] * (1.0 + bracRes[j]["mzDevPPM"] / 1e6)))
                fout.write('    <pt x="%f" y="%f" />\n' % (featureMLEndRT, bracRes[j]["meanMZ"] * (1.0 + bracRes[j]["mzDevPPM"] / 1e6)))
                fout.write('    <pt x="%f" y="%f" />\n' % (featureMLEndRT, bracRes[j]["meanMZ"] * (1.0 - bracRes[j]["mzDevPPM"] / 1e6)))
                fout.write("  </convexhull>\n")

                for samplei, sample in enumerate(bracRes[j]["sampleHulls"]):
                    fout.write('  <convexhull nr="%d">\n' % (samplei + 1))
                    fout.write('    <pt x="%f" y="%f" />\n' % (bracRes[j]["sampleHulls"][sample][0][0], bracRes[j]["sampleHulls"][sample][0][1]))
                    fout.write('    <pt x="%f" y="%f" />\n' % (bracRes[j]["sampleHulls"][sample][1][0], bracRes[j]["sampleHulls"][sample][1][1]))
                    fout.write('    <pt x="%f" y="%f" />\n' % (bracRes[j]["sampleHulls"][sample][2][0], bracRes[j]["sampleHulls"][sample][2][1]))
                    fout.write('    <pt x="%f" y="%f" />\n' % (bracRes[j]["sampleHulls"][sample][3][0], bracRes[j]["sampleHulls"][sample][3][1]))
                    fout.write("  </convexhull>\n")

                fout.write("</feature>\n")

            fout.write("    </featureList>\n")
            fout.write("  </featureMap>\n")

    def write_consensus_spectrum_to_featureML_file_per_sample(self, widthRT=40):
        """
        export the consensus results to a featureML file for easy visualization in TOPPView. A separate featureML file will be generated for each sample.
        The path of the file will be the path of the mlML file with the replaced extension '.featureML'

        Parameters
        ----------
        widthRT : int, optional
            the with of the chronogram spots. Defaults to 40.
        """
        self.add_data_processing_step("exporting consensus spectra to featureML files", "exporting consensus spectra to featureML files")
        for samplei, sample in tqdm.tqdm(enumerate(self.get_sample_names()), total=len(self.get_sample_names()), desc="exporting to featureML"):
            with open(os.path.join(".", "%s.featureML" % (sample)).replace(":", "_"), "w") as fout:
                msDataObj = self.get_msDataObj_for_sample(sample)
                spectra = [spectrum for k, spectrum in msDataObj.get_spectra_iterator()]
                assert len(spectra) == 1

                spectrum = spectra[0]

                fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
                fout.write(
                    '  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
                )
                fout.write('    <dataProcessing completion_time="%s">\n' % datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                fout.write('      <software name="tidyMS.dartMS module" version="%s" />\n' % (__tidyMSdartmsVersion__))
                fout.write("    </dataProcessing>\n")
                fout.write('    <featureList count="%d">\n' % (spectrum.mz.shape[0]))

                for j in range(spectrum.mz.shape[0]):
                    fout.write('<feature id="%s">\n' % j)
                    fout.write('  <position dim="0">%f</position>\n' % (spectrum.time + widthRT / 2))
                    fout.write('  <position dim="1">%f</position>\n' % spectrum.mz[j])
                    fout.write("  <intensity>%f</intensity>\n" % spectrum.spint[j])
                    fout.write('  <quality dim="0">0</quality>\n')
                    fout.write('  <quality dim="1">0</quality>\n')
                    fout.write("  <overallquality>0</overallquality>\n")
                    fout.write("  <charge>1</charge>\n")
                    fout.write('  <convexhull nr="0">\n')
                    fout.write('    <pt x="%f" y="%f" />\n' % (spectrum.time, spectrum.mz[j]))
                    fout.write('    <pt x="%f" y="%f" />\n' % (spectrum.time + widthRT, spectrum.mz[j]))
                    fout.write("  </convexhull>\n")
                    fout.write("</feature>\n")

                fout.write("    </featureList>\n")
                fout.write("  </featureMap>\n")

    #####################################################################################################
    # General functions
    #

    def get_groups(self):
        return list(set(self.groups))

    def get_sample_names(self):
        return self.assay.manager.get_sample_names()

    def get_msDataObj_for_sample(self, sample):
        return self.assay.get_ms_data(sample)

    def get_metaData_for_sample(self, sample, metaData):
        return self.assay.manager.get_sample_metadata().loc[sample].loc[metaData]

    def print_sample_overview(self):
        """
        Prints an overview of the samples
        """
        temp = None

        for samplei, sample in enumerate(self.get_sample_names()):
            msDataObj = self.get_msDataObj_for_sample(sample)

            temp = _add_row_to_pandas_creation_dictionary(
                temp,
                sample=sample,
                spectra=msDataObj.get_n_spectra(),
                mzs=sum((spectrum.mz.shape[0] for k, spectrum in msDataObj.get_spectra_iterator())),
            )

        temp = pd.DataFrame(temp)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            # more options can be specified also
            print(temp.to_markdown())

    def plot_sample_TICs(self, separate=True, separate_by="group"):
        """
        Plots the detected spots of the chronograms

        Parameters
        ----------
        separate : bool, optional
            indicator if a facetted plot shall be used or not. Defaults to True.
        separate_by : str, optional
            the variable used for grouping the results (can be file, group, or batch). Defaults to "group".
        """
        temp = None
        for samplei, sample in enumerate(self.get_sample_names()):
            msDataObj = self.get_msDataObj_for_sample(sample)
            for k, spectrum in msDataObj.get_spectra_iterator():
                temp = _add_row_to_pandas_creation_dictionary(
                    temp,
                    sample=sample,
                    file=sample.split("::")[0],
                    group=self.get_metaData_for_sample(sample, "group"),
                    batch=self.get_metaData_for_sample(sample, "batch"),
                    time=spectrum.time - msDataObj.get_spectrum(0).time,
                    kSpectrum=k,
                    totalIntensity=np.sum(spectrum.spint),
                )

        temp = pd.DataFrame(temp)
        if separate:
            temp["file"] = pd.Categorical(temp["file"], ordered=True, categories=natsort.natsorted(set(temp["file"])))
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="time", y="totalIntensity", colour="group", group="sample"))
                + p9.geom_line(alpha=0.8)
                + p9.geom_point(data=temp.loc[temp.groupby("sample").time.idxmin()], colour="black", size=2)
                + p9.geom_point(data=temp.loc[temp.groupby("sample").time.idxmax()], colour="black", size=2)
                + p9.facet_wrap(separate_by)
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.15, "left": 0.15})
                + p9.ggtitle("TIC of chronogram samples")
            )
        else:
            temp["sample"] = pd.Categorical(temp["sample"], ordered=True, categories=natsort.natsorted(set(temp["sample"])))
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="time", y="totalIntensity", colour="group", group="sample"))
                + p9.geom_line(alpha=0.8)
                + p9.geom_point(data=temp.loc[temp.groupby("sample").time.idxmin()], colour="black", size=2)
                + p9.geom_point(data=temp.loc[temp.groupby("sample").time.idxmax()], colour="black", size=2)
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.15, "left": 0.15})
                + p9.ggtitle("TIC of chronogram samples")
            )

        print(p)

    #####################################################################################################
    # Chronogram import and separation
    #

    @staticmethod
    def _subset_MSData_chronogram(msData, startInd, endInd):
        """
        Function subsets a MSData object by chronogram time into a new MSData_subset_spectra object via an internal reference

        Parameters
        ----------
        msData : MSData
            The MSData object to subset
        startInd : int
            Index of first spectrum (included)
        endInd : int
            Index of last spectrum (included)

        Returns
        -------
        MSData
            The new MSData subset object
        """
        return fileio.MSData_subset_spectra(start_ind=startInd, end_ind=endInd, from_MSData_object=msData)

    @staticmethod
    def _get_separate_chronogram_indices(
        msData, msData_ID, spotsFile, intensityThreshold=0.00001, startTime_seconds=0, endTime_seconds=1e6, addNewSeparationIndices=False
    ):
        """
        Function separats a chronogram MSData object into spots that are defined as being continuously above the set threshold.
        Spots are either automatically detected (when the sportsFile is not available) or user-guided (when the spotsFile exists)
        There is deliberately no option to superseed the automated extraction of the spots to not remove an existing spotsFile by accident. If the user wishes to automatically find the spots, the spotsFile file should be deleted by them

        Parameters
        ----------
        msData : MSData
            The chronogram MSData object to separate into spots
        msData_ID : string
            The name of the chronogram object
        spotsFile : string
            The file to which the spots information will be written to. Furthermore, if the file already exists, information provided there will superseed the automated detection of the spots.
        intensityThreshold : float, optional
            the intensity threshold for separate spots. Defaults to 0.00001.
        startTime_seconds : int, optional
            the minimum time to be used. Defaults to 0.
        endTime_seconds : int, optional
            the maximum time to be used. Defaults to 1E6.
        addNewSeparationIndices : bool, optional
            do not change!

        Raises
        ======
        ValueError
            raised when an invalid spots file is provided

        Returns
        -------
        list of (startInd, endInd, spotName, group, class, batch)
            Detected of user-guided spots.
        """
        if spotsFile is None:
            raise ValueError("Parameter spotsFile must be specified either to save extracted spots to or to read from there. ")

        spots = None
        if type(spotsFile) is str:
            if Path(spotsFile).exists() and os.path.isfile(spotsFile):
                spots = pd.read_csv(spotsFile, sep="\t")
            else:
                spots = pd.DataFrame(
                    {
                        "msData_ID": [],
                        "spotInd": [],
                        "include": [],
                        "name": [],
                        "group": [],
                        "class": [],
                        "batch": [],
                        "startRT_seconds": [],
                        "endRT_seconds": [],
                        "comment": [],
                    }
                )
        else:
            raise ValueError("Parameter spotsFile must be a str")

        spotsCur = spots[spots["msData_ID"] == msData_ID]
        separationInds = []
        if spotsCur.shape[0] == 0:
            logging.info(
                "       .. no spots defined for file. Spots will be detected automatically, but not used for now. Please modify the spots file '%s' to include or modify them"
                % (spotsFile)
            )
            ticInts = [sum(msData.get_spectrum(i).spint) for i in range(msData.get_n_spectra())]
            startInd = None
            endInd = None
            for i, inte in enumerate(ticInts):
                time = msData.get_spectrum(i).time
                if inte >= intensityThreshold and time >= startTime_seconds and time <= endTime_seconds:
                    if startInd is None:
                        startInd = i
                    endInd = i

                else:
                    if startInd is not None:
                        # spots are not automatically added
                        logging.info(
                            "       .. found new automatically detected spot from %.2f to %.2f seconds"
                            % (msData.get_spectrum(startInd).time, msData.get_spectrum(endInd).time)
                        )
                        separationInds.append((startInd, endInd, "Spot_%d" % len(separationInds), "unknown", "unknown", 1))
                        startInd = None
                        endInd = None

            if startInd is not None:
                pass
                # spots are not automatically added
                logging.info(
                    "       .. found new automatically detected spot from %.2f to %.2f seconds"
                    % (msData.get_spectrum(startInd).time, msData.get_spectrum(endInd).time)
                )
                separationInds.append((startInd, endInd, "Spot_%d" % len(separationInds), "unknown", "unknown", 1))

            for spotInd, spot in enumerate(separationInds):
                temp = pd.DataFrame(
                    {
                        "msData_ID": [msData_ID],
                        "spotInd": [int(spotInd)],
                        "include": [False],
                        "name": [spot[2]],
                        "group": [spot[3]],
                        "class": [spot[4]],
                        "batch": [spot[5]],
                        "startRT_seconds": [msData.get_spectrum(spot[0]).time],
                        "endRT_seconds": [msData.get_spectrum(spot[1]).time],
                        "comment": [
                            "spot automatically extracted by _get_separate_chronogram_indices(msData, '%s', intensityThreshold = %f, startTime_seconds = %f, endTime_seconds = %f)"
                            % (msData_ID, intensityThreshold, startTime_seconds, endTime_seconds)
                        ],
                    }
                )
                spotsCur = pd.concat([spotsCur, temp], axis=0)
            spots = pd.concat([spots, spotsCur], axis=0, ignore_index=True).reset_index(drop=True)
            spots["include"] = spots["include"].astype("bool")
            spots.to_csv(spotsFile, sep="\t", index=False)
            if not addNewSeparationIndices:
                separationInds = []

        else:
            for index, row in spotsCur.iterrows():
                if row["include"]:
                    startInd, timeDiff_start = msData.get_closest_spectrum_to_RT(row["startRT_seconds"])
                    endInd, timeDiff_end = msData.get_closest_spectrum_to_RT(row["endRT_seconds"])
                    separationInds.append((startInd, endInd, row["name"], row["group"], row["class"], row["batch"]))

        return separationInds

    def _add_chronograms_samples_to_assay(self, sepInds, msData, filename, fileNameChangeFunction=None):
        """
        Function adds spots from a chronogram file to an existing assay

        Parameters
        ----------
        assay : Assay
            Assay object to add the spots to
        sepInds : list of (startInd, endInd, spotName, group, class, batch)
            Information of spots
        msData : MSData
            Chronogram from which the spots are generated
        filename : str
            Name of the chronogram sample
        """
        if fileNameChangeFunction is None:

            def fileNameChangeFunction(x):
                return x

        for subseti, _ in enumerate(sepInds):
            subset_name = fileNameChangeFunction("VIRTUAL(%s::%s)" % (os.path.splitext(os.path.basename(filename))[0], sepInds[subseti][2]))
            logging.info(
                "       .. adding subset %4d with name '%35s' (group '%s', class '%s'), width %6.1f sec, RTs %6.1f - %6.1f"
                % (
                    subseti,
                    subset_name,
                    sepInds[subseti][3],
                    sepInds[subseti][4],
                    msData.get_spectrum(sepInds[subseti][1]).time - msData.get_spectrum(sepInds[subseti][0]).time,
                    msData.get_spectrum(sepInds[subseti][0]).time,
                    msData.get_spectrum(sepInds[subseti][1]).time,
                )
            )
            subset = fileio.MSData_Proxy(DartMSAssay._subset_MSData_chronogram(msData, sepInds[subseti][0], sepInds[subseti][1]))
            self.assay.add_virtual_sample(
                MSData_object=subset,
                virtual_name=subset_name,
                sample_metadata=pd.DataFrame(
                    {
                        "sample": [subset_name],
                        "group": sepInds[subseti][3],
                        "class": sepInds[subseti][4],
                        "order": [1],
                        "batch": sepInds[subseti][5],
                        "basefile": [os.path.splitext(os.path.basename(filename))[0]],
                        "extracted_spectra_indices": [
                            "%.2f - %.2f seconds" % (msData.get_spectrum(sepInds[subseti][0]).time, msData.get_spectrum(sepInds[subseti][1]).time)
                        ],
                        "spotwidth_seconds": [msData.get_spectrum(sepInds[subseti][1]).time - msData.get_spectrum(sepInds[subseti][0]).time],
                    }
                ),
            )

    @staticmethod
    def show_sample_overview(filenames, ms_mode, instrument, separation_intensity=1e3):
        """
        Generates an overview of the data to be imported in subsequent steps

        Parameters
        ----------
        filenames : list of str
            File path of the chronograms
        """
        for filename in filenames:
            logging.info("    .. processing input file '%s'" % (filename))

            msData = fileio.MSData.create_MSData_instance(
                path=filename,
                ms_mode=ms_mode,
                instrument=instrument,
                separation="None/DART",
                data_import_mode=_constants.MEMORY,
            )

            tic = {}
            maxInt = 0
            for spectrumi, spectrum in msData.get_spectra_iterator():
                inte = np.sum(spectrum.spint)
                maxInt = max(maxInt, inte)
                tic = _add_row_to_pandas_creation_dictionary(tic, intensity=inte, time=spectrum.time)

            tmpName = None
            while tmpName is None:
                tmpName = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
                if Path(tmpName).exists():
                    tmpName = None

            sepInds = DartMSAssay._get_separate_chronogram_indices(
                msData,
                None,
                tmpName,
                intensityThreshold=separation_intensity,
                startTime_seconds=0,
                endTime_seconds=1e6,
                addNewSeparationIndices=True,
            )
            t = {}
            t2 = {}
            for spotInd, (startInd, endInd, _, _, _, _) in enumerate(sepInds):
                min_, max_ = len(tic["time"]) - 1, 0
                for spectrumi, spectrum in msData.get_spectra_iterator():
                    if startInd <= spectrumi <= endInd:
                        t = _add_row_to_pandas_creation_dictionary(
                            t, intensity=tic["intensity"][spectrumi], time=tic["time"][spectrumi], spotNumber=spotInd
                        )
                        min_ = min(min_, spectrumi)
                        max_ = max(max_, spectrumi)
                t = _add_row_to_pandas_creation_dictionary(t, intensity=tic["intensity"][min_], time=tic["time"][min_], spotNumber=spotInd)
                t = _add_row_to_pandas_creation_dictionary(t, intensity=tic["intensity"][max_], time=tic["time"][max_], spotNumber=spotInd)
                t2 = _add_row_to_pandas_creation_dictionary(
                    t2, label=spotInd, spotNumber=spotInd, atTime=(tic["time"][min_] + tic["time"][max_]) / 2.0, atIntensity=maxInt * 0.9
                )

            temp = pd.DataFrame(tic)
            t = pd.DataFrame(t)
            t2 = pd.DataFrame(t2)
            p = (
                p9.ggplot()
                + p9.geom_line(data=temp, mapping=p9.aes(x="time", y="intensity"))
                + p9.geom_area(data=t, mapping=p9.aes(x="time", y="intensity", group="spotNumber", colour="spotNumber", fill="spotNumber", alpha=0.2))
                + p9.geom_text(data=t2, mapping=p9.aes(x="atTime", y="atIntensity", group="spotNumber", label="label"))
                + p9.theme_minimal()
                + p9.ggtitle("TIC of '%s'" % (filename))
            )
            print(p)

            with contextlib.suppress(BaseException):
                Path(tmpName).unlink()

    def plot_signal_neighborhood(self):
        temp = None

        for samplei, sample in enumerate(self.get_sample_names()):
            msData = self.get_msDataObj_for_sample(sample)
            group = self.get_metaData_for_sample(sample, "group")
            for k, spectrum in msData.get_spectra_iterator():
                prevMZPPM = np.concatenate(
                    (
                        [250],
                        (spectrum.mz[1 : spectrum.mz.shape[0]] - spectrum.mz[0 : (spectrum.mz.shape[0] - 1)])
                        / spectrum.mz[0 : (spectrum.mz.shape[0] - 1)]
                        * 1e6,
                    )
                )
                nextMZPPM = np.concatenate(
                    (
                        (spectrum.mz[1 : spectrum.mz.shape[0]] - spectrum.mz[0 : (spectrum.mz.shape[0] - 1)])
                        / spectrum.mz[1 : spectrum.mz.shape[0]]
                        * 1e6,
                        [250],
                    )
                )
                prevMZPPMInt = np.concatenate(
                    (
                        [0],
                        spectrum.spint[1 : spectrum.spint.shape[0]] / spectrum.spint[0 : (spectrum.spint.shape[0] - 1)],
                    )
                )
                nextMZPPMInt = np.concatenate(
                    (
                        spectrum.spint[1 : spectrum.spint.shape[0]] / spectrum.mz[0 : (spectrum.mz.shape[0] - 1)],
                        [0],
                    )
                )

                prevSpectrumMZPPM = []
                if k > 0:
                    oSpectrum = msData.get_spectrum(k - 1)
                    for mz in spectrum.mz:
                        a = np.argmin(np.abs(oSpectrum.mz - mz))
                        prevSpectrumMZPPM.append((oSpectrum.mz[a] - mz) / mz * 1e6)
                else:
                    prevSpectrumMZPPM = [0 for i in range(spectrum.mz.shape[0])]

                nextSpectrumMZPPM = []
                if k < msData.get_n_spectra() - 1:
                    oSpectrum = msData.get_spectrum(k + 1)
                    for mz in spectrum.mz:
                        a = np.argmin(np.abs(oSpectrum.mz - mz))
                        nextSpectrumMZPPM.append((oSpectrum.mz[a] - mz) / mz * 1e6)
                else:
                    nextSpectrumMZPPM = [0 for i in range(spectrum.mz.shape[0])]

                for i in range(spectrum.mz.shape[0]):
                    temp = _add_row_to_pandas_creation_dictionary(
                        temp,
                        sample=sample,
                        group=group,
                        mz=spectrum.mz[i],
                        intensity=spectrum.spint[i],
                        scan_prev_signal_devPPM=prevMZPPM[i],
                        scan_next_signal_devPPM=nextMZPPM[i],
                        scan_prev_signal_intRatio=prevMZPPMInt[i] if prevMZPPMInt[i] < 1 else 1 / prevMZPPMInt[i],
                        scan_next_signal_intRatio=nextMZPPMInt[i] if nextMZPPMInt[i] < 1 else 1 / nextMZPPMInt[i],
                        spectrum_prev_signal_devPPM=prevSpectrumMZPPM[i],
                        spectrum_next_signal_devPPM=nextSpectrumMZPPM[i],
                    )

        temp = pd.DataFrame(temp)

        p = (
            p9.ggplot()
            + p9.geom_point(
                data=temp[temp["scan_next_signal_devPPM"] <= 250],
                mapping=p9.aes(y="mz", x="scan_next_signal_devPPM", alpha="scan_next_signal_intRatio"),
            )
            + p9.geom_point(
                data=temp[temp["scan_prev_signal_devPPM"] <= 250],
                mapping=p9.aes(y="mz", x="-scan_prev_signal_devPPM", alpha="scan_prev_signal_intRatio"),
            )
            + p9.scale_alpha(range=(0, 0.01))
            + p9.xlab("Distance previous/next feature in the same scan (ppm)")
            + p9.ggtitle("Previous/next signal in the same scan")
            + p9.theme(legend_position="none")
        )
        print(p)

        p = (
            p9.ggplot()
            + p9.geom_point(
                data=temp[abs(temp["spectrum_next_signal_devPPM"]) <= 250], mapping=p9.aes(y="mz", x="spectrum_next_signal_devPPM"), alpha=0.002
            )
            + p9.geom_point(
                data=temp[abs(temp["spectrum_prev_signal_devPPM"]) <= 250], mapping=p9.aes(y="mz", x="-spectrum_prev_signal_devPPM"), alpha=0.002
            )
            + p9.xlab("Distance of the 'same' signal in the previous/next scan (ppm)")
            + p9.ggtitle("Similar features in neighboring scans")
        )
        print(p)

    @staticmethod
    def create_assay_from_chronogramFiles(
        assay_name,
        filenames,
        spot_file,
        ms_mode,
        instrument,
        centroid_profileMode=True,
        fileNameChangeFunction=None,
        use_signal_function=None,
        rewriteRTinFiles=False,
        rewriteDeleteUnusedScans=True,
        intensity_threshold_spot_extraction=0,
        import_filters=None,
    ):
        """
        Generates a new assay from a series of chronograms and a spot_file

        Parameters
        ----------
        filenames : list of str
            File path of the chronograms
        spot_file : str
            File path of the spot file
        centroid_profileMode : bool
            indicates if profile mode data shall be centroided automatically

        Returns
        -------
        Assay
            The new generated assay object with the either automatically or user-guided spots from the spot_file
        """

        if import_filters is None:
            import_filters = []

        assay = Assay.Assay(
            data_path=None,
            sample_metadata=None,
            ms_mode="centroid",
            instrument="qtof",
            separation="uplc",
            data_import_mode="memory",
            n_jobs=2,
            cache_MSData_objects=True,
        )
        dartMSAssay = DartMSAssay(assay_name)
        dartMSAssay.assay = assay
        logging.info(" Creating empty assay")
        dartMSAssay.add_data_processing_step(
            "Created empty DartMSAssay",
            "created empty DartMSAssay with create_assay_from_chronogramFiles",
            {
                "filenames": filenames,
                "spot_file": spot_file,
                "ms_mode": ms_mode,
                "instrument": instrument,
                "centroid_profileMode": centroid_profileMode,
                "fileNameChangeFunction": fileNameChangeFunction,
                "use_signal_function": use_signal_function,
                "rewriteRTinFiles": rewriteRTinFiles,
                "rewriteDeleteUnusedScans": rewriteDeleteUnusedScans,
                "intensity_threshold_spot_extraction": intensity_threshold_spot_extraction,
                "import_filters": import_filters,
            },
        )

        rtShiftToApply = 0
        for filename in filenames:
            logging.info("    .. processing input file '%s'" % (filename))

            msData = fileio.MSData.create_MSData_instance(
                path=filename,
                ms_mode=ms_mode,
                instrument=instrument,
                separation="None/DART",
                data_import_mode=_constants.MEMORY,
            )

            for importFilter in import_filters:
                msData = importFilter(msData=msData)

            if rewriteRTinFiles:
                lastRT = 0
                earliestRT = 0
                for spectrumi, spectrum in msData.get_spectra_iterator():
                    lastRT = max(lastRT, spectrum.time)
                    spectrum.time = spectrum.time
                lastRT = lastRT + 10

                if rewriteDeleteUnusedScans:
                    if Path(spot_file).exists() and os.path.isfile(spot_file):
                        spots = pd.read_csv(spot_file, sep="\t")
                        lastRT = 0
                        earliestRT = 1e9

                        for rowi, row in spots.iterrows():
                            if spots.at[rowi, "msData_ID"] == os.path.basename(filename).replace(".mzML", "") and spots.at[rowi, "include"]:
                                lastRT = max(lastRT, spots.at[rowi, "endRT_seconds"])
                                earliestRT = min(earliestRT, spots.at[rowi, "startRT_seconds"])

                        if lastRT > earliestRT:
                            lastRT += 5
                            earliestRT = max(0, earliestRT - 5)
                            logging.info("       .. using only scans from RTs %.1f - %.1f seconds" % (earliestRT, lastRT))
                        else:
                            logging.info("       .. no spots to be used in sample, skipping")
                            continue

                    else:
                        raise ValueError("Can only delete unused scans if the spots file has been created before and manually checked.")

                rtShiftToApply = rtShiftToApply - earliestRT + 10
                logging.info("       .. shifting RT by %.1f seconds" % (rtShiftToApply))

                # Reading data from the xml file
                data = Path(filename).read_text()

                bs_data = bs4.BeautifulSoup(data, "xml")

                # <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="0.033849999309" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
                toDel = []
                tagU = False
                tagInd = 0
                for tag in bs_data.find_all("spectrum"):
                    scanlist = tag.find("scanList")
                    if scanlist is not None:
                        scan = scanlist.find("scan")
                        if scan is not None:
                            cvParam = scan.find("cvParam", {"accession": "MS:1000016", "name": "scan start time"}, recursive=True)
                            if cvParam is not None:
                                rt = float(cvParam["value"])
                                if rt < earliestRT / 60.0 or rt > lastRT / 60.0:
                                    tagU = False
                                    toDel.append(tag)
                                else:
                                    tagU = True
                    if tagU:
                        tag["id"] = tag["id"].replace("scan=%d" % (int(tag["index"]) + 1), "scan=%d" % (tagInd + 1))
                        tag["index"] = tagInd
                        tagInd += 1
                spectrumList = bs_data.find("spectrumList")
                spectrumList["count"] = tagInd
                for delTag in toDel:
                    delTag.extract()
                for tag in bs_data.find_all("cvParam", {"accession": "MS:1000016", "name": "scan start time"}):
                    tag["value"] = float(tag["value"]) + rtShiftToApply / 60.0

                with open(filename.replace(".mzML", "_rtShifted.mzML"), "w", newline="\n") as fout:
                    fout.write(bs_data.prettify().replace("\r", ""))

                if Path(spot_file).exists() and os.path.isfile(spot_file):
                    spots = pd.read_csv(spot_file, sep="\t")

                    for rowi, row in spots.iterrows():
                        if spots.at[rowi, "msData_ID"] == os.path.basename(filename).replace(".mzML", ""):
                            spots.at[rowi, "msData_ID"] = spots.at[rowi, "msData_ID"] + "_rtShifted"
                            spots.at[rowi, "startRT_seconds"] = spots.at[rowi, "startRT_seconds"] + rtShiftToApply
                            spots.at[rowi, "endRT_seconds"] = spots.at[rowi, "endRT_seconds"] + rtShiftToApply

                    spots.to_csv(spot_file, sep="\t", index=False)

                rtShiftToApply += lastRT

                msData = fileio.MSData.create_MSData_instance(
                    path=filename.replace(".mzML", "_rtShifted.mzML"),
                    ms_mode=ms_mode,
                    instrument=instrument,
                    separation="None/DART",
                    data_import_mode=_constants.MEMORY,
                )

            if centroid_profileMode and ms_mode == "profile":
                logging.info("       .. centroiding")
                for k, spectrum in msData.get_spectra_iterator():
                    mzs, intensities = spectrum.find_centroids()

                    spectrum.mz = mzs
                    spectrum.spint = intensities
                    spectrum.centroid = True

            if use_signal_function is not None:
                for spectrumi, spectrum in msData.get_spectra_iterator():
                    useInds = use_signal_function(spectrum.mz, spectrum.spint)
                    spectrum.mz = spectrum.mz[useInds]
                    spectrum.spint = spectrum.spint[useInds]

            sepInds = DartMSAssay._get_separate_chronogram_indices(
                msData,
                os.path.basename(filename).replace(".mzML", "") + ("" if not rewriteRTinFiles else "_rtShifted"),
                spot_file,
                intensityThreshold=intensity_threshold_spot_extraction,
            )
            if len(sepInds) == 0:
                logging.warning("       .. no spots to extract")
            else:
                dartMSAssay._add_chronograms_samples_to_assay(sepInds, msData, filename, fileNameChangeFunction=fileNameChangeFunction)

        return dartMSAssay

    #####################################################################################################
    # Spectra selection
    #

    def drop_lower_spectra(self, drop_rate):
        """
        A function to restrict chronogram spots to only certain spectra in the dataset (e.g., use the 'core' of the spot).
        From the all spectra assigned to the spot, only the drop_rate % will be used
        Use with caution, as a variable number of scans over the spot might affect abundances, especially when aggregation_method = "sum" is used

        Parameters
        ----------
        drop_rate : float
            the ratio of the highest-abundant spectra to be used.
        """
        self.add_data_processing_step("drop lower spectra", "drop lower spectra", {"drop_rate": drop_rate})
        for samplei, sample in enumerate(self.get_sample_names()):
            totalInt = []
            msDataObj = self.get_msDataObj_for_sample(sample)
            for k, spectrum in msDataObj.get_spectra_iterator():
                totalInt.append(np.sum(spectrum.spint))

            sampleObjNew = fileio.MSData_in_memory.generate_from_MSData_object(msDataObj)
            ordInte = np.argsort(np.array(totalInt))
            ordInte = ordInte[0 : math.floor(sampleObjNew.get_n_spectra() * drop_rate)]
            ordInte = np.sort(ordInte)[::-1]
            c = 0
            while c < ordInte.shape[0] and msDataObj.get_n_spectra() > 0:
                sampleObjNew.delete_spectrum(ordInte[c])
                c = c + 1
            msDataObj.to_MSData_object = sampleObjNew

    def select_top_n_spectra(self, n):
        """
        A function to restrict chronogram spots to only certain spectra in the dataset (e.g., use the 'core' of the spot).
        From all spectra assigned to the spot, only the n most abundant will be used
        Parameters
        ----------
        n : integer
            the number of highest-abundant spectra to be used.
        """
        self.add_data_processing_step("select top n spectra", "select top n spectra", {"n": n})
        if n is not None:
            for samplei, sample in enumerate(self.get_sample_names()):
                tic = []
                msDataObj = self.get_msDataObj_for_sample(sample)
                for k, spectrum in msDataObj.get_spectra_iterator():
                    tic.append(np.sum(spectrum.spint))
                # print("   ... TIC of sample '%s' is %s" % (sample, str(sorted(tic))))

                sampleObjNew = fileio.MSData_in_memory.generate_from_MSData_object(msDataObj)
                ordInte = np.argsort(np.array(tic))
                ordInte = ordInte[0 : ordInte.shape[0] - n]
                ordInte = np.sort(ordInte)[::-1]
                c = 0
                while c < ordInte.shape[0] and msDataObj.get_n_spectra() > 0:
                    sampleObjNew.delete_spectrum(ordInte[c])
                    c = c + 1
                msDataObj.to_MSData_object = sampleObjNew

                # tic = []
                # for k, spectrum in msDataObj.get_spectra_iterator():
                #    tic.append(np.sum(spectrum.spint))
                # print("      ... used spctra of sample '%s' are %s" % (sample, str(sorted(tic))))

    #####################################################################################################
    # Sample normalization
    #

    def normalize_samples_by_TIC(self, multiplication_factor=1):
        """
        abundances of spot spectra can be normalized by the toal intensity of the spectra

        Parameters
        ----------
        multiplication_factor : int, optional
            a factor that is applied on top of the normalization (i.e., shifts the maximum abundance to this value). Defaults to 1.
        """
        self.add_data_processing_step("normalize to sample TICs", "normalize to sample TICs", {"muliplication_factor": multiplication_factor})
        for samplei, sample in enumerate(self.get_sample_names()):
            totalInt = []
            msDataObj = self.get_msDataObj_for_sample(sample)
            for k, spectrum in msDataObj.get_spectra_iterator():
                totalInt.append(np.sum(spectrum.spint))
            totalInt = np.sum(np.array(totalInt))

            sampleObjNew = fileio.MSData_in_memory.generate_from_MSData_object(msDataObj)
            msDataObj.to_MSData_object = sampleObjNew

            if totalInt > 0:
                for k, spectrum in msDataObj.get_spectra_iterator():
                    spectrum.spint = spectrum.spint / totalInt * multiplication_factor
            else:
                logging.error("   .. Error: cannot normalize sample '%35s' to TIC as it is zero" % (sample))

    def normalize_to_internal_standard(self, std, multiplication_factor=1, plot=False):
        """
        Abundances of spot spectra are normalized by the abundance of a selected internal standard

        Parameters
        ----------
        std : float
            standard to normalize to
        multiplication_factor : int, optional
            Defaults to 1.
        plot : bool, optional
            Defaults to False.
        """
        self.add_data_processing_step(
            "normalize to internal standard", "normalize to internal standard", {"std": std, "multiplication_factor": multiplication_factor}
        )
        stdMZmin, stdMZmax = std

        temp = None
        for samplei, sample in enumerate(self.get_sample_names()):
            sampleType = self.get_metaData_for_sample(sample, "group")
            totalSTDInt = 0
            msDataObj = self.get_msDataObj_for_sample(sample)
            for k, spectrum in msDataObj.get_spectra_iterator():
                use = np.logical_and(spectrum.mz >= stdMZmin, spectrum.mz <= stdMZmax)
                if np.sum(use) > 0:
                    totalSTDInt = totalSTDInt + np.sum(spectrum.spint[use])

            if totalSTDInt > 0:
                logging.info("    .. sample '%35s' STD intensity (sum) %12.1f * %12.1f" % (sample, totalSTDInt, multiplication_factor))
                for k, spectrum in msDataObj.get_spectra_iterator():
                    spectrum.spint = spectrum.spint / totalSTDInt * multiplication_factor
                temp = _add_row_to_pandas_creation_dictionary(temp, sample=sample, group=sampleType, istdAbundance=totalSTDInt)
            else:
                logging.error("   .. Error: cannot normalize sample '%35s' to internal standard as no signals for it have been found" % (sample))

        if plot:
            temp = pd.DataFrame(temp)
            temp["sample"] = pd.Categorical(temp["sample"], ordered=True, categories=natsort.natsorted(set(temp["sample"])))
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="sample", y="istdAbundance", group="group", colour="group"))
                + p9.geom_point()
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.05, "left": 0.05})
                + p9.guides(alpha=False, colour=False)
                + p9.ggtitle("Abundance of internal standard (mz %.5f - %.5f)" % (std[0], std[1]))
            )
            print(p)

    def batch_correction(self, by_group, plot=True):
        """
        Correct the abundances of all detected features in different batches.
        The algorithm is as follows
        An overall mean-QC-overall-value is derived from all QC samples (regardless of the batch) using all features detected in these QC samples.
        For each batch, a mean-QC-sample-value is derived from QC samples in each batch using all features detected in these QC samples.
        All samples in the batch are corrected by this mean-QC-sample-value. For this, all feature abundances are divided by this value
        Furthermore, this corrected abundance values are multiplied by the mean-QC-overall-value to achieve similar abundance values than before the correction.

        Parameters
        ----------
        by_group : string
            the name of the group to be used for the batch correction
        plot : bool, optional
            show the correction results as plots. Defaults to True.
        """
        self.add_data_processing_step("batch correction", "batch correction", {"by_group": by_group})
        datCorr = np.copy(self.dat)

        correctedFeatures = 0
        correctedBatches = 0

        batchCorrectionValues = {"batch": [], "featurei": [], "correctionValue": [], "meanValue": []}

        qcInds = [i for i in range(len(self.groups)) if self.groups[i] == by_group]
        for featurei in range(datCorr.shape[1]):
            vals = datCorr[qcInds, featurei]
            if np.any(np.logical_not(np.isnan(vals))):
                qcMean = np.mean(vals[~np.isnan(vals)])

                if not np.isnan(qcMean) and qcMean > 0:
                    correctedFeatures += 1

                    for batch in set(self.batches):
                        batchInds = [i for i in range(len(self.batches)) if self.batches[i] == batch]
                        cInds = list(set(batchInds).intersection(set(qcInds)))

                        vals = datCorr[cInds, featurei]
                        if np.any(np.logical_not(np.isnan(vals))):
                            qcCurMean = np.mean(vals[~np.isnan(vals)])

                            if not np.isnan(qcCurMean) and qcCurMean > 0:
                                batchCorrectionValues["batch"].append(batch)
                                batchCorrectionValues["featurei"].append(featurei)
                                batchCorrectionValues["correctionValue"].append(qcMean / qcCurMean)
                                batchCorrectionValues["meanValue"].append(qcCurMean)
                                correctedBatches += 1

        batchCorrectionValues = pd.DataFrame(batchCorrectionValues)
        batchCorrectionValuesGrouped = batchCorrectionValues.groupby(["batch"]).aggregate("median").reset_index()

        if plot:
            p = (
                p9.ggplot(data=batchCorrectionValues, mapping=p9.aes(x="batch", y="np.log2(correctionValue)", group="featurei"))
                + p9.geom_jitter(width=0.4, height=0, alpha=0.15, colour="slategrey")
                + p9.geom_point(data=batchCorrectionValuesGrouped, colour="Firebrick", size=3)
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.15, "left": 0.15})
                + p9.ylim(-2, 2)
                + p9.ggtitle("Before batch correction. Note: some large differences have been clipped in this illustration")
            )
            print(p)

        for batch in set(self.batches):
            if batch in list(batchCorrectionValuesGrouped["batch"]):
                corrVal = batchCorrectionValuesGrouped["correctionValue"][batchCorrectionValuesGrouped["batch"] == batch].iloc[0]
                batchInds = [i for i in range(len(self.batches)) if self.batches[i] == batch]

                for featurei in range(datCorr.shape[1]):
                    datCorr[batchInds, featurei] = datCorr[batchInds, featurei] * corrVal

        logging.info(
            " Batch effect correction carried out. %d / %d features were eligible for correction and %d / %d batches have been corrected. "
            % (correctedFeatures, self.dat.shape[1], correctedBatches, self.dat.shape[1] * len(set(self.batches)))
        )
        logging.info(" Batch correction values are")
        logging.info(batchCorrectionValuesGrouped.to_markdown())

        self.dat = datCorr

    #####################################################################################################
    # MZ shift correction
    #

    def _calculate_mz_offsets(self, referenceMZs=[165.078978594 + 1.007276], max_mz_deviation_absolute=0.1, selection_criteria="mostAbundant"):
        """
        Function to calculate the mz offsets of several reference features in the dataset.
        A signal for a feature on the referenceMZs list is said to be found, if it is within the max_mz_deviation_absolute parameter. The feature with the closest mz difference will be used in cases where several features are present in the search window

        Parameters
        ----------
        self : DartMSAssay
            The self of the experiment
        referenceMZs : list of MZ values (floats) or dict
            The reference features for which the offsets shall be calculated.
                       Defaults to [165.078978594 + 1.007276].
                       If the user provides an mz value, these mz values will be used in all samples and the search mz is the refernce mz
                       If the user provides a dictionary, it must consist of the entries 'observedMZ', 'referenceMZ' and samples.
                       observedMZ is the mz to search for, referenceMZ to correct to and samples is a list of sample names to be used for this particular correction
        max_mz_deviation_absolute : float, optional
            Maximum deviation used for the search. Defaults to 0.1.

        Returns
        -------
        Pandard dataframe
            The calculated offsets for each reference
        """

        if selection_criteria.lower() not in ("closestMZ".lower(), "mostAbundant".lower()):
            raise ValueError("Unknown parameter selection_criteria, must be either of ['mostAbundant', 'closestMZ]")

        temp = None

        for sample in self.get_sample_names():
            msDataObj = self.get_msDataObj_for_sample(sample)
            for i, referenceMZ in enumerate(referenceMZs):
                observedMZ = referenceMZ
                forSamples = None

                if type(referenceMZ) == float:
                    pass

                elif type(referenceMZ) == dict:
                    observedMZ = referenceMZ["observedMZ"]
                    forSamples = referenceMZ["forSamples"]
                    referenceMZ = referenceMZ["referenceMZ"]

                else:
                    raise RuntimeError(
                        "Unknown parameter type for referenceMZ (%s). Must be either an mz value or a dict with 'observedMZ': float, 'forSamples': [List of sample names], 'referenceMZ': float"
                        % (type(referenceMZ))
                    )

                if forSamples is None or sample in forSamples:
                    # print("correcting sample ", sample, " from ", observedMZ, " to ", referenceMZ)
                    for spectrumi, spectrum in msDataObj.get_spectra_iterator():
                        ind = None
                        if selection_criteria.lower() == "closestMZ".lower():
                            ind, curMZ, deltaMZ, deltaMZPPM, inte = spectrum.get_closest_mz(observedMZ, max_offset_absolute=max_mz_deviation_absolute)

                        elif selection_criteria.lower() == "mostAbundant".lower():
                            ind, curMZ, deltaMZ, deltaMZPPM, inte = spectrum.get_most_abundant_signal_in_range(
                                observedMZ, max_offset_absolute=max_mz_deviation_absolute
                            )

                        # print("   .. found correction by ", curMZ - referenceMZ, (curMZ - referenceMZ) / referenceMZ * 1e6)
                        if ind is not None:
                            temp = _add_row_to_pandas_creation_dictionary(
                                temp,
                                referenceMZ=referenceMZ,
                                mz=curMZ,
                                mzDeviation=curMZ - referenceMZ,
                                mzDeviationPPM=(curMZ - referenceMZ) / referenceMZ * 1e6,
                                time=spectrum.time,
                                intensity=inte,
                                sample=sample,
                                file=sample.split("::")[0],
                                chromID="%s %.4f" % (sample, referenceMZ),
                            )

        return pd.DataFrame(temp)

    def _reverse_applied_mz_offset(self, mz, correctby, *args, **kwargs):
        """
        Function to reverse the corrected mz values in a corrected spectrum

        Parameters
        ----------
        mz : float
            the spectrums mz value to be reverse corrected
        correctby : string
            the correction type applied to the mz value

        Raises
        ======
        ValueError
            raised when an invalid option for correctby is provided

        Returns
        -------
        float
            the reverse corrected mz value
        """
        if correctby == "mzDeviationPPM":
            transformFactor = kwargs["transformFactor"]
            return mz / (1.0 - transformFactor / 1e6)
        elif correctby == "mzDeviation":
            transformFactor = kwargs["transformFactor"]
            return mz + transformFactor
        else:
            raise ValueError("Unknown correctby option '%s' specified. Must be either of ['mzDeviationPPM', 'mzDeviation']" % (correctby))

    def correct_MZ_shift_across_samples(
        self,
        referenceMZs=[165.078978594 + 1.007276],
        max_mz_deviation_absolute=0.1,
        correctby="mzDeviationPPM",
        max_deviationPPM_to_use_for_correction=80,
        selection_criteria="mostAbundant",
        correct_on_level="file",
        plot=False,
    ):
        """
        Function to correct systematic shifts of mz values in individual spot samples
        Currently, only a constant MZ offset relative to several reference features can be corrected. The correction is carried out by calculating the median error relative to the reference features' and then apply either the aboslute or ppm devaition to all mz values in the spot sample.
        A signal for a feature on the referenceMZs list is said to be found, if it is within the max_mz_deviation_absolute parameter. The feature with the closest mz difference will be used in cases where several features are present in the search window

        Parameters
        ----------
        assay : Assay
            The assay object of the experiment
        referenceMZs : list of MZ values (float), optional
            The reference features for which the offsets shall be calculated. Defaults to [165.078978594 + 1.007276].
        max_mz_deviation_absolute : float, optional
            Maximum deviation used for the search. Defaults to 0.1.
        correctby : str, optional
            Either "mzDeviation" for correcting by a constant mz offset or "mzDeviationPPM" to correct by a constant PPM offset. Defaults to "mzDeviationPPM".
        plot : ool, optional
            Indicates if a plot shall be generated and returned. Defaults to False.

        Returns
        -------
        pandas.DataFrame, plot
            Overview of the correction and plot (if it shall be generated)
        """

        self.add_data_processing_step(
            "correct mz shift across samples",
            "correct mz shift across samples",
            {
                "referenceMZs": referenceMZs,
                "max_mz_deviation_absolute": max_mz_deviation_absolute,
                "correctby": correctby,
                "max_deviationPPM_to_use_for_correction": max_deviationPPM_to_use_for_correction,
                "selection_criteria": selection_criteria,
                "correct_on_level": correct_on_level,
            },
        )

        if not correct_on_level.lower() in ("file", "sample"):
            raise ValueError("Parameter correct_on_level has an unknown value '%s', must be either of ['file', 'sample']" % (correct_on_level))

        temp = self._calculate_mz_offsets(
            referenceMZs=referenceMZs, max_mz_deviation_absolute=max_mz_deviation_absolute, selection_criteria=selection_criteria
        )
        temp["mode"] = "original MZs"

        tempMod = temp.copy()
        tempMod["mode"] = "corrected MZs (by ppm mz deviation)"
        transformFactors = None
        if correctby.lower() == "mzDeviationPPM".lower():
            transformFactors = (
                tempMod[(np.abs(tempMod["mzDeviationPPM"]) <= max_deviationPPM_to_use_for_correction)]
                .groupby(correct_on_level)["mzDeviationPPM"]
                .median()
            )
            tempMod["mz"] = tempMod["mz"] * (
                1
                - tempMod[(np.abs(tempMod["mzDeviationPPM"]) <= max_deviationPPM_to_use_for_correction)]
                .groupby(correct_on_level)["mzDeviationPPM"]
                .transform("median")
                / 1e6
            )  # * (1. - (temp.groupby("chromID")["mzDeviationPPM"].transform("median")) / 1E6)  ## Error

        elif correctby.lower() == "mzDeviation".lower():
            transformFactors = (
                tempMod[(np.abs(tempMod["mzDeviationPPM"]) <= max_deviationPPM_to_use_for_correction)]
                .groupby(correct_on_level)["mzDeviation"]
                .median()
            )
            tempMod["mz"] = tempMod["mz"] - tempMod[(np.abs(tempMod["mzDeviationPPM"]) <= max_deviationPPM_to_use_for_correction)].groupby(
                correct_on_level
            )["mzDeviation"].transform("median")

        else:
            raise ValueError("Unknown option for correctby parameter. Must be 'mzDeviation' or 'mzDeviationPPM'")

        tempMod["mzDeviationPPM"] = (tempMod["mz"] - tempMod["referenceMZ"]) / tempMod["referenceMZ"] * 1e6
        tempMod["mzDeviation"] = tempMod["mz"] - tempMod["referenceMZ"]

        for samplei, sample in enumerate(self.get_sample_names()):
            msDataObj = self.get_msDataObj_for_sample(sample)
            if issubclass(type(msDataObj), fileio.MSData_in_memory):
                raise RuntimeError(
                    "Function correct_MZ_shift_across_samples only works with objects of class fileio.MSData_in_memory. Please switch data_import_mode to _constancts.MEMORY"
                )

            transformFactor = None
            if correct_on_level.lower() == "sample" and sample in transformFactors.index:
                transformFactor = transformFactors.loc[sample]
            elif correct_on_level.lower() == "file" and sample.split("::")[0] in transformFactors.index:
                transformFactor = transformFactors.loc[sample.split("::")[0]]

            if transformFactor is None:
                logging.error(
                    "Error: Sample %3d / %3d (%45s) could not be corrected as no reference MZs were detected in it"
                    % (samplei + 1, len(self.get_sample_names), sample)
                )
                for k, spectrum in msDataObj.get_spectra_iterator():
                    spectrum.original_mz = spectrum.mz
            else:
                for k, spectrum in msDataObj.get_spectra_iterator():
                    if "reverseMZ" not in dir(spectrum):
                        spectrum.reverseMZ = None
                        spectrum.reverseMZDesc = "None"
                        spectrum.original_mz = spectrum.mz

                    refFun = None
                    refFunDesc = None
                    if correctby.lower() == "mzDeviationPPM".lower():
                        spectrum.mz = spectrum.mz * (1.0 - transformFactor / 1e6)
                        refFun = functools.partial(self._reverse_applied_mz_offset, correctby="mzDeviationPPM", transformFactor=transformFactor)
                        refFunDesc = "mzDeviationPPM by %.5f (%s)" % (transformFactor, spectrum.reverseMZ)

                    elif correctby.lower() == "mzDeviation".lower():
                        spectrum.mz = spectrum.mz - transformFactor
                        refFun = functools.partial(self._reverse_applied_mz_offset, correctby="mzDeviation", transformFactor=transformFactor)
                        refFunDesc = "mzDeviation by %.5f (%s)" % (transformFactor, spectrum.reverseMZ)

                    ## chain reverse mz functions if needed
                    if spectrum.reverseMZ is not None:
                        spectrum.reverseMZ = functools.partial(functools.reduce, lambda p, f: f(p), (refFun, spectrum.reverseMZ))
                        spectrum.reverseMZDesc = refFunDesc + ";" + spectrum.reverseMZDesc
                    else:
                        spectrum.reverseMZ = refFun
                        spectrum.reverseMZDesc = refFunDesc

                logging.info(
                    "     .. Sample %3d / %3d (%45s) correcting by %.1f (%s)"
                    % (samplei + 1, len(self.get_sample_names()), sample, transformFactor, correctby)
                )

        tempMod["mode"] = "corrected MZs (by %s)" % (correctby)
        temp_ = pd.concat([temp, tempMod], axis=0, ignore_index=True).reset_index(drop=False)

        p = None
        if plot:
            temp_["file"] = pd.Categorical(temp_["file"], ordered=True, categories=natsort.natsorted(set(temp_["file"])))
            p = (
                p9.ggplot(
                    data=temp_[(~(temp_["intensity"].isna())) & (np.abs(temp_["mzDeviationPPM"]) <= 100)],
                    mapping=p9.aes(x="referenceMZ", y="mzDeviationPPM", group="chromID", colour="sample", alpha="intensity"),
                )
                + p9.geom_hline(yintercept=0, size=1, colour="Black", alpha=0.25)
                + p9.geom_line()
                + p9.geom_point()
                + p9.facet_wrap("~ mode + file", ncol=12)
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.05, "left": 0.05})
                + p9.guides(alpha=False, colour=False)
                + p9.ggtitle("MZ deviation before and after correction for each sample/chronogram file")
            )
            print(p)

    #####################################################################################################
    # Clustering of mz values functionality
    # used for consensus calculations and bracketing
    #

    def _crude_clustering_for_mz_list(self, sample, mz, intensity, min_difference_ppm):
        """
        Function for a crude clustering of similar mz values in a spot sample

        Parameters
        ----------
        mz : numpy array
            All mz values of a spot sample
        intensity : numpy array
            All intensity values associated with the mz values in the parameter mz
        min_difference_ppm : float
            Minimum difference in PPM required to separate into different clusters
        return_details_object : bool, optional
            Indicator if the . Defaults to False.

        Returns
        -------
        np array
            Array with cluster IDs for each signal (i.e., mz value and intensity in the parameters mz and intensity)
        """
        mzOrd = np.argsort(mz)
        mz_ = mz[mzOrd]
        intensity_ = intensity[mzOrd]
        elems = mz.shape[0]
        diffsPPM = (mz_[1:elems] - mz_[0 : (elems - 1)]) / mz_[0 : (elems - 1)] * 1e6
        clust = np.concatenate([[0], np.cumsum(diffsPPM > min_difference_ppm)], axis=0)

        return clust[np.argsort(mzOrd)]

    def _reindex_cluster(self, cluster):
        """
        Function to reindex a cluster if certain cluster IDs have been deleted previously.
        Clusters will be ascendingly processed by lexiographic sorting resulting in new IDs for any cluster that has an ID higher than a deleted cluster ID

        For example:
        cluster = [0,0,0,1,2,4,4,4,5,6]
        The cluster 3 has been deleted, therefore the cluster IDs 4,5, and 6 will be shifted by -1 each resulting in the new clusters
        returns: [0,0,0,1,2,3,3,3,4,5]

        Note
        Even negative cluster IDs (-1) will be reindexed

        Parameters
        ----------
        cluster : numpy array
            Cluster IDs for any clustering

        Returns
        -------
        numpy array
            The new cluster IDs for each cluster
        """
        newClust = np.zeros(cluster.shape[0], dtype=int)

        clustInds, ns = np.unique(cluster, return_counts=True)

        use = 0
        for i, clustInd in enumerate(clustInds):
            if clustInd == -1:
                pass
            else:
                newClust[cluster == clustInd] = use
                use += 1

        return newClust

    #####################################################################################################
    # Consensus spectra calculation
    #

    def _describe_mz_cluster(self, mz, intensity, clust):
        """
        Function to calculate summary information about each mz cluster

        Parameters
        ----------
        mz : numpy array
            The mz values of each signal
        intensity : numpy array
            The intensity values of each signal
        clust : numpy array
            The cluster IDs of each signal

        Returns
        -------
        numpy matrix
            Each row in the matrix corresponds to one clusters in the clusterd signal space. The columns of the matrix indicate the cluster IDs, the number of signals, the minimum, average and maximum MZ values, the MZ deviation and the sum of the intensities of the respective signals.
        """
        mzOrd = np.argsort(mz)

        uniqClusts, ns = np.unique(clust, return_counts=True)
        # clusterID = rowInd:   0: clusterID   1: Ns   2: Min.MZ   3: Avg.MZ   4: Max.MZ   5: MZ.Dev   6: sum.Int.
        mzDesc = np.zeros([uniqClusts.shape[0], 7])
        mzDesc[:, 2] = np.Inf
        for i in range(mz.shape[0]):
            j = clust[i]
            mzDesc[j, 0] = j
            mzDesc[j, 1] = mzDesc[j, 1] + 1
            mzDesc[j, 2] = np.minimum(mz[i], mzDesc[j, 2])
            mzDesc[j, 3] = mzDesc[j, 3] + mz[i]
            mzDesc[j, 4] = np.maximum(mz[i], mzDesc[j, 4])
            mzDesc[j, 6] = mzDesc[j, 6] + intensity[i]

        mzDesc[:, 3] = mzDesc[:, 3] / mzDesc[:, 1]
        mzDesc[:, 5] = (mzDesc[:, 4] - mzDesc[:, 2]) / mzDesc[:, 3] * 1e6
        return mzDesc

    def _collapse_mz_cluster(self, mz, original_mz, intensity, time, cluster, intensity_collapse_method="average"):
        """
        Function to collapse several spectra (provided as different lists) into a consensus spectrum

        Parameters
        ----------
        mz : numpy array of mz values
            the mz values to collapse
        original_mz : list of mz values
            list of mz values used for the mz cluster
        intensity : numpy array of intensity values
            the intensity values to collapse
        time : numpy array of chronogram time values
            the chronogram time values to collapse
        cluster : numpy array of cluster ids
            the clusters the individual signals have been assigned to
        intensity_collapse_method : str, optional
            The method to be used for calculating the consensus signal intensity. Defaults to "average".

        Raises
        ======
        ValueError
            raised when an unknown option for intensity_collapse_method is provided

        Returns
        -------
        mz, intensity, used-features
            returns a tuple of mz and intensity values and the mz values used for collapsing the signals
        """
        clusts, ns = np.unique(cluster, return_counts=True)
        if -1 in clusts:
            clusts = clusts[1:]
            ns = ns[1:]

        mz_ = np.zeros(clusts.shape[0], dtype=np.float32)
        intensity_ = np.zeros(clusts.shape[0], dtype=np.float32)

        usedFeatures = {}
        for i in range(clusts.shape[0]):
            usedFeatures[i] = np.zeros((ns[i], 4), dtype=np.float32)

        for i in range(mz.shape[0]):
            j = cluster[i]

            assert j >= 0

            mz_[j] += mz[i] * intensity[i]
            intensity_[j] += intensity[i]

            toPut = 0
            while usedFeatures[j][toPut, 0] > 0:
                toPut = toPut + 1
                assert toPut < usedFeatures[j].shape[0]

            usedFeatures[j][toPut, 0] = mz[i]
            usedFeatures[j][toPut, 1] = intensity[i]
            usedFeatures[j][toPut, 2] = time[i]
            usedFeatures[j][toPut, 3] = original_mz[i]

        mz_ = mz_ / intensity_
        if intensity_collapse_method.lower() == "average".lower():
            intensity_ = intensity_ / ns
        elif intensity_collapse_method.lower() == "sum".lower():
            pass
        elif intensity_collapse_method.lower() == "max".lower():
            intensity_ = np.max(intensity_)
        else:
            raise ValueError("Unknown option for parameter intensity_collapse_method, must be either of ['average', 'sum', 'max']")

        ord = np.argsort(mz_)
        return mz_[ord], intensity_[ord], [usedFeatures[i] for i in ord]

    def calculate_consensus_spectra_for_samples(
        self,
        min_difference_ppm=30,
        closest_signal_max_deviation_ppm=20,
        max_mz_deviation_ppm=20,
        min_signals_per_cluster=10,
        minimum_intensity_for_signals=0,
        cluster_quality_check_functions=None,
        aggregation_function="average",
        exportAsFeatureML=True,
        featureMLlocation=".",
    ):
        """
        Function to collapse several spectra into a single consensus spectrum per spot

        Parameters
        ----------
        min_difference_ppm : float, optional
            Minimum difference in PPM required to separate into different clusters. Defaults to 30.
        min_signals_per_cluster : int, optional
            Minimum number of signals for a certain MZ cluster for it to be used in the collapsed spectrum. Defaults to 10.
        """

        self.add_data_processing_step(
            "calculate consensus spectra for samples",
            "calculate consensus spectra for samples",
            {
                "min_difference_ppm": min_difference_ppm,
                "min_signals_per_cluster": min_signals_per_cluster,
                "minimum_intensity_for_signals": minimum_intensity_for_signals,
                "cluster_quality_check_functions": cluster_quality_check_functions,
                "aggregation_function": aggregation_function,
                "featureMLlocation": featureMLlocation,
            },
        )

        if cluster_quality_check_functions is None:
            cluster_quality_check_functions = []

        for samplei, sample in enumerate(self.get_sample_names()):
            temp = {"sample": [], "spectrumInd": [], "time": [], "mz": [], "original_mz": [], "intensity": []}
            msDataObj = self.get_msDataObj_for_sample(sample)
            summary_totalSpectra = 0
            for k, spectrum in msDataObj.get_spectra_iterator():
                temp["sample"].extend((sample for i in range(spectrum.mz.shape[0])))
                temp["spectrumInd"].extend((k for i in range(spectrum.mz.shape[0])))
                temp["time"].extend((spectrum.time for i in range(spectrum.mz.shape[0])))
                temp["mz"].append(spectrum.mz)
                temp["original_mz"].append(spectrum.original_mz)
                temp["intensity"].append(spectrum.spint)
                summary_totalSpectra += 1

            temp["sample"] = np.array(temp["sample"])
            temp["spectrumInd"] = np.array(temp["spectrumInd"])
            temp["time"] = np.array(temp["time"])
            temp["mz"] = np.concatenate(temp["mz"], axis=0, dtype=np.float64)
            temp["original_mz"] = np.concatenate(temp["original_mz"], axis=0)
            temp["intensity"] = np.concatenate(temp["intensity"], axis=0, dtype=np.float64)
            summary_totalSignals = len(temp["mz"])

            temp["cluster"] = self._crude_clustering_for_mz_list(sample, temp["mz"], temp["intensity"], min_difference_ppm=min_difference_ppm)
            summary_clusterAfterCrude = np.unique(temp["cluster"]).shape[0]

            # remove any cluster with less than min_signals_per_cluster signals
            clustInds, ns = np.unique(temp["cluster"], return_counts=True)
            clustNs = ns[temp["cluster"]]
            temp["cluster"][clustNs < min_signals_per_cluster] = -1
            temp["cluster"][temp["intensity"] <= minimum_intensity_for_signals] = -1

            keep = temp["cluster"] >= 0
            temp["sample"] = temp["sample"][keep]
            temp["spectrumInd"] = temp["spectrumInd"][keep]
            temp["time"] = temp["time"][keep]
            temp["mz"] = temp["mz"][keep]
            temp["original_mz"] = temp["original_mz"][keep]
            temp["intensity"] = temp["intensity"][keep]
            temp["cluster"] = self._reindex_cluster(temp["cluster"][keep])

            # refine cluster
            temp["cluster"] = _refine_clustering_for_mz_list(
                sample,
                temp["mz"],
                temp["intensity"],
                temp["spectrumInd"],
                temp["cluster"],
                closest_signal_max_deviation_ppm=closest_signal_max_deviation_ppm,
                max_mz_deviation_ppm=max_mz_deviation_ppm,
            )
            temp["cluster"] = self._reindex_cluster(temp["cluster"])
            summary_clusterAfterFine = np.unique(temp["cluster"]).shape[0]

            # remove any cluster with less than min_signals_per_cluster signals
            clustInds, ns = np.unique(temp["cluster"], return_counts=True)
            clustNs = ns[temp["cluster"]]
            temp["cluster"][clustNs < min_signals_per_cluster] = -1
            for cluster_quality_check_function in cluster_quality_check_functions:
                temp["cluster"] = cluster_quality_check_function(
                    sample, msDataObj, temp["spectrumInd"], temp["time"], temp["mz"], temp["intensity"], temp["cluster"]
                )
            summary_clusterAfterQualityFunctions = np.unique(temp["cluster"]).shape[0]

            keep = temp["cluster"] >= 0
            temp["sample"] = temp["sample"][keep]
            temp["spectrumInd"] = temp["spectrumInd"][keep]
            temp["time"] = temp["time"][keep]
            temp["mz"] = temp["mz"][keep]
            temp["original_mz"] = temp["original_mz"][keep]
            temp["intensity"] = temp["intensity"][keep]
            temp["cluster"] = self._reindex_cluster(temp["cluster"][keep])

            if len(temp["cluster"]) == 0:
                logging.error("   .. Error: no signals to be used for sample '%35s'" % (sample))
                next

            if exportAsFeatureML:
                with open(os.path.join(featureMLlocation, "%s.featureML" % (sample)).replace(":", "_"), "w") as fout:
                    minRT = np.min(temp["time"])
                    maxRT = np.max(temp["time"])

                    ns = np.unique(temp["cluster"])

                    fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
                    fout.write(
                        '  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
                    )
                    fout.write('    <dataProcessing completion_time="%s">\n' % datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    fout.write('      <software name="tidyms" version="%s" />\n' % (__tidyMSdartmsVersion__))
                    fout.write('      <software name="tidycalculate_consensus_spectra_for_samples" version="%s" />\n' % (__tidyMSdartmsVersion__))
                    fout.write("    </dataProcessing>\n")
                    fout.write('    <featureList count="%d">\n' % (ns.shape[0]))

                    for j in range(ns.shape[0]):
                        clust = ns[j]
                        mzs = np.copy(temp["mz"][temp["cluster"] == clust])
                        original_mzs = np.copy(temp["original_mz"][temp["cluster"] == clust])
                        ints = np.copy(temp["intensity"][temp["cluster"] == clust])
                        fout.write('<feature id="%s">\n' % j)
                        fout.write('  <position dim="0">%f</position>\n' % ((maxRT + minRT) / 2))
                        fout.write('  <position dim="1">%f</position>\n' % np.average(mzs, weights=ints))
                        fout.write("  <intensity>%f</intensity>\n" % np.sum(temp["intensity"][temp["cluster"] == clust]))
                        fout.write('  <quality dim="0">0</quality>\n')
                        fout.write('  <quality dim="1">0</quality>\n')
                        fout.write("  <overallquality>0</overallquality>\n")
                        fout.write("  <charge>1</charge>\n")
                        fout.write('  <convexhull nr="0">\n')
                        fout.write('    <pt x="%f" y="%f" />\n' % (minRT, np.min(original_mzs)))
                        fout.write('    <pt x="%f" y="%f" />\n' % (minRT, np.max(original_mzs)))
                        fout.write('    <pt x="%f" y="%f" />\n' % (maxRT, np.max(original_mzs)))
                        fout.write('    <pt x="%f" y="%f" />\n' % (maxRT, np.min(original_mzs)))
                        fout.write("  </convexhull>\n")
                        fout.write("</feature>\n")

                    fout.write("    </featureList>\n")
                    fout.write("  </featureMap>\n")

            logging.info(
                "    .. Sample %4d / %4d (%45s) spectra %3d, signals %6d, cluster after crude %6d, fine %6d, quality control %6d, final number of features %6d"
                % (
                    samplei + 1,
                    len(self.get_sample_names()),
                    sample,
                    summary_totalSpectra,
                    summary_totalSignals,
                    summary_clusterAfterCrude,
                    summary_clusterAfterFine,
                    summary_clusterAfterQualityFunctions,
                    np.unique(temp["cluster"]).shape[0],
                )
            )

            mzs, intensities, usedFeatures = self._collapse_mz_cluster(
                temp["mz"], temp["original_mz"], temp["intensity"], temp["time"], temp["cluster"], intensity_collapse_method=aggregation_function
            )

            sampleObjNew = fileio.MSData_in_memory.generate_from_MSData_object(msDataObj)
            startRT = sampleObjNew.get_spectrum(0).time
            endRT = sampleObjNew.get_spectrum(sampleObjNew.get_n_spectra() - 1).time
            sampleObjNew.delete_spectra(ns=[i for i in range(1, msDataObj.get_n_spectra())])
            spectrum = sampleObjNew.get_spectrum(0)
            spectrum.mz = mzs
            spectrum.spint = intensities
            spectrum.usedFeatures = usedFeatures
            spectrum.startRT = startRT
            spectrum.endRT = endRT

            msDataObj.original_MSData_object = msDataObj.to_MSData_object
            msDataObj.to_MSData_object = sampleObjNew

    #####################################################################################################
    # Bracketing of several samples
    #

    def bracket_consensus_spectrum_samples(self, closest_signal_max_deviation_ppm=20, max_ppm_deviation=25, show_diagnostic_plots=False):
        """
        Function to bracket consensus spectra across different samples

        Parameters
        ----------
        max_ppm_deviation : float, optional
            the maximum allowed devation (in ppm) a consensus group is allowed to have. Defaults to 25.
        show_diagnostic_plots : bool, optional
            indicator if a diagnostic plot shall be shown. Defaults to False.
        """
        self.add_data_processing_step(
            "bracket consensus spectrum per sample", "bracket consensus spectrum per sample", {"max_ppm_deviation": max_ppm_deviation}
        )
        temp = {
            "sample": [],
            "spectrumInd": [],
            "time": [],
            "mz": [],
            "intensity": [],
            "original_usedFeatures": [],
            "startRT": [],
            "endRT": [],
            "cluster": [],
        }

        for samplei, sample in tqdm.tqdm(enumerate(self.get_sample_names()), total=len(self.get_sample_names()), desc="bracketing: fetching data"):
            msDataObj = self.get_msDataObj_for_sample(sample)
            for k, spectrum in msDataObj.get_spectra_iterator():
                temp["sample"].extend((sample for i in range(spectrum.mz.shape[0])))
                temp["spectrumInd"].extend((k for i in range(spectrum.mz.shape[0])))
                temp["time"].extend((spectrum.time for i in range(spectrum.mz.shape[0])))
                temp["mz"].extend((mz for mz in spectrum.mz))
                temp["intensity"].extend((np.log10(inte) for inte in spectrum.spint))
                temp["original_usedFeatures"].extend((spectrum.usedFeatures[i] for i in range(len(spectrum.usedFeatures))))
                temp["startRT"].extend((spectrum.startRT for mz in spectrum.mz))
                temp["endRT"].extend((spectrum.endRT for mz in spectrum.mz))
                temp["cluster"].extend((0 for inte in spectrum.spint))

        temp["sample"] = np.array(temp["sample"])
        temp["spectrumInd"] = np.array(temp["spectrumInd"])
        temp["time"] = np.array(temp["time"])
        temp["mz"] = np.array(temp["mz"], dtype=np.float64)
        temp["intensity"] = np.array(temp["intensity"], dtype=np.float64)
        # temp["original_usedFeatures"] = temp["original_usedFeatures"]
        temp["startRT"] = np.array(temp["startRT"])
        temp["endRT"] = np.array(temp["endRT"])
        temp["cluster"] = np.array(temp["cluster"])

        if False:  # used for development purposes TODO remove all code in this if
            print("Restricting", end="")
            keep = np.abs(temp["mz"] - 358.3079) < 0.06
            useInds = np.argwhere(keep)[:, 0]

            temp["sample"] = temp["sample"][useInds]
            temp["spectrumInd"] = temp["spectrumInd"][useInds]
            temp["time"] = temp["time"][useInds]
            temp["mz"] = temp["mz"][useInds]
            temp["intensity"] = temp["intensity"][useInds]
            temp["original_usedFeatures"] = [temp["original_usedFeatures"][i] for i in useInds]
            temp["startRT"] = temp["startRT"][useInds]
            temp["endRT"] = temp["endRT"][useInds]
            temp["cluster"] = temp["cluster"][useInds]
            print("... done")

        # clusering v2
        # Iterative cluster generation with the same algorithm used for calculating the consensus spectra.
        # This algorithm is greedy and extends clusters based on their mean mz value and the difference to the next closest feature
        if True:
            logging.info("    .. clustering with method 2")
            min_difference_ppm = 100
            min_signals_per_cluster = 2
            temp["cluster"] = self._crude_clustering_for_mz_list(sample, temp["mz"], temp["intensity"], min_difference_ppm=min_difference_ppm)

            # remove any cluster with less than min_signals_per_cluster signals
            clustInds, ns = np.unique(temp["cluster"], return_counts=True)
            clustNs = ns[temp["cluster"]]
            temp["cluster"][clustNs < min_signals_per_cluster] = -1
            keep = temp["cluster"] >= 0

            temp["sample"] = temp["sample"][keep]
            temp["spectrumInd"] = temp["spectrumInd"][keep]
            temp["time"] = temp["time"][keep]
            temp["mz"] = temp["mz"][keep]
            temp["intensity"] = temp["intensity"][keep]
            temp["original_usedFeatures"] = [temp["original_usedFeatures"][i] for i in range(temp["cluster"].shape[0]) if keep[i]]
            temp["startRT"] = temp["startRT"][keep]
            temp["endRT"] = temp["endRT"][keep]
            temp["cluster"] = self._reindex_cluster(temp["cluster"][keep])

            # refine cluster
            temp["cluster"] = _refine_clustering_for_mz_list(
                "bracketed_list",
                temp["mz"],
                temp["intensity"],
                temp["spectrumInd"],
                temp["cluster"],
                closest_signal_max_deviation_ppm=closest_signal_max_deviation_ppm,
                max_mz_deviation_ppm=max_ppm_deviation,
            )

            # remove wrong cluster
            keep = temp["cluster"] >= 0

            temp["sample"] = temp["sample"][keep]
            temp["spectrumInd"] = temp["spectrumInd"][keep]
            temp["time"] = temp["time"][keep]
            temp["mz"] = temp["mz"][keep]
            temp["intensity"] = temp["intensity"][keep]
            temp["original_usedFeatures"] = [temp["original_usedFeatures"][i] for i in range(temp["cluster"].shape[0]) if keep[i]]
            temp["startRT"] = temp["startRT"][keep]
            temp["endRT"] = temp["endRT"][keep]
            temp["cluster"] = self._reindex_cluster(temp["cluster"][keep])

        # Clustering version 1
        # This algorithm starts with the highest abundant features and adds all other features to that particular cluster.
        # It has the drawback that certain overlapping mz values can be predatory.
        if False:
            cclust = 0
            while np.sum(temp["cluster"] == 0) > 0:
                c = np.argmax(temp["intensity"] * (temp["cluster"] == 0))
                cmz = temp["mz"][c]

                assign = np.where(np.abs(temp["mz"] - cmz) / temp["mz"] * 1e6 <= max_ppm_deviation)
                temp["cluster"][assign] = cclust

                cclust = cclust + 1

        tempClusterInfo = {"cluster": [], "meanMZ": [], "minMZ": [], "maxMZ": [], "mzDevPPM": [], "uniqueSamples": [], "featureMLInfo": []}

        temp["cluster"] = self._reindex_cluster(temp["cluster"])
        clusts = np.unique(temp["cluster"])
        for clust in tqdm.tqdm(clusts, desc="bracketing: saving for featureML export"):
            assign = [i for i in range(len(temp["cluster"])) if temp["cluster"][i] == clust]

            tempClusterInfo["cluster"].append(clust)
            mzs = temp["mz"][assign]
            tempClusterInfo["meanMZ"].append(np.mean(mzs))
            tempClusterInfo["minMZ"].append(np.min(mzs))
            tempClusterInfo["maxMZ"].append(np.max(mzs))
            tempClusterInfo["mzDevPPM"].append((np.max(mzs) - np.min(mzs)) / np.mean(mzs) * 1e6)
            tempClusterInfo["uniqueSamples"].append(np.unique(temp["sample"][assign]).shape[0])
            tempClusterInfo["featureMLInfo"].append({})

            tempClusterInfo["featureMLInfo"][clust]["overallquality"] = tempClusterInfo["uniqueSamples"][clust]
            tempClusterInfo["featureMLInfo"][clust]["meanMZ"] = tempClusterInfo["meanMZ"][clust]
            tempClusterInfo["featureMLInfo"][clust]["mzDevPPM"] = tempClusterInfo["mzDevPPM"][clust]

            fs = [temp["original_usedFeatures"][i] for i in range(len(temp["original_usedFeatures"])) if temp["cluster"][i] == clust]
            startRTs = temp["startRT"][temp["cluster"] == clust]
            endRTs = temp["endRT"][temp["cluster"] == clust]
            samples = temp["sample"][temp["cluster"] == clust]

            tempClusterInfo["featureMLInfo"][clust]["sampleHulls"] = {}
            for samplei, sample in enumerate(self.get_sample_names()):
                use = samples == sample
                if np.sum(use) > 0:
                    domzs = np.ndarray.flatten(np.concatenate([np.array(fs[i][:, 3]) for i in range(len(fs)) if use[i]]))
                    startRT = startRTs[use][0]
                    endRT = endRTs[use][0]
                    tempClusterInfo["featureMLInfo"][clust]["sampleHulls"][sample] = [
                        (startRT, np.min(domzs)),
                        (startRT, np.max(domzs)),
                        (endRT, np.max(domzs)),
                        (endRT, np.min(domzs)),
                    ]

        if show_diagnostic_plots:
            temp = None
            for featurei in tqdm.tqdm(range(len(tempClusterInfo["minMZ"])), desc="bracketing: generating plots"):
                for samplei, sample in enumerate(self.get_sample_names()):
                    msDataObj = self.get_msDataObj_for_sample(sample)
                    for k, spectrum in msDataObj.get_spectra_iterator():
                        usei = np.where(
                            np.logical_and(spectrum.mz >= tempClusterInfo["minMZ"][featurei], spectrum.mz <= tempClusterInfo["maxMZ"][featurei])
                        )[0]
                        if usei.size > 0:
                            for i in usei:
                                temp = _add_row_to_pandas_creation_dictionary(
                                    temp,
                                    rt=spectrum.time,
                                    mz=spectrum.mz[i],
                                    intensity=spectrum.spint[i],
                                    sample=sample,
                                    file=sample.split("::")[0],
                                    group=self.get_metaData_for_sample(sample, "group"),
                                    type="consensus",
                                    feature=featurei,
                                )

                                for j in range(spectrum.usedFeatures[i].shape[0]):
                                    temp = _add_row_to_pandas_creation_dictionary(
                                        temp,
                                        rt=spectrum.usedFeatures[i][j, 2],
                                        mz=spectrum.usedFeatures[i][j, 0],
                                        intensity=spectrum.usedFeatures[i][j, 1],
                                        sample=sample,
                                        file=sample.split("::")[0],
                                        group=self.get_metaData_for_sample(sample, "group"),
                                        type="non-consensus",
                                        feature=featurei,
                                    )

                                for j in range(spectrum.usedFeatures[i].shape[0]):
                                    temp = _add_row_to_pandas_creation_dictionary(
                                        temp,
                                        rt=spectrum.usedFeatures[i][j, 2],
                                        mz=spectrum.usedFeatures[i][j, 3],
                                        intensity=spectrum.usedFeatures[i][j, 1],
                                        sample=sample,
                                        file=sample.split("::")[0],
                                        group=self.get_metaData_for_sample(sample, "group"),
                                        type="raw-onlyFeatures",
                                        feature=featurei,
                                    )

                    for k, spectrum in msDataObj.original_MSData_object.get_spectra_iterator():
                        usei = np.where(
                            np.logical_and(
                                spectrum.original_mz >= spectrum.reverseMZ(self.features[featurei][0]),
                                spectrum.original_mz <= spectrum.reverseMZ(self.features[featurei][2]),
                            )
                        )[0]
                        if usei.size > 0:
                            for i in usei:
                                temp = _add_row_to_pandas_creation_dictionary(
                                    temp,
                                    rt=spectrum.time,
                                    mz=spectrum.original_mz[i],
                                    intensity=spectrum.spint[i],
                                    sample=sample,
                                    file=sample.split("::")[0],
                                    group=self.get_metaData_for_sample(sample, "group"),
                                    type="raw-allSignals",
                                    feature=featurei,
                                )

            temp = pd.DataFrame(temp)
            dat = pd.concat(
                [
                    temp.groupby(["feature", "type"])["mz"].count(),
                    temp.groupby(["feature", "type"])["mz"].min(),
                    temp.groupby(["feature", "type"])["mz"].mean(),
                    temp.groupby(["feature", "type"])["mz"].max(),
                    temp.groupby(["feature", "type"])["intensity"].mean(),
                    temp.groupby(["feature", "type"])["intensity"].std(),
                ],
                axis=1,
            )
            dat.set_axis(["Count", "mzMin", "mzMean", "mzMax", "intMean", "intStd"], axis=1, inplace=True)
            dat = dat.reset_index()
            dat["mzDevPPM"] = (dat["mzMax"] - dat["mzMin"]) / dat["mzMean"] * 1e6
            dat["intMean"] = np.log2(dat["intMean"])
            dat = dat[dat["Count"] > 1]

            p = (
                p9.ggplot(data=dat, mapping=p9.aes(x="mzDevPPM", group="type"))
                + p9.geom_histogram()
                + p9.facet_wrap("~ type")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.05, 'left':0.15})
                + p9.theme_minimal()
                + p9.ggtitle("Deviation of feature cluster in ppm")
                + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
                + p9.theme(legend_position="bottom")
            )
            # print(p)

            p = (
                p9.ggplot(data=dat, mapping=p9.aes(x="intMean", y="mzDevPPM", group="type"))
                + p9.geom_point(alpha=0.3)
                + p9.facet_wrap("~ type")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.05, 'left':0.15})
                + p9.theme_minimal()
                + p9.ggtitle("Deviation of feature cluster in ppm")
                + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
                + p9.theme(legend_position="bottom")
            )
            # print(p)

            p = (
                p9.ggplot(data=dat, mapping=p9.aes(x="mzMean", y="mzDevPPM", group="type", colour="type"))
                + p9.geom_point(alpha=0.15)
                + p9.facet_wrap("~ type")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.05, 'left':0.15})
                + p9.theme_minimal()
                + p9.ggtitle("Deviation of feature cluster")
                + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
                + p9.theme(legend_position="bottom")
            )
            # print(p)

        self.features = [
            e for e in zip(tempClusterInfo["minMZ"], tempClusterInfo["meanMZ"], tempClusterInfo["maxMZ"], tempClusterInfo["featureMLInfo"])
        ]

    #####################################################################################################
    # Generate data matrix from bracketing information
    # This setp also automatically re-integrates the results
    #

    def build_data_matrix(self, on="originalData", originalData_mz_deviation_multiplier_PPM=0, aggregation_fun="average"):
        """
        generates a data matrix from corrected, consensus spectra and bracketed features

        Parameters
        ----------
        on : str, optional
            the data used to derived abundance values from. with 'processedData' the consensus spectra will be used, while with 'originalData' the raw-data will be used. Defaults to "originalData".
        originalData_mz_deviation_multiplier_PPM : int, optional
            an optional mz deviation allowed for the raw-data integration. Defaults to 0.
        aggregation_fun : str, optional
            the method to calculate the derived abundance on integration of raw data. Defaults to "average".

        Raises
        ======
        ValueError
            raised if parameters on and aggregation_fun have invalid values
        """
        self.add_data_processing_step(
            "build data matrix", "build data matrix", {"on": on, "originalData_mz_deviation_multiplier_PPM": originalData_mz_deviation_multiplier_PPM}
        )

        if on.lower() not in ("processedData".lower(), "originalData".lower()):
            raise ValueError("Unknown option for parameter on. Must be either of ['processedData', 'originalData']")
        if on.lower() == "originalData".lower() and aggregation_fun.lower() not in ("average".lower(), "sum".lower(), "max".lower()):
            raise ValueError("Unknown aggregation method provided, must be either of ['average', 'sum', 'max']")

        sampleNames = self.get_sample_names()
        sampleNamesToRowI = dict(((sample, i) for i, sample in enumerate(sampleNames)))

        dataMatrix = np.zeros((len(sampleNames), len(self.features)))
        for samplei, sample in tqdm.tqdm(enumerate(sampleNames), total=len(sampleNames), desc="data matrix: gathering data"):
            msDataObj = self.get_msDataObj_for_sample(sample)
            spectrum = msDataObj.get_spectrum(0)

            for braci, (mzmin, mzmean, mzmax, _) in enumerate(self.features):
                if on.lower() == "processedData".lower():
                    use = np.logical_and(spectrum.mz >= mzmin, spectrum.mz <= mzmax)
                    if np.sum(use) > 0:
                        dataMatrix[sampleNamesToRowI[sample], braci] = np.sum(spectrum.spint[use])
                    else:
                        dataMatrix[sampleNamesToRowI[sample], braci] = np.nan

                elif on.lower() == "originalData".lower():
                    s = np.array(())
                    for oSpectrumi, oSpectrum in msDataObj.original_MSData_object.get_spectra_iterator():
                        _mzmin, _mzmean, _mzmax = oSpectrum.reverseMZ(mzmin), oSpectrum.reverseMZ(mzmean), oSpectrum.reverseMZ(mzmax)
                        _mzmin, _mzmax = _mzmin * (1.0 - originalData_mz_deviation_multiplier_PPM / 1e6), _mzmax * (
                            1.0 + originalData_mz_deviation_multiplier_PPM / 1e6
                        )

                        use = np.logical_and(oSpectrum.original_mz >= _mzmin, oSpectrum.original_mz <= _mzmax)
                        if np.sum(use) > 0:
                            s = np.concatenate((s, oSpectrum.spint[use]))

                    if s.shape[0] > 0:
                        if aggregation_fun.lower() == "average".lower():
                            s = np.average(s)
                        elif aggregation_fun.lower() == "sum".lower():
                            s = np.sum(s)
                        elif aggregation_fun.lower() == "max".lower():
                            s = np.max(s)
                        dataMatrix[sampleNamesToRowI[sample], braci] = s

                    else:
                        dataMatrix[sampleNamesToRowI[sample], braci] = np.nan

        self.samples = sampleNames
        self.groups = [self.get_metaData_for_sample(sample, "group") for sample in self.samples]
        self.batches = [self.get_metaData_for_sample(sample, "batch") for sample in self.samples]
        self.dat = dataMatrix

    #####################################################################################################
    # Blank subtraction
    #

    def blank_subtraction(self, blankGroup, toTestGroups, foldCutoff=2, pvalueCutoff=0.05, minDetected=2, plot=False):
        """
        Method to remove background features from the datamatrix. Repeated calls with different blank groups are possible.
        Inspired by the background-subtraction module of MZmine3

        Parameters
        ----------
        blankGroup : string
            the name of the blank group
        toTestGroups : list of str
            the name of the groups to test against the blank group
        foldCutoff : int, optional
            the minimum fold-change between at least one test-group and the blank group in order for a feature to not be considered a background. Defaults to 2.
        pvalueCutoff : float, optional
            the alpha-threshold for the ttest. Defaults to 0.05.
        minDetected : int, optional
            the minimum number a feature must be detected in the background samples in order to be considered a background features. Defaults to 2.
        plot : bool, optional
            indicator whether the subtraction shall be plotted as a volcano plot. Defaults to False.

        Raises
        ======
        NotImplementedError
            should never be raised, but is if the algorithm's implementation is incorrect
        """
        self.add_data_processing_step(
            "blank subtraction",
            "blank subtraction",
            {
                "blankGroup": blankGroup,
                "toTestGroups": toTestGroups,
                "foldCutoff": foldCutoff,
                "pvalueCutoff": pvalueCutoff,
                "minDetected": minDetected,
            },
        )

        keeps = [0 for i in range(self.dat.shape[1])]

        temp = None

        for featurei in range(self.dat.shape[1]):
            blankInds = [i for i, group in enumerate(self.groups) if group == blankGroup]
            valsBlanks = self.dat[blankInds, featurei]
            notInBlanks = False

            if np.sum(~np.isnan(self.dat[:, featurei])) < minDetected:
                continue

            if np.all(np.isnan(valsBlanks)):
                notInBlanks = True

            valsBlanks = valsBlanks[~np.isnan(valsBlanks)]

            for toTestGroup in toTestGroups:
                toTestInds = [i for i, group in enumerate(self.groups) if group == toTestGroup]
                valsGroup = self.dat[toTestInds, featurei]

                if np.sum(~np.isnan(valsGroup)) < minDetected:
                    pass

                elif notInBlanks:
                    assert keeps[featurei] <= 0
                    keeps[featurei] -= 1

                    temp = _add_row_to_pandas_creation_dictionary(
                        temp, pvalues=-np.inf, folds=np.inf, sigIndicators="only in group", comparisons="'%s' vs '%s'" % (toTestGroup, blankGroup)
                    )

                else:
                    valsGroup = valsGroup[~np.isnan(valsGroup)]
                    if valsBlanks.shape[0] > 1:
                        pval = scipy.stats.ttest_ind(valsGroup, valsBlanks, equal_var=False, alternative="greater", trim=0)[1]
                    elif valsBlanks.shape[0] == 1:
                        pval = scipy.stats.ttest_1samp(valsGroup, valsBlanks[0], alternative="greater")[1]
                    else:
                        raise NotImplementedError("No blank values available, implementation is incorrect")
                    fold = np.mean(valsGroup) / np.mean(valsBlanks)
                    sigInd = pval <= pvalueCutoff and fold >= foldCutoff

                    assert keeps[featurei] >= 0
                    if sigInd:
                        keeps[featurei] += 1

                    temp = _add_row_to_pandas_creation_dictionary(
                        temp,
                        pvalues=-np.log10(pval),
                        folds=np.log2(fold),
                        sigIndicators="group >> blank" if sigInd else "-",
                        comparisons="'%s' vs '%s'" % (toTestGroup, blankGroup),
                    )

        if plot:
            temp = pd.DataFrame(temp)
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="folds", y="pvalues", colour="sigIndicators"))
                + p9.geom_point(alpha=0.3)
                + p9.geom_hline(yintercept=-np.log10(0.05), alpha=0.3, colour="black")
                + p9.geom_vline(xintercept=[np.log2(foldCutoff)], alpha=0.3, colour="black")
                + p9.facet_wrap("comparisons")
                + p9.theme_minimal()
                # + p9.theme(legend_position = "bottom")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.15, 'left':0.15})
                + p9.ggtitle(
                    "Blank subtraction volcano plots\n%d features in blanks and samples; %d not in blanks (not illustrated)"
                    % (np.sum(np.array(keeps) > 0), np.sum(np.array(keeps) < 0))
                )
            )
            print(p)

        for i in sorted(list(set(keeps))):
            if i < 0:
                logging.info(
                    "    .. %d features not found in any of the blank samples, but at least in %d samples of %d groups and thus these features will be used"
                    % (sum([k == i for k in keeps]), minDetected, -i)
                )
            elif i == 0:
                logging.info(
                    "    .. %d features found in None of the blank comparisons with higher abundances in the samples. These features will be removed"
                    % (sum([k == i for k in keeps]))
                )
            else:
                logging.info(
                    "    .. %d features found in %d of the blank comparisons with higher abundances in the samples and in at least %d samples. These features will be used"
                    % (sum([k == i for k in keeps]), i, minDetected)
                )
        logging.info(
            " Significance criteria are pval <= pvalueCutoff (%.3f) and fold >= foldCutoff (%.1f) and detected in at least %d samples of a non-Blank group"
            % (pvalueCutoff, foldCutoff, minDetected)
        )

        keeps = [i for i in range(len(keeps)) if keeps[i] != 0]
        self.subset_features(keeps)

        logging.info("    .. %d features remain in the dataset" % (self.dat.shape[1]))

    #####################################################################################################
    # Feature annotation
    #

    def annotate_features(self, useGroups=None, max_deviation_ppm=100, search_ions=None, remove_other_ions=True, plot=False):
        """
        Function to annotate the bracketed features with different sister ions (adducts, isotopologs, etc.) relative to parent ions (mostly [M+H]+ or [M-H]-)

        Parameters
        ----------
        useGroups : list of str, optional
            groups to be used for the annotation (important for testing the ratio). Defaults to None.
        max_deviation_ppm : int, optional
            the maximum allowed deviation between the calculated and observed mz value of a sister ion. Defaults to 100.
        search_ions : dict, optional
            the ions to search for. keys are ion names, values are mz increments (no decrements allowed). Defaults to None.
        remove_other_ions : bool, optional
            indicator if annotated sister ions should be removed. Defaults to True.
        plot : bool, optional
            indicator if annotation results should be plotted. Defaults to False.
        """
        self.add_data_processing_step(
            "annotate features", "annotate features", {"useGroups": useGroups, "max_deviation_ppm": max_deviation_ppm, "search_ions": search_ions}
        )

        if search_ions is None:
            # , "arbi1": 0.5, "arbi2": 0.75, "arbi3": 0.9, "arbi4": 1.04, "arbi5": 1.1}
            searchIons = {"+Na": 22.989218 - 1.007276, "+NH4": 18.033823 - 1.007276, "+CH3OH+H": 33.033489 - 1.007276}
            for cn in range(1, 5):
                searchIons["[13C%d]" % cn] = 1.00335484 * cn

        if useGroups is None:
            useGroups = list(set(self.groups))

        features_ = np.array(self.features)
        order = np.argsort(features_[:, 1])
        features_ = features_[order, :]
        dat_ = self.dat[:, order]

        annotations = [[] for i in range(features_.shape[0])]
        temp = None
        for featurei in tqdm.tqdm(range(features_.shape[0]), desc="annotating features"):
            mz = features_[featurei, 1]
            iAmParent = False

            intensities = np.array([dat_[samplei, featurei] for samplei, sample in enumerate(self.samples) if not np.isnan(dat_[samplei, featurei])])

            if len(annotations[featurei]) == 0:
                for searchIon in searchIons:
                    searchMZ = mz + searchIons[searchIon]

                    ind, deviationPPM = _find_feature_by_mz(features_, searchMZ, max_deviation_ppm=max_deviation_ppm)
                    if ind is not None:
                        ratios = []
                        for samplei, sample in enumerate(self.samples):
                            if useGroups is None or self.groups[samplei] in useGroups:
                                ratio = dat_[samplei, ind] / dat_[samplei, featurei]
                                if not np.isnan(ratio) and ratio > 0:
                                    ratios.append(ratio)
                        ratios = np.array(ratios)

                        if ratios.shape[0] > 10 and np.mean(ratios) > 2.0 and np.mean(ratios) < 200.0 and np.std(ratios) < 50.0:
                            annotations[ind].append(
                                {
                                    "i_am": searchIon,
                                    "ratiosMean": np.mean(ratios),
                                    "ratiosSD": np.std(ratios),
                                    "parentMZ": mz,
                                    "mzDeviationPPM": (searchMZ - mz) / mz * 1e6,
                                }
                            )

                            if not iAmParent:
                                annotations[featurei].append(
                                    {
                                        "i_am": "parent",
                                    }
                                )
                                iAmParent = True

                        temp = _add_row_to_pandas_creation_dictionary(
                            temp,
                            searchIon="%s (%.4f)" % (searchIon, searchIons[searchIon]),
                            cns="%s (%.4f)" % (searchIon, searchIons[searchIon]),
                            MZs=mz,
                            intensityMeans=np.mean(intensities) if intensities.shape[0] > 0 else 0,
                            deviations=_mz_deviationPPM_between(features_[ind, 1], searchMZ),
                            ratiosMean=np.mean(ratios) if ratios.shape[0] > 0 else 0,
                            ratiosSTD=np.std(ratios) if ratios.shape[0] > 1 else 0,
                            ratiosRSTD=np.std(ratios) / np.mean(ratios) * 100.0 if ratios.shape[0] > 1 else 0,
                            ratiosCount=ratios.shape[0],
                        )
        if plot:
            temp = pd.DataFrame(temp)
            temp["searchIon"] = temp["searchIon"].astype(object)
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="MZs", y="deviations", colour="searchIon"))
                + p9.geom_point(alpha=0.05)
                + p9.facet_wrap("~searchIon")
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.15, "left": 0.15})
                + p9.ggtitle("Distribution of mz deviation of sister ions")
            )
            print(p)
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="np.log10(intensityMeans)", y="ratiosRSTD", colour="ratiosCount"))
                + p9.geom_point(alpha=0.1)
                + p9.geom_hline(yintercept=50, colour="firebrick")
                # + p9.facet_wrap("~ratiosCount")
                + p9.theme_minimal()
                + p9.ylim(0, 300)
                + p9.theme(legend_position="bottom")
                + p9.theme(subplots_adjust={"wspace": 0.15, "hspace": 0.25, "top": 0.93, "right": 0.99, "bottom": 0.15, "left": 0.15})
                + p9.ggtitle("Distribution of ratio deviation of sister ions")
            )
            print(p)

            temp["intensityMeansCUT"] = pd.cut(np.log10(temp["intensityMeans"]), 10)
            logging.info(temp[temp["searchIon"] == "[13C1] (1.0034)"].groupby(["intensityMeansCUT"])[["ratiosRSTD"]].describe().to_markdown())

        annotations = [{"Adducts": annotations[i]} if len(annotations[i]) > 0 else {} for i in np.argsort(order)]
        if self.featureAnnotations is None:
            self.featureAnnotations = annotations
        else:
            for featurei in range(len(self.features)):
                if len(annotations[featurei]) > 0:
                    x = annotations[featurei]
                    if "Adducts" in self.featureAnnotations[featurei]:
                        x = self.featureAnnotations[featurei]["Adducts"] | annotations[featurei]["Adducts"]
                    if len(x) > 0:
                        self.featureAnnotations[featurei]["Adducts"] = x

        if remove_other_ions:
            keeps_indices = []
            for i in range(self.dat.shape[1]):
                annos = self.featureAnnotations[i]["Adducts"]

                if len(annos) == 0 or (len(annos) == 1 and annos[0]["i_am"] == "parent"):
                    keeps_indices.append(i)

            print(
                "   .. removing %d (%.1f%%) features annotated as sister-ions of a parent"
                % (self.dat.shape[1] - len(keeps_indices), (self.dat.shape[1] - len(keeps_indices)) / self.dat.shape[1] * 100)
            )
            self.subset_features(keep_features_with_indices=keeps_indices)

    #####################################################################################################
    # Feature annotation
    #

    def restrict_to_high_quality_features__most_n_abundant(self, n_features):
        """
        Function to select high-quality features (after bracketing)
        This function selects the top-n most abundant features

        Parameters
        ----------
        n_features : integer
            the number of features to select
        """
        self.add_data_processing_step("restricting to n most abundant features", "restricting to most abundant features", {"n_features": n_features})
        logging.info(" Restricting data matrix to a %d of the most abundant features" % (n_features))

        abundances = [-1 for i in range(self.dat.shape[1])]
        for featurei in range(self.dat.shape[1]):
            vals = self.dat[:, featurei]
            vals = vals[~np.isnan(vals)]
            if vals.shape[0] > 0:
                inte = np.mean(vals)
                abundances[featurei] = inte / self.dat.shape[0]

        abundances = np.array(abundances)
        keeps = np.argsort(abundances)[::-1][0:n_features]

        self.subset_features(keep_features_with_indices=keeps)
        logging.info("    .. using %d features" % (self.dat.shape[1]))

    def restrict_to_high_quality_features__found_in_replicates(self, test_groups, minimum_ratio_found, found_in_type="anyGroup"):
        """
        Function to select high-quality features (after bracketing)
        This function selects only features that are found in at least % of replicates

        Parameters
        ----------
        test_groups : list of str
            the groups to be tested
        minimum_ratio_found : float
            the minimum ratio of all samples in a group in which the feature must have been detected
        found_in_type : str, optional
            indicator if the feature must be present in all or any group in at least % samples. Defaults to "anyGroup".
        """
        self.add_data_processing_step(
            "restricting to found in replicates features",
            "restricting to found in replicates features",
            {"test_groups": test_groups, "minimum_ratio_found": minimum_ratio_found, "found_in_type": found_in_type},
        )
        logging.info(
            " Using only features that are present in more than %.1f%% replicates of %s to test"
            % (minimum_ratio_found * 100, "all groups" if found_in_type.lower() == "allGroups".lower() else "any group")
        )

        if found_in_type.lower() not in ("allGroups".lower(), "anyGroup".lower()):
            raise ValueError("Unknown option for parameter found_in_type, must be either of ['anyGroup', 'allGroups']")

        foundIn = [0 for i in range(self.dat.shape[1])]
        for featurei in range(self.dat.shape[1]):
            for grp in test_groups:
                groupInd = [i for i, group in enumerate(self.groups) if group == grp]

                vals = self.dat[groupInd, featurei]

                if np.sum(~np.isnan(vals)) / vals.shape[0] >= minimum_ratio_found:
                    foundIn[featurei] += 1

            if found_in_type.lower() == "allGroups".lower() and foundIn[featurei] < len(test_groups):
                foundIn[featurei] = 0

        keeps = np.where(np.array(foundIn) > 0)[0]

        self.subset_features(keep_features_with_indices=keeps)
        logging.info("    .. using %d features" % (self.dat.shape[1]))

    def restrict_to_high_quality_features__minimum_intensity_filter(self, test_groups, minimum_intensity):
        """
        Function to select high-quality features (after bracketing)
        This function selects only features that have a minimum intensity in at least one samples of a group

        Parameters
        ----------
        test_groups : list of str
            the groups to test
        minimum_intensity : float
            the mimimum requested signal intensity
        """
        self.add_data_processing_step(
            "restricting to minimum intensity filter",
            "restricting to minimum intensity filter",
            {"test_groups": test_groups, "minimum_intensity": minimum_intensity},
        )
        logging.info(" Using only features that have a minimum intensity of %.1f in at least one test-group" % (minimum_intensity))

        use = [False for i in range(self.dat.shape[1])]
        for featurei in range(self.dat.shape[1]):
            for grp in test_groups:
                groupInd = [i for i, group in enumerate(self.groups) if group == grp]

                vals = self.dat[groupInd, featurei]

                if np.sum(~np.isnan(vals)) > 0 and np.mean(vals[~np.isnan(vals)]) > minimum_intensity:
                    use[featurei] = True

        keeps = np.where(np.array(use))[0]

        self.subset_features(keep_features_with_indices=keeps)
        logging.info("    .. using %d features" % (self.dat.shape[1]))

    def restrict_to_high_quality_features__low_RSD_in_groups(self, test_groups, maximum_RSD):
        """
        Function to select high-quality features (after bracketing)
        This function selects only features that have a low within-group variability

        Parameters
        ----------
        test_groups : list of str
            the groups to be tested
        maximum_RSD : float
            the maximum allowed rsd for a feature to be used, must be lower in all groups
        """
        self.add_data_processing_step(
            "restricting to maximum rsd ", "restricting to maximum rsd", {"test_groups": test_groups, "maximum_RSD": maximum_RSD}
        )
        logging.info(" Using only features that have a maximum RSD of %.1f in all test-group" % (maximum_RSD))

        use = [True for i in range(self.dat.shape[1])]
        for featurei in range(self.dat.shape[1]):
            for grp in test_groups:
                groupInd = [i for i, group in enumerate(self.groups) if group == grp]

                vals = self.dat[groupInd, featurei]
                vals[np.isnan(vals)] = 0

                if _relative_standard_deviation(vals) > maximum_RSD:
                    use[featurei] = False

        keeps = np.where(np.array(use))[0]

        self.subset_features(keep_features_with_indices=keeps)
        logging.info("    .. using %d features" % (self.dat.shape[1]))

    #####################################################################################################
    # Summary
    #

    def print_results_overview(self):
        """
        prints an overview of the bracketed results to the console
        """
        print("There are %d features (columns) and %d samples (rows) in the dataset" % (self.dat.shape[1], self.dat.shape[0]))
        print(
            "   .. %d (%.1f%%) features have at least one missing value (np.nan)"
            % (np.sum(np.isnan(self.dat).any(axis=0)), np.sum(np.isnan(self.dat).any(axis=0)) / self.dat.shape[1] * 100)
        )

        maxGroupSize = max((sum((grp == group for grp in self.groups)) for group in set(self.groups)))
        maxGroupLabelSize = max((len(group) for group in self.groups))
        a = {}
        for grp in set(self.groups):
            groupInd = [i for i, group in enumerate(self.groups) if group == grp]
            a[grp] = dict([(i, 0) for i in range(maxGroupSize + 1)])
            total = 0
            for featurei in range(self.dat.shape[1]):
                vals = self.dat[groupInd, featurei]
                f = np.sum(~np.isnan(vals))
                a[grp][f] += 1
                if f > 0:
                    total += 1
            a[grp]["total"] = total

        print(
            "This table displays the number of features detected in a set of experimental conditions, arranged in rows representing the number of samples and columns representing the different conditions. Each cell in the table contains a numerical value indicating the number of features that were detected in exactly the corresponding number of samples and under the corresponding experimental condition. Overall, the table provides a summary of the number of features that were identified across the different experimental conditions and sample sizes."
        )
        print("Detected   ", end="")
        for group in sorted(list(set(self.groups))):
            print("%%%ds  " % (maxGroupLabelSize) % (group), end="")
        print()

        for det in range(0, maxGroupSize + 1):
            print("%%%dd   " % (maxGroupLabelSize) % (det), end="")
            for group in sorted(list(set(self.groups))):
                print("%%%ds  " % (maxGroupLabelSize) % (a[group][det] if a[group][det] > 0 else ""), end="")
            print()
        print("%%%ds   " % (maxGroupLabelSize) % ("total"), end="")
        for group in sorted(list(set(self.groups))):
            print("%%%ds  " % (maxGroupLabelSize) % (a[group]["total"] if a[group]["total"] > 0 else ""), end="")
        print()

    def plot_sample_abundances(self):
        """
        plots an overview of the feature abundances per sample an dgroup

        Returns
        -------
        plotnine plot
            the generated plot
        """

        dat = {}
        for samplei, sample in enumerate(self.get_sample_names()):
            group = self.get_metaData_for_sample(sample, "group")
            batch = self.get_metaData_for_sample(sample, "batch")
            for featureInd in range(self.dat.shape[1]):
                if not np.isnan(self.dat[samplei, featureInd]):
                    dat = _add_row_to_pandas_creation_dictionary(
                        dat, sample=sample, group=group, batch=batch, feature=featureInd, abundance=self.dat[samplei, featureInd]
                    )

        dat = pd.DataFrame(dat)

        p = (
            p9.ggplot(data=dat, mapping=p9.aes(x="np.log2(abundance)", colour="group", group="sample"))
            + p9.geom_freqpoly(bins=24)
            + p9.facet_wrap("~ group")
            + p9.ggtitle("Overview of feature abundances per sample")
        )

        return p

    def plot_mz_deviation_overview(self, show=None, random_fraction=1):
        """
        plots an overview of the mz deviation in the bracketed results

        Parameters
        ----------
        show : _type_, optional
            indicator which plots should be shown. Defaults to None.
        random_fraction : int, optional
            use only a random fraction of the features. Defaults to 1.

        Raises
        ======
        ValueError
            Unknown option(s) specified
        """
        if show is None:
            show = ["consensus", "non-consensus", "raw"]
        elif type(show) == str:
            show = [show]
        if type(show) == list and not ("consensus" in show or "non-consensus" in show or "raw" in show):
            raise ValueError("Unknown option(s) for show parameter. Must be a list with entries ['consensus', 'non-consensus', 'raw']")

        dat = None
        useFeatures = list(range(self.dat.shape[1]))
        if random_fraction < 1:
            useFeatures = random.sample(useFeatures, int(len(useFeatures) * random_fraction))

        for featureInd in tqdm.tqdm(range(len(useFeatures)), desc="plot: gathering data"):
            temp = None
            for samplei, sample in enumerate(self.get_sample_names()):
                msDataObj = self.get_msDataObj_for_sample(sample)
                for k, spectrum in msDataObj.get_spectra_iterator():
                    usei = np.where(np.logical_and(spectrum.mz >= self.features[featureInd][0], spectrum.mz <= self.features[featureInd][2]))[0]
                    if usei.size > 0:
                        for i in usei:
                            if "consensus" in show:
                                temp = _add_row_to_pandas_creation_dictionary(
                                    temp,
                                    rt=spectrum.time,
                                    mz=spectrum.mz[i],
                                    intensity=spectrum.spint[i],
                                    sample=sample,
                                    file=sample.split("::")[0],
                                    group=self.get_metaData_for_sample(sample, "group"),
                                    type="consensus",
                                    feature=featureInd,
                                )

                            if "non-consensus" in show:
                                for j in range(spectrum.usedFeatures[i].shape[0]):
                                    temp = _add_row_to_pandas_creation_dictionary(
                                        temp,
                                        rt=spectrum.usedFeatures[i][j, 2],
                                        mz=spectrum.usedFeatures[i][j, 0],
                                        intensity=spectrum.usedFeatures[i][j, 1],
                                        sample=sample,
                                        file=sample.split("::")[0],
                                        group=self.get_metaData_for_sample(sample, "group"),
                                        type="non-consensus",
                                        feature=featureInd,
                                    )

                            if "raw" in show:
                                for j in range(spectrum.usedFeatures[i].shape[0]):
                                    temp = _add_row_to_pandas_creation_dictionary(
                                        temp,
                                        rt=spectrum.usedFeatures[i][j, 2],
                                        mz=spectrum.usedFeatures[i][j, 3],
                                        intensity=spectrum.usedFeatures[i][j, 1],
                                        sample=sample,
                                        file=sample.split("::")[0],
                                        group=self.get_metaData_for_sample(sample, "group"),
                                        type="raw",
                                        feature=featureInd,
                                    )

            temp = pd.DataFrame(temp)
            for name, group in temp.groupby(["type", "feature"]):
                avgMZW, stdMZW = _average_and_std(group["mz"], group["intensity"])
                avgInt, stdInt = np.mean(group["intensity"]), np.std(group["intensity"])

                dat = _add_row_to_pandas_creation_dictionary(
                    dat,
                    avgMZ=avgMZW,
                    stdMZ=stdMZW,
                    avgInt=avgInt,
                    stdInt=stdInt,
                    type=name[0],
                    feature=name[1],
                    calcType="weighted",
                )

                dat = _add_row_to_pandas_creation_dictionary(
                    dat,
                    avgMZ=np.mean(group["mz"]),
                    stdMZ=np.std(group["mz"]),
                    avgInt=avgInt,
                    stdInt=stdInt,
                    type=name[0],
                    feature=name[1],
                    calcType="unweighted",
                )

        dat = pd.DataFrame(dat)

        dat["stdMZPPM"] = dat["stdMZ"] / dat["avgMZ"] * 1e6
        dat = dat[dat["stdMZPPM"] > 1]
        p = (
            p9.ggplot(data=dat, mapping=p9.aes(x="avgMZ", y="stdMZPPM", colour="type"))
            + p9.geom_point(alpha=0.3)
            + p9.facet_grid("calcType ~ type")
            + p9.theme_minimal()
            + p9.ggtitle("Overview of mz average and std (weighted) of bracketed features")
            + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
            + p9.theme(legend_position="bottom")
        )
        print(p)

        dat["avgInt"] = np.log2(dat["avgInt"])
        p = (
            p9.ggplot(data=dat, mapping=p9.aes(x="avgInt", y="stdMZPPM", colour="type"))
            + p9.geom_point(alpha=0.3)
            + p9.facet_grid("calcType ~ type")
            + p9.theme_minimal()
            + p9.ggtitle("Overview of mz average and std (weighted) of bracketed features")
            + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
            + p9.theme(legend_position="bottom")
        )
        print(p)

    def plot_feature_mz_deviations(
        self,
        featureInd,
        refMZ=None,
        types=None,
        keep_samples=None,
        remove_samples=None,
        keep_groups=None,
        remove_groups=None,
        keep_batches=None,
        remove_batches=None,
    ):
        """
        plots the mz deviation for a particular features

        Parameters
        ----------
        featureInd : index
            the index of the feature to plot
        types : list of str, optional
            plots to include. Defaults to None.
        """
        if types is None:
            types = ["consensus", "raw-corrected", "raw"]
        temp = None

        if refMZ is None:
            refMZ = self.features[featureInd][1]

        for samplei, sample in enumerate(self.get_sample_names()):
            sgroup = self.get_metaData_for_sample(sample, "group")
            sbatch = self.get_metaData_for_sample(sample, "batch")
            if (
                (keep_samples is None or sample in keep_samples)
                and (remove_samples is None or sample not in remove_samples)
                and (keep_groups is None or sgroup in keep_groups)
                and (remove_groups is None or sgroup not in remove_groups)
                and (keep_batches is None or sbatch in keep_batches)
                and (remove_batches is None or sbatch not in remove_batches)
            ):
                msDataObj = self.get_msDataObj_for_sample(sample)
                for k, spectrum in msDataObj.get_spectra_iterator():
                    usei = np.where(np.logical_and(spectrum.mz >= self.features[featureInd][0], spectrum.mz <= self.features[featureInd][2]))[0]
                    if usei.size > 0:
                        for i in usei:
                            if "consensus" in types:
                                temp = _add_row_to_pandas_creation_dictionary(
                                    temp,
                                    rt=spectrum.time,
                                    mz=spectrum.mz[i],
                                    intensity=spectrum.spint[i],
                                    sample=sample,
                                    file=sample.split("::")[0],
                                    group=self.get_metaData_for_sample(sample, "group"),
                                    type="consensus",
                                    feature=featureInd,
                                )

                            if "raw-corrected" in types:
                                for j in range(spectrum.usedFeatures[i].shape[0]):
                                    temp = _add_row_to_pandas_creation_dictionary(
                                        temp,
                                        rt=spectrum.usedFeatures[i][j, 2],
                                        mz=spectrum.usedFeatures[i][j, 0],
                                        intensity=spectrum.usedFeatures[i][j, 1],
                                        sample=sample,
                                        file=sample.split("::")[0],
                                        group=self.get_metaData_for_sample(sample, "group"),
                                        type="raw-corrected",
                                        feature=featureInd,
                                    )

                if "raw" in types:
                    for k, spectrum in msDataObj.original_MSData_object.get_spectra_iterator():
                        usei = np.where(
                            np.logical_and(
                                spectrum.original_mz >= spectrum.reverseMZ(self.features[featureInd][0]),
                                spectrum.original_mz <= spectrum.reverseMZ(self.features[featureInd][2]),
                            )
                        )[0]
                        if usei.size > 0:
                            for i in usei:
                                temp = _add_row_to_pandas_creation_dictionary(
                                    temp,
                                    rt=spectrum.time,
                                    mz=spectrum.original_mz[i],
                                    intensity=spectrum.spint[i],
                                    sample=sample,
                                    file=sample.split("::")[0],
                                    group=self.get_metaData_for_sample(sample, "group"),
                                    type="raw",
                                    feature=featureInd,
                                )

        temp = pd.DataFrame(temp)
        p = (
            p9.ggplot(data=temp, mapping=p9.aes(x="rt", y="mz", colour="group"))
            + p9.geom_hline(
                data=temp.groupby(["type"]).mean("mz").reset_index(), mapping=p9.aes(yintercept="mz"), size=1.5, alpha=0.5, colour="slategrey"
            )
            + p9.geom_hline(
                data=temp.groupby(["type"]).min("mz").reset_index(), mapping=p9.aes(yintercept="mz"), size=1.25, alpha=0.5, colour="lightgrey"
            )
            + p9.geom_hline(
                data=temp.groupby(["type"]).max("mz").reset_index(), mapping=p9.aes(yintercept="mz"), size=1.25, alpha=0.5, colour="lightgrey"
            )
            + p9.geom_hline(yintercept=refMZ, size=1.5, alpha=0.5, colour="firebrick")
            + p9.geom_point()
            + p9.facet_wrap("type")
            + p9.theme_minimal()
            + p9.ggtitle(
                "Feature %.5f (%.5f - %.5f)\nmz deviation overall: %.1f (ppm, non-consensus) and %.1f (ppm, consensus), relative to reference: %.1f ppm"
                % (
                    self.features[featureInd][1],
                    self.features[featureInd][0],
                    self.features[featureInd][2],
                    (temp[temp["type"] == "non-consensus"]["mz"].max() - temp[temp["type"] == "non-consensus"]["mz"].min())
                    / temp[temp["type"] == "non-consensus"]["mz"].mean()
                    * 1e6,
                    (temp[temp["type"] == "consensus"]["mz"].max() - temp[temp["type"] == "consensus"]["mz"].min())
                    / temp[temp["type"] == "consensus"]["mz"].mean()
                    * 1e6,
                    (self.features[featureInd][1] - refMZ) / refMZ * 1e6,
                )
            )
            + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
            + p9.theme(legend_position="bottom")
        )
        print(p)

    #####################################################################################################
    # RSD overview
    #

    def plot_RSDs_per_group(self, uGroups=None, include=None, plotType="points", scales="free_y"):
        """
        Plots an rsd distribution per group

        Parameters
        ----------
        uGroups : list of str, optional
            the groups to be included in the overview. Defaults to None.
        include : list of str, optional
            missing values replacement strategies to be used. Defaults to None.
        plotType : str, optional
            type of plot (either points or histogram). Defaults to "points".
        scales : str, optional
            parameter for plotnine and the y-scale of facetted plots. Defaults to "free_y".

        Raises
        ======
        ValueError
            raised if an unknown plottype is provided
        """
        if uGroups is None:
            uGroups = set(self.groups)

        if include is None:
            include = ["wo.nan", "nan=0"]
        temp = {"rsd": [], "mean": [], "sd": [], "featurei": [], "group": [], "type": []}

        for grp in uGroups:
            if type(grp) == list and len(grp) == 2 and type(grp[0]) == type(grp[1]) == str and grp[1] == "Batch":
                for batch in list(set(self.batches)):
                    groupInd = list(
                        set([i for i, group in enumerate(self.groups) if group == grp[0]]).intersection(
                            set([i for i, b in enumerate(self.batches) if b == batch])
                        )
                    )

                    if len(groupInd) > 0:
                        for featurei in range(self.dat.shape[1]):
                            vals = self.dat[groupInd, featurei]

                            if np.all(np.isnan(vals)):
                                next

                            if "wo.nan" in include:
                                vals_ = np.copy(vals)
                                vals_ = vals_[~np.isnan(vals_)]
                                if vals_.shape[0] > 1:
                                    temp["rsd"].append(_relative_standard_deviation(vals_))
                                    temp["mean"].append(np.log2(np.mean(vals_)))
                                    temp["sd"].append(np.log2(np.std(vals_)))
                                    temp["featurei"].append(featurei)
                                    temp["group"].append(grp[0] + "_B" + str(batch))
                                    temp["type"].append("wo.nan")

                            if "nan=0" in include:
                                vals_ = np.copy(vals)
                                vals_[np.isnan(vals_)] = 0
                                if np.sum(vals_ > 0) > 1:
                                    temp["rsd"].append(_relative_standard_deviation(vals_))
                                    temp["mean"].append(np.log2(np.mean(vals_)))
                                    temp["sd"].append(np.log2(np.std(vals_)))
                                    temp["featurei"].append(featurei)
                                    temp["group"].append(grp[0] + "_B" + str(batch))
                                    temp["type"].append("nan=0")

                grp = grp[0]

            if type(grp) == str:
                groupInd = [i for i, group in enumerate(self.groups) if group == grp]

                for featurei in range(self.dat.shape[1]):
                    vals = self.dat[groupInd, featurei]

                    if np.all(np.isnan(vals)):
                        next

                    if "wo.nan" in include:
                        vals_ = np.copy(vals)
                        vals_ = vals_[~np.isnan(vals_)]
                        if vals_.shape[0] > 1:
                            temp["rsd"].append(_relative_standard_deviation(vals_))
                            temp["mean"].append(np.log2(np.mean(vals_)))
                            temp["sd"].append(np.log2(np.std(vals_)))
                            temp["featurei"].append(featurei)
                            temp["group"].append(grp)
                            temp["type"].append("wo.nan")

                    if "nan=0" in include:
                        vals_ = np.copy(vals)
                        vals_[np.isnan(vals_)] = 0
                        if np.sum(vals_ > 0) > 1:
                            temp["rsd"].append(_relative_standard_deviation(vals_))
                            temp["mean"].append(np.log2(np.mean(vals_)))
                            temp["sd"].append(np.log2(np.std(vals_)))
                            temp["featurei"].append(featurei)
                            temp["group"].append(grp)
                            temp["type"].append("nan=0")

        temp = pd.DataFrame(temp)
        p = None
        if plotType == "histogram":
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="rsd", fill="group"))
                + p9.geom_histogram()
                + p9.facet_grid("type~group")
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.15, 'left':0.15})
                + p9.ggtitle("RSD plots")
            )

        elif plotType == "points":
            p = (
                p9.ggplot(data=temp, mapping=p9.aes(x="mean", y="rsd", colour="group"))
                + p9.geom_point(alpha=0.15)
                # + p9.geom_abline(slope = 0.15, intercept = 0, colour = "slategrey")
                # + p9.geom_abline(slope = 0.5, intercept = 0, colour = "black")
                # + p9.geom_abline(slope = 1, intercept = 0, colour = "firebrick")
                + p9.facet_wrap("type + group", scales=scales)
                + p9.theme_minimal()
                + p9.theme(legend_position="bottom")
                # + p9.theme(subplots_adjust={'wspace':0.15, 'hspace':0.25, 'top':0.93, 'right':0.99, 'bottom':0.15, 'left':0.15})
                + p9.ggtitle("RSD plots")
            )

        else:
            raise ValueError("Unknown plot type. Must be 'histogram' or 'points'")

        return p, temp

    #####################################################################################################
    # Convenience functions for a quick, statistical overview
    #

    def calc_volcano_plots(
        self,
        comparisons,
        alpha_critical=0.05,
        minimum_fold_change=2,
        keep_features=None,
        remove_features=None,
        highlight_features=None,
        min_ratio_samples_for_found=0.75,
        min_ratio_samples_for_not_found=0.25,
        sig_color="firebrick",
        not_different_color="cadetblue",
    ):
        """
        generate a volcano plot from the results

        Parameters
        ----------
        comparisons : list of tuple of : str, str)
            the names of the two groups to be compared
        alpha_critical : float, optional
            the critical alpha value for a significant difference. Defaults to 0.05.
        minimum_fold_change : int, optional
            the mimimum required fold-change for a significant difference. Defaults to 2.
        keep_features : _type_, optional
            a list of indices to be used for the uni-variate comparison. Defaults to None.
        remove_features : _type_, optional
            a list of features to not be used for the uni-variate comparison. Defaults to None.
        highlight_features : list of featre inds, optional
            features to highlight, indices according to self.dat matrix and self.features must be provided.
        sig_color : str, optional
            the name of the color used for plotting significantly different features. Defaults to "firebrick".
        not_different_color : str, optional
            the name of the color used for plotting not significantly different features. Defaults to "cadetblue".

        Returns
        -------
        Pandard dataframe
            the data matrix for the volcano plot
        """
        # get data
        dat, features, featureAnnotations, samples, groups, batches = self.get_data_matrix_and_meta(
            keep_features=keep_features, remove_features=remove_features, copy=True
        )

        if highlight_features is None:
            highlight_features = []

        temp = {}

        for grp1, grp2 in comparisons:
            testName = "'%s' vs. '%s'" % (grp1, grp2)

            notTested = 0

            grp1Inds = [i for i, group in enumerate(groups) if group == grp1]
            grp2Inds = [i for i, group in enumerate(groups) if group == grp2]

            for featurei in tqdm.tqdm(range(dat.shape[1])):
                valsGrp1 = dat[grp1Inds, featurei]
                valsGrp2 = dat[grp2Inds, featurei]

                valsGrp1[np.isnan(valsGrp1)] = 0
                valsGrp2[np.isnan(valsGrp2)] = 0

                if (
                    (
                        sum(valsGrp1 != 0) / valsGrp1.shape[0] >= min_ratio_samples_for_found
                        and sum(valsGrp2 != 0) / valsGrp2.shape[0] >= min_ratio_samples_for_found
                    )
                    or (
                        sum(valsGrp1 != 0) / valsGrp1.shape[0] >= min_ratio_samples_for_found
                        and sum(valsGrp2 != 0) / valsGrp2.shape[0] <= min_ratio_samples_for_not_found
                    )
                    or (
                        sum(valsGrp1 != 0) / valsGrp1.shape[0] <= min_ratio_samples_for_not_found
                        and sum(valsGrp2 != 0) / valsGrp2.shape[0] >= min_ratio_samples_for_found
                    )
                ):
                    if np.all(valsGrp1 == 0) and not np.all(valsGrp2 == 0):
                        fold = 0
                    elif not np.all(valsGrp1 == 0) and np.all(valsGrp2 == 0):
                        fold = np.inf
                    else:
                        fold = np.mean(valsGrp1) / np.mean(valsGrp2)

                    pval = scipy.stats.ttest_ind(valsGrp1, valsGrp2, equal_var=False, alternative="two-sided", trim=0)[1]
                    pvalWRST = scipy.stats.ranksums(
                        valsGrp1,
                        valsGrp2,
                        alternative="two-sided",
                    ).pvalue
                    cohensD = cohen_d(valsGrp1, valsGrp2)
                    meanAbundance = np.mean(np.concatenate((valsGrp1, valsGrp2), axis=0))
                    sigInd = (
                        "sig. diff." if pval <= alpha_critical and (fold >= minimum_fold_change or fold <= 1.0 / minimum_fold_change) else "not diff."
                    )

                    temp = _add_row_to_pandas_creation_dictionary(
                        temp,
                        pvalues=pval,
                        pvalues_WilcoxonRankTest=pvalWRST,
                        folds=fold,
                        trans_pvalues=-np.log10(pval),
                        trans_folds=np.log2(fold) if fold > 0 else -np.inf,
                        effectSizes=cohensD,
                        detectionsGrp1=np.sum(valsGrp1 > 0),
                        detectionsGrp2=np.sum(valsGrp2 > 0),
                        zerosGrp1=np.sum(valsGrp1 == 0),
                        zerosGrp2=np.sum(valsGrp2 == 0),
                        meanAbundanceGrp1=np.mean(valsGrp1),
                        meanAbundanceGrp2=np.mean(valsGrp2),
                        stdAbundanceGrp1=np.std(valsGrp1),
                        stdAbundanceGrp2=np.std(valsGrp2),
                        sigIndicators=sigInd,
                        tests=testName,
                        feature="%d mz %8.4f" % (featurei, self.features[featurei][1]),
                        featuremz=self.features[featurei][1],
                        featurei=featurei,
                        highlightFeature=featurei in highlight_features,
                        meanFeatureAbundance=meanAbundance,
                    )

                else:
                    notTested += 1

        temp = pd.DataFrame(temp)
        p = (
            p9.ggplot(
                data=temp,
                mapping=p9.aes(
                    x="trans_folds", y="trans_pvalues", fill="sigIndicators", colour="sigIndicators", size="np.log2(meanFeatureAbundance)"
                ),
            )
            + p9.geom_hline(yintercept=-np.log10(alpha_critical), alpha=0.3, colour="darkgrey")
            + p9.geom_vline(xintercept=[np.log2(minimum_fold_change), np.log2(1 / minimum_fold_change)], alpha=0.3, colour="darkgrey")
            + p9.geom_point(alpha=0.15, colour="ghostwhite")
            + p9.geom_point(data=temp[temp["highlightFeature"]], colour="slategrey")
            + p9.facet_wrap("~tests")
            + p9.scale_fill_manual(values={"sig. diff.": sig_color, "not diff.": not_different_color})
            + p9.ggtitle("Volcano plots (%d comparisons, %d not tested)" % (len(temp.index), notTested))
        )

        return p, temp

    def generate_feature_abundance_plot(
        self, feature_index, keep_samples=None, remove_samples=None, keep_groups=None, remove_groups=None, keep_batches=None, remove_batches=None
    ):
        """
        Shows a single feature

        Parameters
        ----------
        feature_index : index
            the index of the feature to be plotted

        Returns
        -------
        Pandas DataFrame
            the data matrix for the plot
        """
        temp = pd.DataFrame({"abundance": self.dat[:, feature_index], "sample": self.samples, "group": self.groups, "batch": self.batches})
        if keep_samples:
            temp = temp[temp["sample"].isin(keep_samples)]
        if remove_samples:
            temp = temp[~temp["sample"].isin(remove_samples)]
        if keep_groups:
            temp = temp[temp["group"].isin(keep_groups)]
        if remove_groups:
            temp = temp[~temp["group"].isin(remove_groups)]
        if keep_batches:
            temp = temp[temp["batch"].isin(keep_batches)]
        if remove_batches:
            temp = temp[~temp["batch"].isin(remove_batches)]

        p = (
            p9.ggplot(data=temp, mapping=p9.aes(x="group", y="abundance", colour="group"))
            + p9.geom_boxplot()
            + p9.geom_jitter()
            + p9.ggtitle("Feature %d (meanmz: %.5f)" % (feature_index, self.features[feature_index][1]))
        )

        return p, temp

    def generate_feature_raw_plot(
        self,
        refMZ,
        reverseCorrectMZ=True,
        ppmDev=20,
        keep_samples=None,
        remove_samples=None,
        keep_groups=None,
        remove_groups=None,
        keep_batches=None,
        remove_batches=None,
    ):
        """
        generates a raw data plot for a detected feature

        Parameters
        ----------
        refMZ : float
            the mz value of the feature to plot after mz correction (reverseCorrectMZ = True) or in the raw data (reverseCorrectMZ = False)
        reverseCorrectMZ : boolean, optional
            indicates if the refMZ value is after (True) or before (False) mz correction
        ppmDev : float, optional
            the allowed mz deviation for the feature

        Raises
        ======
        ValueError
            raised if parameters on and aggregation_fun have invalid values
        """
        sampleNames = self.get_sample_names()
        temp = {}

        for samplei, sample in enumerate(sampleNames):
            sgroup = self.get_metaData_for_sample(sample, "group")
            sbatch = self.get_metaData_for_sample(sample, "batch")
            if (
                (keep_samples is None or sample in keep_samples)
                and (remove_samples is None or sample not in remove_samples)
                and (keep_groups is None or sgroup in keep_groups)
                and (remove_groups is None or sgroup not in remove_groups)
                and (keep_batches is None or sbatch in keep_batches)
                and (remove_batches is None or sbatch not in remove_batches)
            ):
                msDataObj = self.get_msDataObj_for_sample(sample)
                for oSpectrumi, oSpectrum in msDataObj.original_MSData_object.get_spectra_iterator():
                    _refMZ = oSpectrum.reverseMZ(refMZ) if reverseCorrectMZ else refMZ
                    _mzmin, _mzmax = _refMZ * (1.0 - ppmDev / 1e6), _refMZ * (1.0 + ppmDev / 1e6)

                    use = np.argwhere(np.logical_and(oSpectrum.original_mz >= _mzmin, oSpectrum.original_mz <= _mzmax))[:, 0]
                    if len(use) > 0:
                        for mzi in use:
                            temp = _add_row_to_pandas_creation_dictionary(
                                temp,
                                sample=sample,
                                group=self.get_metaData_for_sample(sample, "group"),
                                batch=self.get_metaData_for_sample(sample, "batch"),
                                spectrum=oSpectrumi,
                                chronogramTime=oSpectrum.time,
                                mz=oSpectrum.original_mz[mzi],
                                intensity=oSpectrum.spint[mzi],
                            )

        temp = pd.DataFrame(temp)

        if len(temp.index) > 0:
            p1 = (
                p9.ggplot(
                    data=temp,
                    mapping=p9.aes(x="chronogramTime", y="mz", colour="group", alpha="intensity"),
                )
                + p9.geom_hline(yintercept=refMZ, alpha=0.3, colour="slategrey")
                + p9.geom_point()
                + p9.ggtitle("raw data plot of %.4f (+- %f ppm)" % (refMZ, ppmDev))
            )
            p2 = (
                p9.ggplot(
                    data=temp,
                    mapping=p9.aes(x="chronogramTime", y="intensity", colour="group"),
                )
                + p9.geom_point()
                + p9.ggtitle("raw data plot of %.4f (+- %f ppm)" % (refMZ, ppmDev))
            )
            temph = temp.groupby(["sample"]).agg({"intensity": "sum", "chronogramTime": "mean", "group": "first", "batch": "first"})
            temph["type"] = "sum of intensity"
            tempj = temp.groupby(["sample"]).agg({"intensity": "mean", "chronogramTime": "mean", "group": "first", "batch": "first"})
            tempj["type"] = "average intensity"
            temph = pd.concat([temph, tempj])
            p3 = (
                p9.ggplot(
                    data=temph,
                    mapping=p9.aes(x="chronogramTime", y="intensity", colour="group"),
                )
                + p9.geom_point()
                + p9.facet_wrap("~ type", scales="free_y")
                + p9.ggtitle("raw data plot of %.4f (+- %f ppm), all signals have been aggregated (sum and average)" % (refMZ, ppmDev))
            )
            pa = (
                p9.ggplot(
                    data=temph,
                    mapping=p9.aes(x="group", y="intensity", colour="group"),
                )
                + p9.geom_boxplot()
                + p9.geom_jitter(height=0, width=0.5)
                + p9.facet_wrap("~ type", scales="free_y")
                + p9.ggtitle("raw data plot of %.4f (+- %f ppm), all signals have been aggregated (sum and average)" % (refMZ, ppmDev))
            )

            return p1, p2, p3, pa, temp

        return None, None, None, None, None

    def calc_2D_Embedding(
        self,
        keep_features=None,
        remove_features=None,
        keep_samples=None,
        remove_samples=None,
        keep_groups=None,
        remove_groups=None,
        keep_batches=None,
        remove_batches=None,
        imputation="zero",
        scaling="standard",
        embedding="pca",
    ):
        """
        Calculates a two-dimensional embedding of the data matrix (or a subset) and illustrates it as a scores/component plot

        Parameters
        ----------
        keep_features : list of indices, optional
            features to be included for the embedding. Defaults to None.
        remove_features : list of indices, optional
            features to not be included for the embedding. Defaults to None.
        keep_samples : list of str, optional
            samples to be included for the embedding. Defaults to None.
        remove_samples : list of str, optional
            samples to not be included for the embedding. Defaults to None.
        keep_groups : list of str, optional
            groups to be included for the embedding. Defaults to None.
        remove_groups : list of str, optional
            groups to not be included for the embedding. Defaults to None.
        keep_batches : list of int, optional
            batches to be included for the embedding. Defaults to None.
        remove_batches : list of int, optional
            batches to not be included for the embedding. Defaults to None.
        imputation : str, optional
            imputation method for features with missing values (i.e., no signals detected for them). Defaults to "zero", allowed are "zero" and "omitNA".
        scaling : str, optional
            the scaling to be applied before the embedding calculation. Defaults to "standard", allowed are None, "", "standard".
        embedding : str, optional
            the embedding type (). Defaults to "pca", allowed are "pca", "lda", "umap".

        Raises
        ======
        ValueError
            raised if invalid parameters are provided

        Returns
        -------
        plotnine
            the plot
        """
        # get data
        dat, features, featureAnnotations, samples, groups, batches = self.get_data_matrix_and_meta(
            keep_features=keep_features,
            remove_features=remove_features,
            keep_samples=keep_samples,
            remove_samples=remove_samples,
            keep_groups=keep_groups,
            remove_groups=remove_groups,
            keep_batches=keep_batches,
            remove_batches=remove_batches,
            copy=True,
        )

        # missing values imputation
        if imputation.lower() == "zero".lower():
            datImp = np.nan_to_num(dat, nan=0, copy=True)

        elif imputation.lower() == "omitNA".lower():
            keep = ~np.isnan(dat).any(axis=0)
            datImp = dat[:, keep]

        else:
            raise ValueError("Unknown imputation method specified, must be either of ['zero', 'omitNA']")

        # scaling
        if scaling is None or scaling == "" or scaling.lower() == "None".lower():
            datImpSca = datImp

        elif scaling.lower() == "standard".lower():
            scaler = StandardScaler()
            datImpSca = scaler.fit_transform(datImp)

        else:
            raise ValueError("Unknown scaling method specified, must be either of [None, 'standard']")

        # embedding
        if embedding.lower() == "pca".lower():
            pca = PCA(n_components=2)
            pca = pca.fit(datImpSca)
            scores = pca.transform(datImpSca)
            comp1 = scores[:, 0]
            comp2 = scores[:, 1]
            comp1_title = "PC1 (%.1f %% covered variance)" % (pca.explained_variance_ratio_[0] * 100.0)
            comp2_title = "PC2 (%.1f %% covered variance)" % (pca.explained_variance_ratio_[1] * 100.0)

        elif embedding.lower() == "lda".lower():
            lda = LinearDiscriminantAnalysis(n_components=2)
            scores = lda.fit(datImp, groups).transform(datImp)
            comp1 = scores[:, 0]
            comp2 = scores[:, 1]
            comp1_title = "Component 1"
            comp2_title = "Component 2"

        elif embedding.lower() == "umap".lower():
            reducer = umap.UMAP()
            scores = reducer.fit_transform(datImp)
            comp1 = scores[:, 0]
            comp2 = scores[:, 1]
            comp1_title = "Component 1"
            comp2_title = "Component 2"

        else:
            raise ValueError("Unknown embedding method specified, must be either of ['pca', 'lda', 'umap']")

        # plot
        temp = {"pc1": comp1, "pc2": comp2, "file": samples, "group": groups, "batch": batches}
        temp = pd.DataFrame(temp)
        temp["batch"] = temp["batch"].astype(object)
        p = (
            p9.ggplot(data=temp, mapping=p9.aes(x="pc1", y="pc2", colour="group", group="group", label="file"))
            + p9.stat_ellipse(data=temp, alpha=0.5, level=0.95)
            + p9.geom_point(alpha=0.8)
            + p9.xlab(comp2_title)
            + p9.ylab(comp2_title)
            + p9.ggtitle("%s with %s imputation and %s scaling" % (embedding, imputation, scaling))
        )
        return p, temp

    def plot_heatmap(
        self,
        keep_features=None,
        remove_features=None,
        keep_samples=None,
        remove_samples=None,
        keep_groups=None,
        remove_groups=None,
        keep_batches=None,
        remove_batches=None,
        linkage_method="ward",
        distance_metric="euclidean",
    ):
        """
        Calculates and plots a heatmap.

        Parameters
        ----------
        keep_features : list of indices, optional
            features to be included for the embedding. Defaults to None.
        remove_features : list of indices, optional
            features to not be included for the embedding. Defaults to None.
        keep_samples : list of str, optional
            samples to be included for the embedding. Defaults to None.
        remove_samples : list of str, optional
            samples to not be included for the embedding. Defaults to None.
        keep_groups : list of str, optional
            groups to be included for the embedding. Defaults to None.
        remove_groups : list of str, optional
            groups to not be included for the embedding. Defaults to None.
        keep_batches : list of int, optional
            batches to be included for the embedding. Defaults to None.
        remove_batches : list of int, optional
            batches to not be included for the embedding. Defaults to None.
        linkage_method : str, optional
            linkage method to be used for generating the feature clustering. Defaults to 'ward', options are from scipy.linkage
        distance_metric : str, optional
            distnace method to be used for generating the feature clustering. Defaults to 'euclidean', options are from scipy.linkage

        Raises
        ======
        ValueError
            raised if invalid parameters are provided

        Returns
        -------
        plot
            the heatmap
        """
        dat, features, featureAnnotations, samples, groups, batches = self.get_data_matrix_and_meta(
            keep_features=keep_features,
            remove_features=remove_features,
            keep_samples=keep_samples,
            remove_samples=remove_samples,
            keep_groups=keep_groups,
            remove_groups=remove_groups,
            keep_batches=keep_batches,
            remove_batches=remove_batches,
            copy=True,
        )

        datImp = np.nan_to_num(dat, nan=0, copy=True)

        scaler = StandardScaler()
        datImpSca = scaler.fit_transform(datImp)

        temp = datImpSca
        temp = np.transpose(temp)
        temp = scaler.fit_transform(temp)
        linkage_data = dendrogram(linkage(temp, method=linkage_method, metric=distance_metric), no_plot=True)

        temp = {}
        for samplei in range(datImpSca.shape[0]):
            sample, group, batch = samples[samplei], groups[samplei], batches[samplei]
            for leavei, featurei in enumerate(linkage_data["leaves"]):
                featurei = int(featurei)
                val = datImpSca[samplei, featurei]

                temp = _add_row_to_pandas_creation_dictionary(
                    temp, sample=sample, group=group, batch=batch, feature=featurei, leave=leavei, value=val
                )

        temp = pd.DataFrame(temp)

        p = (
            p9.ggplot(temp, p9.aes(x="leave", y="sample", fill="value"))
            + p9.theme_minimal()
            + p9.geom_tile(color="white", size=0.1)
            + p9.theme(legend_position="bottom", plot_title=p9.element_text(size=14), axis_text_y=p9.element_text(size=6))
        )
        print(p)

    #####################################################################################################
    # Convenience function for semi-automated parameter optimization
    #

    def get_summary_of_results(self, reference_features=None, reference_features_allowed_deviationPPM=20.0):
        """Show a summary of the results.

        Parameters
        ----------
        reference_features : list of float, optional
            reference mz values to be used. Defaults to None.
        reference_features_allowed_deviationPPM : float, optional
            allowed mz deviation. Defaults to 20.0.

        Returns
        -------
        Pandas DataFrame
            Summary table
        """
        if reference_features is None:
            reference_features = []

        results = {}

        ## total number of detected features
        results["n(feats)"] = self.dat.shape[1]

        ## ppm deviation of features
        mzDevs = np.array([(self.features[i][2] - self.features[i][0]) / self.features[i][1] * 1e6 for i in range(len(self.features))])
        avg, std = _average_and_std(mzDevs)
        results["avg(feats.MZDevPPM)"] = avg
        results["std(feats.MZDevPPM)"] = std

        ## number of missing values in data matrix
        results["n(nan)"] = np.sum(np.isnan(self.dat))
        results["r(nan)"] = np.sum(np.isnan(self.dat) / (self.dat.shape[0] * self.dat.shape[1]))

        ## number of detected reference features
        found = 0
        ppmDeviations = []
        mzs_ = np.array([self.features[i][1] for i in range(len(self.features))])
        for refi in range(len(reference_features)):
            referenceMZmin, referenceMZ, referenceMZmax = (
                reference_features[refi] * (1.0 - reference_features_allowed_deviationPPM / 1.0e6),
                reference_features[refi],
                reference_features[refi] * (1.0 + reference_features_allowed_deviationPPM / 1.0e6),
            )
            if any((referenceMZmin <= self.features[i][1] <= referenceMZmax for i in range(len(self.features)))):
                ind = np.argmin(np.abs(mzs_ - referenceMZ))
                mzDevPPM = (mzs_[ind] - referenceMZ) / referenceMZ * 1e6
                ppmDeviations.append(mzDevPPM)
                found += 1
        results["n(ref)"] = found
        results["avg(ref.MZDevPPM)"] = np.average(np.array(ppmDeviations)) if found > 0 else -1
        results["std(ref.mzDevPPM)"] = np.std(np.array(ppmDeviations)) if found > 0 else -1

        return results

    #####################################################################################################
    # Database search
    #

    @staticmethod
    def generate_database_template(to_tsv_file):
        """Generates a template for the database search

        Parameters
        ----------
        to_tsv_file : str
            file to which the template shall be written
        """
        with open(to_tsv_file, "w") as fout:
            fout.write("\t".join(["_ID", "CompoundName", "InChi", "SMILES", "ChemicalFormula", "Adducts", "MZ", "IonMode"]))
            fout.write("\n")
            fout.write("# Lines starting with the hash (#) symbol are comments")
            fout.write("\n")
            fout.write("# Provide either ")
            fout.write("\n")
            fout.write("#  - a compound's sum formula and adducts to be searched for (no MZ or IonMode required)")
            fout.write("\n")
            fout.write("#  - a compound's MZ value and the IonMode (+ or -)")
            fout.write("\n")
            fout.write("#")
            fout.write("\n")
            fout.write(
                "\t".join(
                    [
                        "1",
                        "L-Phenylalanine",
                        "InChI=1S/C9H11NO2/c10-8(9(11)12)6-7-4-2-1-3-5-7/h1-5,8H,6,10H2,(H,11,12)/t8-/m0/s1/i6D2",
                        "[2H]C([2H])(c1ccccc1)[C@@H](C(=O)O)N",
                        "C9H11NO2",
                        "[M+H]+, [M-H]-",
                        "",
                        "",
                    ]
                )
            )
            fout.write("\n")
            fout.write("\t".join(["2", "Unknown 2", "", "", "", "", "557.4069", "-"]))
            fout.write("\n")

    def annotate_with_compounds(self, tsv_file, max_ppm_dev=15.0, adducts=None, delimiter="\t", quote_character="", comment_character="#"):
        """Annotation of detected features with compounds from a database.

        Parameters
        ----------
        tsv_file : str
            Path to a tab-separated file containing the database.
        max_ppm_dev : float, optional
            Maximum allowed mz difference in ppm relative to the theoretical value. Defaults to 15.0.
        adducts : dict, optional
            key: stri, value: tuple of (charge number, mz increment). Defaults to None.
        delimiter : str, optional
            delimter character of the database. Defaults to "\t".
        quote_character : str, optional
            quote character of the database. Defaults to "".
        comment_character : str, optional
            comment character of the database (not allowed in first row/header). Defaults to "#".
        """
        tempAdducts = {
            "[M+H]+": (1, 1.007276),
            "[M+Na]+": (1, 22.989218),
            "[M+NH4]+": (1, 18.033823),
            "[M]+": (1, -0.00054857990924),
            "[M-H]-": (1, -1.007276),
            "[M+Cl]-": (1, 34.969402),
            "[M]-": (1, 0.00054857990924),
        }

        if adducts is not None:
            adducts = tempAdducts | adducts
        else:
            adducts = tempAdducts

        if self.featureAnnotations is None:
            self.featureAnnotations = [[] for _ in range(len(self.features))]

        with open(tsv_file, "r") as fin:
            tsvReader = None
            if quote_character != "":
                tsvReader = csv.reader(fin, delimiter=delimiter, quotechar=quote_character)
            else:
                tsvReader = csv.reader(fin, delimiter=delimiter)
            headers = None
            headersDict = {}

            for rowi, row in enumerate(tsvReader):
                if rowi == 0:
                    headers = row
                    headersDict = dict(((header, i) for i, header in enumerate(row)))

                elif row[0].startswith(comment_character):
                    pass

                else:
                    types = []
                    mzs = []
                    mz = row[headersDict["MZ"]]
                    if mz is None or mz == "":
                        m = Formula(row[headersDict["ChemicalFormula"]]).get_exact_mass()
                        for add in row[headersDict["Adducts"]].replace(" ", "").split(","):
                            addInfo = adducts[add]
                            mz = m / addInfo[0] + addInfo[1]
                            mzs.append(mz)
                            types.append("%s as %s" % (row[headersDict["MZ"]], add))

                    else:
                        mzs.append(float(mz))
                        types.append("%s as direct mz match" % (row[headersDict["CompoundName"]]))

                    for typei in range(len(types)):
                        refMZ = mzs[typei]
                        typ = types[typei]

                        for featurei in range(len(self.features)):
                            meanMZ = self.features[featurei][1]

                            if abs(meanMZ - refMZ) / refMZ * 1e6 <= max_ppm_dev:
                                t = []
                                if "Database" in self.featureAnnotations[featurei]:
                                    t = self.featureAnnotations[featurei]["Database"]
                                else:
                                    self.featureAnnotations[featurei]["Database"] = t

                                t.append(
                                    {
                                        "DatabaseFile": tsv_file,
                                        "Compound": row[headersDict["CompoundName"]],
                                        "InChi": row[headersDict["InChi"]],
                                        "SMILES": row[headersDict["SMILES"]],
                                        "ChemicalFormula": row[headersDict["ChemicalFormula"]],
                                        "Adduct": typ,
                                        "MZDev_ppm": (meanMZ - refMZ) / refMZ * 1e6,
                                    }
                                )
