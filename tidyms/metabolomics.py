"""
Common routines to work with raw MS data from metabolomics experiments.

Functions
---------
detect_features(path_list) : Perform feature detection on several samples.
feature_correspondence(feature_data) : Match features across different samples
using a combination of clustering algorithms.

"""

import pandas as pd
import numpy as np
from .fileio import MSData
from .container import DataContainer
from .lcms import Roi
from . import validation
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn import mixture
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple, List, Dict, Union
from IPython.display import clear_output


__all__ = ["detect_features", "feature_correspondence", "make_data_container"]


def detect_features(path: Union[Path, List[str]], separation: str = "uplc",
                    instrument: str = "qtof", roi_params: Optional[dict] = None,
                    smoothing_strength: Optional[float] = 1.0,
                    find_peaks_params: Optional[dict] = None,
                    descriptors: Optional[dict] = None,
                    filters: Optional[dict] = None,
                    verbose: bool = True
                    ) -> Tuple[Dict[str, List[Roi]], pd.DataFrame]:
    """
    Perform feature detection on LC-MS centroid samples.

    Parameters
    ----------
    path: Path or List[str]
        Path can be a list of strings of absolute path representations to mzML
        files in centroid mode or a Path object. Path objects can be used in
        two ways: It can point to a mzML file or to a directory. in the second
        case all mzML files inside the directory will be analyzed.
    separation: {"uplc", "hplc"}
        Analytical platform used for separation. Used to set default the values
        of `detect_peak_params`, `roi_params` and `filter_params`.
    instrument: {"qtof". "orbitrap"}
        MS instrument used for data acquisition. Used to set default value
        of `roi_params`.
    roi_params: dict, optional
        parameters to pass to :py:meth:`tidyms.MSData.make_roi`
    smoothing_strength: positive number, optional
        Width of a gaussian window used to smooth the ROI. If None, no
        smoothing is applied.
    find_peaks_params : dict, optional
        parameters to pass to :py:func:`tidyms.peaks.detect_peaks`
    descriptors : dict, optional
        descriptors to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
    filters : dict, optional
        filters to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
    descriptors : dict, optional
        pass custom descriptors to :py:func:`tidyms.peaks.get_peak_descriptors`
    filters : dict, optional
        pass custom filters to :py:func:`tidyms.peaks.get_peak_descriptors`
    verbose: bool

    Returns
    -------
    roi_dict: dict
        dictionary of sample names to a list of ROI.
    feature_table: DataFrame
        A Pandas DataFrame where each row is a feature detected in a sample and
        each column is a feature descriptor. By default the following
        descriptors are computed:

        mz
            weighted average of the m/z in the peak region.
        mz std
            standard deviation of the m/z in the peak region.
        rt
            weighted average of the retention time in the peak region.
        width
            Chromatographic peak width.
        height
            Height of the chromatographic peak minus the baseline.
        area
            Area of the chromatographic peak. minus the baseline area.
        sample
            The sample name where the feature was detected.

        Also, two additional columns have information to search each feature
        in its correspondent Roi:

        roi_index :
            index in the list of ROI where the feature was detected.
        peak_index :
            index of the peaks attribute of each ROI associated to the feature.

    Notes
    -----
    Features are detected as follows:

    1.  Default parameters are set based on the values of the parameters
        `instrument` and `separation`.
    2.  Regions of interest (ROI) are detected in each sample. See the
        documentation of :py:meth:`tidyms.fileio.MSData.make_roi` for a detailed
        description of how ROI are created from raw data.
    3.  Features (chromatographic peaks) are detected on each ROI. See
        :py:meth:`tidyms.lcms.Chromatogram.find_peaks` for a detailed
        description of how peaks are detected and how descriptors are computed.

    See Also
    --------
    fileio.MSData.make_roi : Finds ROIs in a mzML sample.
    lcms.ROI.find_peaks : Detect peaks and compute peak estimators for a ROI.

    """
    # parameter validation
    # validation.validate_detect_peaks_params(detect_peak_params)
    validation.validate_descriptors(descriptors)
    validation.validate_filters(filters)
    if roi_params is None:
        roi_params = dict()

    path_list = _get_path_list(path)
    roi_dict = dict()
    ft_table_list = list()
    n_samples = len(path_list)
    for k, sample_path in enumerate(path_list):
        sample_name = sample_path.stem
        sample_path_str = str(sample_path)
        ms_data = MSData(sample_path_str, ms_mode="centroid",
                         instrument=instrument, separation=separation)
        k_roi = ms_data.make_roi(**roi_params)

        if verbose:
            clear_output(wait=True)
            msg = "Processing sample {} ({}/{})."
            msg = msg.format(sample_name, k + 1, n_samples)
            print(msg)
            print("Searching features in {} ROI...".format(len(k_roi)), end=" ")

        k_table = _build_feature_table(k_roi,
                                       smoothing_strength=smoothing_strength,
                                       descriptors=descriptors,
                                       filters=filters,
                                       find_peaks_params=find_peaks_params)

        if verbose:
            msg = "Found {} features".format(k_table.shape[0])
            print(msg)
        k_table["sample"] = sample_name
        roi_dict[sample_name] = k_roi
        ft_table_list.append(k_table)
    feature_table = pd.concat(ft_table_list).reset_index(drop=True)
    feature_table["roi index"] = feature_table["roi index"].astype(int)
    feature_table["peak index"] = feature_table["peak index"].astype(int)
    return roi_dict, feature_table


def _get_path_list(path: Union[str, List[str], Path]) -> List[Path]:
    if isinstance(path, str):
        path = Path(path)

    if isinstance(path, list):
        path_list = [Path(x) for x in path]
        for p in path_list:
            # check if all files in the list exists
            if not p.is_file():
                msg = "{} doesn't exist".format(p)
                raise ValueError(msg)
    else:
        if path.is_dir():
            path_list = list(path.glob("*.mzML"))
        elif path.is_file():
            path_list = [path]
        else:
            msg = ("Path must be a string or Path object pointing to a "
                   "directory with mzML files or a list strings with the "
                   "absolute path to mzML")
            raise ValueError(msg)
    return path_list


def _build_feature_table(roi: List[Roi],
                         smoothing_strength: Optional[float] = 1.0,
                         descriptors: Optional[dict] = None,
                         filters: Optional[dict] = None,
                         find_peaks_params: Optional[dict] = None
                         ) -> pd.DataFrame:
    """
    Builds a DataFrame with feature descriptors.

    Parameters
    ----------
    roi : List[Roi]
    find_peaks_params : dict, optional
            parameters to pass to :py:func:`tidyms.peaks.detect_peaks`
    smoothing_strength: positive number, optional
        Width of a gaussian window used to smooth the signal. If None, no
        smoothing is applied.
    descriptors : dict, optional
        descriptors to pass to :py:func:`tidyms.peaks.get_peak_descriptors`
    filters : dict, optional
        filters to pass to :py:func:`tidyms.peaks.get_peak_descriptors`

    Returns
    -------
    DataFrame

    """
    roi_index_list = list()
    peak_index_list = list()
    mz_mean_list = list()
    mz_std_list = list()
    descriptors_list = list()

    for roi_index, k_roi in enumerate(roi):
        k_roi.fill_nan()
        k_params = k_roi.find_peaks(smoothing_strength=smoothing_strength,
                                    descriptors=descriptors,
                                    filters=filters,
                                    find_peaks_params=find_peaks_params)
        n_features = len(k_params)
        descriptors_list.extend(k_params)
        k_mz_mean, k_mz_std = k_roi.get_peaks_mz()
        roi_index_list.append([roi_index] * n_features)
        peak_index_list.append(range(n_features))
        mz_mean_list.append(k_mz_mean)
        mz_std_list.append(k_mz_std)

    roi_index_list = np.hstack(roi_index_list)
    peak_index_list = np.hstack(peak_index_list)
    mz_mean_list = np.hstack(mz_mean_list)
    mz_std_list = np.hstack(mz_std_list)

    ft_table = pd.DataFrame(data=descriptors_list)
    ft_table = ft_table.rename(columns={"loc": "rt"})
    ft_table["mz"] = mz_mean_list
    ft_table["mz std"] = mz_std_list
    ft_table["roi index"] = roi_index_list
    ft_table["peak index"] = peak_index_list
    ft_table = ft_table.dropna(axis=0)
    ft_table["roi index"] = ft_table["roi index"].astype(int)
    ft_table["peak index"] = ft_table["peak index"].astype(int)
    return ft_table


def feature_correspondence(feature_data: pd.DataFrame, mz_tolerance: float,
                           rt_tolerance: float, min_fraction: float = 0.2,
                           min_likelihood: float = 0.0):
    r"""
    Match features across different samples.

    Feature matching is done using the DBSCAN algorithm and Gaussian mixture
    models. After performing feature correspondence, features that come from the
    same species are clustered together.

    Parameters
    ----------
    feature_data: DataFrame
        Feature descriptors obtained from detect_features function
    mz_tolerance: float
        Maximum distance in m/z between two features in a cluster.
    rt_tolerance: float
        Maximum distance in rt between two features in a cluster.
    min_fraction: float
        Minimum fraction of samples forming a cluster.
    min_likelihood: float
        Minimum likelihood required to recover a missing value. Lower values
        will recover more features, but with lower confidence.

    Returns
    -------
    cluster: Series
        The cluster number for each feature.

    See Also
    --------
    detect_features
    make_data_container

    Notes
    -----
    The correspondence algorithm is as follows:

    1.  Features ares clustered using m/z and rt information with the DBSCAN
        algorithm. Because the dispersion in the m/z and r/t dimension is
        independent the Chebyshev distance is used to make the clusters.
        `rt_tolerance` and `mz_tolerance` are used to build the :math:`\epsilon`
        parameter of the model. rt is scaled using these two parameters to have
        the same tolerance in both dimensions in the following way:

        .. math::

            rt_{scaled} = rt * \frac{mz_{tolerance}}{rt_{tolerance}}

        The min_samples parameter is defined using from the minimum_dr
        (minimum detection rate) and the total number of samples in the
        data. This step gives us a matching of the features, but different
        species can be clustered together if they are close, or some features
        can be considered as noise and removed. These cases are analyzed in the
        following steps.
    2.  In this step the possibility that more than one species is present in a
        cluster is explored. The number of species is estimated computing the
        number of repeated features, :math:`n_{repeated}` in the cluster (that
        is, how many features come from only one sample, hoy many come from
        two, etc...). The fraction of repeated samples is computed using the
        total number of samples and then the number of species,
        :math:`n_{species}` is found as the maximum of repetitions with a
        fraction greater than `min_fraction`. Using :math:`n_{species}`, Each
        cluster is fit to a gaussian mixture model. Once again, because
        dispersion in rt and m/z is orthogonal, we used diagonal covariances
        matrices in the GMM. After this step, for each cluster, subclusters
        may be generated if :math:`n_{species}` is greater than one.
    3.  Each subcluster is analyzed then to remove repeated features. To
        remove repeated features the log-likelihood is evaluated for
        features coming from the same sample. The feature with the greatest
        likelihood is kept and the others are flagged as noise.
    4.  The final step is to search for missing features (a feature is
        missing if some of the samples are not present in a subcluster).
        This is done by searching for features that come from missing samples
        in the features flagged as noise by DBSCAN (or in the previous
        step). If a feature from a missing sample has a log-likelihood
        greater than `min_likelihood` then is added to the subcluster. If
        more than one feature is possible the one with the greatest
        likelihood is chosen.

    """
    # sample names are used to search for missing samples.
    sample_names = feature_data["sample"].unique()

    # DBSCAN clustering
    min_samples = int(sample_names.size * min_fraction + 1)
    cluster = _make_initial_cluster(feature_data, mz_tolerance, rt_tolerance,
                                    min_samples)

    # split feature data into clustered data and noise
    data = feature_data[cluster != -1]
    noise = feature_data[cluster == -1]

    # cluster number is converted to a string. This makes easier to assign
    # subclusters using the notation 0-0, 0-1, etc...
    # TODO : maybe there's a better solution to this and it's not necessary
    #   to convert values to str.
    cluster = cluster.astype(str)
    features_per_cluster = _estimate_n_species_per_cluster(data, cluster)

    for name, group in data.groupby(cluster):
        n_ft = features_per_cluster[name]
        # Here each cluster is split into subclusters, repeated values are
        # removed and missing values are searched in noise.
        # Each change is made changing the values in cluster.
        subcluster = _process_cluster(group, noise, cluster, sample_names, name,
                                      n_species=n_ft,
                                      min_likelihood=min_likelihood)
        cluster[subcluster.index] = subcluster

    # map cluster to numbers again
    cluster_value = np.sort(cluster.unique())
    n_cluster = cluster_value.size
    has_noise = "-1" in cluster_value
    # set a feature code for each feature
    if has_noise:
        cluster_names = _make_feature_names(n_cluster - 1)
        cluster_names = ["noise"] + cluster_names
    else:
        cluster_names = _make_feature_names(n_cluster)

    cluster_mapping = dict(zip(cluster_value, cluster_names))
    cluster = cluster.map(cluster_mapping)
    return cluster


def make_data_container(feature_data: pd.DataFrame, cluster: pd.Series,
                        sample_metadata: pd.DataFrame,
                        fill_na: bool = True) -> DataContainer:
    """
    Organizes the detected and matched features into a DataContainer.

    Parameters
    ----------
    feature_data: DataFrame
        DataFrame obtained from detect_features function.
    cluster: pd.Series
        Series obtained from feature_correspondence function.
    sample_metadata: DataFrame
        DataFrame with information from each analyzed sample. The index must
        be the sample names used in feature_data. A column named "class", with
        the class name of each sample is required. For further data processing
        run order information in a column named "order" and analytical batch
        information in a column named "batch" are recommended.
    fill_na: bool, True
        If True fill missing values in the data matrix with zeros.

    Returns
    -------
    DataContainer
    """

    # remove noise
    feature_data["cluster"] = cluster
    not_noise = cluster != "noise"
    feature_data = feature_data[not_noise]

    # compute aggregate statistics for each feature -> feature metadata
    estimators = {"mz": ["mean", "std", "min", "max"],
                  "rt": ["mean", "std", "min", "max"]}
    feature_metadata = feature_data.groupby("cluster").agg(estimators)
    feature_metadata.columns = _flatten_column_multindex(feature_metadata)
    feature_metadata.index.name = "feature"

    # make data matrix
    data_matrix = feature_data.pivot(index="sample", columns="cluster",
                                     values="area")
    data_matrix.columns.name = "feature"
    if fill_na:
        data_matrix = data_matrix.fillna(0)

    # add samples without features as nan rows
    missing_index = sample_metadata.index.difference(data_matrix.index)
    # TODO: manage data inputting
    missing = pd.DataFrame(data=0, index=missing_index,
                           columns=data_matrix.columns)
    data_matrix = pd.concat((data_matrix, missing))
    data_matrix = data_matrix.loc[sample_metadata.index, :]

    dc = DataContainer(data_matrix, feature_metadata, sample_metadata)
    return dc


def _make_feature_names(n_features: int):
    max_ft_str_length = len(str(n_features))

    def ft_formatter(x):
        return "FT" + str(x + 1).rjust(max_ft_str_length, "0")

    ft_names = [ft_formatter(x) for x in range(n_features)]
    return ft_names


def _flatten_column_multindex(df: pd.DataFrame):
    columns = df.columns
    level_0 = columns.get_level_values(0)
    level_1 = columns.get_level_values(1)
    col_name_map = {"mzmean": "mz", "mzstd": "mz std", "mzmin": "mz min",
                    "mzmax": "mz max", "rtmean": "rt", "rtstd": "rt std",
                    "rtmin": "rt min", "rtmax": "rt max"}
    new_names = [col_name_map[x + y] for x, y in zip(level_0, level_1)]
    return new_names


def _make_initial_cluster(feature_data: pd.DataFrame, mz_tolerance: float,
                          rt_tolerance: float, min_samples: int = 8):
    """
    First guess of correspondence between features using DBSCAN algorithm.
    Auxiliary function to feature_correspondence.

    Parameters
    ----------
    feature_data : DataFrame
        DataFrame obtained from `detect_features` function.
    mz_tolerance : float
        Used to build epsilon parameter of DBSCAN
    rt_tolerance : float
        Used to build epsilon parameter of DBSCAN.
    min_samples : int
        parameter to pass to DBSCAN

    Returns
    -------
    cluster : Series
        The assigned cluster by DBSCAN
    """

    ft_points = feature_data.loc[:, ["mz", "rt"]].copy()
    ft_points["rt"] = ft_points["rt"] * mz_tolerance / rt_tolerance
    dbscan = DBSCAN(eps=mz_tolerance, min_samples=min_samples,
                    metric="chebyshev")
    dbscan.fit(ft_points)
    cluster = pd.Series(data=dbscan.labels_, index=feature_data.index)
    return cluster


def _estimate_n_species_per_cluster(df: pd.DataFrame, cluster: pd. Series,
                                    min_dr: float = 0.2):
    """
    Estimates the number of features that forms a cluster.

    The number of features is estimated as follows:
        1. The number of features per sample is counted and normalized
        to the total number of features.
        2. The number of features in a cluster will be the maximum
        normalized number of features per sample greater than the minimum
        detection rate.

    Parameters
    ----------
    df: DataFrame
        Feature data obtained from feature_correspondence function
    min_dr: float, 0.2
        Minimum detection rate.
    """

    # sample_per_cluster counts the number of features that come from the same
    # sample and express it as a fraction of the total number features
    # the number of features in a cluster is the maximum number of samples
    # in a cluster above the minimum detection rate.

    def find_n_cluster(x):
        return x.index[np.where(x > min_dr)[0]][-1]

    sample_per_cluster = (df["sample"].groupby(cluster)
                          .value_counts()
                          .unstack(-1)
                          .fillna(0)
                          .astype(int)
                          .apply(lambda x: x.value_counts(), axis=1)
                          .fillna(0))
    sample_per_cluster = sample_per_cluster / df["sample"].unique().size

    features_per_cluster = sample_per_cluster.apply(find_n_cluster, axis=1)
    return features_per_cluster


def _make_gmm(ft_data: pd.DataFrame, n_feature: int, cluster_name: str):
    """
    fit a gaussian model and set subcluster names for each feature. Auxiliary
    function to process cluster.

    Parameters
    ----------
    ft_data : DataFrame
        The mz and rt columns of the cluster DataFrame
    n_feature : int
        Number of features estimated in the cluster.
    cluster_name: str

    Returns
    -------
    gmm : GaussianMixtureModel fitted with cluster data
    score: The log-likelihood of each feature.
    subcluster : pd.Series with subcluster labels.
    """
    gmm = mixture.GaussianMixture(n_components=n_feature,
                                  covariance_type="diag")
    gmm.fit(ft_data.loc[:, ["mz", "rt"]].values)
    # scores = pd.Series(data=gmm.score_samples(ft_data), index=ft_data.index)
    ft_data["score"] = gmm.score_samples(ft_data.loc[:, ["mz", "rt"]].values)

    # get index of features in the cases where the number of features is greater
    # than the number of components in the gmm
    noise_index = (ft_data
                   .groupby("sample")
                   .filter(lambda x: x.shape[0] > n_feature))

    if not noise_index.empty:
        noise_index = (noise_index
                       .groupby("sample")
                       .apply(lambda x: _noise_ind(x, n_feature))
                       .droplevel(0)
                       .index)
    else:
        noise_index = noise_index.index

    noise = pd.Series(data="-1", index=noise_index)

    # if the number of features is equal to the number of components in the
    # gmm, each feature is assigned to a cluster using the Hungarian algorithm
    # on the posterior probabilities on each component
    subcluster = (ft_data.loc[ft_data.index.difference(noise_index)]
                  .groupby("sample")
                  .filter(lambda x: x.shape[0] <= n_feature)
                  .groupby("sample")
                  .apply(lambda x: _get_best_cluster(x, gmm))
                  .droplevel(0)
                  .astype(str))
    subcluster = subcluster.apply(lambda x: str(cluster_name) + "-" + x)
    subcluster = pd.concat([noise, subcluster])
    subcluster = subcluster.sort_index()
    # TODO: add here the case where n_features < n_components

    # subcluster = pd.Series(data=gmm.predict(ft_data), index=ft_data.index,
    #                        dtype=str)
    scores = 1
    return gmm, scores, subcluster


def _remove_repeated_features(ft_data: pd.DataFrame, subcluster: pd.Series,
                              sample_data: pd.Series, scores: pd.Series):
    """
    Removes repeated samples from a subcluster. If More than one feature comes
    from the same sample, only the sample with the best log-likelihood is
    conserved, the others are flagged with cluster -1. Auxiliary function of
    _process_cluster

    Parameters
    ----------
    ft_data: DataFrame
        The mz and rt columns of the cluster DataFrame
    sample_data: Series
        The sample column of the cluster DataFrame
    scores: Series
        log-likelihood obtained from the GMM.
    subcluster: pd.Series
        subcluster labels. Obtained from make_gmm
    """
    grouped = ft_data.groupby([subcluster, sample_data])
    for _, repeated_ft in grouped:
        n_ft = repeated_ft.shape[0]
        if n_ft > 1:
            ind = repeated_ft.index
            best_ft = [scores[ind].idxmax()]
            ind = ind.difference(best_ft)
            subcluster[ind] = "-1"


def _search_missing_features(cluster: pd.Series, sample_data: pd.Series,
                             n_feature: int, cluster_name: int,
                             sample_names: List[str], noise: pd.DataFrame,
                             subcluster: pd.Series,
                             gmm: mixture.GaussianMixture,
                             min_likelihood: float):
    """
    Search for missing features in noise. Auxiliary function of
    _process_cluster.

    Parameters
    ----------
    sample_data : Series
        The sample column from cluster data
    n_feature : int
    cluster_name : str
    sample_names : list[str]
        The name of all of the samples used.
    noise : DataFrame
    subcluster : Series
    gmm: GaussianMixture
    min_likelihood: float


    Returns
    -------

    """
    # TODO: this function still needs some work
    for k in range(n_feature):
        k = str(k)
        subc_name = str(cluster_name) + "-" + k
        subcluster_samples = sample_data[subcluster == subc_name]
        missing_samples = np.setdiff1d(sample_names, subcluster_samples)

        # TODO: add some kind of filter of mz, rt to reduce time
        # add cluster == "-1" to not consider taken features
        missing_candidates = noise.loc[
            noise["sample"].isin(missing_samples), ["mz", "rt", "sample"]]
        if not missing_candidates.empty:
            candidates_scores = gmm.score_samples(
                missing_candidates.loc[:, ["mz", "rt"]].values)
            is_candidate = candidates_scores > min_likelihood
            is_any_candidate = is_candidate.any()
            if is_any_candidate:
                missing_candidates = missing_candidates.loc[
                    missing_candidates.index[is_candidate]]
                missing_candidates["score"] = candidates_scores[is_candidate]
                candidate_index = missing_candidates.groupby("sample").apply(
                    lambda x: x["score"].idxmax())
                cluster[candidate_index] = subc_name


def _process_cluster(df: pd.DataFrame, noise: pd.DataFrame, cluster: pd.Series,
                     sample_names: list, cluster_name: str,
                     min_likelihood: float, n_species: int):
    """
    Process each cluster obtained from DBSCAN. Auxiliary function to
    `feature_correspondence`.

    Parameters
    ----------
    df : DataFrame
        feature_data values for a given cluster
    noise : DataFrame
        Features flagged as noise by DBSCAN
    cluster : Series
        Cluster values obtained by DBSCAN.
    sample_names : list[str]
        names of the analyzed samples.
    cluster_name : str
        name of the cluster being analyzed
    min_likelihood : float
    n_species: int
        Number of features in the cluster, estimated with
        `estimate_features_per_cluster`.

    Returns
    -------
    subcluster : Series
        The subcluster values.
    """

    ft_data = df.loc[:, ["mz", "rt", "sample"]]
    sample_data = df["sample"]

    if n_species >= 1:
        # fit a Gaussian mixture model using the cluster data
        gmm, scores, subcluster = _make_gmm(ft_data, n_species, cluster_name)
    else:
        subcluster = pd.Series(data="-1", index=df.index)

    # send repeated samples to noise: only the feature with the best
    # score in the subcluster is conserved
    # _remove_repeated_features(ft_data, subcluster, sample_data, scores)

    to_noise = df[subcluster == "-1"]
    if not to_noise.empty:
        noise = pd.concat([noise, to_noise])

    # search missing samples in noise:
    # _search_missing_features(cluster, sample_data, n_species, cluster_name,
    #                          sample_names, noise, subcluster, gmm,
    #                          min_likelihood)
    return subcluster


def _get_best_cluster(x, gmm):
    """
    Assigns a feature to a cluster the posterior probability to each cluster.
    """
    proba = gmm.predict_proba(x.loc[:, ["mz", "rt"]].values)
    rows, cols = proba.shape
    if rows != cols:
        fill = np.zeros(shape=(cols - rows, cols))
        proba = np.vstack((proba, fill))
    _, best_cluster = linear_sum_assignment(proba)
    best_cluster = best_cluster[:rows]
    best_cluster = pd.Series(data=best_cluster, index=x.index)
    return best_cluster


def _noise_ind(x, n):
    """
    search the index of samples that are going to be considered as noise.
    Reduces the number of features from a sample in a cluster until the size is
    equal to n
    """
    ind = x["score"].sort_values().index[:(x.shape[0] - n)]
    return x.loc[ind, :]
