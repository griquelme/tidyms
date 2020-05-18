"""
Common routines to work with raw MS data from metabolomics experiments.

Functions
---------
detect_features(path_list) : Perform feature detection on several samples.
Returns a DataFrame with feature descriptors and the ROI where each feature was
detected.
feature_correspondence(feature_data) : Match features across different samples
using a combination of clustering algorithms.

"""

import pandas as pd
import numpy as np
from .fileio import MSData
from .lcms import Roi
import os.path
from sklearn.cluster import DBSCAN
from sklearn import mixture
from typing import Optional, Tuple, List, Dict


def detect_features(path_list: List[str], mode: str = "uplc",
                    ms_mode: str = "qtof", roi_params: Optional[dict] = None,
                    cwt_params: Optional[dict] = None, verbose: bool = True
                    ) -> Tuple[Dict[str, Roi], pd.DataFrame]:
    """
    Perform feature detection on several samples using the algorithm described
    in [1].

    Samples are analyzed one at a time using the detect_features method
    of MSData. See this method for a detailed explanation of each parameter.

    Parameters
    ----------
    path_list: List[str]
        List of path to centroided mzML files.
    mode: {"uplc", "hplc"}
        Analytical platform used for separation. Used to set default values
        of `cwt_params` and `roi_params`.
    ms_mode: {"qtof". "orbitrap"}
        MS instrument used for data acquisition. Used to set default values
        of `cwt_params` and `roi_params`.
    roi_params: dict, optional
        Set roi detection parameters in MSData detect_features method.
        Overwrites defaults values set by `mode` and `ms_mode`.
    cwt_params: dict, optional
        Set peak detection parameters in MSData detect_features method.
        Overwrites defaults values set by `mode` and `ms_mode`.
    verbose: bool
        If True prints a message each time a sample is analyzed.

    Returns
    -------
    roi_mapping: dict of sample names to list of ROI
        ROI for each sample where features were detected. Can be used to
        visualize each feature.
        TODO : returning all the ROI can generate memory problems. In this
            case maybe an alternative is writing ROI to disk.

    proto_dm: DataFrame
        A DataFrame with features detected and their associated descriptors.
        Each feature is a row, each descriptor is a column. The descriptors
        are:
        - mz : Mean m/z of the feature
        - mz std : standard deviation of the m/z. Computed as the standard
        deviation of the m/z in the region where the peak was detected.
        - rt : retention time of the feature, computed as the weighted mean
         of the retention time, using as weights the intensity at each time.
        - width : Chromatographic peak width.
        - intensity : Maximum intensity of the chromatographic peak.
        - area : Area of the chromatographic peak.
        Also, two additional columns have information to search each feature
        in its correspondent Roi:
        - Roi index : index in `roi_list` where the feature was detected.
        - peak_index : index of the peaks attribute of each Roi associated
        to the feature.
        - sample : The sample name where the feature was detected.

    References
    ----------
    # TODO : add centwave ref
    """
    roi_mapping = dict()
    ft_df_list = list()
    n_samples = len(path_list)
    for k, sample_path in enumerate(path_list):
        sample_filename = os.path.split(sample_path)[1]
        sample_name = os.path.splitext(sample_filename)[0]
        if verbose:
            msg = "Processing sample {} ({}/{})."
            msg = msg.format(sample_name, k + 1, n_samples)
            print(msg)
        ms_data = MSData(sample_path)
        roi, df = ms_data.detect_features(mode, ms_mode,
                                          make_roi_params=roi_params,
                                          find_peaks_params=cwt_params)
        df["sample"] = sample_name
        roi_mapping[sample_name] = roi
        ft_df_list.append(df)
    proto_dm = pd.concat(ft_df_list).reset_index(drop=True)
    # TODO: need to check performance for concat when n_samples is big.
    return roi_mapping, proto_dm


def feature_correspondence(feature_data: pd.DataFrame, mz_tolerance: float,
                           rt_tolerance: float, min_dr: int = 8,
                           min_likelihood: float = 0.0):
    """
    Match features across different samples.

    Feature matching is done using the DBSCAN algorithm and Gaussian mixture
    models. After performing feature correspondence a new column, called
    cluster, is added to feature data

    Parameters
    ----------
    feature_data: DataFrame
        Feature descriptors obtained from detect_features function
    mz_tolerance: float
        m/z tolerance used to cluster samples with DBSCAN.
    rt_tolerance: float
        rt tolerance used to cluster samples with DBSCAN.
    min_dr: float
        Minimum detection rate. The product of `min_dr` and the number of
        samples is used to determine the minimum number of features in a
        cluster.
    min_likelihood: float
        Minimum likelihood required to recover a missing samples. Lower values
        will recover more features, but with lower confidence.

    Notes
    -----
    The correspondence algorithm is as follows:

        1. Features ares clustered using m/z and rt information with the DBSCAN
        algorithm. Because the dispersion in the m/z and r/t dimension is
        independent (TODO: more details on this)
        the Chebyshev distance is used to make the clusters. `rt_tolerance` and
        `mz_tolerance` are used to build the epsilon parameter of the model.
        rt is scaled using these two parameters to have the same tolerance in
        both dimensions. The min_samples parameter is defined using from the
        minimum_dr (minimum detection rate) and the total number of samples
        in the data. This step gives us an matching of the samples, but
        different groups of features can be clustered together if they are
        close, or some features can be considered as noise and removed.
        These cases are analyzed in the following steps
        2. Once the data is clustered, we evaluate the number of repeated
        features in each cluster (that is, the number of features that come from
        the same sample) and compute the detection rate for each number of
        repetition:

        ..math::

            dr = \frac{n_{repetitions}}{n_{samples}}

        The number of features in a cluster is the maximum number of repeated
        features with a dr greater than `min_dr`
        3. Each clustered is modeled with a gaussian mixture using the number
        of features computed in the previous step. Once again, because
        dispersion in rt and m/z is orthogonal we used diagonal covariances
        matrix in the GMM. After this step, for each cluster, subclusters may be
        generated if the number of features is greater than one.
        4. Each subcluster is analyzed then to remove repeated features. To
        remove repeated features the log-likelihood is evaluated for features
        coming from the same sample. The feature with the greatest likelihood is
        kept and the others are flagged as noise.
        5. The final step is to search for missing features (a feature is
        missing if some of the samples are not present in the subcluster). This
        is done searching for features that come from missing samples in the
        features flagged as noise by DBSCAN (or in the previous step). If a
        feature from a missing sample has a log-likelihood greater than
        `min_likelihood` then is added to the subcluster. If more than one
        feature is possible the one with the greatest likelihood is chosen.
    """
    cluster = _make_initial_cluster(feature_data, mz_tolerance, rt_tolerance,
                                    min_dr)
    # TODO : complete function


def _make_initial_cluster(feature_data: pd.DataFrame, mz_tolerance: float,
                          rt_tolerance: float, min_samples: int = 8):
    """
    First guess of correspondence between features using DBSCAN algorithm.
    Auxiliary function to feature_corespondence.
`   TODO : fill docstring
    Parameters
    ----------
    feature_data
    mz_tolerance
    rt_tolerance
    min_samples

    Returns
    -------

    """

    ft_points = feature_data.loc[:, ["mz", "rt"]].copy()
    ft_points["rt"] = ft_points["rt"] * mz_tolerance / rt_tolerance
    dbscan = DBSCAN(eps=mz_tolerance, min_samples=min_samples,
                    metric="chebyshev")
    dbscan.fit(ft_points)
    return pd.Series(data=dbscan.labels_, index=feature_data.index)


def _estimate_features_per_cluster(df: pd.DataFrame, min_dr: float = 0.2):
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

    def sample_to_count(x):
        return x["sample"].value_counts().value_counts()

    def find_n_cluster(x):
        return x.index[np.where(x > min_dr)[0]][-1]

    sample_per_cluster = (df.groupby("cluster")
                          .apply(sample_to_count)
                          .unstack(-1)
                          .fillna(0)
                          .astype(int)
                          .apply(lambda x: x / x.sum(), axis=1))
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
    gmm.fit(ft_data)
    scores = pd.Series(data=gmm.score_samples(ft_data), index=ft_data.index)
    subcluster = pd.Series(data=gmm.predict(ft_data), index=ft_data.index,
                           dtype=str)
    subcluster = subcluster.apply(lambda x: str(cluster_name) + "-" + x)
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


def _search_missing_features(sample_data: pd.Series, n_feature: int,
                             cluster_name: int, sample_names: List[str],
                             noise: pd.DataFrame, subcluster: pd.Series,
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
    cluster_name : int
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
            noise["sample"].isin(missing_samples), ["mz", "rt"]]
        candidates_scores = gmm.score_samples(missing_candidates)
        missing_candidates = missing_candidates.loc[
            missing_candidates.index[candidates_scores > min_likelihood]]
        missing_candidates["score"] = candidates_scores[
            candidates_scores > min_likelihood]
        if not missing_candidates.empty:
            a = missing_candidates.groupby("sample").apply(
                lambda x: x["score"].idxmax())
            noise.loc[a, "cluster"] = subc_name


def _process_cluster(df: pd.DataFrame, noise: pd.DataFrame, sample_names: list,
                     cluster_name, min_likelihood: float = 0.0,
                     n_feature: int = 1):
    ft_data = df.loc[:, ["mz", "rt"]]
    sample_data = df["sample"]

    # fit a Gaussian mixture model using the cluster data
    gmm, scores, subcluster = _make_gmm(ft_data, n_feature, cluster_name)

    # send repeated samples to noise: only the feature with the best
    # score in the subcluster is conserved
    _remove_repeated_features(ft_data, subcluster, sample_data, scores)

    to_noise = df[subcluster == "-1"]
    noise = pd.concat([noise, to_noise])

    # search missing samples in noise:
    _search_missing_features(sample_data, n_feature, cluster_name, sample_names,
                             noise, subcluster, gmm, min_likelihood)

    return subcluster
