import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from scipy.integrate import trapz
from typing import Generator, Optional, Tuple
from . import _constants as c
from .fileio import MSData
from .lcms import Chromatogram
from .raw_data_utils import make_chromatograms
from .utils import find_closest, get_progress_bar
from sklearn.impute import KNNImputer


missing_data_iterator_type = Generator[
    Tuple[
        MSData,
        str,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.Index
    ],
    None,
    None
]


def fill_missing_lc(
    ms_data_generator: Generator[Tuple[str, MSData], None, None],
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    mz_tolerance: float,
    n_deviations: float,
    estimate_not_found: bool,
    n_jobs: Optional[int] = None,
    verbose: bool = False
):
    """
    Fill missing values by searching missing peaks in the raw data.

    Notes
    -----

    Missing Chromatographic peaks are searched in the raw data using the
    peak descriptors obtained after features extraction and correspondence.
    Initially, chromatograms for missing features in each sample are built by
    using the mean m/z from the features descriptors. A time window for
    searching the peaks is defined based on the mean Rt and the standard
    deviation values features:

    .. math::

        t_{mean} \pm n_{dev} t_{std}

    If a peak is found in the region, its area is used as fill value. If more
    than one peak is found in the region, the closes to the mean Rt is chosen.
    If no peak was found in the region, an estimation of the peak area is
    computed by integrating the region comprised by the mean start and end
    times of the peak in the detected features.

    Parameters
    ----------
    ms_data_generator : Generator
        Yields sample names and MSData objects for raw data samples.
    data_matrix : pd.DataFrame
    feature_metadata : pd.DataFrame
    mz_tolerance : float
        m/z tolerance used to create chromatograms.
    n_deviations : float
        Number of deviations from the mean retention time to search a peak,
        in units of standard deviations.
    estimate_not_found : bool
        If ``True``, and estimation for the peak area in cases where no
        chromatographic peaks are found is done as described in the Notes.
        If ``False``, missing values after peak search are set to zero.
    n_jobs: int or None, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    verbose : bool
        If True, shows a progress bar.

    Returns
    -------
    data_matrix_filled : pd.DataFrame
        Data matrix with missing values imputed
    missing_indicator : pd.DataFrame
        DataFrame with the same shape as the data matrix, with values that
        indicates if a feature was detected (0), filled (1) or estimated (2).

    """
    missing_iterator = _get_missing_data_iterator(
        ms_data_generator, data_matrix, feature_metadata)
    if verbose:
        progress_bar = get_progress_bar()
        total = data_matrix.shape[0]
        print("Filling missing values in {} samples".format(total))
        missing_iterator = progress_bar(
            missing_iterator,
            total=total
        )

    func = partial(
        _get_sample_fill_values,
        mz_tolerance=mz_tolerance,
        n_deviations=n_deviations,
        estimate_not_found=estimate_not_found
    )
    func = delayed(func)
    fill_results = Parallel(n_jobs=n_jobs)(func(*x) for x in missing_iterator)
    missing = pd.DataFrame(
        data=0,
        index=data_matrix.index,
        columns=data_matrix.columns,
        dtype=int
    )
    for res in fill_results:
        sample, found_area, found_index, not_found_area, not_found_index = res
        # TODO: fix after refactoring DataContainer
        data_matrix.loc[sample, found_index] = found_area
        data_matrix.loc[sample, not_found_index] = not_found_area
        missing.loc[sample, found_index] = 1
        missing.loc[sample, not_found_index] = 2
    return data_matrix, missing


def fill_missing_knn(df: pd.DataFrame, class_: pd.Series, n_neighbors: int):
    """
    Fill missing values using the KNN Imputation. Imputation is done using only
    samples from the same class. If a feature is missing in all samples from
    a class, the value is set to zero.


    Parameters
    ----------
    df : pd.DataFrame
        Data matrix with missing values.
    class_ : pd.Series
        Class label of each sample in the data matrix.
    n_neighbors : int, default=1
        Numbers of neighbors used in the KNN imputation.

    Returns
    -------
    df_filled : pd.DataFrame
        DataMatrix with missing values imputed.

    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    return df.groupby(class_).apply(_apply_imputer, knn_imputer)


def _apply_imputer(df: pd.DataFrame, imputer: KNNImputer) -> pd.DataFrame:
    """
    Helper function for data imputation on each class.

    Auxiliary function to fill_missing_knn

    Parameters
    ----------
    df : pd.DataFrame
    imputer : KNNImputer

    Returns
    -------
    df : pd.DataFrame

    """
    all_missing = df.isna().all()
    df.loc[:, all_missing] = 0
    X = imputer.fit_transform(df)
    return pd.DataFrame(data=X, index=df.index, columns=df.columns)


def _get_missing_data_iterator(
    ms_data_generator: Generator[Tuple[str, MSData], None, None],
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame
) -> missing_data_iterator_type:
    mz = feature_metadata[c.MZ]
    rt = feature_metadata[c.RT]
    rt_std = feature_metadata[c.RT_STD]
    rt_start = feature_metadata[c.RT_START]
    rt_end = feature_metadata[c.RT_END]
    for sample, ms_data in ms_data_generator:
        missing_mask = data_matrix.loc[sample].isna()
        missing_data = (
            ms_data,
            sample,
            mz[missing_mask].to_numpy(),
            rt[missing_mask].to_numpy(),
            rt_std[missing_mask].to_numpy(),
            rt_start[missing_mask].to_numpy(),
            rt_end[missing_mask].to_numpy(),
            missing_mask[missing_mask].index
        )
        yield missing_data


def _get_sample_fill_values(
    ms_data: MSData,
    sample: str,
    mz: np.ndarray,
    rt: np.ndarray,
    rt_std: np.ndarray,
    rt_start: np.ndarray,
    rt_end: np.ndarray,
    features: pd.Index,
    mz_tolerance: float,
    n_deviations: float,
    estimate_not_found: bool
) -> Tuple[str, np.ndarray, pd.Index, np.ndarray, pd.Index]:
    """
    Finds fill values for features not detected during feature extraction in a
    sample.

    Parameters
    ----------
    ms_data : MSData
    rt : array
        Mean retention time estimated of the features.
    rt_std : array
        Standard deviation of the retention time of the features
    rt_start : array
        Mean Peak start time of the features.
    rt_end : array
        Mean Peak end time of the features.
    features : pandas.Index
        Indices used for the features
    mz_tolerance : float
    n_deviations : float
    estimate_not_found : bool

    Returns
    -------

    """
    chromatograms = make_chromatograms(ms_data, mz, window=mz_tolerance)
    chromatograms_rt = chromatograms[0].time
    start_index = np.searchsorted(chromatograms_rt, rt_start)
    end_index = np.searchsorted(chromatograms_rt, rt_end)
    iterator = zip(chromatograms, rt, rt_std, start_index, end_index, features)

    found_index = list()
    found_area = list()
    not_found_index = list()
    not_found_area = list()

    for chrom, c_rt, c_rt_std, c_start_index, c_end_index, c_ft in iterator:
        # fill values outside the valid range to 0
        chrom.fill_nan(fill_value=0.0, bounds_error=False)
        area = _get_fill_area(chrom, c_rt, c_rt_std, n_deviations)
        if area is None:
            not_found_index.append(c_ft)
            if estimate_not_found:
                area = trapz(
                    chrom.spint[c_start_index:c_end_index],
                    chrom.time[c_start_index:c_end_index]
                )

            else:
                area = 0.0
            not_found_area.append(area)
        else:
            found_index.append(c_ft)
            found_area.append(area)

    found_area = np.array(found_area)
    not_found_area = np.array(not_found_area)
    found_index = pd.Index(found_index)
    not_found_index = pd.Index(not_found_index)

    return sample, found_area, found_index, not_found_area, not_found_index


def _get_fill_area(
    chromatogram: Chromatogram,
    rt: float,
    rt_std: float,
    n_deviations: float
) -> Optional[float]:
    """
    Search a peak in a region defined between ``rt - rt_std`` and
    ``rt + rt_std``. If a peak is found, the corresponding area is found.
    Otherwise, return None.

    Parameters
    ----------
    chromatogram : Chromatogram
    rt : float
        Expected retention time for the peak.
    rt_std : float
        Standard deviation of the retention time.
    n_deviations : float
        Multiplier for `r_std`, used to build a tolerance window for peak
        search, defined as :math:`t_{mean} \pm n_{dev} t_{std}`

    Returns
    -------
    area : float or None

    """
    chromatogram.extract_features(store_smoothed=True)
    rt_ft = np.array([p.get_rt(chromatogram) for p in chromatogram.features])
    if rt_ft.size:
        closest_feature = find_closest(rt_ft, rt)
        is_valid_peak = abs(rt_ft[closest_feature] - rt) < rt_std * n_deviations
        if is_valid_peak:
            area = chromatogram.features[closest_feature].get_area(chromatogram)
            chromatogram.features = [chromatogram.features[closest_feature]]
        else:
            area = None
            chromatogram.features = []
    else:
        area = None
    return area
