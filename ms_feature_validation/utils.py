import pyopenms
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import os.path
from typing import Tuple, List, Optional, Union
from collections import namedtuple
msexperiment = Union[pyopenms.MSExperiment, pyopenms.OnDiscMSExperiment]


def find_experimental_rt(rt, chromatogram, rt_guess, width, tolerance):
    """
    Finds experimental retention time based on an approximate value.

    Parameters
    ----------
    rt : np.array
    chromatogram : np.array
    rt_guess : float
        retention time used as guess.
    width : (min_width, max_width)
        tuple of valid width range.
    tolerance : float.
        retention time tolerance shift

    Returns
    -------
    exp_rt : float

    Notes
    -----
    If more than oun peak is found, then the closes value to `rt_guess` is
    returned.
    """
    peaks, peaks_properties = find_peaks(chromatogram,
                                         width=width,
                                         rel_height=0.5)
    possible_rts = rt[peaks]
    possible_rts = possible_rts[(possible_rts - rt_guess) < tolerance]
    exp_rt = possible_rts[np.argmin(np.abs(possible_rts - rt_guess))]
    return exp_rt


def gauss(x, mu, sigma, amp=None):
    """
    gaussian curve.

    Parameters
    ----------.sum(axis=0)
    x : np.array
    mu : float
    sigma : float
    amp : float / None
        If None returns a normalized gaussian curve.

    Returns
    -------
    gaussian : np.array
    """
    if amp is None:
        amp = 1 / (np.sqrt(2 * np.pi) * sigma)
    gaussian = amp * np.power(np.e, - 0.5 * ((x - mu) / sigma) ** 2)
    return gaussian


def gaussian_mixture(x: np.array, *params: float) -> np.array:
    """
    Mixture of gaussian curves.

    Parameters
    ----------
    x : np.array
    params: mu1, sigma1, amp1, mu2, sigma2, amp2, ...

    Returns
    -------
    """
    mixture = np.zeros_like(x)
    for i in range(0, len(params), 3):
        mu_i = params[i]
        sigma_i = params[i + 1]
        amp_i = params[i + 2]
        mixture += gauss(x, mu_i, sigma_i, amp_i)
    return mixture


def cluster(s: pd.Series, tolerance: float) -> pd.Series:
    """
    cluster values within a given tolerance

    Parameters
    ----------
    s : pd.Series
    tolerance : float

    Returns
    -------
    cluster_number : pandas.Series

    Notes
    -----
    The clustering algorithm is as follow:

    1. sort values
    2. find successive values within `tolerance`
    3. These values are assigned a cluster number
    """
    cluster_number = (s.sort_values().diff() > tolerance).cumsum()
    return cluster_number


def mean_cluster_value(mz: pd.Series, cluster: pd.Series) -> pd.Series:
    """
    Returns the mean cluster value.

    Parameters
    ----------
    mz : pandas.Series
    cluster : pandas.Series

    Returns
    -------
    mean: pandas.Series
    """
    return mz.groupby(cluster).mean()


def overlap_groups(df: pd.Series, rt_tolerance: float,
                   mz_tolerance: float) -> pd.Series:
    """
    returns index with overlap in Retention Time and Mass-to-charge ratio.

    Parameters
    ----------
    sample_information : pandas.DataFrame
    rt_tolerance : float
    mz_tolerance : float

    Returns
    -------
    overlap_cluster

    """
    mz = df["mz"]
    rt = df["rt"]

    def has_overlap_helper(x, tol):
        xx, xy = np.meshgrid(x, x)
        x_overlap = np.abs(xx - xy) < tol
        x_overlap[np.diag_indices_from(x_overlap)] = False
        return x_overlap

    def overlap_groups_helper(overlap):
        return list(df.index[overlap])

    mz_overlap = has_overlap_helper(mz, mz_tolerance)
    rt_ovelap = has_overlap_helper(rt, rt_tolerance)
    overlap_df = pd.DataFrame(mz_overlap & rt_ovelap,
                              index=df.index,
                              columns=df.index)
    overlap_series = overlap_df.apply(overlap_groups_helper, axis=0)
    return overlap_series

def get_eic_threshold(sp_threshold, mz_res, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = np.arange(-fwhm / 2, fwhm / 2, mz_res)
    y = gauss(x, 0, sigma, sp_threshold)
    threshold = y.sum()
    return threshold


def recommend_samples(df: pd.DataFrame, q: float,
                      min_samples: float) -> list:
    q_values = df > df.quantile(q)
    recommended_samples = list()
    samples_count = pd.Series(data=np.zeros(q_values.columns.size),
                                    index=q_values.columns)
    while q_values.shape[1] > 0:
        add_sample = q_values.sum(axis=1).idxmax()
        recommended_samples.append(add_sample)
        samples_count += q_values.loc[add_sample, :]
        samples_count = samples_count[samples_count < min_samples]
        q_values.drop(index=add_sample, inplace=True)
        q_values = q_values.loc[:, samples_count.index]
    return recommended_samples


def normalize(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Normalize samples using different methods.

    Parameters
    ----------
    df: pandas.DataFrame
    method: {"sum", "max", "euclidean"}
        Normalization method. `sum` normalizes using the sum along each row,
        `max` normalizes using the maximum of each row. `euclidean` normalizes
        using the euclidean norm of the row.

    Returns
    -------
    normalized: pandas.DataFrame

    """
    if method == "sum":
        normalized = df.divide(df.sum(axis=1), axis=0)
    elif method == "max":
        normalized = df.divide(df.max(axis=1), axis=0)
    elif method == "euclidean":
        normalized = df.apply(lambda x: x / np.linalg.norm(x), axis=1)
    else:
        msg = "method must be `sum`, `max`, `euclidean`"
        raise ValueError(msg)
    # replace nans generated by division by zero
    normalized[normalized.isna()] = 0
    return normalized


def scale(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    scales features using different methods.

    Parameters
    ----------
    df: pandas.DataFrame
    method: {"autoscaling", "rescaling", "pareto"}
        Scaling method. `autoscaling` performs mean centering scaling of
        features to unitary variance. `rescaling` scales data to a 0-1 range.
        `pareto` performs mean centering and scaling using the square root of
        the standard deviation

    Returns
    -------
    scaled: pandas.DataFrame
    """
    if method == "autoscaling":
        scaled = (df - df.mean()) / df.std()
    elif method == "rescaling":
        scaled = (df - df.min()) / (df.max() - df.min())
    elif method == "pareto":
        scaled = (df - df.mean()) / df.std().apply(np.sqrt)
    else:
        msg = "Available methods are `autoscaling`, `rescaling` and `pareto`."
        raise ValueError(msg)
    # replace nans generated when dividin by zero
    scaled[scaled.isna()] = 0
    return scaled


def transform(df: pd.DataFrame, method: str,
              center: bool = True) -> pd.DataFrame:
    """
    perform common data transformations.

    Parameters
    ----------
    df: pandas.DataFrame
    method: {"log", "power"}
        transform method. `log` applies the base 10 logarithm on the data.
        `power`

    Returns
    -------
    transformed: pandas.DataFrame
    """
    if method == "log":
        transformed = df.apply(np.log10)
    elif method == "power":
        transformed = df.apply(np.sqrt)
    else:
        msg = "Available methods are `log` and `power`"
        raise ValueError(msg)
    return transformed


def sample_to_path(samples, path):
    """
    map sample names to raw path if available.

    Parameters
    ----------
    samples : Iterable[str].
        samples names
    path : str.
        path to raw sample data.

    Returns
    -------
    d: dict
    """
    available_files = os.listdir(path)
    filenames = [os.path.splitext(x)[0] for x in available_files]
    full_path = [os.path.join(path, x) for x in available_files]
    d = dict()
    for k, name in enumerate(filenames):
        if name in samples:
            d[name] = full_path[k]
    return d


def get_function_parameters(only=None, exclude=None, ignore='self'):
    """Returns a dictionary of the calling functions
       parameter names and values.

       The optional arguments can be used to filter the result:

           only           use this to only return parameters
                          from this list of names.

           exclude        use this to return every parameter
                          *except* those included in this list
                          of names.

           ignore         use this inside methods to ignore
                          the calling object's name. For
                          convenience, it ignores 'self'
                          by default.

    """
    import inspect
    args, varargs, varkw, defaults = \
        inspect.getargvalues(inspect.stack()[1][0])
    if only is None:
        only = args[:]
        if varkw:
            only.extend(defaults[varkw].keys())
            defaults.update(defaults[varkw])
    if exclude is None:
        exclude = []
    exclude.append(ignore)
    return dict([(attrname, defaults[attrname])
        for attrname in only if attrname not in exclude])

    
def cv(df):
    res = df.std() / df.mean()
    res = res.fillna(0)
    return res

def sd(df):
    res = df.std()
    res = res.fillna(0)
    return res

def iqr(df):
    res = (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)
    res = res.fillna(0)
    return res

def rmad(df):
    res = 1.4826 * df.mad() / df.mean()
    res = res.fillna(0)
    return res

def mad(df):
    res = df.mad()
    res = res.fillna(0)
    return res

def d_ratio(qc_df, samples_df):
    """
    Computes the D-Ratio using sample variation and quality control
    variaton [1].
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        DataFrame with quality control samples
    samples_df : pd.DataFrame
        DataFrame with biological samples
    Returns
    -------
    dr : pd.Series:
        D-Ratio for each feature
    
    
    References
    ----------
    .. [1] D.Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies", Metabolomics (2018)
    14:72.    
    """


Roi = namedtuple("Roi", ("mz", "spint", "index"))

Roi.__doc__ = \
    """
    Region of interest of a MS experiment

    Fields
    ------
    mz: np.ndarray
        mz values for each point
    spint: np.ndarray
        intensity values for each point
    index: int
        scan number when roi starts.
    """


def _list_to_roi(mz_list: List[np.ndarray], sp_list: List[np.ndarray],
                 index: int) -> List[Roi]:
    """
    Convert a list of mz and sp to a list of Roi

    Parameters
    ----------
    mz_list: List[np.ndarray]
    sp_list: List[np.ndarray]
    index: int

    Returns
    -------
    roi_list: List[Roi]
    """
    if len(mz_list) > 0:
        roi_list = [Roi(mz, sp, index) for mz, sp in zip(mz_list, sp_list)]
    else:
        roi_list = list()
    return roi_list


def match_mz(mz1: np.ndarray, mz2: np.ndarray,
             tol: float
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to make_roi. Find matching indices for two MS scans.

    Parameters
    ----------
    mz1: np.ndarray
    mz2: np.ndarray
    tol: float

    Returns
    -------
    mz1_match: np.ndarray
        matched index for mz1
    mz1_unmatch: np.ndarray
        unmatched index for mz1
    mz2_match: np.ndarray
        matched index for mz2
    mz2_unmatch: np.ndarray
        unmatched index for mz2

    """

    sorted_index = np.searchsorted(mz1, mz2)
    # search match index in mz1
    # negative index are converted to 0 ,index equal to mz1 size are converted
    # mz1.size - 1
    mz1_match = np.vstack((sorted_index - 1, sorted_index))
    mz1_match = np.where(mz1_match >= mz1.size, mz1.size - 1, mz1_match)
    mz1_match = np.where(mz1_match < 0, 0, mz1_match)
    # selecting indices within tolerance
    delta_mz = np.abs(mz1[mz1_match] - mz2)
    min_mz_index = np.argmin(delta_mz, axis=0)
    tmp = np.arange(min_mz_index.size)
    delta_mz = delta_mz[min_mz_index, tmp]

    mz1_match = mz1_match[min_mz_index, tmp]
    mz1_match = mz1_match[delta_mz < tol]
    mz1_match, ind = np.unique(mz1_match, return_index=True)

    # search non matched indices in mz1
    mz1_non_match = np.setdiff1d(np.arange(mz1.size), mz1_match)

    # search match index in mz2
    mz2_match = np.where(delta_mz < tol)[0]

    # mz2_match = np.unique(mz2_match)
    mz2_match = mz2_match[ind]

    # search non matched indices in mz2
    mz2_non_match = np.setdiff1d(np.arange(mz2.size), mz2_match)

    return mz1_match, mz1_non_match, mz2_match, mz2_non_match


def make_rois(msexp: msexperiment, pmin: int, max_gap: int,
             min_int: float,  tolerance: float,  start: Optional[int] = None,
              end: Optional[int] = None) -> List[Roi]:

    if start is None:
        start = 0

    if end is None:
        end = msexp.getNrSpectra()

    rois = list()
    roi_maker_list = list()

    # roi_maker_list_initialization
    mz, spint = msexp.getSpectrum(start).get_peaks()
    _, unique_index = np.unique(mz, return_index=True)
    mz = mz[unique_index]
    spint = spint[unique_index]
    roi_maker = RoiMaker(mz, spint, start, end, pmin, max_gap, min_int)
    roi_maker_list.append(roi_maker)

    for k_scan in range(start + 1, end):
        completed = list()  # completed roi
        mz, spint = msexp.getSpectrum(k_scan).get_peaks()
        # remove duplicates, bug in proteowizard
        _, unique_index = np.unique(mz, return_index=True)
        mz = mz[unique_index]
        spint = spint[unique_index]

        for k, roi_maker in enumerate(roi_maker_list):
            mz1_match, mz1_nomatch, mz2_match, mz2_nomatch = \
                match_mz(roi_maker.mz_mean, mz, tolerance)
            # mz1 = np.ones_like(roi_maker.mz_mean) * np.nan
            mz1 = np.zeros_like(roi_maker.mz_mean)
            mz1[mz1_match] = mz[mz2_match]
            sp1 = np.zeros_like(roi_maker.mz_mean)
            # sp1 = np.ones_like(roi_maker.mz_mean) * np.nan
            sp1[mz1_match] = spint[mz2_match]
            mz = mz[mz2_nomatch]
            spint = spint[mz2_nomatch]
            roi_maker.add(mz1, sp1)
            rois.extend(roi_maker.make_roi())
            if roi_maker.completed:
                completed.append(k)

        if mz.size > 0:
            roi_maker_tmp = RoiMaker(mz, spint, k_scan, end, pmin,
                                     max_gap, min_int)
            roi_maker_list.append(roi_maker_tmp)

        while completed:
            del roi_maker_list[completed.pop()]
    return rois



class RoiMaker:
    """
    Class used to make ROI from an MS Data in centroid mode.

    Attributes
    ----------
    mz: np.ndarray
        mz array used to initialize ROI
    spint: np.ndarray
        intensities array used to initialize ROI
    pmin: int
        Minimum number of points in the ROI
    start: int
        scan when roi in initialized
    pmax: int
        Maximum number of points in the ROI
    max_gap: int
        Maximum number of missing points
    min_int: float
        Minimum intensity value in a ROI.
    """

    def __init__(self, mz: np.ndarray, spint: np.ndarray, start: int, end: int,
                 pmin: int, max_gap: int, min_int: float):
        self.pmin = pmin
        self.pmax = end - start
        self.max_gap = max_gap
        self.min_int = min_int
        self.gap = np.zeros(shape=mz.size, dtype=int)
        # self.mz = np.ones((pmax, mz.size)) * np.nan
        # self.spint = np.ones((pmax, mz.size)) * np.nan
        self.mz = np.zeros((self.pmax, mz.size))
        self.spint = np.zeros((self.pmax, mz.size))
        self.index = 0
        self.add(mz, spint)
        self.start = start
        self.mz_mean = mz
        self.completed = False
        self.end = end

    def add(self, mz: np.ndarray, spint: np.ndarray):
        """
        Add mz and spint values.
        """
        self.mz[self.index, :] = mz
        self.spint[self.index, :] = spint
        missing = (self.mz[self.index, :] == 0)
        self.gap = np.where(missing, self.gap + 1, 0)
        self.index += 1
        # mean ignoring zeros
        count = (self.mz[:self.index, ] > 0).sum(axis=0)
        self.mz_mean = self.mz[:self.index, ].sum(axis=0) / count

    def make_roi(self):
        """Make a list of completed ROI and clean non extended ROI"""
        mz_list = list()
        spint_list = list()
        gap_mask = self.gap <= self.max_gap
        spint_mask = self.spint[:self.index, :].max(axis=0) > self.min_int
        n_completed = 0

        # find completed roi
        if (self.index > self.pmin) and (self.index < self.pmax):
            completed_mask = (~gap_mask) & spint_mask
            n_completed = completed_mask.sum()
        elif self.index >= self.pmax:
            completed_mask = spint_mask
            self.completed = True
            n_completed = completed_mask.sum()

        # extract completed roi to a list
        if n_completed > 1:
            mz_tmp = list(self.mz[:self.index, completed_mask].T)
            spint_tmp = list(self.spint[:self.index, completed_mask].T)
            mz_list.extend(mz_tmp)
            spint_list.extend(spint_tmp)
        elif n_completed == 1:
            mz_tmp = self.mz[:self.index, completed_mask].flatten()
            spint_tmp = self.spint[:self.index, completed_mask].flatten()
            mz_list.append(mz_tmp)
            spint_list.append(spint_tmp)
        roi = _list_to_roi(mz_list, spint_list, self.start)

        # remove invalid roi
        self.mz = self.mz[:, gap_mask]
        self.spint = self.spint[:, gap_mask]
        self.gap = self.gap[gap_mask]
        self.mz_mean = self.mz_mean[gap_mask]

        if self.mz_mean.size == 0:
            self.completed = True
        # TODO: check long roi
        return roi

            # list of mz and int, index convert to a named tuple of ROI

