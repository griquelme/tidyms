from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import os.path


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


def iqr(df):
    res = (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)
    res = res.fillna(0)
    return res

def rmad(df):
    res = 1.4826 * df.mad() / df.mean()
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