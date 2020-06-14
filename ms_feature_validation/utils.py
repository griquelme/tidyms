"""
Functions used inside several modules.

Functions
---------
gauss(x, mu, sigma, amp) : creates a gaussian curve
gaussian_mixture(x, params) : create an array with several gaussian curves
normalize(df, method) : adjust row values of a DataFrame
scale(df, method) : adjust column values of a DataFrame
transform(df, method) : perform element-wise transformation on a DataFrame
sample_to_path(samples, path) : find files in path with name equal to samples
cv(df) : Computes the coefficient of variation for columns of a DataFrame
sd(df) : Computes the std of a DataFrame and fills missing values with zeroes.
iqr(df) : Computes the inter-quartile range  of a DataFrame and fills missing
values with zeroes.
mad(df) : Computes the median absolute deviation for column in a DataFrame.
Fill missing values with zeroes.
robust_cv(df) : Estimates the coefficient of variation for columns of a
DataFrame using the MAD and median. Fill missing values with zeroes.
find_closest(x, xq) : Finds the elements in xq closest to x.

"""


import numpy as np
import pandas as pd
import os.path
from typing import Optional, Union


def gauss(x: np.ndarray, mu: float, sigma: float, amp: float):
    """
    gaussian curve.

    Parameters
    ----------.sum(axis=0)
    x : np.array
    mu : float
    sigma : float
    amp : float

    Returns
    -------
    gaussian : np.array
    """
    gaussian = amp * np.power(np.e, - 0.5 * ((x - mu) / sigma) ** 2)
    return gaussian


def gaussian_mixture(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of gaussian curves.

    Parameters
    ----------
    x : np.array
    params: np.ndarray
        parameter for each curve the shape of the array is n_curves by
        3. Each row has parameters for one curve (mu, sigma, amp)

    Returns
    -------
    mixture: np.ndarray
        array with gaussian curves. Each row is a gaussian curve. The shape
        of the array is `params.shape[0]` by `x.size`.
    """
    mixture = np.zeros((params.shape[0], x.size))
    for k_row, param in enumerate(params):
        mixture[k_row] = gauss(x, *param)
    return mixture


def normalize(df: pd.DataFrame, method: str,
              feature: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize samples using different methods.

    Parameters
    ----------
    df: pandas.DataFrame
    method: {"sum", "max", "euclidean", "feature"}
        Normalization method. `sum` normalizes using the sum along each row,
        `max` normalizes using the maximum of each row. `euclidean` normalizes
        using the euclidean norm of the row. `feature` normalizes area using
        the value of an specified feature
    feature: str, optional
        Feature used for normalization in `feature` mode.

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
    elif method == "feature":
        normalized = df.divide(df[feature], axis=0)
    else:
        msg = "method must be `sum`, `max`, `euclidean` or `feature`."
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


def transform(df: pd.DataFrame, method: str) -> pd.DataFrame:
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

    
def cv(df: pd.DataFrame):
    """Computes the Coefficient of variation for each column"""
    res = df.std() / df.mean()
    res = res.fillna(0)
    return res


def sd(df):
    """
    Computes the standard deviation for each column. Fill missing values
    with zero
    """
    res = df.std()
    res = res.fillna(0)
    return res


def iqr(df):
    """Computes the inter-quartile range for each column."""
    res = (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)
    res = res.fillna(0)
    return res


def robust_cv(df):
    """
    Estimation of the coefficient of variation using the MAD and median.
    Assumes a normal distribution.
    """

    # 1.4826 is used to estimate sigma in an unbiased way assuming a normal
    # distribution for each feature.
    res = 1.4826 * df.mad() / df.median()
    res = res.fillna(0)
    return res


def mad(df):
    """
    Computes the median absolute deviation for each column. Fill missing
    values with zero.
    """
    res = df.mad()
    res = res.fillna(0)
    return res


def _find_closest_sorted(x: np.ndarray,
                         xq: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Find the index in x closest to each xq element. Assumes that x is sorted.

    Parameters
    ----------
    x: numpy.ndarray
        Sorted vector
    xq: numpy.ndarray
        search vector

    Returns
    -------
    ind: numpy.ndarray
        array with the same size as xq with indices closest to x.

    Raises
    ------
    ValueError: when x or xq are empty.
    """

    if isinstance(xq, (float, int)):
        xq = np.array(xq)

    if (x.size == 0) or (xq.size == 0):
        msg = "`x` and `xq` must be non empty arrays"
        raise ValueError(msg)

    ind = np.searchsorted(x, xq)

    if ind.size == 1:
        if ind == 0:
            return ind
        elif ind == x.size:
            return ind - 1
        else:
            return ind - ((xq - x[ind - 1]) < (x[ind] - xq))

    else:
        # cases where the index is between 1 and x.size - 1
        mask = (ind > 0) & (ind < x.size)
        ind[mask] -= (xq[mask] - x[ind[mask] - 1]) < (x[ind[mask]] - xq[mask])
        # when the index is x.size, then the closest index is x.size -1
        ind[ind == x.size] = x.size - 1
        return ind


def find_closest(x: np.ndarray, xq: Union[np.ndarray, float, int],
                 is_sorted: bool = True) -> np.ndarray:
    if is_sorted:
        return _find_closest_sorted(x, xq)
    else:
        sorted_index = np.argsort(x)
        closest_index = _find_closest_sorted(x[sorted_index], xq)
        return sorted_index[closest_index]


def get_filename(fullpath: str) -> str:
    """
    get the filename from a full path.

    Parameters
    ----------
    fullpath: str

    Returns
    -------
    filename: str`
    """
    return os.path.splitext(os.path.split(fullpath)[1])[0]
