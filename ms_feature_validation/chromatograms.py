from scipy.signal import find_peaks
import numpy as np


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


def gauss(x, mu, sigma, amp):
    """
    Normalized gaussian curve.

    Parameters
    ----------
    x : np.array
    mu : float
    sigma : float
    amp : float

    Returns
    -------
    gaussian : np.array
    """
    norm = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    gaussian = amp * norm * np.power(np.e, - 2 * ((x - mu) / sigma) ** 2)
    return gaussian


def gaussian_mixture(x, mu, sigma, amp):
    """
    Mixture of gaussian curves.

    Parameters
    ----------
    x : np.array
    mu : np.array
    sigma : np.array
    amp : np.array

    Returns
    -------
    """
    mixture = np.zeros((len(mu), x.size))
    for k, m, s, a in zip(range(len(mu)), mu, sigma, amp):
        mixture[k, :] = gauss(x, m, s, a)
    mixture = mixture.sum(axis=0)
    return mixture


def cluster(s, tolerance):
    """
    cluster values within a given tolerance

    Parameters
    ----------
    s : pandas.Series
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


