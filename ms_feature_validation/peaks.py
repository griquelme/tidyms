"""
Utilities to work with peaks
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from . import utils


def pick(x, y, fwhm=None, height=None, asymmetry=False,
         tailing=False, integrate=False, integrate_height=0.95):

    peak_params = dict()
    peaks, _ = find_peaks(y, height)
    if fwhm:
        fwhm_index = get_peak_widths(y, peaks, 0.5)
        peaks_fwhm = x[fwhm_index[1, :]] - x[fwhm_index[0, :]]
        peaks = peaks[(peaks_fwhm >= fwhm[0]) & (peaks_fwhm <= fwhm[1])]
        peak_params["fwhm left"] = x[fwhm_index[0, :]]
        peak_params["fwhm right"] = x[fwhm_index[1, :]]
        peak_params["fwhm"] = peaks_fwhm
        peak_params["fwhm overlap"] = find_overlap(fwhm_index)
    if asymmetry:
        peak_params["asymmetry"] = analyse_peak_shape(x, y, peaks, "asymmetry")
    if tailing:
        peak_params["tailing"] = analyse_peak_shape(x, y, peaks, "tailing")
    if integrate:
        lims = get_peak_widths(y, peaks, integrate_height)
        area = np.zeros(lims.shape[1])
        for k in range(area.size):
            y_lims = y[lims[0, k]:(lims[1, k] + 1)]
            x_lims = x[lims[0, k]:(lims[1, k] + 1)]
            area[k] = trapz(y_lims, x_lims)
        peak_params["area"] = area
        peak_params["area left"] = x[lims[0, :]]
        peak_params["area right"] = x[lims[1, :]]
        peak_params["area overlap"] = find_overlap(lims)
    peak_params["index"] = peaks
    peak_params["loc"] = x[peaks]
    peak_params["height"] = y[peaks]
    return peak_params


def get_peak_widths(y, peaks, rel_height):

    w = peak_widths(y, peaks, rel_height)
    left_index = np.round(w[2]).astype(int)
    right_index = np.round(w[3]).astype(int)
    return np.vstack((left_index, right_index))


def analyse_peak_shape(x, y, peaks, mode):
    """
    computes peak asymmetry or peak tailing factor.

    Parameters
    ----------
    x : np.array
    y : np.array
    peaks : peaks index
    mode : {"asymmetry", "tailing"}

    Returns
    -------
    factor : np.array
    """
    w = get_peak_widths(y, peaks, 0.1)
    left_width = x[peaks] - x[w[0, :]]
    right_width = x[w[1, :]] - x[peaks]
    if mode == "asymmetry":
        factor = left_width / right_width
    elif mode == "tailing":
        factor = (left_width + right_width) / (2 * left_width)
    else:
        raise ValueError("mode should be `asymmetry` or `tailing`")
    return factor


def find_overlap(intervals):
    reshaped = intervals.T.reshape(intervals.size)
    has_overlap = np.where(np.diff(reshaped)[1::2] < 0)[0]
    has_overlap = np.hstack((has_overlap, has_overlap + 1))
    overlap = np.zeros(intervals.shape[1], dtype=bool)
    overlap[has_overlap] = True
    return overlap


def fit_gaussian(x, y):
    """
    Fit multiple gaussians.

    Parameters
    ----------
    x: np.array
    y: np.array

    Returns
    -------
    result
    """
    # initial parameters guess
    peak_list = pick(x, y, fwhm=[0, 1], asymmetry=True)
    mu = peak_list["loc"]
    # correct peaks fwhm in cases of overlap
    fwhm = guess_fwhm(peak_list["fwhm"], peak_list["fwhm overlap"])
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    amp = peak_list["height"]
    guess = np.vstack((mu, sigma, amp)).T.flatten()
    popt, pcov = curve_fit(utils.gaussian_mixture, x, y, p0=guess)
    return popt, pcov


def overlap_groups(overlap):
    """
    Group contiguous overlapped peaks.

    Parameters
    ----------
    overlap: list[bool].
        returned from pick.

    Returns
    -------
    groups: list[int].
        List of peaks with overlap.
    """
    groups = list()
    group = list()
    for i, has_overlap in enumerate(overlap):
        group.append(i)
        if not has_overlap:
            groups.append(group)
            group = list()
    if group:
        groups.append(group)
    return groups


def guess_fwhm(fwhm, overlap):
    groups = overlap_groups(overlap)
    guess = np.zeros_like(fwhm)
    for group in groups:
        guess[group] = fwhm[group].mean()
    return guess
