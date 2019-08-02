"""
Functions to read Raw LC-MS data using pyopenms and functions to create
chromatograms and accumulate spectra.
"""

import pyopenms
import numpy as np
from scipy.interpolate import interp1d


def read(path):
    """
    Load `path` file into an OnDiskExperiment. If the file is not indexed, load
    the file.
    Parameters
    ----------
    path

    Returns
    -------

    """
    exp_reader = pyopenms.OnDiscMSExperiment()
    try:
        exp_reader.openFile(path)
    except RuntimeError:
        msg = "{} is not an indexed mzML file, switching to MSExperiment"
        print(msg.format(path))
        exp_reader = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(path, exp_reader)
    return exp_reader


def chromatogram(msexp, mz, tolerance=0.005):
    """
    Calculates the EIC for the msexperiment
    Parameters
    ----------
    msexp: MSExp or OnDiskMSExp.
    mz: iterable[float]
        mz values used to build the EICs.
    tolerance: float.
               Tolerance to build the EICs.
    Returns
    -------
    rt, chromatograms: tuple
        rt is an array of retention times. chromatograms is an array with rows
        of EICs.
    """
    if not isinstance(mz, np.ndarray):
        mz = np.array(mz)
    mz_intervals = (np.vstack((mz - tolerance, mz + tolerance))
                    .T.reshape(mz.size * 2))
    nsp = msexp.getNrSpectra()
    chromatograms = np.zeros((mz.size, nsp))
    rt = np.zeros(nsp)
    for ksp in range(nsp):
        sp = msexp.getSpectrum(ksp)
        rt[ksp] = sp.getRT()
        mz_sp, int_sp = sp.get_peaks()
        ind_sp = np.searchsorted(mz_sp, mz_intervals)
        # elements added at the end of mz_sp raise IndexError
        ind_sp[ind_sp >= int_sp.size] = int_sp.size - 1
        chromatograms[:, ksp] = np.add.reduceat(int_sp, ind_sp)[::2]
    return rt, chromatograms


def accumulate_spectra(msexp, scans, ref, subtract=None, accumulator="sum"):
    """
    accumulates a spectra into a single spectrum.

    Parameters
    ----------
    msexp : pyopenms.MSEXperiment, pyopenms.OnDiskMSExperiment
    scans : tuple[int] : start, end
        scan range to accumulate. `start` and `end` are used as slice index.
    ref: int.
        Scan used to pass to the interpolator.
    subtract : None or Tuple[int], left, right
        Scans regions to substract. `left` must be smaller than `start` and
        `right` greater than `end`.
    accumulator : {"sum", "mean"}

    Returns
    -------
    accum_mz, accum_int : tuple[np.array]
    """
    accumulator_functions = {"sum": np.sum, "mean": np.mean}
    accumulator = accumulator_functions[accumulator]

    if subtract is not None:
        if (subtract[0] > scans[0]) or (subtract[-1] < scans[-1]):
            raise ValueError("subtract region outside scan region.")
    else:
        subtract = scans

    rows = scans[1] - subtract[0]
    mz_ref, int_ref = msexp.getSpectrum(ref).get_peaks()
    interp_int = np.zeros((rows, mz_ref.size))
    for krow, scan in zip(range(rows), range(*subtract)):
        mz_scan, int_scan = msexp.getSpectrum(scan).get_peaks()
        interpolator = interp1d(mz_scan, int_scan)
        interp_int[krow, :] = interpolator(mz_ref)

    # substract indices to match interp_int rows
    scans = scans[0] - subtract[0], scans[1] - subtract[0]
    subtract = 0, subtract[1] - subtract[0]

    accum_int = (accumulator(interp_int[scans[0]:scans[1]], axis=0)
                 - accumulator(interp_int[subtract[0]:scans[0]], axis=0)
                 - accumulator(interp_int[scans[1]:subtract[1]], axis=0))
    accum_mz = mz_ref

    return accum_mz, accum_int
