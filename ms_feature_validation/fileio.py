import pyopenms
import numpy as np


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
                       rt is an array of retention times. chromatograms is an array with rows of EICs.
    """
    if not isinstance(mz, np.ndarray):
        mz = np.array(mz)

    nsp = msexp.getNrSpectra()
    chromatograms = list()
    rt = list()
    for k in range(nsp):
        spectrum = exp.getSpectrum(k_sp)
        if len(rt) > 1 and (rt[-1] > spectrum.getRT()):
            break
        if spectrum.getMSLevel() == mslevel:
            rt.append(spectrum.getRT())
            current_mz, current_int = spectrum.get_peaks()
            mz_index = _select_roi(current_mz, mz, tolerance)


    return int_list, mz_list, scan_times, num_scan_times


def _select_roi(x, x_roi, tolerance):
    """
    returns an mask for x with regions [x_roi - tolerance, x_roi + tolerance].
    Parameters
    ----------
    x: array[float].
       sorted array of floats
    x_roi: array[float]
       x values of interest
    tolerance: float
               tolerance to build the mask
    Returns
    -------
    x_tolerance:
    """
    x_vals = np.zeros(x_roi.size * 2, dtype=x_roi.dtype)
    x_vals[0::2] = x_roi - tolerance
    x_vals[1::2] = x_roi + tolerance
    x_tol = np.searchsorted(x, x_vals)
    x_tol = np.reshape(x_tol, (x_roi.size, 2))
    return x_tol