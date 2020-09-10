"""
Functions and objects for working with LC-MS data

Objects
-------
Chromatogram
MSSpectrum
Roi

"""

import numpy as np
import pandas as pd
import pyopenms
from scipy.interpolate import interp1d
from typing import Optional, Iterable, Tuple, Union, List, Callable
from . import peaks
from . import validation
import bokeh.plotting
from bokeh.palettes import Set3
from collections import namedtuple

from .utils import find_closest

ms_experiment_type = Union[pyopenms.MSExperiment, pyopenms.OnDiscMSExperiment]


def reader(path: str, on_disc: bool = True):
    """
    Load `path` file into an OnDiskExperiment. If the file is not indexed, load
    the file.

    Parameters
    ----------
    path : str
        path to read mzML file from.
    on_disc : bool
        if True doesn't load the whole file on memory.

    Returns
    -------
    pyopenms.OnDiskMSExperiment or pyopenms.MSExperiment
    """
    if on_disc:
        try:
            exp_reader = pyopenms.OnDiscMSExperiment()
            exp_reader.openFile(path)
        except RuntimeError:
            msg = "{} is not an indexed mzML file, switching to MSExperiment"
            print(msg.format(path))
            exp_reader = pyopenms.MSExperiment()
            pyopenms.MzMLFile().load(path, exp_reader)
    else:
        exp_reader = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(path, exp_reader)
    return exp_reader


def make_chromatograms(ms_experiment: ms_experiment_type, mz: Iterable[float],
                       window: float = 0.005, start: Optional[int] = None,
                       end: Optional[int] = None,
                       accumulator: str = "sum"
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes extracted ion chromatograms for a list of m/z values from raw
    data.

    Parameters
    ----------
    ms_experiment : MSExp or OnDiskMSExp.
    mz : iterable[float]
        mz values used to build the EICs.
    start : int, optional
        first scan to build the chromatograms
    end : int, optional
        last scan to build the chromatograms. The scan with `number` end is not
        included in the chromatograms.
    window : positive number.
               Tolerance to build the EICs.
    accumulator : {"sum", "mean"}
        "mean" divides the intensity in the EIC using the number of points in
        the window.
    Returns
    -------
    rt : array of retention times
    eic : array with rows of EICs.
    """
    nsp = ms_experiment.getNrSpectra()

    if not isinstance(mz, np.ndarray):
        mz = np.array(mz)

    if start is None:
        start = 0

    if end is None:
        end = nsp

    # validate params
    params = {"start": start, "end": end, "window": window, "mz": mz,
              "accumulator": accumulator}
    validation.validate_make_chromatograms_params(nsp, params)

    # mz_intervals has this shape to be compatible with reduce at
    mz_intervals = (np.vstack((mz - window, mz + window))
                    .T.reshape(mz.size * 2))

    eic = np.zeros((mz.size, end - start))
    rt = np.zeros(end - start)
    for ksp in range(start, end):
        # find rt, mz and intensity values of the current scan
        sp = ms_experiment.getSpectrum(ksp)
        rt[ksp - start] = sp.getRT()
        mz_sp, int_sp = sp.get_peaks()
        ind_sp = np.searchsorted(mz_sp, mz_intervals)

        # check if the slices aren't empty
        has_mz = (ind_sp[1::2] - ind_sp[::2]) > 0
        # elements added at the end of mz_sp raise IndexError
        ind_sp[ind_sp >= int_sp.size] = int_sp.size - 1
        # this function adds the values between two consecutive indices
        tmp_eic = np.where(has_mz, np.add.reduceat(int_sp, ind_sp)[::2], 0)
        if accumulator == "mean":
            norm = ind_sp[1::2] - ind_sp[::2]
            norm[norm == 0] = 1
            tmp_eic = tmp_eic / norm
        eic[:, ksp - start] = tmp_eic
    return rt, eic


def accumulate_spectra(ms_experiment: ms_experiment_type, start: int, end: int,
                       subtract_left: Optional[int] = None,
                       subtract_right: Optional[int] = None,
                       kind: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
    """
    accumulates a spectra into a single spectrum.

    Parameters
    ----------
    ms_experiment : pyopenms.MSExperiment, pyopenms.OnDiskMSExperiment
    start : int
        start slice for scan accumulation
    end : int
        end slice for scan accumulation.
    subtract_left : int, optional
        Scans between `subtract_left` and `start` are subtracted from the
        accumulated spectrum.
    subtract_right : int, optional
        Scans between `subtract_right` and `end` are subtracted from the
        accumulated spectrum.
    kind : str
        kind of interpolator to use with scipy interp1d.

    Returns
    -------
    accumulated_mz : array of m/z values
    accumulated_int : array of cumulative intensities.
    """
    if subtract_left is None:
        subtract_left = start

    if subtract_right is None:
        subtract_right = end

    # parameter validation
    params = {"start": start, "end": end, "subtract_left": subtract_left,
              "subtract_right": subtract_right, "kind": kind}
    n_sp = ms_experiment.getNrSpectra()
    validation.validate_accumulate_spectra_params(n_sp, params)

    # creates a common mz reference value for the scans
    mz, _ = ms_experiment.getSpectrum(start).get_peaks()
    accumulated_mz = _get_uniform_mz(mz)
    accumulated_sp = np.zeros_like(accumulated_mz)

    # interpolates each scan to the reference. Removes values outside the
    # min and max of the reference.
    for scan in range(subtract_left, subtract_right):
        mz_scan, int_scan = ms_experiment.getSpectrum(scan).get_peaks()
        mz_min, mz_max = mz_scan.min(), mz_scan.max()
        min_ind, max_ind = np.searchsorted(accumulated_mz, [mz_min, mz_max])
        interpolator = interp1d(mz_scan, int_scan, kind=kind)
        tmp_sp = interpolator(accumulated_mz[min_ind:max_ind])
        # accumulate scans
        if (scan < start) or (scan > end):
            accumulated_sp[min_ind:max_ind] -= tmp_sp
        else:
            accumulated_sp[min_ind:max_ind] += tmp_sp

    is_positive_sp = accumulated_sp > 0
    accumulated_mz = accumulated_mz[is_positive_sp]
    accumulated_sp = accumulated_sp[is_positive_sp]
    return accumulated_mz, accumulated_sp


def _get_uniform_mz(mz: np.ndarray):
    """returns a new uniformly sampled m/z array."""
    mz_min = mz.min()
    mz_max = mz.max()
    mz_res = np.diff(mz).min()
    uniform_mz = np.arange(mz_min, mz_max, mz_res)
    return uniform_mz


def make_widths_lc(mode: str) -> np.ndarray:
    """
    Create an array of widths to use in CWT peak picking of LC data.

    Parameters
    ----------
    mode: {"hplc", "uplc"}

    Returns
    -------
    widths: array
    """
    if mode == "uplc":
        widths = [np.linspace(0.25, 5, 20), np.linspace(6, 20, 8),
                  np.linspace(25, 60, 8)]
        widths = np.hstack(widths)
    elif mode == "hplc":
        widths = [np.linspace(1, 10, 20), np.linspace(10, 30, 8),
                  np.linspace(40, 90, 8)]
        widths = np.hstack(widths)
    else:
        msg = "Valid modes are `hplc` or `uplc`."
        raise ValueError(msg)
    return widths


def make_widths_ms(mode: str) -> np.ndarray:
    """
    Create an array of widths to use in CWT peak picking of MS data.

    Parameters
    ----------
    mode : {"qtof", "orbitrap"}

    Returns
    -------
    widths : array
    """
    if mode == "qtof":
        min_width = 0.005
        middle = 0.1
        max_width = 0.2
    elif mode == "orbitrap":
        min_width = 0.0005
        middle = 0.001
        max_width = 0.005
    else:
        msg = "mode must be `orbitrap` or `qtof`"
        raise ValueError(msg)
    # [:-1] prevents repeated value
    widths = np.hstack((np.linspace(min_width, middle, 20)[:-1],
                        np.linspace(middle, max_width, 10)))
    return widths


def get_lc_cwt_params(mode: str) -> dict:
    """
    Return sane default values for performing CWT based peak picking on LC data.

    Parameters
    ----------
    mode : {"hplc", "uplc"}
        HPLC assumes typical experimental conditions for HPLC experiments:
        longer columns with particle size greater than 3 micron. UPLC is for
        data acquired with short columns with particle size lower than 3 micron.

    Returns
    -------
    cwt_params : dict
        parameters to pass to .peak.pick_cwt function.
    """
    cwt_params = {"snr": 10, "min_length": 5, "max_distance": 1,
                  "gap_threshold": 1, "estimators": "default"}

    if mode == "hplc":
        cwt_params["min_width"] = 10
        cwt_params["max_width"] = 90
    elif mode == "uplc":
        cwt_params["min_width"] = 5
        cwt_params["max_width"] = 60
    else:
        msg = "`mode` must be `hplc` or `uplc`"
        raise ValueError(msg)
    return cwt_params


def get_ms_cwt_params(mode: str) -> dict:
    """
    Return sane default values for performing CWT based peak picking on MS data.

    Parameters
    ----------
    mode : {"qtof", "orbitrap"}
        qtof assumes a peak width in the range of 0.01-0.05 Da. `orbitrap`
        assumes a peak width in the range of 0.001-0.005 Da.

    Returns
    -------
    cwt_params : dict
        parameters to pass to .peak.pick_cwt function.
    """
    cwt_params = {"snr": 10, "min_length": 5, "gap_threshold": 1,
                  "estimators": "default"}

    if mode == "qtof":
        cwt_params["min_width"] = 0.01
        cwt_params["max_width"] = 0.2
        cwt_params["max_distance"] = 0.005
    elif mode == "orbitrap":
        cwt_params["min_width"] = 0.0005
        cwt_params["max_width"] = 0.005
        cwt_params["max_distance"] = 0.0025
    else:
        msg = "`mode` must be `qtof` or `orbitrap`"
        raise ValueError(msg)
    return cwt_params


def get_roi_params(separation: str = "uplc", instrument: str = "qtof"):
    """
    Creates a dictionary with recommended parameters for the make_roi function
    in different use cases.

    Parameters
    ----------
    separation : {"uplc", "hplc"}
        Mode in which the data was acquired. Used to set minimum length of the
        roi and number of missing values.
    instrument : {"qtof", "orbitrap"}
        Type of MS instrument. Used to set the tolerance.

    Returns
    -------
    roi_parameters : dict
    """
    roi_params = {"min_intensity": 500, "multiple_match": "reduce"}

    if separation == "uplc":
        roi_params.update({"max_missing": 1, "min_length": 10})
    elif separation == "hplc":
        roi_params.update({"max_missing": 1, "min_length": 20})
    else:
        msg = "valid `separation` are uplc and hplc"
        raise ValueError(msg)

    if instrument == "qtof":
        roi_params.update({"tolerance": 0.01})
    elif instrument == "orbitrap":
        roi_params.update({"tolerance": 0.005})
    else:
        msg = "valid `instrument` are qtof and orbitrap"
        raise ValueError(msg)

    roi_params["mode"] = separation

    return roi_params


def _find_isotopic_distribution_aux(mz: np.ndarray, mz_ft: float,
                                    q: int, n_isotopes: int,
                                    tol: float):
    """
    Finds the isotopic distribution for a given charge state. Auxiliary function
    to find_isotopic_distribution.
    Isotopes are searched based on the assumption that the mass difference
    is due to the presence of a 13C atom.

    Parameters
    ----------
    mz : numpy.ndarray
        List of peaks
    mz_ft : float
        Monoisotopic mass
    q : charge state of the ion
    n_isotopes : int
        Number of isotopes to search in the distribution
    tol: float
        Mass tolerance, in absolute units

    Returns
    -------
    match_ind : np.ndarray
        array of indices for the isotopic distribution.
    """
    # TODO: Remove this function when the isotope finder module is added.

    mono_index = find_closest(mz, mz_ft)
    mz_mono = mz[mono_index]
    if abs(mz_mono - mz_ft) > tol:
        match_ind = np.array([])
    else:
        dm = 1.003355
        mz_theoretic = mz_mono + np.arange(n_isotopes) * dm / q
        closest_ind = find_closest(mz, mz_theoretic)
        match_ind = np.where(np.abs(mz[closest_ind] - mz_theoretic) <= tol)[0]
        match_ind = closest_ind[match_ind]
    return match_ind


def _find_isotopic_distribution(mz: np.ndarray, mz_mono: float,
                                q_max: int, n_isotopes: int,
                                tol: float):
    """
    Finds the isotopic distribution within charge lower than q_max.
    Isotopes are searched based on the assumption that the mass difference
    is due to the presence of a 13C atom. If multiple charge states are
    compatible with an isotopic distribution, the charge state with the largest
    number of isotopes detected is kept.

    Parameters
    ----------
    mz : numpy.ndarray
        List of peaks
    mz_mono : float
        Monoisotopic mass
    q_max : int
        max charge to analyze
    n_isotopes : int
        Number of isotopes to search in the distribution
    tol : float
        Mass tolerance, in absolute units

    Returns
    -------
    best_peaks: numpy.ndarray

    """
    # TODO: Remove this function when the isotope finder module is added.
    best_peaks = np.array([], dtype=int)
    n_peaks = 0
    for q in range(1, q_max + 1):
        tmp = _find_isotopic_distribution_aux(mz, mz_mono, q,
                                              n_isotopes, tol)
        if tmp.size > n_peaks:
            best_peaks = tmp
    return best_peaks


class Chromatogram:
    """
    Representation of a chromatogram. Manages plotting and peak picking.

    Attributes
    ----------
    rt : array
        retention time in each scan.
    spint : array
        intensity in each scan.
    mode : str
        used to set default parameter for peak picking.

    """

    def __init__(self, rt: np.ndarray, spint: np.ndarray,
                 mode: Optional[str] = None):
        """
        Constructor of the Chromatogram.

        Parameters
        ----------
        spint : array of non negative numbers.
            Intensity values of each scan
        rt : array of positive numbers.
            Retention time values.
        mode : {"uplc", "hplc"}, optional
            used to set default parameters in peak picking. If None, `mode` is
            set to uplc.
        """
        if mode is None:
            self.mode = "uplc"
        elif mode in ["uplc", "hplc"]:
            self.mode = mode
        else:
            msg = "mode must be None, uplc or hplc"
            raise ValueError(msg)

        self.rt = rt
        self.spint = spint
        self.peaks = None

    def find_peaks(self, cwt_params: Optional[dict] = None) -> dict:
        """
        Find peaks with the modified version of the cwt algorithm described in
        the centWave algorithm. Peaks are added to the peaks
        attribute of the Chromatogram object.

        Parameters
        ----------
        cwt_params : dict
            key-value parameters to overwrite the defaults in the pick_cwt
            function. The default are obtained using the mode attribute.

        Returns
        -------
        params : dict
            dictionary of peak parameters

        See Also
        --------
        peaks.detect_peaks : peak detection using the CWT algorithm.
        lcms.get_lc_cwt_params : set default parameters for pick_cwt.

        """
        default_params = get_lc_cwt_params(self.mode)

        if cwt_params:
            default_params.update(cwt_params)

        widths = make_widths_lc(self.mode)
        peak_list, peak_params = \
            peaks.detect_peaks(self.rt, self.spint, widths, **default_params)
        self.peaks = peak_list
        return peak_params

    def plot(self, draw: bool = True, fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        Plot the chromatogram.

        Parameters
        ----------
        draw : bool, optional
            if True run bokeh show function.
        fig_params : dict
            key-value parameters to pass into bokeh figure function.
        line_params : dict
            key-value parameters to pass into bokeh line function.

        Returns
        -------
        bokeh Figure
        """
        default_line_params = {"line_width": 1, "line_color": "black",
                               "alpha": 0.8}
        cmap = Set3[12]

        if line_params is None:
            line_params = default_line_params
        else:
            for params in line_params:
                default_line_params[params] = line_params[params]
            line_params = default_line_params

        default_fig_params = {"aspect_ratio": 1.5}
        if fig_params is None:
            fig_params = default_fig_params
        else:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params

        fig = bokeh.plotting.figure(**fig_params)
        fig.line(self.rt, self.spint, **line_params)
        if self.peaks:
            for k, peak in enumerate(self.peaks):
                fig.varea(self.rt[peak.start:(peak.end + 1)],
                          self.spint[peak.start:(peak.end + 1)], 0,
                          fill_alpha=0.8, fill_color=cmap[k % 12])
                # k % 12 is used to cycle over the colormap

        #  figure appearance
        fig.xaxis.axis_label = "Rt [s]"
        fig.yaxis.axis_label = "intensity [au]"
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.yaxis.formatter.precision = 2
        fig.xaxis.axis_label_text_font_style = "bold"

        if draw:
            bokeh.plotting.show(fig)
        return fig


class MSSpectrum:
    """
    Representation of a Mass Spectrum in profile mode. Manages conversion to
    centroids and plotting of data.

    Attributes
    ----------
    mz : array of m/z values
    spint : array of intensity values.
    mode : str
        MS instrument type. Used to set default values in peak picking.

    """
    def __init__(self, mz: np.ndarray, spint: np.ndarray,
                 mode: Optional[str] = None):
        """
        Constructor of the MSSpectrum.

        Parameters
        ----------
        mz: array
            m/z values.
        spint: array
            intensity values.

        """
        self.mz = mz
        self.spint = spint
        self.peaks = None

        if mode is None:
            self.mode = "qtof"
        elif mode in ["qtof", "orbitrap"]:
            self.mode = mode
        else:
            msg = "mode must be qtof or orbitrap"
            raise ValueError(msg)

    def find_centroids(self, snr: float = 10,
                       min_distance: Optional[float] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Find centroids in the spectrum.

        Centroids are found as local maxima above a noise value. See notes for
        implementation details.

        Parameters
        ----------
        snr : positive number
            Minimum signal to noise ratio of the peaks. Overwrites values
            set by mode.
        min_distance : positive number, optional
            Minimum distance between consecutive peaks. If None, sets the value
            to 0.01 if the `mode` attribute is qtof. If the `mode` is orbitrap,
            sets the value to 0.005

        Returns
        -------
        centroids : array of peak centroids
        area : array of peak area
        centroid_index : index of the centroids in `mz`

        Notes
        -----
        Peaks are found as local maxima in the signal. To remove low intensity
        values, a baseline and noise is estimated assuming that y has additive
        contributions from signal,  baseline and noise:

        .. math::

            y[n] = s[n] + b[n] + \epsilon

        Where :math:`\epsilon \sim N(0, \sigma)`. A peak is valid only if

        .. math::

            \frac{y[n_{peak}] - b[n_{peak}]}{\sigma} \geq SNR

        The extension of the peak is computed as the closest minimum to the
        peak. If two peaks are closer than `min_distance`, the peaks are merged.

        """
        params = {"snr": snr}

        if min_distance is not None:
            params["min_distance"] = min_distance
        else:
            if self.mode == "qtof":
                md = 0.01
            elif self.mode == "orbitrap":
                md = 0.005
            else:
                raise ValueError
            params["min_distance"] = md

        centroids, area, centroid_index = \
            peaks.find_centroids(self.mz, self.spint, **params)

        return centroids, area, centroid_index

    def plot(self, draw: bool = True, fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None) -> bokeh.plotting.Figure:
        """
        Plot the spectrum.

        Parameters
        ----------
        draw : bool, optional
            if True run bokeh show function.
        fig_params : dict
            key-value parameters to pass into bokeh figure function.
        line_params : dict
            key-value parameters to pass into bokeh line function.

        Returns
        -------
        bokeh Figure
        """
        default_line_params = {"line_width": 1, "line_color": "black",
                               "alpha": 0.8}
        cmap = Set3[12]

        if line_params is None:
            line_params = default_line_params
        else:
            for params in line_params:
                default_line_params[params] = line_params[params]
            line_params = default_line_params

        default_fig_params = {"aspect_ratio": 1.5}
        if fig_params is None:
            fig_params = default_fig_params
        else:
            default_fig_params.update(fig_params)
            fig_params = default_fig_params

        fig = bokeh.plotting.figure(**fig_params)
        fig.line(self.mz, self.spint, **line_params)
        if self.peaks:
            for k, peak in enumerate(self.peaks):
                fig.varea(self.mz[peak.start:(peak.end + 1)],
                          self.spint[peak.start:(peak.end + 1)], 0,
                          fill_alpha=0.8, fill_color=cmap[k % 12])
                # k % 12 is used to cycle over the colormap

        #  figure appearance
        fig.xaxis.axis_label = "m/z"
        fig.yaxis.axis_label = "intensity [au]"
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.yaxis.formatter.precision = 2
        fig.xaxis.axis_label_text_font_style = "bold"

        if draw:
            bokeh.plotting.show(fig)
        return fig


_TempRoi = namedtuple("TempRoi", ["mz", "sp", "scan"])


def _make_empty_temp_roi():
    return _TempRoi(mz=list(), sp=list(), scan=list())


class Roi(Chromatogram):
    """
    mz traces where a chromatographic peak may be found. Subclassed from
    Chromatogram. To be used with the detect_features method of MSData.

    Attributes
    ----------
    rt : array
        retention time in each scan.
    spint : array
        intensity in each scan.
    mz : array
        m/z in each scan.
    first_scan : int
        first scan in the raw data where the ROI was detected.

    """
    def __init__(self, spint: np.ndarray, mz: np.ndarray, rt: np.ndarray,
                 first_scan: int, mode: Optional[str] = None):
        super(Roi, self).__init__(rt, spint, mode=mode)
        self.mz = mz
        self.first_scan = first_scan

    def fill_nan(self):
        """
        fill missing intensity values using linear interpolation.
        """
        missing = np.isnan(self.spint)
        interpolator = interp1d(self.rt[~missing], self.spint[~missing])
        mz_mean = np.nanmean(self.mz)
        self.mz[missing] = mz_mean
        self.spint[missing] = interpolator(self.rt[missing])

    def get_peaks_mz(self):
        """
        Computes the weighted mean of the m/z for each peak and the m/z
        standard deviation
        Returns
        -------
        mean_mz : array
        mz_std : array
        """
        mz_std = np.zeros(len(self.peaks))
        mz_mean = np.zeros(len(self.peaks))
        for k, peak in enumerate(self.peaks):
            # missing = np.isnan(self.mz[peak.start:peak.end + 1])
            # print("wat")
            # peak_mz = self.mz[peak.start:peak.end + 1][~missing]
            # peak_spint = self.spint[peak.start:peak.end + 1][~missing]
            peak_mz = self.mz[peak.start:peak.end + 1]
            peak_spint = self.spint[peak.start:peak.end + 1]
            mz_mean[k] = np.average(peak_mz, weights=peak_spint)
            mz_std[k] = peak_mz.std()
        return mz_mean, mz_std


class _RoiProcessor:
    """
    Class used by make_roi function to generate Roi instances.

    Attributes
    ----------
    mz_mean: numpy.ndarray
        mean value of mz for a given row in mz_array. Used to add new values
        based on a tolerance. its updated after adding a new column
    n_missing: numpy.ndarray
        number of consecutive missing values. Used to detect finished rois
    roi: list[ROI]
    """

    def __init__(self, mz_seed: np.ndarray, max_missing: int = 1,
                 min_length: int = 5, min_intensity: float = 0,
                 tolerance: float = 0.005, multiple_match: str = "closest",
                 mz_reduce: Union[str, Callable] = "mean",
                 sp_reduce: Union[str, Callable] = "sum",
                 mode: Optional[str] = None):
        """

        Parameters
        ----------
        mz_seed: numpy.ndarray
            initial values to build rois
        max_missing: int
            maximum number of missing consecutive values. when a row surpass
            this number the roi is flagged as finished.
        min_length: int
            The minimum length of a finished roi to be considered valid before
            being added to the roi list.
        min_intensity: float
        tolerance: float
            mz tolerance used to connect values.
        multiple_match: {"closest", "reduce"}
            how to match peaks when there is more than one match. If mode is
            `closest`, then the closest peak is assigned as a match and the
            others are assigned to no match. If mode is `reduce`, then a unique
            mz and intensity value is generated using the reduce function in
            `mz_reduce` and `spint_reduce` respectively.
        mz_reduce: str or callable
            function used to reduce mz values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is reduce. See the following prototype:

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

        sp_reduce: str or callable
            function used to reduce spint values. Can be a function accepting
            numpy arrays and returning numbers. Only used when `multiple_match`
            is reduce. To use custom functions see the prototype shown on
            `mz_reduce`.
        mode: str, optional
            Mode used to create ROI.
        """
        if len(mz_seed.shape) != 1:
            msg = "array must be a vector"
            raise ValueError(msg)

        if multiple_match not in ["closest", "reduce"]:
            msg = "Valid modes are closest or reduce"
            raise ValueError(msg)

        if mz_reduce == "mean":
            self._mz_reduce = np.mean
        else:
            self._mz_reduce = mz_reduce

        if sp_reduce == "mean":
            self._spint_reduce = np.mean
        elif sp_reduce == "sum":
            self._spint_reduce = np.sum
        else:
            self._spint_reduce = sp_reduce

        self.mz_mean = mz_seed.copy()
        self.roi_index = np.arange(mz_seed.size)
        self.n_missing = np.zeros_like(mz_seed, dtype=int)
        self.max_intensity = np.zeros_like(mz_seed)
        self.length = np.zeros_like(mz_seed, dtype=int)
        self.index = 0
        self.temp_roi_dict = {x: _make_empty_temp_roi() for x in self.roi_index}
        self.roi = list()
        self.min_intensity = min_intensity
        self.max_missing = max_missing
        self.min_length = min_length
        self.tolerance = tolerance
        self.multiple_match = multiple_match
        self.mode = mode

    def add(self, mz: np.ndarray, sp: np.ndarray, targeted: bool = False):
        """
        Adds new mz and spint values to temporal roi.
        """

        # find matching values
        match_index, mz_match, sp_match, mz_no_match, sp_no_match = \
            _match_mz(self.mz_mean, mz, sp, self.tolerance,
                      self.multiple_match, self._mz_reduce, self._spint_reduce)

        for k, k_mz, k_sp in zip(match_index, mz_match, sp_match):
            k_temp_roi = self.temp_roi_dict[self.roi_index[k]]
            k_temp_roi.mz.append(k_mz)
            k_temp_roi.sp.append(k_sp)
            k_temp_roi.scan.append(self.index)

        # update mz_mean and missing values
        updated_mean = ((self.mz_mean[match_index] * self.length[match_index]
                         + mz_match) / (self.length[match_index] + 1))

        self.length[match_index] += 1
        self.n_missing += 1
        self.n_missing[match_index] = 0
        self.max_intensity[match_index] = \
            np.maximum(self.max_intensity[match_index], sp_match)
        if not targeted:
            self.mz_mean[match_index] = updated_mean
            self.extend(mz_no_match, sp_no_match)
        self.index += 1

    def append_to_roi(self, rt: np.ndarray, targeted: bool = False):
        """
        Remove completed ROI. Valid ROI are appended toi roi attribute.
        """

        # check completed rois
        is_completed = self.n_missing > self.max_missing

        # the most common case are short rois that must be discarded
        is_valid_roi = ((self.length >= self.min_length) &
                        (self.max_intensity >= self.min_intensity))

        # add completed roi
        completed_index = np.where(is_completed)[0]
        for ind in completed_index:
            roi_ind = self.roi_index[ind]
            finished_roi = self.temp_roi_dict.pop(roi_ind)
            if is_valid_roi[ind]:
                roi = tmp_roi_to_roi(finished_roi, rt, mode=self.mode)
                self.roi.append(roi)
        if targeted:
            self.n_missing[is_completed] = 0
            self.length[is_completed] = 0
            self.max_intensity[is_completed] = 0
            max_roi_ind = self.roi_index.max()
            n_completed = is_completed.sum()
            new_indices = np.arange(max_roi_ind + 1,
                                    max_roi_ind + 1 + n_completed)
            self.roi_index[is_completed] = new_indices
            new_tmp_roi = {k: _make_empty_temp_roi() for k in new_indices}
            self.temp_roi_dict.update(new_tmp_roi)
        else:
            self.mz_mean = self.mz_mean[~is_completed]
            self.n_missing = self.n_missing[~is_completed]
            self.length = self.length[~is_completed]
            self.roi_index = self.roi_index[~is_completed]
            self.max_intensity = self.max_intensity[~is_completed]

    def extend(self, mz: np.ndarray, sp: np.ndarray):
        """adds new mz values to mz_mean"""
        max_index = self.roi_index.max()
        new_indices = np.arange(mz.size) + max_index + 1
        mz_mean_tmp = np.hstack((self.mz_mean, mz))
        roi_index_tmp = np.hstack((self.roi_index, new_indices))
        sorted_index = np.argsort(mz_mean_tmp)
        n_missing_tmp = np.zeros_like(new_indices, dtype=int)
        n_missing_tmp = np.hstack((self.n_missing, n_missing_tmp))
        length_tmp = np.ones_like(new_indices, dtype=int)
        length_tmp = np.hstack((self.length, length_tmp))
        max_int_tmp = np.zeros_like(new_indices, dtype=float)
        max_int_tmp = np.hstack((self.max_intensity, max_int_tmp))

        for k_index, k_mz, k_sp in zip(new_indices, mz, sp):
            new_roi = _TempRoi(mz=[k_mz], sp=[k_sp], scan=[self.index])
            self.temp_roi_dict[k_index] = new_roi
        self.mz_mean = mz_mean_tmp[sorted_index]
        self.roi_index = roi_index_tmp[sorted_index]
        self.n_missing = n_missing_tmp[sorted_index]
        self.length = length_tmp[sorted_index]
        self.max_intensity = max_int_tmp[sorted_index]

    def flag_as_completed(self):
        self.n_missing[:] = self.max_missing + 1


def _compare_max(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """
    returns the element-wise maximum between old and new

    Parameters
    ----------
    old: numpy.ndarray
    new: numpy.ndarray
        can have nan

    Returns
    -------
    numpy.ndarray
    """
    new[np.isnan(new)] = 0
    return np.maximum(old, new)


def _match_mz(mz1: np.ndarray, mz2: np.ndarray, sp2: np.ndarray,
              tolerance: float, mode: str, mz_reduce: Callable,
              sp_reduce: Callable):
    """
    aux function to add method in _RoiProcessor. Find matched values.

    Parameters
    ----------
    mz1: numpy.ndarray
        _RoiProcessor mz_mean
    mz2: numpy.ndarray
        mz values to match
    sp2: numpy.ndarray
        intensity values associated to mz2
    tolerance: float
        tolerance used to match values
    mode: {"closest", "merge"}
        Behaviour when more more than one peak in mz2 matches with a given peak
        in mz1. If mode is `closest`, then the closest peak is assigned as a
        match and the others are assigned to no match. If mode is `merge`, then
        a unique mz and int value is generated using the average of the mz and
        the sum of the intensities.

    Returns
    ------
    match_index: numpy.ndarray
        index when of peaks matching in mz1.
    mz_match: numpy.ndarray
        values of mz2 that matches with mz1
    sp_match: numpy.ndarray
        values of sp2 that matches with mz1
    mz_no_match: numpy.ndarray
    sp_no_match: numpy.ndarray
    """
    closest_index = find_closest(mz1, mz2)
    dmz = np.abs(mz1[closest_index] - mz2)
    match_mask = (dmz <= tolerance)
    no_match_mask = ~match_mask
    match_index = closest_index[match_mask]

    # check multiple_matches
    unique, first_index, count_index = np.unique(match_index,
                                                 return_counts=True,
                                                 return_index=True)

    # set match values
    match_index = unique
    sp_match = sp2[match_mask][first_index]
    mz_match = mz2[match_mask][first_index]

    # compute matches for duplicates
    multiple_match_mask = count_index > 1
    first_index = first_index[multiple_match_mask]
    if first_index.size > 0:
        first_index_index = np.where(count_index > 1)[0]
        count_index = count_index[multiple_match_mask]
        iterator = zip(first_index_index, first_index, count_index)
        if mode == "closest":
            rm_index = list()   # list of duplicate index to remove
            mz_replace = list()
            spint_replace = list()
            for first_ind, index, count in iterator:
                # check which of the duplicate is closest, the rest are removed
                closest = \
                    np.argmin(dmz[match_mask][index:(index + count)]) + index
                mz_replace.append(mz2[match_mask][closest])
                spint_replace.append(sp2[match_mask][closest])
                remove = np.arange(index, index + count)
                remove = np.setdiff1d(remove, closest)
                rm_index.extend(remove)
            no_match_mask[rm_index] = True
            mz_match[first_index_index] = mz_replace
            sp_match[first_index_index] = spint_replace
        elif mode == "reduce":
            for first_ind, index, count in iterator:
                # check which of the duplicate is closest
                mz_multiple_match = mz2[match_mask][index:(index + count)]
                sp_multiple_match = sp2[match_mask][index:(index + count)]
                mz_match[first_ind] = mz_reduce(mz_multiple_match)
                sp_match[first_ind] = sp_reduce(sp_multiple_match)
        else:
            msg = "mode must be `closest` or `merge`"
            raise ValueError(msg)

    mz_no_match = mz2[no_match_mask]
    sp_no_match = sp2[no_match_mask]
    return match_index, mz_match, sp_match, mz_no_match, sp_no_match


def tmp_roi_to_roi(tmp_roi: _TempRoi, rt: np.ndarray,
                   mode: Optional[str] = None) -> Roi:
    first_scan = tmp_roi.scan[0]
    last_scan = tmp_roi.scan[-1]
    size = last_scan + 1 - first_scan
    mz_tmp = np.ones(size) * np.nan
    spint_tmp = mz_tmp.copy()
    tmp_index = np.array(tmp_roi.scan) - tmp_roi.scan[0]
    rt_tmp = rt[first_scan:(last_scan + 1)]
    mz_tmp[tmp_index] = tmp_roi.mz
    spint_tmp[tmp_index] = tmp_roi.sp
    roi = Roi(spint_tmp, mz_tmp, rt_tmp, first_scan, mode=mode)
    return roi


def make_roi(ms_experiment: ms_experiment_type, tolerance: float,
             max_missing: int, min_length: int, min_intensity: float,
             multiple_match: str, targeted_mz: Optional[np.ndarray] = None,
             start: Optional[int] = None, end: Optional[int] = None,
             mz_reduce: Union[str, Callable] = "mean",
             sp_reduce: Union[str, Callable] = "sum",
             mode: Optional[str] = None
             ) -> List[Roi]:
    """
    Make Region of interest from MS data in centroid mode.
    Used by MSData to as the first step of the centWave algorithm.

    Parameters
    ----------
    ms_experiment: pyopenms.MSExperiment
    max_missing : int
        maximum number of missing consecutive values. when a row surpass this
        number the roi is considered as finished and is added to the roi list if
        it meets the length and intensity criteria.
    min_length : int
        The minimum length of a roi to be considered valid.
    min_intensity : float
        Minimum intensity in a roi to be considered valid.
    tolerance : float
        mz tolerance to connect values across scans
    start : int, optional
        First scan to analyze. If None starts at scan 0
    end : int, optional
        Last scan to analyze. If None, uses the last scan number.
    multiple_match : {"closest", "reduce"}
        How to match peaks when there is more than one match. If mode is
        `closest`, then the closest peak is assigned as a match and the
        others are assigned to no match. If mode is `reduce`, then unique
        mz and intensity values are generated using the reduce function in
        `mz_reduce` and `sp_reduce` respectively.
    mz_reduce : "mean" or Callable
        function used to reduce mz values. Can be a function accepting
        numpy arrays and returning numbers. Only used when `multiple_match`
        is reduce. See the following prototype:

        .. code-block:: python

            def mz_reduce(mz_match: np.ndarray) -> float:
                pass

    sp_reduce : {"mean", "sum"} or Callable
        function used to reduce intensity values. Can be a function accepting
        numpy arrays and returning numbers. Only used when `multiple_match`
        is reduce. To use custom functions see the prototype shown on
        `mz_reduce`.
    targeted_mz : numpy.ndarray, optional
        if a list of mz is provided, roi are searched only using this list.

    mode : str, optional
        mode used to create Roi objects.

    Returns
    -------
    roi: list[Roi]

    Notes
    -----
    To create a ROI, m/z values in consecutive scans are connected if they are
    within the tolerance`. If there's more than one possible m/z value to
    connect in the next scan, two different strategies are available, using the
    `multiple_match` parameter: If "closest" is used, then m/z values are
    matched to the closest ones, and the others are used to create new ROI. If
    "reduce" is used, then all values within the tolerance are combined. m/z and
    intensity values are combined using the `mz_reduce`  and `sp_reduce`
    parameters respectively. If no matching value has be found in a scan, a NaN
    is added to the ROI. If no matching values are found in `max_missing`
    consecutive scans the ROI is flagged as finished. In this stage, two
    checks are made before the ROI is considered valid:

    1.  The number of non missing values must be higher than `min_length`.
    2.  The maximum intensity value in the ROI must be higher than
        `min_intensity`.

    If the two conditions are meet, the ROI is added to the list of valid ROI.

    """
    if start is None:
        start = 0

    if end is None:
        end = ms_experiment.getNrSpectra()

    if targeted_mz is None:
        mz_seed, _ = ms_experiment.getSpectrum(start).get_peaks()
        targeted = False
    else:
        mz_seed = targeted_mz
        targeted = True

    size = end - start
    rt = np.zeros(size)
    processor = _RoiProcessor(mz_seed, max_missing=max_missing,
                              min_length=min_length,
                              min_intensity=min_intensity, tolerance=tolerance,
                              multiple_match=multiple_match,
                              mz_reduce=mz_reduce, sp_reduce=sp_reduce,
                              mode=mode)
    for k_scan in range(start, end):
        sp = ms_experiment.getSpectrum(k_scan)
        rt[k_scan - start] = sp.getRT()
        mz, spint = sp.get_peaks()
        processor.add(mz, spint, targeted=targeted)
        processor.append_to_roi(rt, targeted=targeted)

    # add roi not completed during last scan
    processor.flag_as_completed()
    processor.append_to_roi(rt)
    return processor.roi


def detect_roi_peaks(roi: List[Roi],
                     cwt_params: Optional[dict] = None) -> pd.DataFrame:
    if cwt_params is None:
        cwt_params = dict()

    roi_index_list = list()
    peak_index_list = list()
    mz_mean_list = list()
    mz_std_list = list()
    peak_params = list()

    for roi_index, k_roi in enumerate(roi):
        k_roi.fill_nan()
        k_params = k_roi.find_peaks(cwt_params=cwt_params)
        n_features = len(k_params)
        peak_params.extend(k_params)
        k_mz_mean, k_mz_std = k_roi.get_peaks_mz()
        roi_index_list.append([roi_index] * n_features)
        peak_index_list.append(range(n_features))
        mz_mean_list.append(k_mz_mean)
        mz_std_list.append(k_mz_std)

    roi_index_list = np.hstack(roi_index_list)
    peak_index_list = np.hstack(peak_index_list)
    mz_mean_list = np.hstack(mz_mean_list)
    mz_std_list = np.hstack(mz_std_list)

    peak_params = pd.DataFrame(data=peak_params)
    peak_params = peak_params.rename(columns={"loc": "rt"})
    peak_params["mz"] = mz_mean_list
    peak_params["mz std"] = mz_std_list
    peak_params["roi index"] = roi_index_list
    peak_params["peak index"] = peak_index_list
    peak_params = peak_params.dropna(axis=0)
    return peak_params
