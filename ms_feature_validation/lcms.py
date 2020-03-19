"""
Functions for working with LC-MS data
"""

import numpy as np
import pandas as pd
import pyopenms
from scipy.interpolate import interp1d
from typing import Optional, Iterable, Tuple, Union
from . import peaks
import bokeh.plotting
from bokeh.palettes import Set3
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool

from .utils import find_closest

msexperiment = Union[pyopenms.MSExperiment, pyopenms.OnDiscMSExperiment]


def reader(path: str, on_disc: bool = True):
    """
    Load `path` file into an OnDiskExperiment. If the file is not indexed, load
    the file.
    Parameters
    ----------
    path: str
        path to read mzML file from.
    on_disc:
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


def chromatogram(msexp: msexperiment, mz: Iterable[float],
                 tolerance: float = 0.005, start: Optional[int] = None,
                 end: Optional[int] = None,
                 accumulator: str = "sum") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the EIC for the msexperiment
    Parameters
    ----------
    msexp: MSExp or OnDiskMSExp.
    mz: iterable[float]
        mz values used to build the EICs.
    start: int, optional
        first scan to build the chromatogram
    end: int, optional
        last scan to build the chromatogram.
    tolerance: float.
               Tolerance to build the EICs.
    accumulator: {"sum", "mean"}
        "mean" divides the intensity in the EIC using the number of points in
        the window.
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

    if start is None:
        start = 0

    if end is None:
        end = nsp

    chromatograms = np.zeros((mz.size, end - start))
    rt = np.zeros(end - start)
    for ksp in range(start, end):
        sp = msexp.getSpectrum(ksp)
        rt[ksp] = sp.getRT()
        mz_sp, int_sp = sp.get_peaks()
        ind_sp = np.searchsorted(mz_sp, mz_intervals)
        # elements added at the end of mz_sp raise IndexError
        ind_sp[ind_sp >= int_sp.size] = int_sp.size - 1
        chromatograms[:, ksp] = np.add.reduceat(int_sp, ind_sp)[::2]
        if accumulator == "mean":
            norm = ind_sp[1::2] - ind_sp[::2]
            norm[norm == 0] = 1
            chromatograms[:, ksp] = chromatograms[:, ksp] / norm
        elif accumulator == "sum":
            pass
        else:
            msg = "accumulator possible values are `mean` and `sum`."
            raise ValueError(msg)
    return rt, chromatograms


def accumulate_spectra(msexp: msexperiment, start: int,
                       end: int, subtract: Optional[Tuple[int, int]] = None,
                       kind: str = "linear",
                       accumulator: str = "sum"
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    accumulates a spectra into a single spectrum.

    Parameters
    ----------
    msexp : pyopenms.MSExperiment, pyopenms.OnDiskMSExperiment
    start: int
        start slice for scan accumulation
    end: int
        end slice for scan accumulation.
    kind: str
        kind of interpolator to use with scipy interp1d.
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
        if (subtract[0] > start) or (subtract[-1] < end):
            raise ValueError("subtract region outside scan region.")
    else:
        subtract = (start, end)

    # interpolate accumulate and substract regions
    rows = subtract[1] - subtract[0]
    mz_ref = _get_mz_roi(msexp, subtract)
    interp_int = np.zeros((rows, mz_ref.size))
    for krow, scan in zip(range(rows), range(*subtract)):
        mz_scan, int_scan = msexp.getSpectrum(scan).get_peaks()
        interpolator = interp1d(mz_scan, int_scan, kind=kind)
        interp_int[krow, :] = interpolator(mz_ref)

    # subtract indices to match interp_int rows
    start = start - subtract[0]
    end = end - subtract[0]
    subtract = 0, subtract[1] - subtract[0]

    accum_int = (accumulator(interp_int[start:end], axis=0)
                 - accumulator(interp_int[subtract[0]:start], axis=0)
                 - accumulator(interp_int[end:subtract[1]], axis=0))
    accum_mz = mz_ref

    return accum_mz, accum_int


def _get_mz_roi(ms_experiment, scans):
    """
    make an mz array with regions of interest in the selected scans.

    Parameters
    ----------
    ms_experiment: pyopenms.MSEXperiment, pyopenms.OnDiskMSExperiment
    scans : tuple[int] : start, end

    Returns
    -------
    mz_ref = numpy.array
    """
    mz_0, _ = ms_experiment.getSpectrum(scans[0]).get_peaks()
    mz_min = mz_0.min()
    mz_max = mz_0.max()
    mz_res = np.diff(mz_0).min()
    mz_ref = np.arange(mz_min, mz_max, mz_res)
    roi = np.zeros(mz_ref.size + 1)
    # +1 used to prevent error due to mz values bigger than mz_max
    for k in range(*scans):
        curr_mz, _ = ms_experiment.getSpectrum(k).get_peaks()
        roi_index = np.searchsorted(mz_ref, curr_mz)
        roi[roi_index] += 1
    roi = roi.astype(bool)
    return mz_ref[roi[:-1]]


def make_widths_lc(x: np.ndarray, max_width: float) -> np.ndarray:
    """
    Create an array of widths to use in CWT peak picking of LC data.

    Parameters
    ----------
    x: numpy.ndarray
        vector of x axis. It's assumed that x is sorted.
    max_width: float
    Returns
    -------
    widths: numpy.ndarray
    """
    min_x_distance = np.diff(x).min()
    n = int((max_width - min_x_distance) / min_x_distance)
    first_half = np.linspace(min_x_distance, 10 * min_x_distance, 40)
    second_half = np.linspace(11 * min_x_distance, max_width, n - 10)
    widths = np.hstack((first_half, second_half))

    return widths


def make_widths_ms(min_width: float, max_width: float) -> np.ndarray:
    """
    Create an array of widths to use in CWT peak picking of MS data.

    Parameters
    ----------
    min_width: float
        Minimum expected width
    max_width: float
        Maximum expected width

    Returns
    -------
    widths: numpy.ndarray
    """
    n = int((max_width - min_width) / min_width)
    widths = np.linspace(min_width, max_width, n)
    return widths


def get_lc_cwt_params(mode: str) -> dict:
    """
    Return sane default values for performing CWT based peak picking on LC data.

    Parameters
    ----------
    mode: {"hplc", "uplc"}
        HPLC assumes typical experimental conditions for HPLC experiments:
        longer columns with particle size greater than 3 micron. UPLC is for
        data acquired with short columns with particle size lower than 3 micron.

    Returns
    -------
    cwt_params: dict
        parameters to pass to .peak.pick_cwt function.
    """
    cwt_params = {"snr": 10, "bl_ratio": 2, "min_length": None,
                  "max_distance": None, "gap_thresh": 1}

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
    mode: {"qtof", "orbitrap"}
        qtof assumes a peak width in the range of 0.01-0.05 Da. `orbitrap`
        assumes a peak width in the range of 0.001-0.005 Da.
        TODO: add ppm scale

    Returns
    -------
    cwt_params: dict
        parameters to pass to .peak.pick_cwt function.
    """
    cwt_params = {"snr": 10, "bl_ratio": 2, "min_length": None,
                  "max_distance": None, "gap_thresh": 1}

    if mode == "qtof":
        cwt_params["min_width"] = 0.005
        cwt_params["max_width"] = 0.05
    elif mode == "orbitrap":
        cwt_params["min_width"] = 0.0005
        cwt_params["max_width"] = 0.005
    else:
        msg = "`mode` must be `qtof` or `orbitrap`"
        raise ValueError(msg)
    return cwt_params


def find_isotopic_distribution_aux(mz: np.ndarray, mz_mono: float,
                                   q: int, n_isotopes: int,
                                   tol: float):
    """
    Finds the isotopic distribution for a given charge state. Auxiliary function
    to find_isotopic_distribution.
    Isotopes are searched based on the assumption that the mass difference
    is due to the presence of a 13C atom.

    Parameters
    ----------
    mz: numpy.ndarray
        List of peaks
    mz_mono: float
        Monoisotopic mass
    q: charge state of the ion
    n_isotopes: int
        Number of isotopes to search in the distribution
    tol: float
        Mass tolerance, in absolute units

    Returns
    -------
    match_ind: np.ndarray
        array of indices for the isotopic distribution.
    """
    dm = 1.003355
    mz_theoric = mz_mono + np.arange(n_isotopes) * dm / q
    closest_ind = find_closest(mz, mz_theoric)
    match_ind = np.where(np.abs(mz - mz_theoric[closest_ind]) <= tol)[0]
    return match_ind


def find_isotopic_distribution(mz: np.ndarray, mz_mono: float,
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
    mz: numpy.ndarray
        List of peaks
    mz_mono: float
        Monoisotopic mass
    q_max: int
        max charge to analyze
    n_isotopes: int
        Number of isotopes to search in the distribution
    tol: float
        Mass tolerance, in absolute units

    Returns
    -------
    best_peaks: numpy.ndarray
    """
    best_peaks = None
    n_peaks = 0
    for q in range(1, q_max + 1):
        tmp = find_isotopic_distribution_aux(mz, mz_mono, q,
                                             n_isotopes, tol)
        if tmp.size > n_peaks:
            best_peaks = tmp

    # check if the monoisotopic peak is in the distribution
    mz_mono_ind = find_closest(mz, mz_mono)
    if abs(mz[mz_mono_ind] - mz_mono) > tol:
        best_peaks = np. array([])
    return best_peaks


class Chromatogram:
    """
    Manages chromatograms plotting and peak picking

    Attributes
    ----------
    spint: numpy.ndarray
        intensity in each scan
    mz: numpy.ndarray or float
        mz value for each scan. Used to estimate mean and deviation of the
        mz in the chromatogram
    start: int, optional
        scan number where chromatogram starts
    end: int, optional
    """

    def __init__(self, spint: np.ndarray, rt: np.ndarray,
                 mz: Union[np.ndarray, float], start: Optional[int] = None,
                 end: Optional[int] = None):

        self.rt = rt
        self.spint = spint
        self.mz = mz
        self.peaks = None

        if start is None:
            self.start = 0
        if end is None:
            self.end = rt.size

    def find_peaks(self, mode: str = "uplc", **cwt_params) -> None:
        """
        Find peaks with the modified version of the cwt algorithm described in
        the CentWave algorithm [1]. Peaks are added to the peaks
        attribute of the Chromatogram object.

        Parameters
        ----------
        mode: {"hplc", "uplc"}
            Set peak picking parameters assuming HPLC or UPLC experimental
            conditions. HPLC assumes longer columns with particle size greater
            than 3 micron (min_width is set to 10 seconds and `max_width` is set
             to 90 seconds). UPLC is for data acquired with short columns with
            particle size lower than 3 micron (min_width is set to 5 seconds and
            `max_width` is set to 60 seconds). In both cases snr is set to 10.
        cwt_params:
            key-value parameters to overwrite the defaults in the pick_cwt
            function from the peak module.

        References
        ----------
        .. [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
        """
        default_params = get_lc_cwt_params(mode)
        if cwt_params:
            default_params.update(cwt_params)

        widths = make_widths_lc(self.rt[self.start:self.end],
                                default_params["max_width"])
        peak_list = peaks.pick_cwt(self.rt[self.start:self.end],
                                   self.spint[self.start:self.end],
                                   widths, **default_params)
        self.peaks = peak_list
        if self.start > 0:
            for peak in self.peaks:
                peak.start += self.start
                peak.end += self.start
                peak.loc += self.start

    def get_peak_params(self, subtract_bl: bool = True,
                        rt_estimation: str = "weighted") -> pd.DataFrame:
        """
        Compute peak parameters using retention time and mass-to-charge ratio

        Parameters
        ----------
        subtract_bl: bool
            If True subtracts the estimated baseline from the intensity and
            area.
        rt_estimation: {"weighted", "apex"}
            if "weighted", the peak retention time is computed as the weighted
            mean of rt in the extension of the peak. If "apex", rt is
            simply the value obtained after peak picking.

        Returns
        -------
        peak_params: pandas.DataFrame
        """
        if self.peaks is None:
            msg = "`pick_cwt` method must be runned before using this method"
            raise ValueError(msg)

        peak_params = list()
        for peak in self.peaks:
            tmp = peak.get_peak_params(self.spint, x=self.rt,
                                       subtract_bl=subtract_bl,
                                       center_estimation=rt_estimation)
            tmp["rt"] = tmp.pop("location")
            if isinstance(self.mz, np.ndarray):
                mz_mean = np.average(self.mz, weights=self.spint)
                mz_std = self.mz.std()
                tmp["mz mean"] = mz_mean
                tmp["mz std"] = mz_std
            else:
                tmp["mz mean"] = self.mz
            peak_params.append(tmp)
        return pd.DataFrame(data=peak_params)

    def plot(self, subtract_bl: bool = True, draw: bool = True,
             fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None,
             scatter_params: Optional[dict] = None) -> bokeh.plotting.Figure:

        default_line_params = {"line_width": 1, "line_color": "black",
                               "alpha": 0.8}
        cmap = Set3[12] + Set3[12]

        if line_params is None:
            line_params = default_line_params
        else:
            for params in line_params:
                default_line_params[params] = line_params[params]
            line_params = default_line_params

        if fig_params is None:
            fig_params = dict()

        if scatter_params is None:
            scatter_params = dict()


        fig = bokeh.plotting.figure(**fig_params)
        fig.line(self.rt, self.spint, **line_params)
        if self.peaks:
            source = ColumnDataSource(
                self.get_peak_params(subtract_bl=subtract_bl))
            for k, peak in enumerate(self.peaks):
                fig.varea(self.rt[peak.start:(peak.end + 1)],
                          self.spint[peak.start:(peak.end + 1)], 0,
                          fill_alpha=0.8, fill_color=cmap[k])
            scatter = fig.scatter(source=source, x="rt", y="intensity",
                                  **scatter_params)
            # add hover tool only on scatter points
            tooltips = [("rt", "@rt"), ("mz", "@{mz mean}"),
                        ("intensity", "@intensity"),
                        ("area", "@area"), ("width", "@width")]
            hover = HoverTool(renderers=[scatter], tooltips=tooltips)
            fig.add_tools(hover)

        if draw:
            bokeh.plotting.show(fig)
        return fig


class MSSpectrum:
    """
    Manages peak picking, isotopic distribution analysis and plotting of MS
    data.
    """
    def __init__(self, mz: np.ndarray, spint: np.ndarray):
        self.mz = mz
        self.spint = spint
        self.peaks = None

    def find_peaks(self, mode, **cwt_params):
        """
        Find peaks with the modified version of the cwt algorithm described in
        the CentWave algorithm [1]. Peaks are added to the attribute.

        Parameters
        ----------
        mode: {"qtof", "orbitrap"}
            qtof assumes a peak width in the range of 0.01-0.05 Da. `orbitrap`
            assumes a peak width in the range of 0.001-0.005 Da.
        cwt_params:
            key-value parameters to overwrite the defaults in the pick_cwt
            function from the peak module.

        References
        ----------
        .. [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
        """
        default_params = get_ms_cwt_params(mode)
        if cwt_params:
            default_params.update(cwt_params)

        widths = make_widths_ms(default_params["min_width"],
                                default_params["max_width"])
        peak_list = peaks.pick_cwt(self.mz, self.spint, widths,
                                   **default_params)
        self.peaks = peak_list

    def get_peak_params(self, subtract_bl: bool = True,
                        mz_estimation: str = "weighted") -> pd.DataFrame:
        """
        Compute peak parameters using mass-to-charge ratio and intensity

        Parameters
        ----------
        subtract_bl: bool
            If True subtracts the estimated baseline from the intensity and
            area.
        mz_estimation: {"weighted", "apex"}
            if "weighted", the location of the peak is computed as the weighted
            mean of x in the extension of the peak, using y as weights. If
            "apex", the location is simply the location obtained after peak
            picking.

        Returns
        -------
        peak_params: pandas.DataFrame
        """
        if self.peaks is None:
            msg = "`find_peaks` method must be used first."
            raise ValueError(msg)

        peak_params = [x.get_peak_params(self.spint, self.mz,
                                         subtract_bl=subtract_bl,
                                         center_estimation=mz_estimation)
                       for x in self.peaks]
        peak_params = pd.DataFrame(data=peak_params)
        peak_params.rename(columns={"location": "mz"}, inplace=True)
        return peak_params

    def plot(self, subtract_bl: bool = True, draw: bool = True,
             fig_params: Optional[dict] = None,
             line_params: Optional[dict] = None,
             scatter_params: Optional[dict] = None) -> bokeh.plotting.Figure:

        default_line_params = {"line_width": 1, "line_color": "black",
                               "alpha": 0.8}
        cmap = Set3[12] + Set3[12] + Set3[12] + Set3[12]

        if line_params is None:
            line_params = default_line_params
        else:
            for params in line_params:
                default_line_params[params] = line_params[params]
            line_params = default_line_params

        if fig_params is None:
            fig_params = dict()

        if scatter_params is None:
            scatter_params = dict()


        fig = bokeh.plotting.figure(**fig_params)
        fig.line(self.mz, self.spint, **line_params)
        if self.peaks:
            source = \
                ColumnDataSource(self.get_peak_params(subtract_bl=subtract_bl))
            for k, peak in enumerate(self.peaks):
                fig.varea(self.mz[peak.start:(peak.end + 1)],
                          self.spint[peak.start:(peak.end + 1)], 0,
                          fill_alpha=0.8, fill_color=cmap[k])
            scatter = fig.scatter(source=source, x="mz", y="intensity",
                                  **scatter_params)
            # add hover tool only on scatter points
            tooltips = [("mz", "@{mz}{%0.4f}"),
                        ("intensity", "@intensity"),
                        ("area", "@area"), ("width", "@width")]
            hover = HoverTool(renderers=[scatter], tooltips=tooltips)
            hover.formatters = {"mz": "printf"}
            fig.add_tools(hover)

        if draw:
            bokeh.plotting.show(fig)
        return fig
