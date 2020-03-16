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


def make_widths_ms(min_width: float,
                   max_width: float) -> np.ndarray:
    """
    Create an array of widths to use in CWT peak picking of MS data.

    Parameters
    ----------
    x: numpy.ndarray
        vector of x axis. It's assumed that x is sorted.
    max_width: float
    Returns
    -------
    widths: numpy.ndarray
    """
    n = int((max_width - min_width) / min_width)
    widths = np.linspace(min_width, max_width, n)
    return widths


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
    index: int
        scan number where chromatogram starts
    """

    def __init__(self, spint: np.ndarray, rt: np.ndarray,
                 mz: Union[np.ndarray, float], index: int = 0):
        self.rt = rt
        self.spint = spint
        self.mz = mz
        self.index = index
        self.peaks = None

    def find_peaks(self, snr: float = 10, bl_ratio: float = 2,
                   min_width: float = 5, max_width: float = 30,
                   max_distance: Optional[float] = None,
                   min_length: Optional[int] = None,
                   gap_thresh: int = 1) -> None:
        """
        Find peaks using CentWave algorithm [1]. Peaks are added to the peaks
        attribute of the Chromatogram object.

        Parameters
        ----------
        snr: float
            Minimum signal-to-noise ratio of the peaks
        bl_ratio: float
            minimum signal / baseline ratio
        min_width: float
            min width of the peaks, in rt units.
        max_width: float
            max width of the peaks, in rt units.
        max_distance: float
            maximum distance between peaks used to build ridgelines.
        min_length: int
            minimum number of points in a ridgeline
        gap_thresh: int
            maximum number of missing points in a ridgeline

        References
        ----------
        .. [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
        """
        widths = make_widths_lc(self.rt, max_width)
        peak_list = peaks.pick_cwt(self.rt, self.spint, widths, snr=snr,
                                   bl_ratio=bl_ratio, min_width=min_width,
                                   max_width=max_width,
                                   max_distance=max_distance,
                                   min_length=min_length,
                                   gap_thresh=gap_thresh)
        self.peaks = peak_list

    def get_peak_params(self, subtract_bl: bool = True) -> pd.DataFrame:
        """
        Compute peak parameters using retention time and mass-to-charge ratio

        Parameters
        ----------
        subtract_bl: bool
            If True subtracts the estimated baseline from the intensity and
            area.

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
                                       subtract_bl=subtract_bl)
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

        source = ColumnDataSource(self.get_peak_params(subtract_bl=subtract_bl))
        fig = bokeh.plotting.figure(**fig_params)
        fig.line(self.rt, self.spint, **line_params)
        for k, peak in enumerate(self.peaks):
            fig.varea(self.rt[peak.start:(peak.end + 1)],
                      self.spint[peak.start:(peak.end + 1)], 0,
                      fill_alpha=0.8, fill_color=cmap[k])
        scatter = fig.scatter(source=source, x="rt", y="intensity",
                              **scatter_params)
        # add hovertool only on scatter points
        tooltips = [("rt", "@rt"), ("mz", "@{mz mean}"),
                    ("intensity", "@intensity"),
                    ("area", "@area"), ("width", "@width")]
        hover = HoverTool(renderers=[scatter], tooltips=tooltips)
        fig.add_tools(hover)
        if draw:
            bokeh.plotting.show(fig)
        return fig
