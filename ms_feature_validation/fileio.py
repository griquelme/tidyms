"""
Functions to read Raw LC-MS data using pyopenms and functions to create
chromatograms and accumulate spectra.
"""

import pyopenms
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional, Iterable, Tuple, Union, List
from . import utils
from .data_container import DataContainer
from . import validation
msexperiment = Union[pyopenms.MSExperiment, pyopenms.OnDiscMSExperiment]


def read_progenesis(path):
    """
    Read a progenesis file into a DataContainer

    Parameters
    ----------
    path : path to an Progenesis csv output

    Returns
    -------
    dc = DataContainer
    """
    df = pd.read_csv(path, skiprows=2, index_col="Compound")
    df_header = pd.read_csv(path, nrows=2)
    df_header = df_header.fillna(axis=1, method="ffill")
    norm_index = df_header.columns.get_loc("Normalised abundance") - 1
    raw_index = df_header.columns.get_loc("Raw abundance") - 1
    ft_def = df.iloc[:, 0:norm_index]
    data = df.iloc[:, raw_index:(2 * raw_index - norm_index)].T
    sample_info = df_header.iloc[:,
                  (raw_index + 1):(2 * raw_index - norm_index + 1)].T
    sample_info.set_index(sample_info.iloc[:, 1], inplace=True)
    sample_info.drop(labels=[1],  axis=1, inplace=True)

    # rename sample info
    sample_info.index.rename("sample", inplace=True)
    sample_info.rename({sample_info.columns[0]: "class"},
                       axis="columns", inplace=True)
    # rename data matrix
    data.index = sample_info.index
    data.columns.rename("feature", inplace=True)
    # rename features def
    ft_def.index.rename("feature", inplace=True)
    ft_def.rename({"m/z": "mz", "Retention time (min)": "rt"},
                  axis="columns",
                  inplace=True)
    ft_def["rt"] = ft_def["rt"] * 60
    validation.validate_data_container(data, ft_def, sample_info, None)
    dc = DataContainer(data, ft_def, sample_info)
    return dc


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


class MSData:
    """
    Reads mzML files and perform common operations on MS Data.

    Attributes
    ----------
    reader: pyopenms.OnDiscExperiment or pyopenms.MSExperiment
    mode: {"centroid", "profile"}, optional
        The mode in which the data is stored. If None, mode is guessed, but
        it's recommended to supply the mode in the constructor.
    """

    def __init__(self, path: str, mode: Optional[str] = None,
                 on_disc: bool = True):
        self.reader = reader(path, on_disc=on_disc)
        self.mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        if value is None:
            if self._is_centroided():
                self._mode = "centroid"
            else:
                self._mode = "profile"
        elif value in ["centroid", "profile"]:
            self._mode = value
        else:
            msg = "mode must be `centroid` or `profile`"
            raise ValueError(msg)

    def get_eic(self, mz: Iterable[float], tolerance: float = 0.05,
                start: Optional[int] = None, end: Optional[int] = None,
                accumulator: str = "sum",) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Extracted Ion Chromatogram for a list mass-to-charge
        values.

        Parameters
        ----------
        mz: Iterable[float]
            Mass-to-charge values to build EICs.
        tolerance: float
            Mass tolerance in absolute units TODO: merge with functions from
            formula generator.
        start: int, optional
            first scan used to build the chromatogram.
            If None, uses the first scan.
        end: int
            last scan used to build the chromatogram.
            if None, uses the last scan.
        accumulator: {"mean", "sum"}
            accumulator function used to in each scan.

        Returns
        -------
        rt: np.ndarray
            Retention time for each scan
        eic: np.ndarray
            Extracted Ion Chromatogram for each mz value. Each column is a mz
            and each row is a scan.
        """
        return chromatogram(self.reader, mz, tolerance=tolerance, start=start,
                            end=end, accumulator=accumulator)

    def accumulate_spectra(self, start: Optional[int], end: Optional[int],
                           subtract: Optional[Tuple[int, int]] = None,
                           kind: str = "linear", accumulator: str = "sum"
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        accumulates a spectra into a single spectrum.

        Parameters
        ----------
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
        accum_mz: numpy.ndarray
            array of accumulated mz
        accum_int: numpy.ndarray
            array of accumulated intensities.
        """
        accum_mz, accum_int = accumulate_spectra(self.reader, start, end,
                                                 subtract=subtract, kind=kind,
                                                 accumulator=accumulator)
        return accum_mz, accum_int

    def _is_centroided(self) -> bool:
        """
        Hack to guess if data is centroided.

        Returns
        -------
        bool
        """
        mz, spint = self.reader.getSpectrum(0).get_peaks()
        dmz = np.diff(mz)
        return dmz.min() > 0.008

    def find_roi(self, pmin: int, min_int: float, max_gap: int,
                 tolerance: float, start: Optional[int] = None,
                 end: Optional[int] = None) -> List[utils.Roi]:
        """
        Find region of interests (ROI) in the data.
        A ROI is built finding close mz values in consecutive scans.

        Parameters
        ----------
        pmin: int
            Minimum lenght of a ROI.
        min_int: float
            Minimum intensity of the maximum in the ROI.
        max_gap: int
            Maximum number of consecutive points in a ROI where no peak
            was found.
        tolerance: float
            maximum distance to add a point to a ROI.
        start: int, optional
            First scan used to search ROI. If None, starts from the first scan.
        end: int, optional
            Last scan used to search ROI. If None, end is set to the last scan.

        Notes
        -----

        The algorithm used to build the ROI is described in [1].

        [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive feature
        detection for high resolution LC/MS. BMC Bioinformatics 9, 504 (2008).
        https://doi.org/10.1186/1471-2105-9-504
        """
        return utils.make_rois(self.reader, pmin, max_gap, min_int,
                               tolerance, start=start, end=end)
    def get_rt(self):
        """
        Retention time vector for the experiment.

        Returns
        -------
        rt: np.ndarray
        """
        nsp = self.reader.getNrSpectra()
        rt = np.array([self.reader.getSpectrum(k).getRT() for k in range(nsp)])
        return rt
