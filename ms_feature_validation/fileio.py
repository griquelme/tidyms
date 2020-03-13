"""
Functions to read Raw LC-MS data using pyopenms and functions to create
chromatograms and accumulate spectra.
"""

import numpy as np
import pandas as pd
from typing import Optional, Iterable, Tuple, Union, List
from .data_container import DataContainer
from . import lcms
from . import utils
from . import validation


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
        self.reader = lcms.reader(path, on_disc=on_disc)
        self.mode = mode
        self.chromatograms = None
        self.features = None

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

    def make_chromatograms(self, mz: List[float], tolerance: float = 0.05,
                           start: Optional[int] = None,
                           end: Optional[int] = None,
                           accumulator: str = "sum"):
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
        rt, spint = lcms.chromatogram(self.reader, mz, tolerance=tolerance,
                                      start=start, end=end,
                                      accumulator=accumulator)
        chromatograms = list()
        for row in range(spint.shape[0]):
            tmp = lcms.Chromatogram(spint[row, :], rt, mz[row], index=start)
            chromatograms.append(tmp)
        self.chromatograms = chromatograms

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
        accum_mz, accum_int = lcms.accumulate_spectra(self.reader, start, end,
                                                      subtract=subtract,
                                                      kind=kind,
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

        ..[1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
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

    def detect_features(self, snr: float = 10, bl_ratio: float = 2,
                        min_width: float = 5, max_width: float = 30,
                        max_distance: Optional[float] = None,
                        min_length: Optional[int] = None,
                        gap_thresh: int = 1, subtract_bl: bool = True) -> None:
        """
        Find peaks in all chromatograms using the CentWave algorithm [1].
        Peaks are added to the peaks attribute of the Chromatogram object.

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
        subtract_bl: bool
            If True subtracts the estimated baseline from the intensity and
            area.

        References
        ----------
        ..[1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
        """
        features = list()
        for chrom in self.chromatograms:
            chrom.find_peaks(snr=snr, bl_ratio=bl_ratio,
                             min_width=min_width, max_width=max_width,
                             max_distance=max_distance, min_length=min_length,
                             gap_thresh=gap_thresh)
            features.append(chrom.get_peak_params(subtract_bl=subtract_bl))

        # organize features into a DataFrame
        roi_ind = [np.ones(y.shape[0]) * x for x, y in enumerate(features)]
        roi_ind = np.hstack(roi_ind)
        features = pd.concat(features)
        features["roi"] = roi_ind.astype(int)
        features = features.reset_index()
        features.rename(columns={"index": "peak"}, inplace=True)
        self.features = features
