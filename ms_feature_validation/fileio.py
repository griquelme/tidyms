"""
Functions to read Raw LC-MS data using pyopenms and functions to create
chromatograms and accumulate spectra.
"""

import numpy as np
import pandas as pd
from typing import Optional, Iterable, Tuple, Union, List, BinaryIO, TextIO
from .data_container import DataContainer
from . import lcms
from . import utils
from . import validation
import pickle


def read_pickle(path: Union[str, BinaryIO]) -> DataContainer:
    """
    read a DataContainer stored as a pickle

    Parameters
    ----------
    path: str or filelike
        path to read DataContainer

    Returns
    -------
    DataContainer
    """
    if hasattr(path, "read"):
        with path as fin:
            result = pickle.load(fin)
    else:
        with open(path, "rb") as fin:
            result = pickle.load(fin)
    return result


def read_progenesis(path: Union[str, TextIO]):
    """
    Read a progenesis file into a DataContainer

    Parameters
    ----------
    path : path to an Progenesis csv output

    Returns
    -------
    dc = DataContainer
    """
    # df = pd.read_csv(path, skiprows=2, index_col="Compound")
    # df_header = pd.read_csv(path, nrows=2)
    df_header = pd.read_csv(path, low_memory=False)
    df = df_header.iloc[2:].copy()
    col_names = df_header.iloc[1].values
    df.columns = col_names
    df = df.set_index("Compound")
    df_header = df_header.iloc[:1].copy()
    #------------------
    df_header = df_header.fillna(axis=1, method="ffill")
    norm_index = df_header.columns.get_loc("Normalised abundance") - 1
    raw_index = df_header.columns.get_loc("Raw abundance") - 1
    ft_def = df.iloc[:, 0:norm_index].copy()
    data = df.iloc[:, raw_index:(2 * raw_index - norm_index)].T
    sample_info = df_header.iloc[:,
                  (raw_index + 1):(2 * raw_index - norm_index + 1)].T
    # sample_info.set_index(sample_info.iloc[:, 1], inplace=True)
    # sample_info.drop(labels=[1],  axis=1, inplace=True)

    # rename sample info
    # sample_info.index.rename("sample", inplace=True)
    # sample_info.rename({sample_info.columns[0]: "class"},
    #                    axis="columns", inplace=True)
    # rename data matrix
    # data.index = sample_info.index
    data.index.rename("sample", inplace=True)
    data.columns.rename("feature", inplace=True)
    data = data.astype(float)
    # rename sample info
    sample_info.index = data.index
    sample_info.rename({sample_info.columns[0]: "class"},
                       axis="columns", inplace=True)
    # rename features def
    ft_def.index.rename("feature", inplace=True)
    ft_def.rename({"m/z": "mz", "Retention time (min)": "rt"},
                  axis="columns",
                  inplace=True)
    ft_def = ft_def.astype({"rt": float, "mz": float})
    ft_def["rt"] = ft_def["rt"] * 60
    validation.validate_data_container(data, ft_def, sample_info, None)
    dc = DataContainer(data, ft_def, sample_info)
    return dc


def _convert_to_intebatch_order(order: pd.Series,
                               batch: pd.Series) -> pd.Series:
    """
    Convert the order values from a per-batch order to a interbatch order.

    Parameters
    ----------
    order: pandas.Series
        order and batch must share the same index, be of the same size and of
        dtype int.
    batch: pandas.Series

    Returns
    -------
    interbatch_order: pandas.Series

    Raises
    ------
    ValueError: if the order values are already unique.

    Examples
    --------
    order = pd.Series([1, 2, 3, 1, 2, 3])
    batch = pd.Series([1, 1, 1, 2, 2, 2])
    convert_to_interbatch_order(order, batch)
    pd.Series([1, 2, 3, 4, 5, 6])
    """

    if order.unique().size == order.size:
        msg = "order values are already unique"
        raise ValueError(msg)

    # find a value to add to each batch to make unique and sorted order values
    max_order = order.groupby(batch).max()
    add_to_order = np.roll(max_order, 1)
    add_to_order[0] = 0
    add_to_order = add_to_order.cumsum()
    add_to_order = pd.Series(data=add_to_order, index=max_order.index)
    add_to_order = batch.map(add_to_order)
    interbatch_order = order + add_to_order
    return interbatch_order


def add_order_from_csv(dc: DataContainer, path: Union[str, TextIO],
                       interbatch_order: bool = True) -> None:
    """
    adds sample order and sample batch using information from a csv file.
    A column with the name `sample`  with the same values as the index of
    the DataContainer sample_metadata must be provided.
    order information is taken from a column with name `order` and the same
    is done with batch information. order data must be positive integers
    and each batch must have unique values. Each batch must be identified
    with a positive integer.

    Parameters
    ----------
    path: str
        path to the file with order data. Data format is inferred from the
        file extension.
    interbatch_order: bool
        If True converts the order value to a unique value for the whole
        DataContainer. This makes plotting the data as a function of order
        easier.

    """
    # if hasattr(path, "read"):
    #     filename = path.name
    # else:
    #     filename = path
    # ext = filename.split(".")[-1]
    #
    # if ext == "csv":
    df = pd.read_csv(path, index_col="sample")
    # elif ext in ["xls", "xlsx"]:
    #     df = pd.read_excel(path)
    order = df["order"].astype(int)
    batch = df["batch"].astype(int)

    if interbatch_order:
        try:
            order = _convert_to_intebatch_order(order, batch)
        except ValueError:
            # order is already unique
            pass
    dc.order = order
    dc.batch = batch


def read_data_matrix(path: Union[str, TextIO, BinaryIO],
                     format: str) -> DataContainer:
    """
    Read different Data Matrix formats into a DataContainer
    Parameters
    ----------
    path: str
        path to the data matrix file.
    format: {"progenesis", "pickle"}

    Returns
    -------
    DataContainer
    """
    if format == "progenesis":
        return read_progenesis(path)
    elif format == "pickle":
        return read_pickle(path)
    else:
        msg = "Invalid Format"
        raise ValueError(msg)


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

    def make_tic(self, mode: str = "tic") -> lcms.Chromatogram:
        """
        Make a total ion chromatogram.

        Parameters
        ----------
        mode: {"tic", "bpi"}

        Returns
        -------
        rt: np.ndarray
        tic: np.ndarray
        """
        if mode == "tic":
            reduce = np.sum
        elif mode == "bpi":
            reduce = np.max
        else:
            msg = "valid modes are tic or bpi"
            raise ValueError(msg)

        n_scan = self.reader.getNrSpectra()
        rt = np.zeros(n_scan)
        tic = np.zeros(n_scan)
        for k_scan in range(n_scan):
            sp = self.reader.getSpectrum(k_scan)
            rt[k_scan] = sp.getRT()
            _, spint = sp.get_peaks()
            tic[k_scan] = reduce(spint)
        return lcms.Chromatogram(tic, rt, None)

    def make_chromatograms(self, mz: List[float], window: float = 0.05,
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
        window: float
            Mass window in absolute units TODO: merge with functions from
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
        # parameter validation
        params = {"window": window, "accumulator": accumulator,
                  "start": start, "end": end}
        validation.validate(params,
                            validation.make_make_chromatogram_validator(self))
        #----------------------
        rt, spint = lcms.chromatogram(self.reader, mz, window=window,
                                      start=start, end=end,
                                      accumulator=accumulator)
        chromatograms = list()
        for row in range(spint.shape[0]):
            tmp = lcms.Chromatogram(spint[row, :], rt, mz[row], start=start)
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

    def detect_features(self, mode: str = "uplc", subtract_bl: bool = True,
                        rt_estimation: str = "weighted",
                        **cwt_params) -> None:
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
        subtract_bl: bool
            If True subtracts the estimated baseline from the intensity and
            area.
        rt_estimation: {"weighted", "apex"}
            if "weighted", the peak retention time is computed as the weighted
            mean of rt in the extension of the peak. If "apex", rt is
            simply the value obtained after peak picking.
        cwt_params:
            key-value parameters to overwrite the defaults in the pick_cwt
            function from the peak module.

        References
        ----------
        .. [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
        feature detection for high resolution LC/MS. BMC Bioinformatics 9,
        504 (2008). https://doi.org/10.1186/1471-2105-9-504
        """

        features = list()
        for chrom in self.chromatograms:
            chrom.find_peaks(mode, **cwt_params)
            peaks_params = chrom.get_peak_params(subtract_bl=subtract_bl,
                                                 rt_estimation=rt_estimation)
            features.append(peaks_params)

        # organize features into a DataFrame
        roi_ind = [np.ones(y.shape[0]) * x for x, y in enumerate(features)]
        roi_ind = np.hstack(roi_ind)
        features = pd.concat(features)
        features["roi"] = roi_ind.astype(int)
        features = features.reset_index()
        features.rename(columns={"index": "peak"}, inplace=True)
        self.features = features
