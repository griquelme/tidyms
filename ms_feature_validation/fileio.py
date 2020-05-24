"""
Functions and objects to work with mzML data and tabular data obtained from
third party software used to process Mass Spectrometry data.

Objects
-------
MSData: reads raw MS data in the mzML format. Manages Chromatograms and
MSSpectrum creation. Performs feature detection on centroided data.

Functions
---------
read_pickle(path): Reads a DataContainer stored as a pickle.
read_progenesis(path): Reads data matrix in a csv file generated with
Progenesis software.
read_data_matrix(path, mode): Reads data matrix in several formats. Calls other
read functions.
functions.

See Also
--------
Chromatogram
MSSpectrum
DataContainer
Roi
"""

import numpy as np
import pandas as pd
from typing import Optional, Iterable, Tuple, Union, List, BinaryIO, TextIO
from .data_container import DataContainer
from . import lcms
from . import validation
import pickle


def read_pickle(path: Union[str, BinaryIO]) -> DataContainer:
    """
    read a DataContainer stored as a pickle

    Parameters
    ----------
    path: str or file
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
    path : str or file

    Returns
    -------
    dc : DataContainer
    """
    df_header = pd.read_csv(path, low_memory=False)
    df = df_header.iloc[2:].copy()
    col_names = df_header.iloc[1].values
    df.columns = col_names
    df = df.set_index("Compound")
    df_header = df_header.iloc[:1].copy()
    df_header = df_header.fillna(axis=1, method="ffill")
    norm_index = df_header.columns.get_loc("Normalised abundance") - 1
    raw_index = df_header.columns.get_loc("Raw abundance") - 1
    ft_def = df.iloc[:, 0:norm_index].copy()
    data = df.iloc[:, raw_index:(2 * raw_index - norm_index)].T
    sample_info = \
        df_header.iloc[:, (raw_index + 1):(2 * raw_index - norm_index + 1)].T

    # rename data matrix
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


def _convert_to_interbatch_order(order: pd.Series,
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
    >>>order = pd.Series([1, 2, 3, 1, 2, 3])
    >>>batch = pd.Series([1, 1, 1, 2, 2, 2])
    >>>_convert_to_interbatch_order(order, batch)
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
    dc: DataContainer
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
            order = _convert_to_interbatch_order(order, batch)
        except ValueError:
            # order is already unique
            pass
    dc.order = order
    dc.batch = batch


def read_data_matrix(path: Union[str, TextIO, BinaryIO],
                     data_matrix_format: str) -> DataContainer:
    """
    Read different Data Matrix formats into a DataContainer.

    Parameters
    ----------
    path: str
        path to the data matrix file.
    data_matrix_format: {"progenesis", "pickle"}

    Returns
    -------
    DataContainer

    Examples
    --------
    >>> data = read_data_matrix("data_path.csv", "progenesis")
    """
    if data_matrix_format == "progenesis":
        return read_progenesis(path)
    elif data_matrix_format == "pickle":
        return read_pickle(path)
    else:
        msg = "Invalid Format"
        raise ValueError(msg)


class MSData:
    """
    Reads mzML files and perform common operations on MS Data.

    Attributes
    ----------
    reader : pyopenms.OnDiscExperiment or pyopenms.MSExperiment
        pyopenms object used to read raw data.
    ms_mode : {"centroid", "profile"}
        The mode in which the MS data is stored. If None, mode is guessed, but
        it's recommended to supply the mode in the constructor.
    instrument : {"qtof". "orbitrap"}, optional
        The MS instrument type used to acquire the experimental data.
    separation : {"uplc", "hplc"}, optional
        The separation technique used before MS analysis

    Methods
    -------
    make_tic(mode) : Computes the total ion chromatogram.
    make_chromatograms(mz) : Computes extracted ion chromatograms.
    accumulate_spectra(start, end) : Merge several scans into one.
    get_rt() : returns the retention time vector of the experiment.


    """

    def __init__(self, path: str, ms_mode: Optional[str] = None,
                 instrument: Optional[str] = None,
                 separation: Optional[str] = None):
        """
        Constructor for MSData

        Parameters
        ----------
        path : str
            Path to a mzML file.
        ms_mode : {"centroid", "profile"}, optional
            Mode of the MS data. if None, the data mode is guessed, but it's
            better to provide a mode. The `ms_mode` is used to set default
            parameters on the methods.
        instrument : {"qtof", "orbitrap"}, optional
            MS instrument type used for data acquisition. Used to set default
            parameters in the methods.
        separation: {"uplc", "hplc"}, optional
            Type of separation technique used in the experiment. Used to set
            default parameters in the methods.

        """
        self.reader = lcms.reader(path, on_disc=True)
        self.ms_mode = ms_mode
        if separation is None:
            self.separation_technique = "uplc"
        if instrument is None:
            self.ms_instrument = "qtof"

    @property
    def ms_mode(self) -> str:
        return self._mode

    @ms_mode.setter
    def ms_mode(self, value: Optional[str]):
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
            Mass window in absolute units
            TODO: add tolerance in different units (Da, ppm).
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
        chromatograms : list[Chromatograms]

        """
        # parameter validation
        params = {"window": window, "accumulator": accumulator,
                  "start": start, "end": end}
        validation.validate(params,
                            validation.make_make_chromatogram_validator(self))

        rt, spint = lcms.chromatogram(self.reader, mz, window=window,
                                      start=start, end=end,
                                      accumulator=accumulator)
        chromatograms = list()
        for row in range(spint.shape[0]):
            tmp = lcms.Chromatogram(spint[row, :], rt, mz[row], start=start)
            chromatograms.append(tmp)
        return chromatograms

    def accumulate_spectra(self, start: Optional[int], end: Optional[int],
                           subtract: Optional[Tuple[int, int]] = None,
                           kind: str = "linear", accumulator: str = "sum"
                           ) -> lcms.MSSpectrum:
        """
        accumulates a spectra into a single spectrum.

        Parameters
        ----------
        start: int
            First scan number to accumulate
        end: int
            Last scan number to accumulate.
        kind: str
            kind of interpolator to use with scipy interp1d.
        subtract : None or Tuple[int], left, right
            Scans regions to subtract. `left` must be smaller than `start` and
            `right` greater than `end`.
        accumulator : {"sum", "mean"}

        Returns
        -------
        MSSpectrum
        """
        accum_mz, accum_int = \
            lcms.accumulate_spectra(self.reader, start, end, subtract=subtract,
                                    kind=kind, accumulator=accumulator)
        sp = lcms.MSSpectrum(accum_mz, accum_int)
        return sp

    def _is_centroided(self) -> bool:
        """
        Hack to guess if the data is centroided.

        Returns
        -------
        bool

        """
        mz, spint = self.reader.getSpectrum(0).get_peaks()
        dmz = np.diff(mz)
        # if the data is in profile mode, mz values are going to be closer.
        return dmz.min() > 0.008

    def get_spectrum(self, scan: int):
        """
        get the spectrum for the given scan number.

        Parameters
        ----------
        scan: int
            scan number

        Returns
        -------
        MSSpectrum
        """
        mz, sp = self.reader.getSpectrum(scan).get_peaks()
        return lcms.MSSpectrum(mz, sp)

    def get_rt(self):
        """
        Gets the retention time vector for the experiment.

        Returns
        -------
        rt: np.ndarray

        """
        nsp = self.reader.getNrSpectra()
        rt = np.array([self.reader.getSpectrum(k).getRT() for k in range(nsp)])
        return rt

    def detect_features(self, mode: str = "uplc", ms_mode: str = "qtof",
                        subtract_bl: bool = True,
                        rt_estimation: str = "weighted",
                        make_roi_params: Optional[dict] = None,
                        find_peaks_params: Optional[dict] = None
                        ) -> Tuple[List[lcms.Roi], pd.DataFrame]:
        """
        Find features (mz, rt pairs) in the data using the algorithm described
        in [1]_. MS data must be in centroid mode.

        Feature detection is done in three steps:

        1.  Region of interest (ROI) are detected in the data. Each ROI
            consists in m/z traces where a chromatographic peak may be found.
            It has the detected mz and intensity associated to each scan and the
            time where it was detected.
        2.  For each ROI, chromatographic peaks are searched. Each
            chromatographic peak is a feature. Several descriptors associated to
            each feature are computed.
        3.  Features are organized in a DataFrame, where each row is a
            feature and each column is a feature descriptor.


        Parameters
        ----------
        mode : {"hplc", "uplc"}
            HPLC assumes longer columns with particle size greater than 3 micron
            (min_width is set to 10 seconds and `max_width` is set to 90
            seconds). UPLC is for data acquired with short columns with particle
            size lower than 3 micron (min_width is set to 5 seconds and
            `max_width` is set to 60 seconds). Used to generate default
            `make_roi_params` and `find_peak_params`.
        ms_mode : {"qtof", "orbitrap"}
            MS instrument used to generate de data. Used to set mass tolerance
            params in make_roi_params.
        subtract_bl : bool
            If True subtracts the estimated baseline from the intensity and
            area.
        rt_estimation : {"weighted", "apex"}
            if "weighted", the peak retention time is computed as the weighted
            mean of rt in the extension of the peak. If "apex", rt is
            simply the value obtained after peak picking.
        make_roi_params : dict, optional
            Parameters to pass to the make_roi_function. Overwrites default
            parameters. See function function documentation for a detailed
            description of each parameter.
        find_peaks_params : dict, optional
            Parameters to pass to the find_peaks_function. Overwrites default
            parameters. See function function documentation for a detailed
            description of each parameter.

        Returns
        -------
        roi_list : list[Roi]
            A list with the detected regions of interest.
        peak_data : DataFrame
            A DataFrame with features detected and their associated descriptors.
            Each feature is a row, each descriptor is a column. The descriptors
            are:

            mz :
                Mean m/z of the feature.
            mz std :
                standard deviation of the m/z. Computed as the standard
                deviation of the m/z in the region where the peak was detected.
            rt :
                retention time of the feature, computed as the weighted mean
                of the retention time, using as weights the intensity at each
                time.
            width :
                Chromatographic peak width.
            intensity :
                Maximum intensity of the chromatographic peak.
            area :
                Area of the chromatographic peak.


            Also, two additional columns have information to search each feature
            in its correspondent Roi:

            roi_index :
                index in `roi_list` where the feature was detected.
            peak_index :
                index of the peaks attribute of each Roi associated to the
                feature.

        See Also
        --------
        lcms.make_roi
        lcms.Roi

        References
        ----------
        ..  [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
            feature detection for high resolution LC/MS. BMC Bioinformatics 9,
            504 (2008). https://doi.org/10.1186/1471-2105-9-504

        """

        # TODO: subtract_bl, rt_estimation, should be moved to cwt params.

        # step 1: detect ROI
        tmp_roi_params = lcms.get_make_roi_params(mode, ms_mode)
        if make_roi_params is not None:
            tmp_roi_params.update(make_roi_params)
        make_roi_params = tmp_roi_params
        roi_list = lcms.make_roi(self.reader, **make_roi_params)

        # step 2 and 3: find peaks and make DataFrame
        peak_data = lcms.detect_roi_peaks(roi_list, mode=mode,
                                          subtract_bl=subtract_bl,
                                          rt_estimation=rt_estimation,
                                          cwt_params=find_peaks_params)
        return roi_list, peak_data
