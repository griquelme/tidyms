"""
Functions and objects to work with mzML data and tabular data obtained from
third party software used to process Mass Spectrometry data.

Objects
-------
MSData: reads raw MS data in the mzML format. Creates Chromatograms, ROI and
MSSpectrum from ra data.

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

import os
import pickle
import requests
from typing import Optional, Callable, Union, List, BinaryIO, TextIO, Tuple
from typing import Generator
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .container import DataContainer
from ._names import *
from . import lcms
from . import validation as v
from .utils import get_tidyms_path, gaussian_mixture, find_closest
from ._mzml import build_offset_list, get_spectrum, get_chromatogram


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
    sample_info.rename({sample_info.columns[0]: _sample_class},
                       axis="columns", inplace=True)

    # rename features def
    ft_def.index.rename("feature", inplace=True)
    ft_def.rename({"m/z": "mz", "Retention time (min)": "rt"},
                  axis="columns",
                  inplace=True)
    ft_def = ft_def.astype({"rt": float, "mz": float})
    ft_def["rt"] = ft_def["rt"] * 60
    v.validate_data_container(data, ft_def, sample_info)
    dc = DataContainer(data, ft_def, sample_info)
    return dc


def read_mzmine(data: Union[str, TextIO],
                sample_metadata: Union[str, TextIO]) -> DataContainer:
    """
    read a MZMine2 csv file into a DataContainer.

    Parameters
    ----------
    data : str or file
        csv file generated with MZMine.
    sample_metadata : str, file or DataFrame
        csv file with sample metadata. The following columns are required:
        * sample : the same sample names used in `data`
        * class : the sample classes
        Columns with run order and analytical batch information are optional.
        Must be names "order" and "batch"

    Returns
    -------
    DataContainer

    """
    df = pd.read_csv(data)
    col_names = pd.Series(df.columns)
    sample_mask = ~df.columns.str.startswith("row")

    # remove filename extensions
    col_names = col_names.str.split(".").apply(lambda x: x[0])
    # remove "row " in columns
    col_names = col_names.str.replace("row ", "")
    col_names = col_names.str.replace("Peak area", "")
    col_names = col_names.str.strip()
    df.columns = col_names
    df = df.rename(columns={"m/z": "mz", "retention time": "rt"})
    ft_metadata = df.loc[:, ["mz", "rt"]]

    # make feature metadata
    n_ft = df.shape[0]
    ft_len = len(str(n_ft))
    ft_names = ["FT" + str(x).rjust(ft_len, "0") for x in (range(1, n_ft + 1))]
    ft_metadata.index = ft_names

    # data matrix
    data_matrix = df.loc[:, sample_mask]
    data_matrix = data_matrix.loc[:, ~data_matrix.isna().all()]
    data_matrix.index = ft_names
    data_matrix = data_matrix.T

    if not isinstance(sample_metadata, pd.DataFrame):
        sample_metadata = pd.read_csv(sample_metadata)
        sample_metadata = sample_metadata.set_index("sample")

    dc = DataContainer(data_matrix, ft_metadata, sample_metadata)
    return dc


def read_xcms(data_matrix: str, feature_metadata: str,
              sample_metadata: str, class_column: str = "class",
              sep: str = "\t"):
    """
    Reads tabular data generated with xcms.

    Parameters
    ----------
    data_matrix : str
        Path to a tab-delimited data matrix generated with the R package
        SummarizedExperiment assay method.
    feature_metadata : str
        Path to a tab-delimited  file with feature metadata, (called feature
        definitions in XCMS) generated with the R package SummarizedExperiment
        colData method.
    sample_metadata : str
        Path to a tab-delimited  file with sample metadata, generated with the
        R package SummarizedExperiment colData method. A column named class is
        required. If the class information is under another name, it must be
        specified in the `class_column` parameter.
    class_column : str
        Column name which holds sample class information in the sample metadata.
    sep : str
        Separator used in the files. As the feature metadata generated with XCMS
        has comma characters, the default value for the separator is "\t".

    Returns
    -------
    DataContainer
    """

    # data matrix
    dm = pd.read_csv(data_matrix, sep=sep)
    dm = dm.T
    dm.columns.name = "feature"
    dm.index.name = "sample"

    # feature metadata
    fm = pd.read_csv(feature_metadata, sep=sep)
    fm.index.name = "feature"
    fm = fm.rename(columns={"mzmed": "mz", "rtmed": "rt"})
    fm = fm.loc[:, ["mz", "rt"]]
    # TODO : include information from CAMERA package

    # sample_metadata
    sm = pd.read_csv(sample_metadata, sep=sep, index_col=0)
    sm.index.name = "sample"
    sm = sm.rename(columns={class_column: "class"})
    dc = DataContainer(dm, fm, sm)
    return dc


def read_data_matrix(path: Union[str, TextIO, BinaryIO],
                     data_matrix_format: str,
                     sample_metadata: Optional[str] = None
                     ) -> DataContainer:
    """
    Read different Data Matrix formats into a DataContainer.

    Parameters
    ----------
    path: str
        path to the data matrix file.
    data_matrix_format: {"progenesis", "pickle", "mzmine"}
    sample_metadata : str, file or DataFrame.
        Required for mzmine data.

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
    elif data_matrix_format == "mzmine":
        return read_mzmine(path, sample_metadata)
    else:
        msg = "Invalid Format"
        raise ValueError(msg)


class MSData:
    """
    Reader object for raw MS data.

    Manages chromatogram, roi and spectra creation.

    Attributes
    ----------
    path : str
        Path to a mzML file.
    ms_mode : {"centroid", "profile"}, default="centroid"
        The mode in which the MS data is stored.
    instrument : {"qtof". "orbitrap"}, default="qtof"
        The MS instrument type used to acquire the experimental data. Used to
        set default parameters in the methods.
    separation : {"uplc", "hplc"}, default="uplc"
        The separation technique used before MS analysis. Used to
        set default parameters in the methods.

    """

    def __init__(self, path: str, ms_mode: str = "centroid",
                 instrument: str = "qtof", separation: str = "uplc"):
        sp_offset, chrom_offset, index_offset = build_offset_list(path)
        self._spectra_offset = sp_offset
        self._chromatogram_offset = chrom_offset
        self._index_offset = index_offset
        self.path = path

        self.ms_mode = ms_mode
        self.instrument = instrument
        self.separation = separation

    @property
    def ms_mode(self) -> str:
        return self._ms_mode

    @ms_mode.setter
    def ms_mode(self, value: Optional[str]):
        if value in ["centroid", "profile"]:
            self._ms_mode = value
        else:
            msg = "mode must be `centroid` or `profile`. Got `{}`."
            raise ValueError(msg.format(value))

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value: str):
        valid_instruments = ["qtof", "orbitrap"]
        if value in valid_instruments:
            self._instrument = value
        else:
            msg = "`instrument` must be `orbitrap` or `qtof`. Got `{}`."
            msg = msg.format(value)
            raise ValueError(msg)

    @property
    def separation(self) -> str:
        return self._separation

    @separation.setter
    def separation(self, value: str):
        valid_separation = ["uplc", "hplc"]
        if value in valid_separation:
            self._separation = value
        else:
            msg = "`separation` must be any of {}".format(valid_separation)
            raise ValueError(msg)

    def get_n_chromatograms(self) -> int:
        """
        Get the chromatogram count in the file

        Returns
        -------
        n_chromatograms : int

        """
        return len(self._chromatogram_offset)

    def get_n_spectra(self) -> int:
        """
        Get the spectra count in the file

        Returns
        -------
        n_spectra : int

        """
        return len(self._spectra_offset)

    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        """
        Get the nth chromatogram stored in the file.

        Parameters
        ----------
        n : int

        Returns
        -------
        name : str
        chromatogram : lcms.Chromatogram

        """
        chrom_data = get_chromatogram(
            self.path,
            self._spectra_offset,
            self._chromatogram_offset,
            self._index_offset,
            n
        )
        name = chrom_data["name"]
        chromatogram = lcms.Chromatogram(
            chrom_data["time"], chrom_data["spint"], mode=self.separation)
        return name, chromatogram

    def get_spectrum(self, scan: int) -> lcms.MSSpectrum:
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
        sp_data = get_spectrum(
            self.path,
            self._spectra_offset,
            self._chromatogram_offset,
            self._index_offset,
            scan
        )
        sp_data["is_centroid"] = self.ms_mode == "centroid"
        return lcms.MSSpectrum(**sp_data)

    @v.validated_ms_data(v.spectra_iterator_schema)
    def get_spectra_iterator(
            self,
            ms_level: int = 1,
            start: int = 0,
            end: Optional[int] = None,
            start_time: float = 0.0,
            end_time: Optional[float] = None
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        """
        Yields the spectra in the file.

        Parameters
        ----------
        ms_level : int, default=1
            Use data from this ms level.
        start : int, default=0
            Starts iteration at this scan number.
        end : int or None, default=None
            Ends iteration at this scan number. By default, uses all scans.
        start_time : float, default=0.0
            Ignore scans with acquisition times lower than this value.
        end_time : float or None, default=None
            Ignore scans with acquisition times higher than this value.

        Yields
        ------
        scan_number: int
        spectrum : lcms.MSSpectrum

        """
        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = ((start_time <= sp.time) and
                             ((end_time is None) or (end_time > sp.time)))
            if is_valid_level and is_valid_time:
                yield k, sp

    @v.validated_ms_data(v.make_chromatogram_schema)
    def make_chromatograms(
            self,
            mz: np.ndarray,
            window: Optional[float] = None,
            accumulator: str = "sum",
            ms_level: int = 1,
            start: int = 0,
            end: Optional[int] = None,
            start_time: float = 0.0,
            end_time: Optional[float] = None,
    ) -> List[lcms.Chromatogram]:
        """
        Computes extracted ion chromatograms using a list of m/z values.

        Parameters
        ----------
        mz : array
            m/z values used to build the EICs.
        window : positive number or None, default=None
            m/z tolerance used to build the EICs. If ``self.instrument`` is
            ``"qtof"``, the default value is ``0.05``. If ``self.instrument`` is
            ``"orbitrap"`` the default value is ``0.005``.
        accumulator : {"sum", "mean"}, default="sum"
            Mode used to accumulate the values inside the m/z window. ``"sum"``
            computes the total intensity inside the window. ``"mean"`` divides
            the total intensity using the number of points inside the window.
        ms_level : int, default=1
            ms level used to build the chromatograms.
        start : int, default=0
            Create chromatograms starting at this scan number.
        end : int or None, default=None
            End chromatograms at this scan number. By default, uses all scans.
        start_time : float, default=0.0
            include scans starting at this acquisition time.
        end_time : float or None, default=None
            Stops when the acquisition time is higher than this value.

        Returns
        -------
        chromatograms : List of Chromatograms

        """
        n_sp = self.get_n_spectra()

        # mz_intervals has this shape to be compatible with reduce at
        mz_intervals = (np.vstack((mz - window, mz + window))
                        .T.reshape(mz.size * 2))

        eic = np.zeros((mz.size, n_sp))
        rt = np.zeros(n_sp)
        valid_index = list()
        sp_iterator = self.get_spectra_iterator(
            ms_level, start, end, start_time, end_time)
        for scan, sp in sp_iterator:
            valid_index.append(scan)
            rt[scan] = sp.time
            sp_size = sp.mz.size

            # prevents error when working with empty spectra
            if sp_size == 0:
                continue

            # values for each eic in the current scan
            ind_sp = np.searchsorted(sp.mz, mz_intervals)  # slices for each eic
            has_mz = (ind_sp[1::2] - ind_sp[::2]) > 0  # find non-empty slices
            # elements added at the end of mz_sp raise IndexError
            ind_sp[ind_sp >= sp_size] = sp_size - 1
            # this adds the values between two consecutive indices
            tmp_eic = np.where(
                has_mz, np.add.reduceat(sp.spint, ind_sp)[::2], 0)
            if accumulator == "mean":
                norm = ind_sp[1::2] - ind_sp[::2]
                norm[norm == 0] = 1
                tmp_eic = tmp_eic / norm
            eic[:, scan] = tmp_eic
        valid_index = np.array(valid_index)
        rt = rt[valid_index]
        eic = eic[:, valid_index]

        chromatograms = list()
        for row in eic:
            chromatogram = lcms.Chromatogram(
                rt.copy(), row, mode=self.separation)
            chromatograms.append(chromatogram)
        return chromatograms

    def make_tic(self,
                 kind: str = "tic",
                 ms_level: int = 1,
                 start: int = 0,
                 end: Optional[int] = None,
                 start_time: float = 0.0,
                 end_time: Optional[float] = None,
                 ) -> lcms.Chromatogram:
        """
        Creates a total ion chromatogram.

        Parameters
        ----------
        kind: {"tic", "bpi"}, default="tic"
            `tic` computes the total ion chromatogram. `bpi` computes the base
            peak chromatogram.
        ms_level : int, default=1
            ms level used to build the chromatogram.
        start : int, default=0
            Create the chromatogram starting at this scan
        end : int or None, default=None
            End chromatograms at this scan. By default, use all scans.
        start_time : float, default=0.0
            include scans starting at this acquisition time.
        end_time : float or None, default=None
            Stops when the acquisition time is higher than this value. If None,
            it doesn't filter scans by time.

        Returns
        -------
        chromatograms : lcms.Chromatograms

        """
        if kind == "tic":
            reduce = np.sum
        elif kind == "bpi":
            reduce = np.max
        else:
            msg = "valid modes are tic or bpi"
            raise ValueError(msg)

        n_scan = self.get_n_spectra()
        rt = np.zeros(n_scan)
        tic = np.zeros(n_scan)
        valid_index = list()
        # it is not possible to know a priori how many scans of each level are
        # available in a given file without iterating over it. valid_index holds
        # the index related to the selected level and is used to remove scans
        # from other levels.
        sp_iterator = self.get_spectra_iterator(
            ms_level, start, end, start_time, end_time)
        for scan, sp in sp_iterator:
            valid_index.append(scan)
            rt[scan] = sp.time
            if sp.spint.size:
                tic[scan] = reduce(sp.spint)
            else:
                tic[scan] = 0
        tic = tic[valid_index]
        rt = rt[valid_index]
        return lcms.Chromatogram(rt, tic, self.separation)

    @v.validated_ms_data(v.accumulate_spectra_schema)
    def accumulate_spectra(
        self,
        start: int,
        end: int,
        subtract_left: Optional[int] = None,
        subtract_right: Optional[int] = None,
        ms_level: int = 1
    ) -> lcms.MSSpectrum:
        """
        accumulates a series of consecutive spectra into a single spectrum.

        Parameters
        ----------
        start: int
            Start accumulating scans at this scan number.
        end: int
            Ends accumulation at this scan number. Scans in the range
            [start:end) are used
        subtract_left : int or None, default=None
            Scans in the range [subtract_left:start) are subtracted from the
            accumulated spectrum. If ``None``, don't subtract anything.
        subtract_right : int, or None, default=None
            Scans in the range [end:subtract_right) are subtracted from the
            accumulated spectrum. If ``None``, don't subtract anything.
        ms_level : int, default=1
            ms level used to build the accumulated spectrum.

        Returns
        -------
        MSSpectrum

        """
        if self.ms_mode == "centroid":
            sp = self._accumulate_spectra_centroid(
                start, end, subtract_left, subtract_right, ms_level)
        else:   # profile
            sp = self._accumulate_spectra_profile(
                start, end, subtract_left, subtract_right, ms_level)
        return sp

    def get_rt(
            self,
            start: int = 0,
            end: Optional[int] = None,
            ms_level: int = 1,
            start_time: float = 0.0,
            end_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Gets the retention time array for the experiment.

        Parameters
        ----------
        start : int, default=0
            include values starting from this scan number.
        end : int or None, default=None
            include values until this scan number. If None, ends after the last
            scan.
        ms_level : int, default=1
            Use scans from this ms level.
        start_time : float, default=0.0
            Use scans with acquisition time greater than this value.
        end_time : float or None, defaults=None
            If specified, includes scans with acquisition times lower than
            this value.

        Returns
        -------
        rt : np.ndarray

        ms_level : int, optional
            data level to use. If None returns a retention time array for all
            levels.

        """
        sp_iter = self.get_spectra_iterator(
            ms_level, start, end, start_time, end_time)
        rt = list()
        for _, sp in sp_iter:
            rt.append(sp.time)
        return np.array(rt)

    @v.validated_ms_data(v.make_roi_schema)
    def make_roi(
        self,
        tolerance: Optional[float] = None,
        max_missing: Optional[int] = None,
        min_length: Optional[int] = None,
        min_intensity: float = 0.0,
        multiple_match: str = "reduce",
        mz_reduce: Union[str, Callable] = "mean",
        sp_reduce: Union[str, Callable] = "sum",
        targeted_mz: Optional[np.ndarray] = None,
        pad: Optional[int] = None,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        min_snr: float = 10,
        min_distance: Optional[float] = None
    ) -> List[lcms.Roi]:
        """
        Builds regions of interest from raw data.

        See the Notes for a detailed description of the algorithm used.

        Parameters
        ----------
        tolerance : positive number or None, default=None
            m/z tolerance to connect values across scans. If None, the value is
            set based on the value of ``self.instrument``. If
            ``self.instrument`` is ``"qtof"``, the tolerance is ``0.01``. If
            ``self.instrument`` is ``"orbitrap"``, the tolerance is ``0.005``
        max_missing : non-negative integer or None, default=None
            maximum number of consecutive missing values in a valid  ROI. If
            ``None``, the value is set to ``1``.
        min_length : positive integer or None, default=None
            The minimum length of a valid ROI. If, ``None``, the value is set
            based on ``self.separation``. If ``self.separation`` is ``"uplc"``,
            the value is set to ``10``. If ``self.separation`` is ``"hplc"``,
            the value is set to ``20``.
        min_intensity : non-negative number , default=0.0
            Minimum intensity in a valid ROI.
        pad: int or None, default=None
            Pad dummy values to the left and right of the ROI. This produces
            better peak picking results when searching low intensity peaks in a
            ROI. Using None set the value to ``2`` if ``self.separation`` is
            ``"uplc"`` or ``"hplc"``.
        multiple_match : {"closest", "reduce"}, default="reduce"
            How peaks are matched when there is more than one valid match. If
            ``"closest"`` is used, the closest peak is assigned as a match and
            the others are used to create new ROIs. If ``"reduce"`` is used,
            unique m/z and intensity values are generated using the reduce
            function in `mz_reduce` and `sp_reduce` respectively.
        mz_reduce : "mean" or Callable, default="mean"
            Function used to reduce m/z values. If ``"mean"`` is used, the mean
            value of all valid m/z is used. Any function that accepts numpy
            arrays and return numbers can be used. Used only when
            `multiple_match` is set to ``"reduce"``. See the following
            prototype:

            .. code-block:: python

                def mz_reduce(mz_match: np.ndarray) -> float:
                    pass

        sp_reduce : {"mean", "sum"} or Callable, default="sum"
            Function used to reduce intensity values. ``"mean"`` computes the
            mean intensity and ``"sum"`` computes the total intensity. Any
            function that accepts numpy arrays and return numbers can be used.
            Only used when `multiple_match` is set to ``"reduce"``. See the
            prototype shown on `mz_reduce`.
        targeted_mz : numpy.ndarray or None, default=None
            A list of m/z values to perform a targeted ROI creation. If this
            value is provided, only ROI with these m/z values will be created.
        ms_level : int, default=1
            ms level used to build the ROI.
        start : int, default=0
            Create ROI starting at this scan
        end : int or None, default=None
            Stop ROI creation at this scan. If None, stops after the last scan
            in the file.
        start_time : float, default=0.0
            Use scans starting at this acquisition time.
        end_time : float or None, default=None
            Stops when the acquisition time is higher than this value.
        min_snr : positive number, default=10.0
            Minimum signal-to-noise ratio of the peaks. Used only to convert
            profile data to centroid mode
        min_distance : positive number or None, default=None
            Minimum distance between consecutive peaks. If ``None``, the value
            is set to 0.01 if ``self.instrument`` is ``"qtof"`` or to 0.005 if
            ``self.instrument`` is ``"orbitrap"``. Used only to convert profile
            data to centroid mode.

        Notes
        -----
        ROIs are built using the method described in [TR08]_ with slight
        modifications.

        A ROI is modelled as a combination of three arrays with the same size:
        m/z, intensity and time. ROIs are created and extended connecting
        m/z values across successive scans using the following method:

        1.  The m/z values in The first scan are used to initialize a list of
            ROI. If `targeted_mz` is used, the ROI are initialized using this
            list.
        2.  m/z values from the next scan extend the ROIs if they are closer
            than `tolerance` to the mean m/z of the ROI. Values that don't match
            any ROI are used to create new ROIs and are appended to the ROI
            list (only if `targeted_mz` is not used).
        3.  If more than one m/z value is within the tolerance threshold,
            m/z and intensity values are computed according to the
            `multiple_match` strategy.
        4.  If a ROI can't be extended with any m/z value from the new scan,
            it is extended using NaNs.
        5.  If the last `max_missing` values of a ROI are NaN, the ROI is
            flagged as finished. If the maximum intensity of a finished ROI
            is greater than `min_intensity` and the number of points is greater
            than `min_length`, then the ROI is flagged as valid. Otherwise, the
            ROI is discarded.
        6.  Repeat from step 2 until no more scans are available.

        Returns
        -------
        roi : list[Roi]
            A list with the detected regions of interest.

        Raises
        ------
        ValueError
            If the data is not in centroid mode.

        See Also
        --------
        lcms.Roi : Representation of a ROI.

        References
        ----------
        .. [TR08] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
            feature detection for high resolution LC/MS. BMC Bioinformatics 9,
            504 (2008). https://doi.org/10.1186/1471-2105-9-504

        """

        if targeted_mz is None:
            sp_seed = self.get_spectrum(start)
            mz_seed, _ = sp_seed.find_centroids(min_snr, min_distance)
            targeted = False
        else:
            mz_seed = targeted_mz
            targeted = True

        rt = np.zeros(self.get_n_spectra())
        processor = lcms.RoiMaker(
            mz_seed,
            max_missing=max_missing,
            min_length=min_length,
            min_intensity=min_intensity,
            tolerance=tolerance,
            multiple_match=multiple_match,
            mz_reduce=mz_reduce,
            sp_reduce=sp_reduce,
            targeted=targeted
        )

        scans = list()  # scan number used in to build ROI
        sp_iterator = self.get_spectra_iterator(
            ms_level,
            start,
            end,
            start_time,
            end_time
        )
        for scan, sp in sp_iterator:
            rt[scan] = sp.time
            scans.append(scan)
            sp_mz, sp_spint = sp.find_centroids(min_snr, min_distance)
            processor.extend_roi(sp_mz, sp_spint, scan)
            processor.store_completed_roi()

        # add roi not completed during the last scan
        processor.flag_as_completed()
        processor.store_completed_roi()

        # extend roi, find rt of each roi and convert to Roi objects
        roi = processor.process_completed_roi(scans, rt, pad, self.separation)
        return roi

    def _accumulate_spectra_centroid(
        self,
        start: int,
        end: int,
        subtract_left: int,
        subtract_right: int,
        ms_level: int = 1
    ) -> lcms.MSSpectrum:
        """
        accumulates a series of consecutive spectra into a single spectrum.

        auxiliary method for accumulate_spectra.

        """
        # don't remove any m/z value when detecting rois
        max_missing = subtract_right - subtract_left

        roi = self.make_roi(
            max_missing=max_missing,
            min_length=1,
            start=subtract_left,
            end=subtract_right,
            ms_level=ms_level
        )

        mz = np.zeros(len(roi))
        spint = mz.copy()

        # set subtract values to negative
        for k, r in enumerate(roi):
            accum_mask = - np.ones(r.scan.size)
            accum_start, accum_end = np.searchsorted(r.scan, [start, end])
            accum_mask[accum_start:accum_end] = 1
            mz[k] = np.nanmean(r.mz)
            spint[k] = np.nansum(r.spint * accum_mask)

        # remove negative values
        pos_values = spint > 0
        mz = mz[pos_values]
        spint = spint[pos_values]

        # sort values
        sorted_index = np.argsort(mz)
        mz = mz[sorted_index]
        spint = spint[sorted_index]

        sp = lcms.MSSpectrum(
            mz, spint, ms_level=ms_level, instrument=self.instrument)
        return sp

    def _accumulate_spectra_profile(
        self,
        start: int,
        end: int,
        subtract_left: int,
        subtract_right: int,
        ms_level: int = 1,
    ) -> lcms.MSSpectrum:
        """
        aux method for accumulate_spectra.

        """
        # The spectra are accumulated in two steps:
        #
        #  1.  iterate through scans to build a grid of m/z values for the
        #      accumulated spectra.
        #  2.  A second iteration is done to interpolate the intensity in each
        #      scan to the m/z grid and generate the accumulated spectrum.
        #
        #  This process is done in two steps to avoid storing the intensity
        #  values from each scan until the grid is built.

        accum_mz = None
        # m/z tol. A small value is used to prevent distortions in the results
        tol = 0.00005
        sp_iter = self.get_spectra_iterator(
            ms_level,
            subtract_left,
            subtract_right
        )
        # first iteration. Builds a grid of m/z values for the accumulated
        # spectrum. The grid is extended using new m/z values that appear
        # in each new scan
        for scan, sp in sp_iter:
            if accum_mz is None:
                accum_mz = sp.mz
            ind = find_closest(accum_mz, sp.mz)
            no_match = np.abs(accum_mz[ind] - sp.mz) > tol
            accum_mz = np.sort(np.hstack((accum_mz, sp.mz[no_match])))

        accum_sp = np.zeros_like(accum_mz)
        sp_iter = self.get_spectra_iterator(
            ms_level,
            subtract_left,
            subtract_right
        )

        for scan, sp in sp_iter:
            interpolator = interp1d(sp.mz, sp.spint, fill_value=0.0)
            if (scan < start) or (scan > end):
                sign = -1
            else:
                sign = 1
            accum_sp += interpolator(accum_mz) * sign

        # set negative values that may result from subtraction to zero
        is_positive_sp = accum_sp > 0
        accum_mz = accum_mz[is_positive_sp]
        accum_sp = accum_sp[is_positive_sp]

        res = lcms.MSSpectrum(
            accum_mz,
            accum_sp,
            instrument=self.instrument,
            ms_level=ms_level,
            is_centroid=False
        )
        return res


class SimulatedMSData(MSData):  # pragma: no cover
    """
    Emulates a MSData using simulated data. Used for tests.

    """
    def __init__(self, mz_values: np.ndarray, rt_values: np.ndarray,
                 mz_params: np.ndarray, rt_params: np.ndarray,
                 ft_noise: Optional[np.ndarray] = None,
                 noise: Optional[float] = None, ms_mode: str = "centroid",
                 separation: str = "uplc", instrument: str = "qtof"):
        """
        Constructor function

        Parameters
        ----------
        mz_values : array
            sorted mz values for each mz scan, used to build spectra in profile
            mode.
        rt_values : array
            sorted rt values, used to set scan time.
        mz_params : array with shape (n, 3)
             Used to build m/z peaks. Each row is the mean, standard deviation
             and amplitude in the m/z dimension.
        rt_params : array with shape (n, 3)
             Used to build rt peaks. Each row is the mean, standard deviation
             and amplitude in the rt dimension
        ft_noise : array_with shape (n, 2), optional
            adds noise to mz and rt values
        noise : positive number, optional
            noise level to add to each scan. the noise is modeled as gaussian
            iid noise in each scan with standard deviation equal to the value
            used. An offset value is added to make the noise contribution
            non-negative. To make the noise values reproducible, each scan has
            a seed value associated to generate always the same noise value in
            a given scan. seed values are generated randomly each time a new
            object is instantiated.

        """
        # MSData params
        self._spectra_offset = None
        self._chromatogram_offset = None
        self._index_offset = None
        self.path = None

        self.ms_mode = ms_mode
        self.instrument = instrument
        self.separation = separation

        # simulation params
        self.mz = mz_values
        self.mz_params = mz_params
        self.rt = rt_values
        self.n_scans = rt_values.size
        self._seeds = None
        self._noise_level = None
        self.ft_noise = ft_noise
        # seeds are used to ensure that each time that a spectra is generated
        # with get_spectra its values are the same
        self._seeds = np.random.choice(self.n_scans * 10, self.n_scans)
        self._noise_level = noise

        if ft_noise is not None:
            np.random.seed(self._seeds[0])
            rt_params[0] += np.random.normal(scale=ft_noise[1])
        self.rt_array = gaussian_mixture(rt_values, rt_params)

    def get_n_spectra(self):
        return self.n_scans

    def get_spectrum(self, scan_number: int):
        is_valid_scan = (0 <= scan_number) and (self.n_scans > scan_number)
        if not is_valid_scan:
            msg = "Invalid scan number."
            raise ValueError(msg)
        rt = self.rt[scan_number]

        # create a mz array for the current scan
        if self.ft_noise is not None:
            np.random.seed(self._seeds[scan_number])
            mz_noise = np.random.normal(scale=self.ft_noise[0])
            mz_params = self.mz_params.copy()
            mz_params[:, 0] += mz_noise
        else:
            mz_params = self.mz_params

        if self.ms_mode == "centroid":
            mz = mz_params[:, 0]
            spint = self.rt_array[:, scan_number] * mz_params[:, 1]
        else:
            mz = self.mz
            mz_array = gaussian_mixture(self.mz, mz_params)
            spint = self.rt_array[:, scan_number][:, np.newaxis] * mz_array
            spint = spint.sum(axis=0)

        if self._noise_level is not None:
            np.random.seed(self._seeds[scan_number])
            noise = np.random.normal(size=mz.size, scale=self._noise_level)
            noise -= noise.min()
            spint += noise
        sp = lcms.MSSpectrum(
            mz, spint, time=rt, ms_level=1, instrument=self.instrument,
            is_centroid=(self.ms_mode == "centroid"))
        return sp


def list_available_datasets(hide_test_data: bool = True) -> List[str]:
    """
    List available example datasets

    Parameters
    ----------
    hide_test_data : bool

    Returns
    -------
    datasets: List[str]
    """
    repo_url = "https://raw.githubusercontent.com/griquelme/tidyms-data/master/"
    dataset_url = repo_url + "datasets.txt"
    r = requests.get(dataset_url)
    datasets = r.text.split()
    if hide_test_data:
        datasets = [x for x in datasets if not x.startswith("test")]
    return datasets


def _check_dataset_name(name: str):
    available = list_available_datasets(False)
    if name not in available:
        msg = "Invalid dataset name. Available datasets are: {}"
        msg = msg.format(available)
        raise ValueError(msg)


def _download_dataset(name: str):
    """Download a dataset from GitHub"""
    _check_dataset_name(name)
    cache_path = get_tidyms_path()
    dataset_path = os.path.join(cache_path, name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    repo_url = "https://raw.githubusercontent.com/griquelme/tidyms-data/master/"
    dataset_url = repo_url + "/" + name
    files = ["feature.csv", "sample.csv", "data.csv"]
    for f in files:
        file_url = dataset_url + "/" + f
        r = requests.get(file_url)
        file_path = os.path.join(dataset_path, f)
        with open(file_path, "w") as fin:
            fin.write(r.text)


def load_dataset(name: str, cache: bool = True, **kwargs) -> DataContainer:
    """
    load example dataset into a DataContainer. Available datasets can be seen
    using the list_datasets function.

    Parameters
    ----------
    name : str
        name of an available dataset.
    cache : bool
        If True tries to read the dataset from a local cache.
    kwargs: additional parameters to pass to the Pandas read_csv function

    Returns
    -------
    data_matrix : DataFrame
    feature_metadata : DataFrame
    sample_metadata : DataFrame

    """
    cache_path = get_tidyms_path()
    dataset_path = os.path.join(cache_path, name)
    sample_path = os.path.join(dataset_path, "sample.csv")
    feature_path = os.path.join(dataset_path, "feature.csv")
    data_matrix_path = os.path.join(dataset_path, "data.csv")

    is_data_found = (os.path.exists(sample_path) and
                     os.path.exists(feature_path) and
                     os.path.exists(data_matrix_path))

    if not (cache and is_data_found):
        _download_dataset(name)
        pass

    sample_metadata = pd.read_csv(sample_path, index_col=0, **kwargs)
    feature_metadata = pd.read_csv(feature_path, index_col=0, **kwargs)
    data_matrix = pd.read_csv(data_matrix_path, index_col=0, **kwargs)
    dataset = DataContainer(data_matrix, feature_metadata, sample_metadata)
    return dataset
