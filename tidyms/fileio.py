"""
Functions and objects to work with mzML data and tabular data obtained from
third party software used to process Mass Spectrometry data.

Objects
-------
MSData: reads raw MS data in the mzML format. Manages Chromatograms and
MSSpectrum creation. Performs feature detection on centroid data.

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
import os
from typing import Optional, Iterable, Callable, Union, List, BinaryIO, TextIO
from .container import DataContainer
from ._names import *
from . import lcms
from . import validation
import pickle
import requests
import warnings
import pyopenms


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
    validation.validate_data_container(data, ft_def, sample_info, None)
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
    Container object for raw MS data.

    Manages chromatogram, roi and spectra creation.

    Attributes
    ----------
    reader : pyopenms.OnDiscExperiment or pyopenms.MSExperiment
        pyopenms object used to read raw data.
    ms_mode : {"centroid", "profile"}
        The mode in which the MS data is stored. If None, mode is guessed, but
        it's recommended to supply the mode in the constructor.
    instrument : {"qtof". "orbitrap"}
        The MS instrument type used to acquire the experimental data. Used to
        set default parameters in the methods.
    separation : {"uplc", "hplc"}
        The separation technique used before MS analysis.  Used to
        set default parameters in the methods.

    """

    def __init__(self, path: str, ms_mode: str = "centroid",
                 instrument: str = "qtof", separation: str = "uplc"):
        """
        Constructor for MSData

        Parameters
        ----------
        path : str
            Path to a mzML file.
        ms_mode : {"centroid", "profile"}
            Mode of the MS data.
        instrument : {"qtof", "orbitrap"}
            MS instrument type used for data acquisition. Used to set default
            parameters in the methods.
        separation: {"uplc", "hplc"}
            Type of separation technique used in the experiment. Used to set
            default parameters in the methods.

        """
        self.reader = _reader(path, on_disc=True)
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
            msg = "mode must be `centroid` or `profile`"
            raise ValueError(msg)

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value: str):
        valid_instruments = ["qtof", "orbitrap"]
        if value in valid_instruments:
            self._instrument = value
        else:
            msg = "`instrument` must be any of {}".format(valid_instruments)
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

    def make_tic(self, kind: str = "tic", ms_level: int = 1
                 ) -> lcms.Chromatogram:
        """
        Makes a total ion chromatogram.

        Parameters
        ----------
        kind: {"tic", "bpi"}
            `tic` computes the total ion chromatogram. `bpi` computes the base
            peak chromatogram.
        ms_level : positive int
            data level used to build the chromatograms. By default, level 1 is
            used.

        Returns
        -------
        tic : lcms.Chromatogram

        """
        return lcms.make_tic(self.reader, kind, self.separation, ms_level)

    def make_chromatograms(self, mz: List[float],
                           window: Optional[float] = None,
                           accumulator: str = "sum",
                           ms_level: int = 1) -> List[lcms.Chromatogram]:
        """
        Computes the Extracted Ion Chromatogram for a list m/z values.

        Parameters
        ----------
        mz: Iterable[float]
            Mass-to-charge values to build EICs.
        window: positive number, optional
            Mass window in absolute units. If None, uses a 0.05  window if the
            `instrument` attribute is qtof or a 0.005 value if the instrument
            is orbitrap.
        accumulator: {"mean", "sum"}
            accumulator function used to in each scan.
        ms_level : positive int
            data level used to build the chromatograms. By default, level 1 is
            used.

        Returns
        -------
        chromatograms : list[Chromatograms]

        """
        if window is None:
            if self.instrument == "qtof":
                window = 0.05
            elif self.instrument == "orbitrap":
                window = 0.005

        n_sp = self.reader.getNrSpectra()
        validation.validate_make_chromatograms_params(
            n_sp, mz, window, 0, n_sp, accumulator, self.separation, ms_level)

        return lcms.make_chromatograms(self.reader, mz, window=window,
                                       chromatogram_mode=self.separation,
                                       accumulator=accumulator,
                                       ms_level=ms_level)

    def accumulate_spectra(self, start: int, end: int,
                           subtract_left: Optional[int] = None,
                           subtract_right: Optional[int] = None,
                           ms_level: int = 1
                           ) -> lcms.MSSpectrum:
        """
        accumulates a series of consecutive spectra into a single spectrum.

        Parameters
        ----------
        start: int
            First scan number to accumulate
        end: int
            Last scan number to accumulate.
        subtract_left : int, optional
            Scans between `subtract_left` and `start` are subtracted from the
            accumulated spectrum.
        subtract_right : int, optional
            Scans between `subtract_right` and `end` are subtracted from the
            accumulated spectrum.
        ms_level : int
            data level used to build the chromatograms. By default, level 1 is
            used.

        Returns
        -------
        MSSpectrum

        """
        n_sp = self.reader.getNrSpectra()
        if subtract_right is None:
            subtract_right = end

        if subtract_left is None:
            subtract_left = start

        validation.validate_accumulate_spectra_params(
            n_sp, start, end, subtract_left, subtract_right, ms_level)

        if self.ms_mode == "profile":
            sp = lcms.accumulate_spectra_profile(self.reader, start, end,
                                                 subtract_left=subtract_left,
                                                 subtract_right=subtract_right,
                                                 ms_level=ms_level,
                                                 instrument=self.instrument)
        else:
            roi_params = lcms.get_roi_params(self.separation, self.instrument)
            tolerance = roi_params["tolerance"]
            sp = lcms.accumulate_spectra_centroid(self.reader, start, end,
                                                  subtract_left, subtract_right,
                                                  tolerance=tolerance)
        return sp

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
        scan = int(scan)  # this prevents a bug in pyopenms with numpy int types
        mz, sp = self.reader.getSpectrum(scan).get_peaks()
        return lcms.MSSpectrum(mz, sp)

    def get_rt(self, ms_level: Optional[int] = None) -> np.ndarray:
        """
        Gets the retention time vector for the experiment.

        Returns
        -------
        rt : np.ndarray
        ms_level : int, optional
            data level to use. If None return a retention time array for all
            levels.

        """
        nsp = self.reader.getNrSpectra()
        rt = np.zeros(nsp)
        level = np.zeros(nsp, dtype=bool)
        for k in range(nsp):
            sp = self.reader.getSpectrum(k)
            rt[k] = sp.getRT()
            if ms_level is not None:
                level[k] = (sp.getMSLevel() == ms_level)

        if ms_level is not None:
            rt = rt[level]
        return rt

    def make_roi(self, tolerance: Optional[float] = None,
                 max_missing: Optional[int] = None,
                 min_length: Optional[int] = None,
                 min_intensity: Optional[float] = None,
                 multiple_match: str = "reduce",
                 mz_reduce: Union[str, Callable] = None,
                 sp_reduce: Union[str, Callable] = "sum",
                 start: Optional[int] = None, end: Optional[int] = None,
                 targeted_mz: Optional[np.ndarray] = None,
                 pad: Optional[int] = None,
                 ms_level: Optional[int] = None) -> List[lcms.Roi]:
        """
        Builds regions of interest from raw data.

        Parameters
        ----------
        tolerance : positive number, optional
            m/z tolerance to connect values across scans
        max_missing : non negative integer, optional
            maximum number of consecutive missing values. when a ROI surpass
            this number the ROI is flagged as finished and is added to the
            ROI list if it meets the length and intensity criteria.
        min_length : positive integer, optional
            The minimum length of a roi to be considered valid.
        min_intensity : non negative number
            Minimum intensity in a ROI to be considered valid.
        start : int, optional
            First scan to analyze. If None starts at scan 0
        end : int, optional
            Last scan to analyze. If None, uses the last scan number.
        pad: int, optional
            Pad dummy values to the left and right of the ROI. This produces
            better peak picking results when searching low intensity peaks in a
            ROI.
        multiple_match : {"closest", "reduce"}
            How to match peaks when there is more than one match. If `closest`
            is used, then the closest peak is assigned as a match and the
            others are used to create a new ROI. If `reduce` is used, then
            unique m/z and intensity values are generated using the reduce
            function in `mz_reduce` and `sp_reduce` respectively.
        mz_reduce : "mean" or Callable
            function used to reduce m/z values. It Can be any function accepting
            numpy arrays and returning numbers. Used only when `multiple_match`
            is set to "reduce". See the following prototype:

            .. code-block:: python

                def mz_reduce(mz_match: np.ndarray) -> float:
                    pass

        sp_reduce : {"mean", "sum"} or Callable
            function used to reduce intensity values. It Can be any function
            accepting numpy arrays and returning numbers. Only used when
            `multiple_match` is set to "reduce". See the prototype shown on
            `mz_reduce`.
        targeted_mz : numpy.ndarray, optional
            A list of m/z values to perform a targeted ROI creation. If this
            value is provided, only ROI with these m/z values will be created.
        ms_level : int, optional
            data level used to build the chromatograms. By default, level 1 is
            used.

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
        roi_list : list[Roi]
            A list with the detected regions of interest.

        Raises
        ------
        ValueError
            If the data is not in centroid mode.

        See Also
        --------
        lcms.Roi : Representation of a ROI.
        lcms.get_make_roi_params : get default parameters for the function

        References
        ----------
        .. [TR08] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
            feature detection for high resolution LC/MS. BMC Bioinformatics 9,
            504 (2008). https://doi.org/10.1186/1471-2105-9-504

        """
        if self.ms_mode != "centroid":
            msg = "Data must be in centroid mode to create ROIs."
            raise ValueError(msg)

        params = {"tolerance": tolerance, "max_missing": max_missing,
                  "min_length": min_length, "min_intensity": min_intensity,
                  "multiple_match": multiple_match, "mz_reduce": mz_reduce,
                  "sp_reduce": sp_reduce, "start": start, "end": end,
                  "targeted_mz": targeted_mz, "mode": self.separation,
                  "ms_level": ms_level, "pad": pad}
        params = {k: v for k, v in params.items() if v is not None}
        roi_params = lcms.get_roi_params(self.separation, self.instrument)
        roi_params.update(params)
        n_spectra = self.reader.getNrSpectra()
        validation.validate_make_roi_params(n_spectra, roi_params)
        roi_list = lcms.make_roi(self.reader, **roi_params)
        return roi_list


def _reader(path: str, on_disc: bool = True):
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
            # this checks if OnDiscMSExperiment can be used else switch
            # to MSExperiment
            exp_reader.getSpectrum(0)
        except RuntimeError:
            msg = "{} is not an indexed mzML file, switching to MSExperiment"
            warnings.warn(msg.format(path))
            exp_reader = pyopenms.MSExperiment()
            pyopenms.MzMLFile().load(path, exp_reader)
    else:
        exp_reader = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(path, exp_reader)
    return exp_reader


def _get_cache_path() -> str:
    cache_path = os.path.join("~", ".tidyms")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


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
    """Download a dataset from github"""
    _check_dataset_name(name)
    cache_path = _get_cache_path()
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
    cache_path = _get_cache_path()
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
