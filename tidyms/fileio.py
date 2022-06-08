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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import BinaryIO, Generator, List, Optional, TextIO, Tuple, Union
from .container import DataContainer
from ._names import *
from . import lcms
from . import validation as v
from .utils import get_tidyms_path, gaussian_mixture
from ._mzml import build_offset_list, get_spectrum, get_chromatogram

# TODO: test read_pickle valid file
# TODO: delete read_data_matrix function
# TODO: test get_spectra iterator raise ValueError when end > n_spectra
# TODO: test download_tidyms_data


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
    r"""
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
        has comma characters, the default value for the separator is "\\t".

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
        The separation technique used before MS analysis. Used to set default
        parameters in the methods.

    """

    def __init__(
        self,
        path: Union[str, Path],
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc"
    ):
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

    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        """
        get the nth spectrum stored in the file.

        Parameters
        ----------
        n: int
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
            n
        )
        sp_data["is_centroid"] = self.ms_mode == "centroid"
        return lcms.MSSpectrum(**sp_data)

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
            Starts iteration at this spectrum index.
        end : int or None, default=None
            Ends iteration at this spectrum index. If None, stops after the
            last spectrum.
        start_time : float, default=0.0
            Ignore scans with acquisition times lower than this value.
        end_time : float or None, default=None
            Ignore scans with acquisition times higher than this value.

        Yields
        ------
        scan_number: int
        spectrum : lcms.MSSpectrum

        """
        if end is None:
            end = self.get_n_spectra()
        elif end > self.get_n_spectra():
            msg = "End must be lower than the number of spectra in the file."
            raise ValueError(msg)

        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = ((start_time <= sp.time) and
                             ((end_time is None) or (end_time > sp.time)))
            if is_valid_level and is_valid_time:
                yield k, sp


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


def download_tidyms_data(
    name: str,
    files: List[str],
    download_dir: Optional[str] = None
):
    """
    Download a list of files from the data repository

    https://github.com/griquelme/tidyms-data

    Parameters
    ----------
    name : str
        Name of the data directory
    files : List[str]
        List of files inside the data directory.
    download_dir : str or None, default=None
        String representation of a path to download the data. If None, downloads
        the data to the `.tidyms` directory

    Examples
    --------

    Download the `data.csv` file from the `reference-materials` directory into
    the current directory:

    >>> import tidyms as ms
    >>> dataset = "reference-materials"
    >>> file_list = ["data.csv"]
    >>> ms.fileio.download_tidyms_data(dataset, file_list, download_dir=".")

    See Also
    --------
    dowload_dataset
    load_dataset

    """
    if download_dir is None:
        download_dir = Path(get_tidyms_path()).joinpath(name)
    else:
        download_dir = Path(download_dir).joinpath(name)

    if not download_dir.exists():
        download_dir.mkdir()

    url = "https://raw.githubusercontent.com/griquelme/tidyms-data/master/"
    dataset_url = url + "/" + name

    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    for f in files:
        file_path = download_dir.joinpath(f)
        if not file_path.is_file():
            file_url = dataset_url + "/" + f
            r = requests.get(file_url)
            # save content
            with open(file_path, "w") as fin:
                fin.write(r.text)

            # convert end line characters for windows
            if os.name == "nt":
                with open(file_path, "rb") as fin:
                    content = fin.read()
                content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

                with open(file_path, "wb") as fin:
                    fin.write(content)


def _get_dataset_files(name: str):
    data_matrix_files = ["feature.csv", "sample.csv", "data.csv"]
    files_dict = {
        "reference-materials": data_matrix_files,
        "test-mzmine": data_matrix_files,
        "test-xcms": data_matrix_files,
        "test-progenesis": data_matrix_files,
        "test-raw-data": [
            "centroid-data-indexed-uncompressed.mzML",
            "centroid-data-zlib-indexed-compressed.mzML",
            "centroid-data-zlib-no-index-compressed.mzML",
            "profile-data-zlib-indexed-compressed.mzML"
        ],
        "test-nist-raw-data": [
            'NZ_20200226_005.mzML', 'NZ_20200226_007.mzML',
            'NZ_20200226_009.mzML', 'NZ_20200226_011.mzML',
            'NZ_20200226_013.mzML', 'NZ_20200226_015.mzML',
            'NZ_20200226_017.mzML', 'NZ_20200226_019.mzML',
            'NZ_20200226_021.mzML', 'NZ_20200226_023.mzML',
            'NZ_20200227_009.mzML', 'NZ_20200227_011.mzML',
            'NZ_20200227_013.mzML', 'NZ_20200227_015.mzML',
            'NZ_20200227_017.mzML', 'NZ_20200227_019.mzML',
            'NZ_20200227_021.mzML', 'NZ_20200227_023.mzML',
            'NZ_20200227_025.mzML', 'NZ_20200227_027.mzML',
            'NZ_20200227_029.mzML', 'NZ_20200227_039.mzML',
            'NZ_20200227_049.mzML', 'NZ_20200227_059.mzML',
            'NZ_20200227_069.mzML', 'NZ_20200227_079.mzML',
            'NZ_20200227_089.mzML', 'NZ_20200227_091.mzML',
            'NZ_20200227_093.mzML', 'NZ_20200227_097.mzML',
            'sample_list.csv'
        ]
    }
    return files_dict[name]


def download_dataset(name: str, download_dir: Optional[str] = None):
    """
    Download a directory from the data repository.

    https://github.com/griquelme/tidyms-data

    Parameters
    ----------
    name : str
        Name of the data directory
    download_dir : str or None, default=None
        String representation of a path to download the data. If None, downloads
        the data to the `.tidyms` directory

    Examples
    --------

    Download the `data.csv` file from the `reference-materials` directory into
    the current directory:

    >>> import tidyms as ms
    >>> dataset = "reference-materials"
    >>> ms.fileio.download_dataset(dataset, download_dir=".")

    See Also
    --------
    dowload_dataset
    load_dataset

    """
    datasets = list_available_datasets(False)
    if name in datasets:
        file_list = _get_dataset_files(name)
        download_tidyms_data(name, file_list, download_dir=download_dir)
    else:
        msg = "{} is not a valid dataset name".format(name)
        raise ValueError(msg)


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
        download_dataset(name)
        pass

    sample_metadata = pd.read_csv(sample_path, index_col=0, **kwargs)
    feature_metadata = pd.read_csv(feature_path, index_col=0, **kwargs)
    data_matrix = pd.read_csv(data_matrix_path, index_col=0, **kwargs)
    dataset = DataContainer(data_matrix, feature_metadata, sample_metadata)
    return dataset
