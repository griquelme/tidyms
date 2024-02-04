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
import abc
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import BinaryIO, Generator, List, Optional, TextIO, Tuple, Union
from .container import DataContainer
from .core import constants as c
from . import lcms
from . import validation as v
from .utils import get_tidyms_path, gaussian_mixture
from ._mzml import MZMLReader

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
    data = df.iloc[:, raw_index : (2 * raw_index - norm_index)].T
    sample_info = df_header.iloc[
        :, (raw_index + 1) : (2 * raw_index - norm_index + 1)
    ].T

    # rename data matrix
    data.index.rename("sample", inplace=True)
    data.columns.rename("feature", inplace=True)
    data = data.astype(float)

    # rename sample info
    sample_info.index = data.index
    sample_info.rename({sample_info.columns[0]: c.CLASS}, axis="columns", inplace=True)

    # rename features def
    ft_def.index.rename("feature", inplace=True)
    ft_def.rename(
        {"m/z": "mz", "Retention time (min)": "rt"}, axis="columns", inplace=True
    )
    ft_def = ft_def.astype({"rt": float, "mz": float})
    ft_def["rt"] = ft_def["rt"] * 60
    v.validate_data_container(data, ft_def, sample_info)
    dc = DataContainer(data, ft_def, sample_info)
    return dc


def read_mzmine(
    data: Union[str, TextIO], sample_metadata: Union[str, TextIO]
) -> DataContainer:
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


def read_xcms(
    data_matrix: str,
    feature_metadata: str,
    sample_metadata: str,
    class_column: str = "class",
    sep: str = "\t",
):
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


def read_data_matrix(
    path: Union[str, TextIO, BinaryIO],
    data_matrix_format: str,
    sample_metadata: Optional[str] = None,
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

    @staticmethod
    def create_MSData_instance(*args, **kwargs):
        data_import_mode = c.DEFAULT_DATA_LOAD_MODE
        if "data_import_mode" in kwargs:
            data_import_mode = kwargs["data_import_mode"]
            del kwargs["data_import_mode"]

        if data_import_mode.lower() == c.SIMULATED:
            mz_values = kwargs["mz_values"]
            del kwargs["mz_values"]
            rt_values = kwargs["rt_values"]
            del kwargs["rt_values"]
            mz_params = kwargs["mz_params"]
            del kwargs["mz_params"]
            rt_params = kwargs["rt_params"]
            del kwargs["rt_params"]
            return MSData_simulated(
                mz_values, rt_values, mz_params, rt_params, **kwargs
            )
        if data_import_mode.lower() == c.INFILE and "path" in kwargs:
            path = Path(kwargs["path"])
            suffix = path.suffix
            if suffix == "":
                return MSData_from_file(*args, **kwargs)
        if data_import_mode.lower() == c.INFILE:
            return MSData_from_file(*args, **kwargs)
        elif data_import_mode.lower() == c.MEMORY:
            return MSData_in_memory.generate_from_file(*args, **kwargs)

        raise Exception(
            "Unknown data_import_mode parameter '%s'. Must be either 'file', 'memory', 'simulated'"
            % (data_import_mode)
        )

    @abc.abstractmethod
    def __init__(
        self,
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
        is_virtual_sample: bool = False,
    ):
        self.ms_mode = ms_mode
        self.instrument = instrument
        self.separation = separation
        self._is_virtual_sample = is_virtual_sample

    @property
    def ms_mode(self) -> str:
        return self._ms_mode

    @ms_mode.setter
    def ms_mode(self, value: Optional[str]):
        if value in c.MS_MODES:
            self._ms_mode = value
        else:
            msg = "mode must be `centroid` or `profile`. Got `{}`."
            raise ValueError(msg.format(value))

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value: str):
        if value in c.MS_INSTRUMENTS:
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
        if value in c.LC_MODES:
            self._separation = value
        else:
            msg = "`separation` must be any of {}. Got `{}`".format(c.LC_MODES, value)
            raise ValueError(msg)

    @property
    def is_virtual_sample(self) -> bool:
        return self._is_virtual_sample

    @is_virtual_sample.setter
    def is_virtual_sample(self, value: bool):
        raise ValueError("Cannot change sample virtuallity after creation.")

    @abc.abstractmethod
    def get_n_chromatograms(self) -> int:
        """
        Get the chromatogram count in the file

        Returns
        -------
        n_chromatograms : int

        """
        pass

    @abc.abstractmethod
    def get_n_spectra(self) -> int:
        """
        Get the spectra count in the file

        Returns
        -------
        n_spectra : int

        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
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
        pass

    def get_closest_spectrum_to_RT(self, time: float = 0.0) -> lcms.MSSpectrum:
        bestK = None
        bestK_timeDiff = None
        for k, sp in self.get_spectra_iterator():
            if bestK is None or abs(sp.time - time) <= abs(bestK_timeDiff):
                bestK = k
                bestK_timeDiff = sp.time - time

        return bestK, bestK_timeDiff


class MSData_Proxy(MSData):
    def __init__(self, to_MSData_object):
        self.to_MSData_object = to_MSData_object

    @property
    def to_MSData_object(self) -> MSData:
        return self._to_MSData_object

    @to_MSData_object.setter
    def to_MSData_object(self, obj: MSData):
        self._to_MSData_object = obj

    @property
    def ms_mode(self) -> str:
        return self._to_MSData_object.ms_mode

    @ms_mode.setter
    def ms_mode(self, value: Optional[str]):
        self.self._to_MSData_object.ms_mode = value

    @property
    def instrument(self) -> str:
        return self._to_MSData_object.instrument

    @instrument.setter
    def instrument(self, value: str):
        self._to_MSData_object.instrument = value

    @property
    def separation(self) -> str:
        return self._to_MSData_object.separation

    @separation.setter
    def separation(self, value: str):
        self._to_MSData_object.separation = value

    @property
    def is_virtual_sample(self) -> bool:
        return self._to_MSData_object.is_virtual_sample

    @is_virtual_sample.setter
    def is_virtual_sample(self, value: bool):
        self._to_MSData_object.is_virtual_sample = value

    def get_n_chromatograms(self) -> int:
        return self._to_MSData_object.get_n_chromatograms()

    def get_n_spectra(self) -> int:
        return self._to_MSData_object.get_n_spectra()

    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        return self._to_MSData_object.get_chromatogram(n)

    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        return self._to_MSData_object.get_spectrum(n)

    @abc.abstractmethod
    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        return self._to_MSData_object.get_spectra_iterator(
            ms_level=ms_level,
            start=start,
            end=end,
            start_time=start_time,
            end_time=end_time,
        )


class MSData_subset_spectra(MSData):
    """
    Subset of another MSData object.


    Attributes
    ----------
    start_ind: integer (including)
    end_ind: integer (including, must be greater or equal to start_ind)
    from_MSData_object: MSData object

    """

    @abc.abstractmethod
    def __init__(self, start_ind: int, end_ind: int, from_MSData_object: MSData):
        if (
            start_ind >= 0
            and start_ind < from_MSData_object.get_n_spectra()
            and end_ind >= 0
            and end_ind < from_MSData_object.get_n_spectra()
            and start_ind <= end_ind
        ):
            # super().__init__(is_virtual_sample = True)
            self._is_virtual_sample = True
            self.start_ind = start_ind
            self.end_ind = end_ind
            self._from_MSData_object = from_MSData_object

        else:
            raise ValueError(
                "Incorrect attributes provided for MSData_subset_spectra object generation (start_ind = %s, end_ind = %s, from_MSData_object with %d spectra)"
                % (start_ind, end_ind, from_MSData_object.get_n_spectra())
            )

    @property
    def ms_mode(self) -> str:
        return self._from_MSData_object.ms_mode

    @ms_mode.setter
    def ms_mode(self, value: Optional[str]):
        msg = "Warning: ms_mode cannot be set on a subset object. No changes will be made."
        print(msg.format(value))

    @property
    def instrument(self) -> str:
        return self._from_MSData_object._instrument

    @instrument.setter
    def instrument(self, value: str):
        msg = "Warning: instrument cannot be set on a subset object. No changes will be made."
        print(msg.format(value))

    @property
    def separation(self) -> str:
        return self._from_MSData_object._separation

    @separation.setter
    def separation(self, value: str):
        msg = "Warning: separation cannot be set on a subset object. No changes will be made."
        print(msg.format(value))

    def get_n_chromatograms(self) -> int:
        return self._from_MSData_object.get_n_chromatograms()

    def get_n_spectra(self) -> int:
        return self.end_ind - self.start_ind + 1

    @abc.abstractmethod
    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        return self._from_MSData_object.get_chromatogram(n)

    @abc.abstractmethod
    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        if (
            n + self.start_ind >= self._from_MSData_object.get_n_spectra()
            or n + self.start_ind > self.end_ind
        ):
            raise ValueError(
                "Spectrum index %d is invalid. There are only %d spectra in the MSData object"
                % (n, self.end_ind - self.start_ind + 1)
            )
        return self._from_MSData_object.get_spectrum(n + self.start_ind)

    @abc.abstractmethod
    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        if end is None:
            end = self.get_n_spectra()
        elif end > self.get_n_spectra():
            msg = "End must be lower than the number of spectra in the file."
            raise ValueError(msg)

        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = (start_time <= sp.time) and (
                (end_time is None) or (end_time > sp.time)
            )
            if is_valid_level and is_valid_time:
                yield k, sp


class MSData_from_file(MSData):
    """
    Class for reading data from files without storing too much data in the memory

    """

    def __init__(
        self,
        path: Union[str, Path],
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
    ):
        super().__init__(
            ms_mode=ms_mode,
            instrument=instrument,
            separation=separation,
            is_virtual_sample=False,
        )
        path = Path(path)
        suffix = path.suffix
        if suffix == ".mzML":
            self._reader = MZMLReader(path)
        elif suffix == "":
            # used to intantiate MSData_simulated
            self._reader = None
        else:
            msg = "{} is not a valid format for MS data".format(suffix)
            raise ValueError(msg)

    def get_n_chromatograms(self) -> int:
        return self._reader.n_chromatograms

    def get_n_spectra(self) -> int:
        return self._reader.n_spectra

    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        chrom_data = self._reader.get_chromatogram(n)
        name = chrom_data["name"]
        chromatogram = lcms.Chromatogram(chrom_data["time"], chrom_data["spint"])
        return name, chromatogram

    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        sp_data = self._reader.get_spectrum(n)
        sp_data["is_centroid"] = self.ms_mode == "centroid"
        return lcms.MSSpectrum(**sp_data)

    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        if end is None:
            end = self.get_n_spectra()
        elif end > self.get_n_spectra():
            msg = "End must be lower than the number of spectra in the file."
            raise ValueError(msg)

        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = (start_time <= sp.time) and (
                (end_time is None) or (end_time > sp.time)
            )
            if is_valid_level and is_valid_time:
                yield k, sp


class MSData_in_memory(MSData):
    """
    Class for reading the entire file once to memory.

    """

    @abc.abstractmethod
    def generate_from_MSData_object(msDataObj):
        mz_perScan = []
        spint_perScan = []
        time_perScan = []
        ms_level_perScan = []
        polarity_perScan = []
        instrument_perScan = []
        is_centroid_perScan = []

        for k, spectrum in msDataObj.get_spectra_iterator():
            mz_perScan.append(spectrum.mz)
            spint_perScan.append(spectrum.spint)
            time_perScan.append(spectrum.time)
            ms_level_perScan.append(spectrum.ms_level)
            polarity_perScan.append(spectrum.polarity)
            instrument_perScan.append(spectrum.instrument)
            is_centroid_perScan.append(spectrum.is_centroid)

        temp = MSData_in_memory(
            ms_mode=msDataObj.ms_mode,
            instrument=msDataObj.instrument,
            separation=msDataObj.separation,
        )

        for i in range(len(mz_perScan)):
            spectrum = lcms.MSSpectrum(
                mz_perScan[i],
                spint_perScan[i],
                time_perScan[i],
                ms_level_perScan[i],
                polarity_perScan[i],
                instrument_perScan[i],
                is_centroid_perScan[i],
            )
            temp._spectra.append(spectrum)

        return temp

    @abc.abstractmethod
    def generate_from_file(
        path,
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
    ):
        temp = MSData_in_memory(
            ms_mode=ms_mode, instrument=instrument, separation=separation
        )

        path = Path(path)
        suffix = path.suffix
        if suffix == ".mzML":
            reader = MZMLReader(path)
            for i in range(reader.n_spectra):
                sp_data = reader.get_spectrum(i)
                sp_data["is_centroid"] = ms_mode == "centroid"
                temp._spectra.append(lcms.MSSpectrum(**sp_data))
        else:
            msg = "{} is not a valid format for MS data".format(suffix)
            raise ValueError(msg)

        return temp

    def __init__(
        self,
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
    ):
        super().__init__(
            ms_mode=ms_mode,
            instrument=instrument,
            separation=separation,
            is_virtual_sample=False,
        )

        self._ms_mode = ms_mode
        self._instrument = instrument
        self._separation = separation

        self._spectra = []

    def get_n_chromatograms(self) -> int:
        return self._reader.n_chromatograms

    def get_n_spectra(self) -> int:
        return len(self._spectra)

    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        chrom_data = self._reader.get_chromatogram(n)
        name = chrom_data["name"]
        chromatogram = lcms.Chromatogram(
            chrom_data["time"], chrom_data["spint"], mode=self.separation
        )
        return name, chromatogram

    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        return self._spectra[n]

    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        if end is None:
            end = self.get_n_spectra()
        elif end > self.get_n_spectra():
            msg = "End must be lower than the number of spectra in the file."
            raise ValueError(msg)

        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = (start_time <= sp.time) and (
                (end_time is None) or (end_time > sp.time)
            )
            if is_valid_level and is_valid_time:
                yield k, sp

    def duplicate_object(self):
        return copy.deepcopy(self)

    def delete_spectrum(self, n):
        if n >= 0 and n < len(self._spectra):
            del self._spectra[n]

    def delete_spectra(self, ns):
        ns = sorted(ns)[::-1]
        if len(ns) < len(set(ns)):
            raise Exception("Error: array contains non-unique indices")

        for n in ns:
            self.delete_spectrum(n)


class MSData_simulated(MSData):  # pragma: no cover
    """
    Emulates a MSData using simulated data. Used for tests.

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        rt_values: np.ndarray,
        mz_params: np.ndarray,
        rt_params: np.ndarray,
        ft_noise: Optional[np.ndarray] = None,
        noise: Optional[float] = None,
        ms_mode: str = "centroid",
        separation: str = "uplc",
        instrument: str = "qtof",
    ):
        super().__init__(
            ms_mode=ms_mode,
            instrument=instrument,
            separation=separation,
            is_virtual_sample=True,
        )
        # MSData params
        self._ms_mode = ms_mode
        self._instrument = instrument
        self._separation = separation

        self._reader = _SimulatedReader(
            mz_values,
            rt_values,
            mz_params,
            rt_params,
            ms_mode,
            instrument,
            ft_noise,
            noise,
        )

    def get_n_chromatograms(self) -> int:
        return self._reader.n_chromatograms

    def get_n_spectra(self) -> int:
        return self._reader.n_spectra

    def get_chromatogram(self, n: int) -> Tuple[str, lcms.Chromatogram]:
        chrom_data = self._reader.get_chromatogram(n)
        name = chrom_data["name"]
        chromatogram = lcms.Chromatogram(
            chrom_data["time"], chrom_data["spint"], mode=self.separation
        )
        return name, chromatogram

    def get_spectrum(self, n: int) -> lcms.MSSpectrum:
        sp_data = self._reader.get_spectrum(n)
        sp_data["is_centroid"] = self.ms_mode == "centroid"
        return lcms.MSSpectrum(**sp_data)

    def get_spectra_iterator(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[Tuple[int, lcms.MSSpectrum], None, None]:
        if end is None:
            end = self.get_n_spectra()
        elif end > self.get_n_spectra():
            msg = "End must be lower than the number of spectra in the file."
            raise ValueError(msg)

        for k in range(start, end):
            sp = self.get_spectrum(k)
            is_valid_level = ms_level == sp.ms_level
            is_valid_time = (start_time <= sp.time) and (
                (end_time is None) or (end_time > sp.time)
            )
            if is_valid_level and is_valid_time:
                yield k, sp


class _SimulatedReader:
    """
    Reader object for simulated data

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        rt_values: np.ndarray,
        mz_params: np.ndarray,
        rt_params: np.ndarray,
        ms_mode: str,
        instrument: str,
        ft_noise: Optional[np.ndarray] = None,
        noise: Optional[float] = None,
    ):
        # simulation params
        self.ms_mode = ms_mode
        self.instrument = instrument
        self.mz = mz_values
        self.mz_params = mz_params
        self.rt = rt_values
        self.n_spectra = rt_values.size
        self._seeds = None
        self.ft_noise = ft_noise
        # seeds are used to ensure that each time that a spectra is generated
        # with get_spectra its values are the same
        self._seeds = np.random.choice(self.n_spectra * 10, self.n_spectra)
        self._noise_level = noise

        if ft_noise is not None:
            np.random.seed(self._seeds[0])
            rt_params[0] += np.random.normal(scale=ft_noise[1])
        self.rt_array = gaussian_mixture(rt_values, rt_params)

    def get_spectrum(self, scan_number: int):
        is_valid_scan = (0 <= scan_number) and (self.n_spectra > scan_number)
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

        sp = {
            "mz": mz,
            "spint": spint,
            "time": rt,
            "ms_level": 1,
            "instrument": self.instrument,
            "is_centroid": (self.ms_mode == "centroid"),
        }
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
    name: str, files: List[str], download_dir: Optional[str] = None
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
    download_dataset
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

    WINDOWS_LINE_ENDING = b"\r\n"
    UNIX_LINE_ENDING = b"\n"

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
            "profile-data-zlib-indexed-compressed.mzML",
        ],
        "test-nist-raw-data": [
            "NZ_20200226_005.mzML",
            "NZ_20200226_007.mzML",
            "NZ_20200226_009.mzML",
            "NZ_20200226_011.mzML",
            "NZ_20200226_013.mzML",
            "NZ_20200226_015.mzML",
            "NZ_20200226_017.mzML",
            "NZ_20200226_019.mzML",
            "NZ_20200226_021.mzML",
            "NZ_20200226_023.mzML",
            "NZ_20200227_009.mzML",
            "NZ_20200227_011.mzML",
            "NZ_20200227_013.mzML",
            "NZ_20200227_015.mzML",
            "NZ_20200227_017.mzML",
            "NZ_20200227_019.mzML",
            "NZ_20200227_021.mzML",
            "NZ_20200227_023.mzML",
            "NZ_20200227_025.mzML",
            "NZ_20200227_027.mzML",
            "NZ_20200227_029.mzML",
            "NZ_20200227_039.mzML",
            "NZ_20200227_049.mzML",
            "NZ_20200227_059.mzML",
            "NZ_20200227_069.mzML",
            "NZ_20200227_079.mzML",
            "NZ_20200227_089.mzML",
            "NZ_20200227_091.mzML",
            "NZ_20200227_093.mzML",
            "NZ_20200227_097.mzML",
            "sample_list.csv",
        ],
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
    download_dataset
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

    is_data_found = (
        os.path.exists(sample_path)
        and os.path.exists(feature_path)
        and os.path.exists(data_matrix_path)
    )

    if not (cache and is_data_found):
        download_dataset(name)
        pass

    sample_metadata = pd.read_csv(sample_path, index_col=0, **kwargs)
    feature_metadata = pd.read_csv(feature_path, index_col=0, **kwargs)
    data_matrix = pd.read_csv(data_matrix_path, index_col=0, **kwargs)
    dataset = DataContainer(data_matrix, feature_metadata, sample_metadata)
    return dataset
