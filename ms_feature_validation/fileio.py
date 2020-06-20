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
import os
from typing import Optional, Iterable, Tuple, Union, List, BinaryIO, TextIO
from .data_container import DataContainer
from . import lcms
from . import validation
from . import utils
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
            sample : the same sample names used in `data`
            class : the sample classes
        Columns with run order and analytical batch information are optional.
        Must be names "order" and "batch"

    Returns
    -------
    DataContainer

    """
    df = pd.read_csv(data)
    colnames = pd.Series(df.columns)
    sample_mask = ~df.columns.str.startswith("row")

    # remove filename extensions
    colnames = colnames.str.split(".").apply(lambda x: x[0])
    # remove "row " in columns
    colnames = colnames.str.replace("row ", "")
    colnames = colnames.str.replace("Peak area", "")
    colnames = colnames.str.strip()
    df.columns = colnames
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
        sample_metadata = pd.DataFrame(data=sample_metadata)
        sample_metadata["sample"] += ".raw"
        sample_metadata["sample"] = (sample_metadata["sample"]
                                     .str.split(".")
                                     .apply(lambda x: x[0]))
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
    sm = pd.read_csv(sample_metadata, sep=sep)
    sm.index.name = "sample"
    sm = sm.rename(columns={class_column: "class"})
    dc = DataContainer(dm, fm, sm)
    return dc

def read_data_matrix(path: Union[str, TextIO, BinaryIO],
                     data_matrix_format: str,
                     sample_metadata: Union[str, TextIO, pd.DataFrame]
                     ) -> DataContainer:
    """
    Read different Data Matrix formats into a DataContainer.

    Parameters
    ----------
    path: str
        path to the data matrix file.
    data_matrix_format: {"progenesis", "pickle", "mzmine"}
    sample_metadta : str, file or DataFrame.
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
        return  read_mzmine(path, sample_metadata)
    else:
        msg = "Invalid Format"
        raise ValueError(msg)


class MSData:
    """
    Container object for raw MS data.

    Manages chromatogram creation, spectra creation and feature detection.

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
        The separation technique used before MS analysis.

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
            self.separation = "uplc"
        elif separation in ["uplc", "hplc"]:
            self.separation = separation
        else:
            msg = "`separation` must be uplc or hplc"
            raise ValueError(msg)
            # TODO : the option None should be possible for DI experiments.
        if instrument is None:
            self.instrument = "qtof"
        elif instrument in ["qtof", "orbitrap"]:
            self.instrument = instrument
        else:
            msg = "`instrument` must be qtof or orbitrap"
            raise ValueError(msg)

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
        return lcms.Chromatogram(rt, tic, None)

    def make_chromatograms(self, mz: List[float],
                           window: Optional[float] = None,
                           accumulator: str = "sum") -> List[lcms.Chromatogram]:
        """
        Computes the Extracted Ion Chromatogram for a list mass-to-charge
        values.

        Parameters
        ----------
        mz: Iterable[float]
            Mass-to-charge values to build EICs.
        window: positive number, optional
            Mass window in absolute units. If None, uses a 0.01  window if the
            `instrument` attribute is qtof or a 0.005 value if the instrument
            is orbitrap.
        accumulator: {"mean", "sum"}
            accumulator function used to in each scan.

        Returns
        -------
        chromatograms : list[Chromatograms]

        """
        # parameter validation
        params = {"accumulator": accumulator}
        if window is None:
            if self.instrument == "qtof":
                window = 0.05
            elif self.instrument == "orbitrap":
                window = 0.005
            params["window"] = window

        rt, spint = lcms.make_chromatograms(self.reader, mz, **params)
        chromatograms = list()
        for row in range(spint.shape[0]):
            tmp = lcms.Chromatogram(rt, spint[row, :], mode=self.separation)
            chromatograms.append(tmp)
        return chromatograms

    def accumulate_spectra(self, start: Optional[int], end: Optional[int],
                           subtract: Optional[Tuple[int, int]] = None,
                           kind: str = "linear", accumulator: str = "sum"
                           ) -> lcms.MSSpectrum:
        """
        accumulates a series of consecutive spectra into a single spectrum.

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
            Accumulation method used to merge the scans.

        Returns
        -------
        MSSpectrum
        """
        # TODO: accumulate spectra needs to have different behaviours for
        #   centroid and profile data.
        accum_mz, accum_int = \
            lcms.accumulate_spectra(self.reader, start, end, subtract=subtract,
                                    kind=kind, accumulator=accumulator)
        sp = lcms.MSSpectrum(accum_mz, accum_int, mode=self.instrument)
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

    def detect_features(self, roi_params: Optional[dict] = None,
                        peaks_params: Optional[dict] = None
                        ) -> Tuple[List[lcms.Roi], pd.DataFrame]:
        """
        Detect features in centroid mode data. Each feature is a chromatographic
        peak represented by m/z, rt, peak area and peak width values.

        Parameters
        ----------
        roi_params : dict, optional
            Parameters to pass to the make_roi function. Overwrites default
            parameters. See function function documentation for a detailed
            description of each parameter. Default parameters are set using
            the `ms_instrument` and `separation_technique` attributes.
        peaks_params : dict, optional
            Parameters to pass to the detect_peaks function. Overwrites
            default parameters. Default values are set using the
            `separation_technique` attribute. See function function
            documentation for a detailed description of each parameter.

        Returns
        -------
        roi_list : list[Roi]
            A list with the detected regions of interest.
        feature_data : DataFrame
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

        Raises
        ------
        ValueError
            If the data is not in centroid mode.

        Notes
        -----

        Feature detection is done in three steps using the algorithm described
        in [1]:

        1.  Regions of interest (ROI) search: ROI are searched in the data. Each
            ROI consists in m/z traces where a chromatographic peak may be
            found. It has the detected mz and intensity associated to each scan
            and the time where it was detected.
        2.  Chromatographic peak detection: For each ROI, chromatographic peaks
            are searched. Each chromatographic peak is a feature.
            Several descriptors associated to each feature are computed.
        3.  Features are organized in a DataFrame, where each row is a
            feature and each column is a feature descriptor.

        See Also
        --------
        lcms.make_roi : Function used to search ROIs.
        lcms.Roi : Representation of a ROI.
        peaks.pick_cwt : Function used to detect chromatographic peaks.
        lcms.get_lc_cwt_params : Creates default parameters for peak picking.
        lcms.get_roi_params : Creates default parameters for roi search.

        References
        ----------
        ..  [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive
            feature detection for high resolution LC/MS. BMC Bioinformatics 9,
            504 (2008). https://doi.org/10.1186/1471-2105-9-504

        """

        if self.ms_mode != "centroid":
            msg = "Data must be in centroid mode for feature detection."
            raise ValueError(msg)

        # step 1: detect ROI
        tmp_roi_params = lcms.get_roi_params(self.separation,
                                             self.instrument)
        if roi_params is not None:
            tmp_roi_params.update(roi_params)
        roi_params = tmp_roi_params
        roi_list = lcms.make_roi(self.reader, **roi_params)

        # step 2 and 3: find peaks and build a DataFrame with the parameters
        feature_data = \
            lcms.detect_roi_peaks(roi_list, cwt_params=peaks_params)
        return roi_list, feature_data


def _get_datasets_path(dataset_type: str = "matrix") -> List[str]:
    """
    List full path to available data sets.

    Parameters
    ----------
    dataset_type : {"matrix", "raw"}
        "matrix" shows available data in matrix form. "raw" lists mzML data
        files.

    Returns
    -------
    datasets_path: list[str]
    """
    module_fullname = __file__
    project_path, _ = os.path.split(module_fullname)
    project_path, _ = os.path.split(project_path)
    data_path = os.path.join(project_path, "datasets", dataset_type)
    datasets = os.listdir(data_path)
    dataset_path = [os.path.join(data_path, x) for x in datasets]
    return dataset_path


def get_available_datasets(dataset_type: str = "matrix") -> List[str]:
    """
    List available datasets

    Parameters
    ----------
    dataset_type : {"matrix", "raw"}
        "matrix" shows available data in matrix form. "raw" lists mzML data
        files.

    Returns
    -------
    datasets: list[str]
    """
    dataset_path = _get_datasets_path(dataset_type)

    datasets = [utils.get_filename(x) for x in dataset_path]
    return datasets


def load_dataset(name: str, dataset_type: str = "matrix"):
    """
    Load a data set

    Parameters
    ----------
    name : str
        name of an available dataset.
    dataset_type : str
        "matrix" shows available data in matrix form. "raw" lists mzML data
        files.

    Returns
    -------
    data: if dataset_type is matrix, then returns a DataContainer. if the
    dataset_type is raw, returns a MSData object.
    """
    data_path = _get_datasets_path(dataset_type=dataset_type)
    datasets = get_available_datasets(dataset_type=dataset_type)
    try:
        ind = datasets.index(name)
        path = data_path[ind]
        if dataset_type == "raw":
            data = MSData(path)
        elif dataset_type == "matrix":
            data = DataContainer.from_progenesis(path)
        else:
            msg = "`dataset_type` must be raw or matrix"
            raise ValueError(msg)
        return data
    except KeyError:
        msg = "{} is not an available data set name".format(name)
        raise ValueError(msg)
    