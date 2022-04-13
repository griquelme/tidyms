import pickle
from pathlib import Path
from typing import Optional, List, Dict, Union, Generator
from .fileio import MSData
from .lcms import Roi
from .utils import get_progress_bar
import numpy as np
import pandas as pd
from .correspondence import feature_correspondence
from .container import DataContainer
from joblib import Parallel, delayed


# TODO: validate data
# TODO: add plotting functions (Create plotter function)
# TODO: add check step function.

# TODO: raise error messages if the functions are called in the wrong order
#   This could be done implementing a tag associated with each processing step
#   If the tag changes, the process should be done again. tag can be generated
#   from the hash of the parameters (delegate this to Assay Metadata).

# TODO: add options to add new_samples

# subclass LCMSAssay limit separation to uplc or hplc, modify methods adding
# docstrings and validation.

class Assay:
    """
    Manages data preprocessing workflows from raw data to data matrix.

    Creates a directory with data after each preprocessing step.

    """

    def __init__(
        self,
        name: str,
        path: Optional[Union[str, List[str], Path]] = None,
        sample_metadata: Optional[Union[pd.DataFrame, str]] = None,
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
    ):
        """
        Constructor method

        Parameters
        ----------
        path : str, List[str] or Path
            Contains the path to mzML files to be analyzed. ``path`` can be a
            string or list of strings of absolute path representations to mzML
            files in centroid mode or a Path object. Path objects can be used in
            two ways: It can point to a mzML file or to a directory. In the
            second case all mzML files inside the directory will be used.
        name : str
            Name of the assay. Used to create a directory where the preprocessed
            data is stored.
        sample_metadata : str or DataFrame or None.
            Provides information associated with each sample that is used during
            preprocessing. If a string is provided, it is assumed that it is the
            path to a csv file with sample metadata information. The other
            columns may contain any kind of data but the following columns have
            reserved uses:

            sample
                This column is mandatory. Must contain each one of the file
                names in ``path``, without the `.mzML` extension.
            class
                The class of each sample.
            order
                Unique positive integer numbers indicating the run order of
                each sample.
            batch
                The batch number where each sample was analyzed. The values
                must be positive integers.

            If a DataFrame is provided, it must have the same structure as csv
            file described above. If ``None`` is provided, the samples are
            assumed to be from the same class and no order and batch
            information is used.
        ms_mode : {"centroid", "profile"}
            The mode in which the data is stored.
        instrument : {"qtof", "orbitrap"}
            The instrument type. Used to set several defaults during data
            preprocessing.
        separation : {"uplc", "hplc"}
            The separation method used. Used to set several defaults during data
            preprocessing.

        """
        if not isinstance(name, Path):
            assay_path = Path(name)
        else:
            assay_path = name

        suffix = ".tidyms-assay"
        if assay_path.suffix != suffix:
            assay_path = Path(assay_path.parent / (assay_path.name + suffix))

        status = _is_assay_dir(assay_path)
        if status == 1:     # valid assay dir, load assay
            metadata_path = assay_path.joinpath("metadata.pickle")
            with open(metadata_path, "rb") as fin:
                self._metadata = pickle.load(fin)
        elif status == 0:   # dir does not exist, create assay
            metadata = _AssayMetadata(
                assay_path,
                path,
                sample_metadata,
                ms_mode,
                separation,
                instrument
            )
            self._metadata = metadata
            self._metadata.create_assay_dir()
            self._metadata.save()
        else:
            msg = "The directory is not a valid Assay directory."
            raise ValueError(msg)

        # sample metadata
        self.sample_metadata = self._metadata.sample_metadata
        self.feature_table = None

    def get_ms_data(self, sample: str):
        """
        Creates a MSData object for the selected sample.

        Parameters
        ----------
        sample: str
            Sample name.

        Returns
        -------
        MSData

        Raises
        ------
        ValueError : if the sample is not found.

        """
        sample_path = self._metadata.get_sample_path(sample)
        return MSData(sample_path, **self._metadata.params["MSData"])

    def detect_features(
        self,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Builds Regions Of Interest (ROI) from raw data. ROI are regions where
        features may be found.

        ROIs are computed and saved to disk. Use ``self.load_roi`` or
        ``self.load_roi_list`` to load ROI from each sample.

        Parameters
        ----------
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs : dict
            Parameters to pass to :py:meth:`tidyms.fileio.MSData.make_roi`.

        See Also
        --------
        lcms.Roi : Abstract ROI representation.
        lcms.MSData.LCRoi : ROI used in LC data.

        """
        # params used in a previous preprocessing step
        # if the same parameters are used, the file is not analyzed again
        # build list of files to process:
        step = "detect_features"
        process_samples = self._metadata.get_process_samples(step, kwargs)
        n_samples = len(process_samples)

        # clear old stored data:
        for sample in process_samples:
            self._metadata.clear_roi_data(sample)

        if n_samples:

            def iter_func():
                for x in self._metadata.sample_to_path:
                    roi_path = self._metadata.get_roi_path(x)
                    ms_data = self.get_ms_data(x)
                    print("x =", x)
                    print("roi_path =", roi_path)
                    yield roi_path, ms_data

            def worker(args):
                roi_path, ms_data = args
                roi_list = self._detect_features_func(ms_data, **kwargs)
                _save_roi_list(roi_path, roi_list)

            worker = delayed(worker)
            iterator = iter_func()
            if verbose:
                print("Creating ROI in {} samples".format(n_samples))
                bar = get_progress_bar()
                iterator = bar(iterator, total=n_samples)

            Parallel(n_jobs=n_jobs)(worker(x) for x in iterator)
            self._metadata.params[step] = kwargs
            self._metadata.flag_completed(step)
            self._metadata.save()
            return self

    def load_roi(self, sample: str, index: int) -> Roi:
        """
        Loads a ROI obtained after feature detection.

        Parameters
        ----------
        sample : str
            sample name
        index : int
            index of the requested ROI.

        Returns
        -------
        ROI

        Raises
        ------
        ValueError : if the ROI data was not found. This error occurs if a
        wrong sample name is used or if `self.detect_features` was not called.

        """
        file_name = "{}.pickle".format(index)
        roi_path = self._metadata.get_roi_path(sample).joinpath(file_name)
        if roi_path.is_file():
            with roi_path.open("rb") as fin:
                roi = pickle.load(fin)
        else:
            msg = (
                "ROI data not found. Run the `detect_features` method before "
                "loading ROI data."
            )
            raise ValueError(msg)
        return roi

    def load_roi_list(self, sample: str) -> List[Roi]:
        """
        Loads all the ROIs detected in a sample.

        Parameters
        ----------
        sample : str
            sample name

        Returns
        -------
        ROI

        Raises
        ------
        ValueError : if the ROI data was not found. This error occurs if a
        wrong sample name is used or if `self.detect_features` was not called.

        """
        roi_list = list()
        for path in self._metadata.get_roi_list_path(sample):
            with open(path, "rb") as fin:
                roi = pickle.load(fin)
                roi_list.append(roi)
        return roi_list

    def load_features(self, sample: str) -> pd.DataFrame:
        """
        Loads a feature table where each row is a feature and each column is a
        feature descriptor.

        Parameters
        ----------
        sample : str
            sample name

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError : if the feature data was not found. This error occurs if a
        wrong sample name is used or if `self.describe_features` was not called.
        """
        ft_path = self._metadata.get_feature_path(sample)
        if ft_path.is_file():
            with ft_path.open("rb") as fin:
                df = pickle.load(fin)
        else:
            msg = (
                "Feature data not found. Run `self.extract_features` and "
                "`self.describe_features` before loading feature data."
            )
            raise ValueError(msg)
        return df

    def extract_features(
        self,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Extract features from the ROIs detected on each sample.

        Parameters
        ----------
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs : dict
            Parameters to pass to :py:meth:`tidyms.LCRoi.extract_features`.

        See Also
        --------
        peaks.find_peaks : Function used to detect peaks in each ROI

        """
        previous_step = "detect_features"
        step = "extract_features"
        self._metadata.check_step(step, previous_step)

        process_samples = self._metadata.get_process_samples(step, kwargs)
        n_samples = len(process_samples)
        if n_samples:

            def iterator():
                for sample in self._metadata.sample_to_path:
                    roi_path = self._metadata.get_roi_path(sample)
                    roi_list = self.load_roi_list(sample)
                    yield roi_path, roi_list

            def worker(args):
                roi_path, roi_list = args
                self._extract_features_func(roi_list, **kwargs)
                _save_roi_list(roi_path, roi_list)

            worker = delayed(worker)
            if verbose:
                print("Extracting features in {} samples".format(n_samples))
                bar = get_progress_bar()
                iterator = bar(iterator(), total=n_samples)
            Parallel(n_jobs=n_jobs)(worker(x) for x in iterator)
            self._metadata.params[step] = kwargs
            self._metadata.flag_completed(step)
            self._metadata.save()
            return self

    def describe_features(
        self,
        n_jobs: Optional[int] = 1,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Compute feature descriptors for the features extracted from the data.

        Parameters
        ----------
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs : dict
            Parameters to pass to :py:meth:`tidyms.LCRoi.describe_features`.

        See Also
        --------
        peaks.find_peaks : Function used to detect peaks in each ROI

        """
        previous_step = "extract_features"
        step = "describe_features"
        self._metadata.check_step(step, previous_step)
        process_samples = self._metadata.get_process_samples(step, kwargs)
        n_samples = len(process_samples)
        if n_samples:

            def iterator():
                for sample in self._metadata.sample_to_path:
                    roi_path = self._metadata.get_roi_path(sample)
                    ft_path = self._metadata.get_feature_path(sample)
                    roi_list = self.load_roi_list(sample)
                    yield roi_path, ft_path, roi_list

            def worker(args):
                roi_path, ft_path, roi_list = args
                ft_table = self._describe_features_func(roi_list, **kwargs)
                _save_roi_list(roi_path, roi_list)
                ft_table.to_pickle(ft_path)

            worker = delayed(worker)
            if verbose:
                msg = "Computing feature descriptors in {} samples."
                print(msg.format(n_samples))
                bar = get_progress_bar()
                iterator = bar(iterator(), total=n_samples)
            else:
                iterator = iterator()
            Parallel(n_jobs=n_jobs)(worker(x) for x in iterator)
            self._metadata.params[step] = kwargs
            self._metadata.flag_completed(step)
            self._metadata.save()
            self.build_feature_table()
            return self

    def build_feature_table(self):
        """
        Concatenates the feature tables from all samples into one DataFrame.
        Used to match features and to build the data matrix.

        Raises
        ------
        ValueError : if the feature table was not built for all samples. This
        occurs if `self.describe_features` was not called.

        """
        # create feature table
        df_list = list()
        for name in self._metadata.sample_to_path:
            df = self.load_features(name)
            df["sample_"] = self._metadata.sample_to_code[name]
            group = self._metadata.sample_to_class.get(name)
            if group is None:
                df["class_"] = 0
            else:
                df["class_"] = group
            df_list.append(df)
        feature_table = pd.concat(df_list).reset_index(drop=True)
        feature_table["roi_index"] = feature_table["roi_index"].astype(int)
        feature_table["ft_index"] = feature_table["ft_index"].astype(int)
        feature_table["sample_"] = feature_table["sample_"].astype(int)
        feature_table["class_"] = feature_table["class_"].astype(int)
        file_name = "feature-table.pickle"
        save_path = self._metadata.assay_path.joinpath(file_name)
        feature_table.to_pickle(save_path)
        self.feature_table = feature_table

    def match_features(self, **kwargs):
        previous_step = "describe_features"
        step = "match_features"
        self._metadata.check_step(step, previous_step)
        self._match_features_func(**kwargs)
        file_name = "feature-table.pickle"
        save_path = self._metadata.assay_path.joinpath(file_name)
        self.feature_table.to_pickle(save_path)
        self._metadata.params[step] = kwargs
        self._metadata.flag_completed(step)
        self._metadata.save()

        return self

    def make_data_matrix(self, **kwargs):
        raise NotImplementedError

    def annotate_isotopologues(self, **kwargs):
        raise NotImplementedError

    def annotate_adducts(self, **kwargs):
        raise NotImplementedError

    def fill_missing(self, **kwargs):
        raise NotImplementedError

    def _match_features_func(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _detect_features_func(ms_data: MSData, **kwargs) -> List[Roi]:
        return ms_data.make_roi(**kwargs)

    @staticmethod
    def _extract_features_func(roi_list: List[Roi], **kwargs):
        for roi in roi_list:
            roi.extract_features(**kwargs)

    @staticmethod
    def _describe_features_func(roi_list: List[Roi], **kwargs) -> pd.DataFrame:
        roi_index_list = list()
        ft_index = list()
        descriptors_list = list()

        for roi_index, roi in enumerate(roi_list):
            descriptors = roi.describe_features(**kwargs)
            n_features = len(descriptors)
            descriptors_list.extend(descriptors)
            roi_index_list.append([roi_index] * n_features)
            ft_index.append(range(n_features))

        roi_index_list = np.hstack(roi_index_list)
        ft_index = np.hstack(ft_index)

        ft_table = pd.DataFrame(data=descriptors_list)
        ft_table["roi_index"] = roi_index_list
        ft_table["ft_index"] = ft_index
        ft_table = ft_table.dropna(axis=0)
        ft_table["roi_index"] = ft_table["roi_index"].astype(int)
        ft_table["ft_index"] = ft_table["ft_index"].astype(int)
        return ft_table


class LCMSAssay(Assay):
    """
    Manages preprocessing workflows for LC-MS data.

    Subclassed from Assay

    """

    def match_features(
            self,
            include_classes: Optional[List[int]] = None,
            mz_tolerance: Optional[float] = None,
            rt_tolerance: Optional[float] = None,
            min_fraction: float = 0.25,
            max_deviation: float = 3,
            n_jobs: Optional[int] = None,
            verbose: bool = False
    ) -> "Assay":
        """
        Match features across samples according the similarity in m/z and Rt
        values.

        A new column is created for the feature table called `cluster_` that
        groups features in different samples according to their chemical
        identity. See the notes for a detailed description of the matching
        algorithm.

        Parameters
        ----------
        include_classes : List[str] or None, default=None
            Classes used to estimate the number of chemical species in each
            cluster.
        mz_tolerance : float or None, default=None
            m/z tolerance used to connect close features in the first stage.
        rt_tolerance : float or None, default=None
            Rt tolerance used to connect close features in the first stage.
        min_fraction : float, default=0.25
            Minimum fraction of samples necessary to build a cluster. If
            `include_classes` is None, the minimum fraction is estimated from
            all samples in the cluster. If `include_classes` were specified,
            fraction in a cluster is computed using samples from these classes.
        max_deviation : float, default=3.0
            The maximum deviation of a feature from the cluster mean, relative
            to the cluster standard deviation. Features with deviations grater
            than this value are assigned as noise.
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.

        Notes
        -----

        """
        return super(LCMSAssay, self).match_features(
            include_classes=include_classes,
            mz_tolerance=mz_tolerance,
            rt_tolerance=rt_tolerance,
            min_fraction=min_fraction,
            max_deviation=max_deviation,
            n_jobs=n_jobs,
            verbose=verbose
        )

    def make_data_matrix(self) -> DataContainer:

        # remove noise
        rm_noise_mask = self.feature_table["cluster_"] > -1
        feature_data = self.feature_table[rm_noise_mask]

        # dict used to rename index and columns
        sample_to_code = self._metadata.sample_to_code
        code_to_sample = {v: k for k, v in sample_to_code.items()}
        n_cluster = feature_data["cluster_"].unique().size
        max_n_chars_cluster = len(str(n_cluster))
        template = "FT-{{:0{}d}}".format(max_n_chars_cluster)
        cluster_to_ft = {k: template.format(k + 1) for k in range(n_cluster)}

        # compute aggregate statistics for each feature -> feature metadata
        estimators = {"mz": ["mean", "std", "min", "max"],
                      "rt": ["mean", "std", "min", "max"]}
        feature_metadata = feature_data.groupby("cluster_").agg(estimators)
        feature_metadata.columns = _flatten_column_multindex(feature_metadata)
        feature_metadata.index = feature_metadata.index.map(cluster_to_ft)
        feature_metadata.index.name = "feature"

        # make data matrix
        data_matrix = feature_data.pivot(index="sample_", columns="cluster_",
                                         values="area")
        data_matrix.columns = data_matrix.columns.map(cluster_to_ft)
        data_matrix.index = data_matrix.index.map(code_to_sample)
        data_matrix.columns.name = "feature"
        data_matrix.index.name = "sample"

        # TODO: deal with nan values.
        data_matrix = data_matrix.fillna(0)

        # add samples without features as nan rows
        missing_index = self.sample_metadata.index.difference(data_matrix.index)
        missing = pd.DataFrame(data=0, index=missing_index,
                               columns=data_matrix.columns)
        data_matrix = pd.concat((data_matrix, missing))
        data_matrix = data_matrix.loc[self.sample_metadata.index, :]

        sm = self.sample_metadata.copy()
        sm.index.name = "sample"
        sm = sm.rename(columns={"class_": "class", "id_": "id"})

        dc = DataContainer(data_matrix, feature_metadata, sm)
        return dc

    def _match_features_func(self, **kwargs):
        feature_correspondence(
            self.feature_table,
            self._metadata.sample_per_class,
            **kwargs
        )


class _AssayMetadata:
    """
    Manages sample metadata information and location of data files

    """

    def __init__(
        self,
        assay_path: Path,
        data_path: Union[str, List[str], Path],
        sample_metadata: Optional[Union[str, pd.DataFrame]],
        ms_mode: str,
        instrument: str,
        separation: str,
    ):
        self.assay_path = assay_path
        self.data_path = data_path

        path_list = _get_path_list(data_path)
        name2path = {x.stem: x for x in path_list}
        self.sample_to_path = name2path

        sample2code = dict(zip(name2path.keys(), range(len(name2path))))
        self.sample_to_code = sample2code
        self.sample_to_class = None
        self.sample_per_class = None
        self.group_factors = None
        self.sample_metadata = sample_metadata
        self.params = _build_params_dict(ms_mode, separation, instrument)

    @property
    def sample_metadata(self) -> pd.DataFrame:
        return self._sample_metadata

    @sample_metadata.setter
    def sample_metadata(self, value: Optional[Union[str, pd.DataFrame]] = None):
        # sample metadata df
        if isinstance(value, str):
            value = pd.read_csv(value)
        elif value is None:
            sample_names = list(self.sample_to_path)
            value = pd.DataFrame(data=sample_names, columns=["sample"])
        elif not isinstance(value, pd.DataFrame):
            msg = (
                "`sample_metadata` must be a path to a csv file, a DataFrame, "
                "or None."
            )
            raise ValueError(msg)
        value = _normalize_sample_metadata(value, self.sample_to_path)
        self._sample_metadata = value
        classes, class_factors = pd.factorize(value.class_)
        counts = np.unique(classes, return_counts=True)
        self.sample_to_class = dict(zip(value.index, classes))
        self.sample_per_class = dict(zip(*counts))
        self.group_factors = class_factors

    def create_assay_dir(self):
        # create assay dir
        self.assay_path.mkdir()

        # roi dir
        roi_dir_path = self.assay_path.joinpath("roi")
        roi_dir_path.mkdir()

        # dir for each sample roi
        for s in self.sample_to_code:
            sample_roi_path = roi_dir_path.joinpath(s)
            sample_roi_path.mkdir()

        # feature tables dir
        ft_dir_path = self.assay_path.joinpath("feature")
        ft_dir_path.mkdir()

    def check_step(self, step: str, previous_step: str):
        check_okay = self.params["preprocess_steps"][previous_step]
        if not check_okay:
            msg = "`{}` must be called before {}".format(previous_step, step)
            raise PreprocessingOrderError(msg)

    def check_sample_name(self, name: str):
        if name not in self.sample_to_code:
            msg = "{} not found in the assay data."
            raise ValueError(msg.format(name, self.data_path))

    def clear_roi_data(self, name: str):
        path_list = self.get_roi_list_path(name)
        for path in path_list:
            if path.is_file():
                path.unlink()

    def flag_completed(self, step: str):
        self.params["preprocess_steps"][step] = True

    def get_sample_path(self, name: str):
        self.check_sample_name(name)
        sample_path = self.sample_to_path.get(name)
        if not sample_path.is_file():
            msg = "{} does not exist."
            raise ValueError(msg)

    def get_roi_path(self, name: str) -> Path:
        self.check_sample_name(name)
        roi_path = self.assay_path.joinpath("roi", name)
        return roi_path

    def get_roi_list_path(self, name) -> Generator[Path, None, None]:
        roi_path = self.get_roi_path(name)
        n_roi = len([x for x in roi_path.glob("*")])
        for k in range(n_roi):
            yield roi_path.joinpath("{}.pickle".format(k))

    def get_feature_path(self, name: str) -> Path:
        self.check_sample_name(name)
        file_name = "{}.pickle".format(name)
        feature_path = self.assay_path.joinpath("feature", file_name)
        return feature_path

    def get_process_samples(self, step: str, kwargs: dict):
        """
        Get the list of samples to process. Checks if the samples has been
        processed already using the parameters specified by kwargs.

        Parameters
        ----------
        step : str
            name of the processing step
        kwargs : dict
            parameters used to call the processing step

        Returns
        -------
        process_samples : List[str]
            List of sample names to be processed

        """
        old_params = self.params.get(step)
        same_params = compare_dict(old_params, kwargs)
        # build list of files to process:
        samples = set(self.sample_to_code.keys())
        if same_params:
            already_processed = self.params["processed_samples"][step]
            process_samples = samples.difference(already_processed)
        else:
            process_samples = samples
        self.params["processed_samples"][step] = samples
        return process_samples

    def save(self):
        save_path = self.assay_path.joinpath("metadata.pickle")
        with open(save_path, "wb") as fin:
            pickle.dump(self, fin)


def _normalize_sample_metadata(df: pd.DataFrame, name_to_path: Dict[str, str]):
    n_rows, _ = df.shape
    df = df.copy()  # avoid modifying the original DataFrame

    if "sample" in df:
        df = df.set_index("sample")
        sample_names = set(name_to_path)
        n_samples = len(sample_names)
        sample_intersection = df.index.intersection(sample_names)
        if sample_intersection.size < n_samples:     # missing samples in df
            msg = "The following samples are missing in the sample metadata: {}"
            missing_samples = sample_names.difference(df.index)
            raise ValueError(msg.format(missing_samples))
        elif sample_intersection.size > n_samples:   # missing samples in path
            missing_samples = df.index.difference(sample_names)
            msg = (
                "The following samples were not found in the path provided "
                "and must be included in the path or removed from the sample "
                "metadata: {}".format(missing_samples)
            )
            raise ValueError(msg)
    else:
        msg = "The `sample` column is missing in the data"
        raise ValueError(msg)

    if "class" not in df:
        df["class"] = 0

    if "order" in df:
        order_dtype = df["order"].dtype
        is_int = order_dtype == int
        if not is_int:
            msg = "order column must contain integer values. Got {}"
            raise ValueError(msg.format(order_dtype))
        else:
            # check all positive and unique
            all_positive = df["order"].min() > 0
            all_unique = np.unique(df["order"]).size == n_rows
            if not (all_unique and all_positive):
                msg = "order values must be unique positive integers"
                raise ValueError(msg)
    else:
        df["order"] = np.arange(n_rows) + 1

    if "batch" in df:
        batch_dtype = df["batch"].dtype
        is_int = batch_dtype == int
        if not is_int:
            msg = "order column must contain integer values. Got {}"
            raise ValueError(msg.format(batch_dtype))
        else:
            all_positive = df["batch"].min() > 0
            all_unique = np.unique(df["order"]).size == df.shape[0]
            if not (all_unique and all_positive):
                msg = "order values must be unique positive integers"
                raise ValueError(msg)
    else:
        df["batch"] = 1

    rename_dict = {"order": "order_", "batch": "batch_", "class": "class_"}
    df = df.rename(columns=rename_dict)
    return df


class PreprocessingOrderError(ValueError):
    """
    Exception raised when the preprocessing methods are called in the wrong
    order.

    """
    pass


def make_data_container(
    feature_data: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    fill_na: bool = True
) -> DataContainer:
    """
    Organizes the detected and matched features into a DataContainer.

    Parameters
    ----------
    feature_data: DataFrame
        DataFrame obtained from detect_features function.
    sample_metadata: DataFrame
        Information from each analyzed sample. The index must be the sample
        name used in feature_data. A column named "class", with
        the class name of each sample is required. For further data processing
        run order information in a column named "order" and analytical batch
        information in a column named "batch" are recommended.
    fill_na: bool, True
        If True fill missing values in the data matrix with zeros.

    Returns
    -------
    DataContainer
    """

    # remove noise
    rm_noise_mask = feature_data["cluster_"] > -1
    feature_data = feature_data[rm_noise_mask]

    # compute aggregate statistics for each feature -> feature metadata
    estimators = {"mz": ["mean", "std", "min", "max"],
                  "rt": ["mean", "std", "min", "max"]}
    feature_metadata = feature_data.groupby("cluster_").agg(estimators)
    feature_metadata.columns = _flatten_column_multindex(feature_metadata)
    feature_metadata.index.name = "feature"

    # make data matrix
    data_matrix = feature_data.pivot(index="sample_", columns="cluster_",
                                     values="area")
    data_matrix.columns.name = "feature"
    data_matrix.index.name = "sample"
    if fill_na:
        data_matrix = data_matrix.fillna(0)

    # add samples without features as nan rows
    missing_index = sample_metadata.index.difference(data_matrix.index)
    # TODO: manage data inputting
    missing = pd.DataFrame(data=0, index=missing_index,
                           columns=data_matrix.columns)
    data_matrix = pd.concat((data_matrix, missing))
    data_matrix = data_matrix.loc[sample_metadata.index, :]

    sm = sample_metadata.copy()
    sm.index.name = "sample"
    sm = sm.rename(columns={"group_": "class", "id_": "id"})

    dc = DataContainer(data_matrix, feature_metadata, sm)
    return dc


def _flatten_column_multindex(df: pd.DataFrame):
    columns = df.columns
    level_0 = columns.get_level_values(0)
    level_1 = columns.get_level_values(1)
    col_name_map = {
        "mzmean": "mz",
        "mzstd": "mz std",
        "mzmin": "mz min",
        "mzmax": "mz max",
        "rtmean": "rt",
        "rtstd": "rt std",
        "rtmin": "rt min",
        "rtmax": "rt max"
    }
    new_names = [col_name_map[x + y] for x, y in zip(level_0, level_1)]
    return new_names


def _get_path_list(path: Union[str, List[str], Path]) -> List[Path]:
    """
    converts path to a list of Path objects pointing to mzML files. Aux function
    to Assay.__init__.

    """
    if isinstance(path, str):
        path = Path(path)

    if isinstance(path, list):
        path_list = [Path(x) for x in path]
        for p in path_list:
            # check if all files in the list exists
            if not p.is_file():
                msg = "{} doesn't exist".format(p)
                raise ValueError(msg)
    else:
        if path.is_dir():
            path_list = list(path.glob("*.mzML"))
        elif path.is_file():
            path_list = [path]
        else:
            msg = (
                "Path must be a string or Path object pointing to a "
                "directory with mzML files or a list strings with the "
                "absolute path to mzML files."
            )
            raise ValueError(msg)
    return path_list


def _load_roi(file_path: Path) -> List[Roi]:
    with file_path.open("rb") as fin:
        roi = pickle.load(fin)
    return roi


def _save_roi(file_path: Path, roi: List[Roi]):
    with file_path.open("wb") as fin:
        pickle.dump(roi, fin)


def _save_roi_list(dir_path: Path, roi_list: List[Roi]):
    for k, roi in enumerate(roi_list):
        save_path = dir_path.joinpath("{}.pickle".format(k))
        with save_path.open("wb") as fin:
            pickle.dump(roi, fin)


def compare_dict(d1: Optional[dict], d2: Optional[dict]) -> bool:
    if (d1 is None) or (d2 is None):
        res = False
    else:
        res = len(d1) == len(d2)    # first check of equality
    if res:
        for k in d1:
            v1 = d1.get(k)
            v2 = d2.get(k)
            # check type
            res = type(v1) == type(v2)
            if not res:
                break

            # check value
            if isinstance(v1, np.ndarray):
                res = np.array_equal(v1, v2)
            else:
                res = v1 == v2

            if not res:
                break
    return res


def _build_params_dict(ms_mode: str, separation: str, instrument: str):
    params_dict = {
        "MSData": {
            "ms_mode": ms_mode,
            "separation": separation,
            "instrument": instrument,
        },
        "detect_features": dict(),
        "extract_features": dict(),
        "describe_features": dict(),
        "processed_samples": {
            "detect_features": set(),
            "extract_features": set(),
            "describe_features": set()
        },
        "preprocess_steps": {
            "detect_features": False,
            "extract_features": False,
            "describe_features": False,
            "match_features": False
        }
    }
    return params_dict


def _build_processed_samples_dict():
    processed_samples_dict = {
        "detect_features": set(),
        "extract_features": set(),
        "describe_features": set()
    }
    return processed_samples_dict


def _is_assay_dir(p: Path) -> int:
    """
    check directory structure.

    Parameters
    ----------
    p: Path

    Returns
    -------
    status: int
        0 if the directory doesn't exist. 1 If it is valid assay directory.
        -1 if the directory exists and is not a valid directory.
    """
    if p.exists():
        roi_dir = p.joinpath("roi")
        ft_dir = p.joinpath("feature")
        metadata_path = p.joinpath("metadata.pickle")
        check_roi_dir = roi_dir.is_dir()
        check_ft_dir = ft_dir.is_dir()
        check_metadata = metadata_path.is_file()
        is_valid_structure = (
                check_roi_dir and
                check_ft_dir and
                check_metadata
        )
        if is_valid_structure:
            status = 1
        else:
            status = -1
    else:
        status = 0
    return status


def _create_assay_dir(p: Path, samples: dict):
    # create assay dir
    p.mkdir()

    # roi dir
    roi_dir_path = p.joinpath("roi")
    roi_dir_path.mkdir()

    # dir for each sample roi
    for s in samples:
        sample_roi_path = roi_dir_path.joinpath(s)
        sample_roi_path.mkdir()

    # feature tables dir
    ft_dir_path = p.joinpath("feature")
    ft_dir_path.mkdir()
