import pickle
import numpy as np
import pandas as pd
from functools import wraps
from inspect import getfullargspec
from joblib import Parallel, delayed
from pathlib import Path
from typing import Callable, List, Optional, Union
from . import _constants as c
from . import validation as val
from . import raw_data_utils
from .container import DataContainer
from .correspondence import match_features
from .fileio import MSData
from .lcms import Roi
from .utils import get_progress_bar
from ._plot_bokeh import _LCAssayPlotter


# TODO: add id_ column to sample metadata
# TODO: add make_roi params to each column in sample metadata for cases where
#   more than one sample are obtained from the same file.
# TODO: test verbose false describe_features, extract_features

def _manage_preprocessing_step(func):
    """
    Checks conditions before a preprocessing step, records processing
    parameters and stores processed data to disk.

    Returns
    -------

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function parameters
        func_arg_spec = getfullargspec(func)
        func_arg_names = func_arg_spec.args[1:]    # exclude self
        params = dict(zip(func_arg_names, args[1:]))
        params.update(kwargs)
        assay = args[0]     # type: Assay

        step = func.__name__
        assay.manager.manage_step_before(step, params)
        results = func(assay, **params)
        assay.manager.manage_step_after(step, params)
        return results
    return wrapper


class Assay:
    """
    Manages data preprocessing workflows from raw data to data matrix.

    See the :ref:`user guide <processing-datasets>` for usage instructions.

    Parameters
    ----------
    data_path : str, List[str] or Path
        Contains the path of mzML files to be analyzed. ``data_path`` can be a
        string or list of strings of absolute path representations to mzML
        files in centroid mode or a Path object. Path objects can be used in
        two ways: It can point to a mzML file or to a directory. In the
        second case all mzML files inside the directory will be used.
    assay_path : str
        Path to store the assay data. If the path does not exist, a new
        directory is created. If an existing assay directory is passed,
        it loads the data from the assay.
    sample_metadata : str, Path, DataFrame or None.
        Provides information associated with each sample. If a string is
        provided, it is assumed that it is the path to a csv file with sample
        metadata information. The other columns may contain any kind of data
        but the following columns have reserved uses:

        sample
            This column is mandatory. Must contain each one of the file
            names in ``data_path``, without the `.mzML` extension.
        class
            The class of each sample.
        order
            A unique positive integer number that indicates the run order of
            each sample.
        batch
            The batch number where each sample was analyzed. The values
            must be positive integers.

        If a DataFrame is provided, it must have the same structure as the csv
        file described above. If ``None`` is provided, the samples are
        assumed to be from the same class and no order and batch information is
        used.
    ms_mode : {"centroid", "profile"}
        The mode in which the data is stored.
    instrument : {"qtof", "orbitrap"}
        The instrument type. Used to set several defaults during data
        preprocessing.
    separation : {"uplc", "hplc"}
        The separation method used. Used to set several defaults during data
        preprocessing.

    """

    def __init__(
        self,
        assay_path: Union[str, Path],
        data_path: Optional[Union[str, List[str], Path]],
        sample_metadata: Optional[Union[pd.DataFrame, str, Path]] = None,
        ms_mode: str = "centroid",
        instrument: str = "qtof",
        separation: str = "uplc",
    ):

        self.manager = _create_assay_manager(
            assay_path,
            data_path,
            sample_metadata,
            ms_mode,
            instrument,
            separation
        )
        self.separation = separation
        self.instrument = instrument
        self.plot = _create_assay_plotter(self)
        self.feature_table = None   # type: Optional[pd.DataFrame]
        self.data_matrix = None  # type: Optional[DataContainer]
        self.feature_metrics = dict()

    @property
    def separation(self) -> str:
        return self._separation

    @separation.setter
    def separation(self, value: str):
        if value in c.LC_MODES:
            self._separation = value
        else:
            msg = "{} is not a valid separation mode. Valid values are: {}."
            raise ValueError(msg.format(value, c.LC_MODES))

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value: str):
        if value in c.MS_INSTRUMENTS:
            self._instrument = value
        else:
            msg = "{} is not a valid instrument. Valid values are: {}."
            raise ValueError(msg.format(value, c.MS_INSTRUMENTS))

    @staticmethod
    def _get_feature_detection_strategy(
        strategy: Union[str, Callable] = "default"
    ) -> Callable:
        """
        Returns the function used for feature detection.
        """
        if callable(strategy):
            func = strategy
        elif strategy == "default":
            func = raw_data_utils.make_roi
        else:
            msg = "{} is not a valid strategy.".format(strategy)
            raise ValueError(msg)
        return func

    @staticmethod
    def _get_feature_extraction_strategy(
        strategy: Union[str, Callable] = "default"
    ) -> Callable:
        """
        Returns the function used for feature extraction.

        """
        if callable(strategy):
            func = strategy
        elif strategy == "default":
            func = _extract_features_default
        else:
            msg = "{} is not a valid strategy.".format(strategy)
            raise ValueError(msg)
        return func

    @staticmethod
    def _get_feature_matching_strategy(
            strategy: Union[str, Callable] = "default"
    ) -> Callable:
        """
        Sets the function used for mathing_features.

        """
        if callable(strategy):
            func = strategy
        elif strategy == "default":
            func = _match_features_default
        else:
            msg = "{} is not a valid strategy.".format(strategy)
            raise ValueError(msg)
        return func

    def get_ms_data(self, sample: str) -> MSData:
        """
        Loads a raw sample file into an MSData object.

        Parameters
        ----------
        sample: str
            Sample name used in the sample metadata.

        Returns
        -------
        MSData

        Raises
        ------
        ValueError : if the sample is not found.

        """
        sample_path = self.manager.get_sample_path(sample)
        return MSData(sample_path, **self.manager.params["MSData"])

    def get_sample_metadata(self) -> pd.DataFrame:
        """
        Creates a DataFrame with the metadata of each sample used in the assay.

        Returns
        -------
        DataFrame

        """
        return self.manager.get_sample_metadata()

    def load_roi(self, sample: str, roi_index: int) -> Roi:
        """
        Loads a ROI from a sample.

        Must be called after performing feature detection.

        Parameters
        ----------
        sample : str
            sample name used in the sample metadata.
        roi_index : int
            index of the requested ROI.

        Returns
        -------
        ROI

        Raises
        ------
        ValueError : If an invalid `name` or `roi_index` were used.
        FileNotFoundError : If a non-existent `roi_index` was used.

        See Also
        --------
        detect_features : Detect ROI in the Assay samples.
        load_roi_list : Loads all ROI from a sample.

        """
        file_name = "{}.pickle".format(roi_index)
        roi_path = self.manager.get_roi_dir_path(sample).joinpath(file_name)

        with roi_path.open("rb") as fin:
            roi = pickle.load(fin)

        return roi

    def load_roi_list(self, sample: str) -> List[Roi]:
        """
        Loads all the ROIs detected in a sample.

        Must be called after performing feature detection.

        Parameters
        ----------
        sample : str.
            sample name used in the sample metadata.

        Returns
        -------
        List[ROI]

        Raises
        ------
        ValueError : if the ROI data was not found. This error occurs if a
        wrong sample name is used or if `self.detect_features` was not called.

        See Also
        --------
        detect_features : Detect ROI in the Assay samples.

        """
        if not self.manager.check_step("detect_features"):
            msg = "`load_roi_list` must be called after `detect_features`."
            raise ValueError(msg)

        roi_list = list()
        for path in self.manager.get_roi_list_path(sample):
            with open(path, "rb") as fin:
                roi = pickle.load(fin)
                roi_list.append(roi)
        return roi_list

    def load_features(self, sample: str) -> pd.DataFrame:
        """
        Loads a table with feature descriptors for a sample.



        Parameters
        ----------
        sample : str
            sample name used in the sample metadata.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError : if the feature data was not found. This error occurs if a
        wrong sample name is used or if `self.describe_features` was not called.

        """
        if not self.manager.check_step("describe_features"):
            msg = "`load_features` must be called after `describe_features`."
            raise ValueError(msg)

        ft_path = self.manager.get_feature_path(sample)
        with ft_path.open("rb") as fin:
            df = pickle.load(fin)

        return df

    @_manage_preprocessing_step
    def detect_features(
        self,
        strategy: Union[str, Callable] = "default",
        n_jobs: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Builds Regions Of Interest (ROI) from raw data for each sample.

        ROIs are computed and saved to disk. Computed ROIs can be recovered
        using ``self.load_roi`` or ``self.load_roi_list``.

        Parameters
        ----------
        strategy : str or callable, default="default"
            If ``default`` is used, then
            :py:meth:`tidyms.raw_data_utils.make_roi` is used to build ROIs in
            each sample. A function can be passed to customize the detection
            process. The following template must be used:

            .. code-block:: python

                def func(ms_data: MSData, **kwargs) -> List[Roi]:
                    ...

        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs :
            Parameters to pass to the underlying function used. See the strategy
            parameter.

        See Also
        --------
        fileio.MSData : mzML reader
        lcms.Roi : abstract ROI
        lcms.LCRoi : ROI used in LC data

        """
        detect_features_func = self._get_feature_detection_strategy(strategy)
        process_samples = self.manager.sample_queue
        n_samples = len(process_samples)

        if n_samples:

            def iter_func(sample_list):
                for x in sample_list:
                    roi_path = self.manager.get_roi_dir_path(x)
                    ms_data = self.get_ms_data(x)
                    yield roi_path, ms_data

            def worker(args):
                roi_path, ms_data = args
                roi_list = detect_features_func(ms_data, **kwargs)
                _save_roi_list(roi_path, roi_list)

            worker = delayed(worker)
            iterator = iter_func(process_samples)
            if verbose:
                print("Creating ROI in {} samples".format(n_samples))
                bar = get_progress_bar()
                iterator = bar(iterator, total=n_samples)

            Parallel(n_jobs=n_jobs)(worker(x) for x in iterator)
        else:
            if verbose:
                print("All samples are processed already.")

        return self

    @_manage_preprocessing_step
    def extract_features(
        self,
        strategy: Union[str, Callable] = "default",
        n_jobs: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Extract features from the ROIs detected on each sample.

        Features are stored in ``features`` attribute of each ROI. ROIs can be
        recovered using ``self.load_roi`` or ``self.load_roi_list``.

        Parameters
        ----------
        strategy : str or callable, default="default"
            If ``default`` is used, then
            :py:meth:`tidyms.lcms.LCRoi.extract_features` is used to extract
            features from each ROI. A function can be passed to customize the
            extraction process. The following template must be used:

            .. code-block:: python

                def func(roi: Roi, **kwargs) -> List[Feature]:
                    ...

        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs :
            Parameters to pass to the underlying function used. See the strategy
            parameter.

        Raises
        ------
        PreprocessingOrderError : if called before ``self.detect_features``.

        See Also
        --------
        lcms.Roi : abstract ROI
        lcms.LCRoi : ROI used in LC data
        lcms.Feature : abstract feature
        lcms.Peak : feature used in LC data

        """
        extract_features_func = self._get_feature_extraction_strategy(strategy)
        process_samples = self.manager.sample_queue
        n_samples = len(process_samples)
        if n_samples:

            def iterator():
                for sample in self.manager.get_sample_names():
                    roi_path = self.manager.get_roi_dir_path(sample)
                    roi_list = self.load_roi_list(sample)
                    yield roi_path, roi_list

            def worker(args):
                roi_path, roi_list = args
                for roi in roi_list:
                    extract_features_func(roi, **kwargs)
                _save_roi_list(roi_path, roi_list)

            worker = delayed(worker)
            iterator = iterator()
            if verbose:
                print("Extracting features in {} samples".format(n_samples))
                bar = get_progress_bar()
                iterator = bar(iterator, total=n_samples)
            Parallel(n_jobs=n_jobs)(worker(x) for x in iterator)
        else:
            if verbose:
                print("All samples are processed already.")
        return self

    @_manage_preprocessing_step
    def describe_features(
        self,
        n_jobs: Optional[int] = 1,
        verbose: bool = True,
        **kwargs
    ) -> "Assay":
        """
        Compute feature descriptors for the features extracted from the data.

        Features descriptors from each sample are organized in a Pandas
        DataFrame and stored to disk and can be recovered using
        ``self.load_features``. Besides the descriptors, these DataFrames
        contain two additional columns: `roi_index` and `ft_index`. `roi_index`
        is used to indentify the ROI where the feature was detected, and
        recovered using the ``load_roi`` method. The `ft_index` value is used
        to identify the feature in the `feature` attribute of the ROI.

        Parameters
        ----------
        n_jobs: int or None, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        verbose : bool, default=True
            If ``True``, displays a progress bar.
        **kwargs : dict
            Parameters to pass to :py:meth:`tidyms.lcms.Roi.describe_features`.

        """
        process_samples = self.manager.sample_queue
        n_samples = len(process_samples)
        if n_samples:

            def iterator():
                for sample in self.manager.get_sample_names():
                    roi_path = self.manager.get_roi_dir_path(sample)
                    ft_path = self.manager.get_feature_path(sample)
                    roi_list = self.load_roi_list(sample)
                    yield roi_path, ft_path, roi_list

            def worker(args):
                roi_path, ft_path, roi_list = args
                ft_table = _describe_features_default(roi_list, **kwargs)
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
        else:
            if verbose:
                print("All samples are processed already.")
            return self

    @_manage_preprocessing_step
    def build_feature_table(self):
        """
        Merges the feature descriptors from all samples into one DataFrame.

        The feature table is stored in ``self.feature_table``. Two additional
        columns are created: `sample_` contains the sample name where the
        feature was detected. `class_` contains the corresponding class name of
        the sample.

        Raises
        ------
        ValueError : if the feature table was not built for all samples. This
        occurs if `self.describe_features` was not called.

        """
        file_name = c.FT_TABLE_FILENAME
        save_path = self.manager.assay_path.joinpath(file_name)
        if save_path.is_file():
            feature_table = pd.read_pickle(save_path)
        else:
            df_list = list()
            for name in self.manager.get_sample_names():
                df = self.load_features(name)
                df[c.SAMPLE] = name
                class_ = self.manager.get_class(name)
                if class_ is None:
                    df[c.CLASS] = 0
                else:
                    df[c.CLASS] = class_
                df_list.append(df)
            feature_table = pd.concat(df_list).reset_index(drop=True)
            feature_table[c.ROI_INDEX] = feature_table[c.ROI_INDEX].astype(int)
            feature_table[c.FT_INDEX] = feature_table[c.FT_INDEX].astype(int)
            feature_table[c.SAMPLE] = feature_table[c.SAMPLE]
            feature_table[c.CLASS] = feature_table[c.CLASS]
            feature_table.to_pickle(str(save_path))
        self.feature_table = feature_table
        return self

    @_manage_preprocessing_step
    def match_features(
        self,
        strategy: Union[str, Callable] = "default",
        **kwargs
    ):
        r"""
        Match features across samples. Each feature is labelled using an integer
        value to assign a common id. Features that do not belong to any group
        are labelled with ``-1``. The label is stored in the `label_` column
        of the feature table.

        Parameters
        ----------
        strategy : str or callable, default="default"
            If ``default`` is used, then
            :py:func:`tidyms.correspondence.match_features` is used to match
            features across samples. A function can be passed to customize the
            matching process. The following template must be used:

            .. code-block:: python

                def func(assay: Assay, **kwargs) -> Dict:
                    ...

            The dictionary must have at least one key called "cluster_",
            containing an 1D numpy with size matching the number of rows in the
            feature table. Each value is used to group features from different
            samples into a data matrix. The value ``-1`` in the array is used
            to signal features that were not matched to any group and are not
            going to be included in the data matrix. Other keys with arbitrary
            names may be used, but the value associated must be a 1D numpy
            array with size equal to the number of different groups in the
            cluster array. These arrays can be used to compute feature matching
            metrics than can be passed to the data matrix construction and used
            to assess the feature matching process.
        **kwargs :
            Parameters to pass to the underlying function used. See the strategy
            parameter.

        """
        match_features_func = self._get_feature_matching_strategy(strategy)
        process_samples = self.manager.sample_queue
        n_samples = len(process_samples)
        if n_samples:
            results = match_features_func(self, **kwargs)
            self.feature_table[c.LABEL] = results.pop(c.LABEL)
            self.feature_metrics.update(results)
            file_name = c.FT_TABLE_FILENAME
            save_path = self.manager.assay_path.joinpath(file_name)
            self.feature_table.to_pickle(str(save_path))
        return self

    def make_data_matrix(self) -> DataContainer:
        """
        Organizes the data from the feature table into a data matrix.

        Returns
        -------
        DataContainer

        """

        feature_table = self.feature_table
        if self.separation in c.LC_MODES:
            data_matrix, feature_metadata = _lc_feature_data_from_feature_table(
                feature_table
            )
        else:
            # TODO: update this code when new modes are included
            raise ValueError
        data_matrix.columns.name = c.FEATURE
        data_matrix.index.name = "sample"

        # TODO: deal with nan values. Move this to self.fill_missing
        data_matrix = data_matrix.fillna(0)

        sample_metadata = self.manager.get_sample_metadata()
        # add samples without features as rows with zeros
        missing_index = sample_metadata.index.difference(data_matrix.index)
        missing = pd.DataFrame(
            data=0,
            index=missing_index,
            columns=data_matrix.columns
        )
        data_matrix = pd.concat((data_matrix, missing))
        # sort data_matrix
        data_matrix = data_matrix.loc[sample_metadata.index, :]
        dc = DataContainer(data_matrix, feature_metadata, sample_metadata)
        self.data_matrix = dc
        return dc

    def annotate_isotopologues(self, **kwargs):
        raise NotImplementedError

    def annotate_adducts(self, **kwargs):
        raise NotImplementedError

    def fill_missing(self, **kwargs):
        raise NotImplementedError


class _AssayManager:
    """
    Manages sample metadata information, location of data files and checking
    the processing order of an Assay.

    Attributes
    ----------
    sample_metadata: DataFrame
        Information associated with each mzML file:

        Index
            Sample name of each mzML file.
        class_
            Class to which the sample belongs. Used to group samples according
            different characteristics (e.g. healthy, blank, qc, ...). The type
            of the values must be ``str``.
        order_
            Injection order of each sample. Not used in the Assay preprocessing
            but is passed to the created Data Matrix, where it is used to
            perform several curation steps. The values are unique positive
            numbers.
        batch_
            Analytical batch of each sample. Not used in the Assay preprocessing
            but is passed to the created Data Matrix, where it is used to
            perform several curation steps.
    assay_path : Path
        Directory where the preprocessed data of the Assay is stored.
    params : dict
        Parameters used in each preprocessing step. Also stores which samples
        have been processed and which preprocessing steps has been applied.

    """
    PROCESSING_STEPS = [
            "detect_features",
            "extract_features",
            "describe_features",
            "build_feature_table",
            "match_features",
            "fill_missing"
    ]

    def __init__(
        self,
        assay_path: Path,
        data_path: Union[str, List[str], Path],
        sample_metadata: Optional[Union[str, Path, pd.DataFrame]],
        ms_mode: str,
        instrument: str,
        separation: str,
    ):
        self.assay_path = assay_path
        self._sample_to_path = None
        self.sample_metadata = None
        self.sample_queue = None
        self.add_samples(data_path, sample_metadata)
        self.params = _build_params_dict(ms_mode, separation, instrument)

    def add_samples(
        self,
        data_path: Optional[Union[str, List[str], Path]],
        sample_metadata: Optional[Union[str, pd.DataFrame]]
    ):
        # set data path and related attributes
        if self._sample_to_path is None:
            self._add_samples_empty_assay(data_path, sample_metadata)
        else:
            self._add_samples_existing_assay(data_path, sample_metadata)

    def _add_samples_empty_assay(
        self,
        data_path: Optional[Union[str, List[str], Path]],
        sample_metadata: Optional[Union[str, pd.DataFrame]]
    ):
        path_list = sorted(_get_path_list(data_path))
        sample_to_path = {x.stem: x for x in path_list}
        self._sample_to_path = sample_to_path

        sample_names = self.get_sample_names()
        sm = _create_sample_metadata(sample_metadata, sample_names)
        sm = _normalize_sample_metadata(sm, sample_names)
        self.sample_metadata = sm

    def _add_samples_existing_assay(
        self,
        data_path: Optional[Union[str, List[str], Path]],
        sample_metadata: Optional[Union[str, pd.DataFrame]]
    ):
        new_path_list = sorted(_get_path_list(data_path))
        new_sample_to_path = dict()
        for path in new_path_list:
            stem = path.stem
            if stem not in self._sample_to_path:
                new_sample_to_path[stem] = path
        self._sample_to_path.update(new_sample_to_path)

        new_sample_names = list(new_sample_to_path)
        sm = _create_sample_metadata(sample_metadata, new_sample_names)
        sm = _normalize_sample_metadata(sm, new_sample_names)
        self.sample_metadata = pd.concat((self.sample_metadata, sm))
        self.clear_data("build_feature_table", new_sample_names)

    def create_assay_dir(self):
        """
        Creates a directory to store the data from a new Assay.
        """
        self.assay_path.mkdir()
        # roi dir
        roi_dir_path = self.assay_path.joinpath(c.ROI_DIR)
        roi_dir_path.mkdir()
        # dir for each sample roi
        for s in self.get_sample_names():
            sample_roi_path = roi_dir_path.joinpath(s)
            sample_roi_path.mkdir()

        # feature tables dir
        ft_dir_path = self.assay_path.joinpath(c.FT_DIR)
        ft_dir_path.mkdir()

    def check_step(self, step: str):
        """
        Checks if the previous required preprocessing steps were executed
        before the current step.

        Parameters
        ----------
        step : str
            name of the preprocessing step

        Raises
        ------
        PreprocessingOrderError : If the previous steps were not executed.

        """
        steps = _AssayManager.PROCESSING_STEPS
        try:
            ind = steps.index(step)
            previous = steps[ind - 1]
        except ValueError:      # manage optional steps
            if step == "annotate_isotopologues":
                previous = steps[2]
            elif step == "annotate_adducts":
                previous = "annotate_isotopologues"
            else:
                msg = "`{}` is not a valid preprocessing step".format(step)
                raise ValueError(msg)

        # skip checking for the first step
        if step == steps[0]:
            check_okay = True
        else:
            check_okay = self.params["preprocess_steps"][previous]

        if not check_okay:
            msg = "`{}` must be called before `{}`"
            msg = msg.format(previous, step)
            raise PreprocessingOrderError(msg)
        return check_okay

    def _check_sample_name(self, name: str):
        if name not in self._sample_to_path:
            msg = "{} not found in the assay data.".format(name)
            raise ValueError(msg)

    def clear_data(self, step: str, samples: List[str]):
        """
        Deletes old processed data.

        """
        if step == "detect_features":
            for sample in samples:
                path_list = self.get_roi_list_path(sample)
                for path in path_list:
                    if path.is_file():
                        path.unlink(missing_ok=True)
        elif step == "describe_features":
            for sample in samples:
                path = self.get_feature_path(sample)
                path.unlink(missing_ok=True)
        elif step == "build_feature_table":
            if samples:
                file_name = c.FT_TABLE_FILENAME
                table_path = self.assay_path.joinpath(file_name)
                table_path.unlink(missing_ok=True)
        self.params["preprocess_steps"][step] = False
        processed_samples = self.params["processed_samples"][step]
        self.params["processed_samples"][step] = \
            processed_samples.difference(samples)

    def flag_completed(self, step: str):
        """
        Flags a processing step as completed.
        """
        self.params["preprocess_steps"][step] = True

    def get_class(self, name: str):
        self._check_sample_name(name)
        return self.sample_metadata.at[name, c.CLASS]

    def get_sample_names(self) -> List[str]:
        return list(self._sample_to_path)

    def get_sample_path(self, name: str) -> Path:
        self._check_sample_name(name)
        return self._sample_to_path.get(name)

    def get_roi_dir_path(self, name: str) -> Path:
        self._check_sample_name(name)
        roi_path = self.assay_path.joinpath(c.ROI_DIR, name)
        return roi_path

    def get_roi_list_path(self, name: str) -> List[Path]:
        roi_path = self.get_roi_dir_path(name)
        n_roi = len([x for x in roi_path.glob("*")])
        roi_list_path = list()
        for k in range(n_roi):
            path = roi_path.joinpath("{}.pickle".format(k))
            roi_list_path.append(path)
        return roi_list_path

    def get_feature_path(self, name: str) -> Path:
        self._check_sample_name(name)
        file_name = "{}.pickle".format(name)
        feature_path = self.assay_path.joinpath(c.FT_DIR, file_name)
        return feature_path

    def get_sample_metadata(self) -> pd.DataFrame:
        # TODO: remove this method after refactoring DataContainer
        sm = self.sample_metadata.copy()
        sm.index.name = "sample"
        sm = sm.rename(columns={c.CLASS: "class", "id_": "id",
                                c.ORDER: "order", c.BATCH: "batch"})
        return sm

    def send_samples_to_queue(self, step: str, kwargs: dict):
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
        samples = set(self.get_sample_names())
        if same_params:
            already_processed = self.params["processed_samples"][step]
            process_samples = samples.difference(already_processed)
        else:
            process_samples = samples
        self.sample_queue = process_samples

    def manage_step_before(self, step: str, kwargs: dict):
        self.check_step(step)
        self.send_samples_to_queue(step, kwargs)
        if step in _AssayManager.PROCESSING_STEPS:
            ind = _AssayManager.PROCESSING_STEPS.index(step)
            for s in _AssayManager.PROCESSING_STEPS[ind:]:
                self.clear_data(s, self.sample_queue)

    def manage_step_after(self, step: str, kwargs: dict):
        self.update_params(step, kwargs)
        self.flag_completed(step)
        self.params["processed_samples"][step] = set(self.get_sample_names())
        self.sample_queue = None
        self.save()

    def save(self):
        save_path = self.assay_path.joinpath(c.MANAGER_FILENAME)
        with open(save_path, "wb") as fin:
            pickle.dump(self, fin)

    def update_params(self, step: str, params: dict):
        self.params[step] = params


class PreprocessingOrderError(ValueError):
    """
    Exception raised when the preprocessing methods are called in the wrong
    order.

    """
    pass


def _extract_features_default(roi: Roi, **kwargs):
    return roi.extract_features(**kwargs)


def _describe_features_default(roi_list: List[Roi], **kwargs) -> pd.DataFrame:
    roi_index_list = list()
    ft_index = list()
    descriptors_list = list()

    for roi_index, roi in enumerate(roi_list):
        descriptors = roi.describe_features(**kwargs)
        n_features = len(descriptors)
        descriptors_list.extend(descriptors)
        roi_index_list.append([roi_index] * n_features)
        ft_index.append(range(n_features))

    if roi_index_list:
        roi_index_list = np.hstack(roi_index_list)
        ft_index = np.hstack(ft_index)

        ft_table = pd.DataFrame(data=descriptors_list)
        ft_table["roi_index"] = roi_index_list
        ft_table["ft_index"] = ft_index
        ft_table = ft_table.dropna(axis=0)
        ft_table["roi_index"] = ft_table["roi_index"].astype(int)
        ft_table["ft_index"] = ft_table["ft_index"].astype(int)
    else:
        ft_table = pd.DataFrame()
    return ft_table


def _match_features_default(assay: Assay, **kwargs):
    # set defaults and validate input
    params = val.match_features_defaults(assay.separation, assay.instrument)
    params.update(kwargs)
    validator = val.ValidatorWithLowerThan(val.match_features_schema())
    params = val.validate(params, validator)

    class_to_n_samples = (
        assay.manager.sample_metadata[c.CLASS]
        .value_counts()
        .to_dict()
    )
    results = match_features(
        assay.feature_table,
        class_to_n_samples,
        **params
    )
    return results


def _create_assay_manager(
    assay_path,
    data_path,
    sample_metadata,
    ms_mode,
    instrument,
    separation
) -> _AssayManager:
    assay_path = _normalize_assay_path(assay_path)

    status = _is_assay_dir(assay_path)
    if status == 1:  # valid assay dir, load assay
        metadata_path = assay_path.joinpath(c.MANAGER_FILENAME)
        with open(metadata_path, "rb") as fin:
            manager = pickle.load(fin)
    elif status == 0:  # dir does not exist, create assay
        manager = _AssayManager(
            assay_path,
            data_path,
            sample_metadata,
            ms_mode,
            instrument,
            separation,
        )
        manager.create_assay_dir()
        manager.save()
    else:
        msg = "The directory is not a valid Assay directory."
        raise ValueError(msg)
    return manager


def _create_assay_plotter(assay: Assay):
    separation = assay.separation
    if separation in c.LC_MODES:
        plotter = _LCAssayPlotter(assay)
    else:
        msg = "No valid plotter found for separation={}".format(separation)
        raise ValueError(msg)
    return plotter


def _create_sample_metadata(
        sample_metadata: Optional[Union[str, pd.DataFrame]],
        sample_names: List[str]
) -> pd.DataFrame:
    # sample metadata df
    if isinstance(sample_metadata, str) or isinstance(sample_metadata, Path):
        df = pd.read_csv(sample_metadata)
    elif sample_metadata is None:
        df = pd.DataFrame(data=sample_names, columns=["sample"])
    elif isinstance(sample_metadata, pd.DataFrame):
        df = sample_metadata
    else:
        msg = (
            "`sample_metadata` must be a path to a csv file, a DataFrame "
            "or None."
        )
        raise ValueError(msg)
    return df


def _normalize_sample_metadata(df: pd.DataFrame, name_to_path: List[str]):
    n_rows, _ = df.shape
    df = df.copy()  # avoid modifying the original DataFrame

    if "sample" in df:
        df = df.set_index("sample")
        samples_in_path = set(name_to_path)
        samples_in_metadata = set(df.index)
        missing_in_metadata = samples_in_path.difference(samples_in_metadata)
        missing_in_path = samples_in_metadata.difference(samples_in_path)
        if len(missing_in_metadata):     # missing samples in df
            msg = "The following samples are missing in the sample metadata: {}"
            raise ValueError(msg.format(missing_in_metadata))
        elif len(missing_in_path):   # missing samples in path
            msg = "The following samples were not found in the data path {}"
            raise ValueError(msg.format(missing_in_path))
    else:
        msg = "The `sample` column is missing in the data"
        raise ValueError(msg)

    if "class" not in df:
        df["class"] = 0

    if "order" in df:
        order_dtype = df["order"].dtype
        is_int = order_dtype == int
        if not is_int:
            msg = "Order column dtype must be `int`. Got `{}`."
            raise TypeError(msg.format(order_dtype))
        else:
            # check all positive and unique
            all_positive = df["order"].min() > 0
            all_unique = np.unique(df["order"]).size == n_rows
            if not (all_unique and all_positive):
                msg = "Order column values must be unique positive integers."
                raise ValueError(msg)
    else:
        df["order"] = np.arange(n_rows) + 1

    if "batch" in df:
        batch_dtype = df["batch"].dtype
        is_int = batch_dtype == int
        if not is_int:
            msg = "Batch column dtype must be `int`. Got `{}`."
            raise TypeError(msg.format(batch_dtype))
        else:
            all_positive = df["batch"].min() > 0
            if not all_positive:
                msg = "Batch values must unique positive integers."
                raise ValueError(msg)
    else:
        df["batch"] = 1

    rename_dict = {"order": c.ORDER, "batch": c.BATCH, "class": c.CLASS}
    df = df.rename(columns=rename_dict)
    return df


def _normalize_assay_path(assay_path: Union[Path, str]):

    if not isinstance(assay_path, Path):
        assay_path = Path(assay_path)

    suffix = ".tidyms-assay"
    if assay_path.suffix != suffix:
        assay_path = Path(assay_path.parent / (assay_path.name + suffix))
    return assay_path.absolute()


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
    elif isinstance(path, Path):
        if path.is_dir():
            path_list = list(path.glob("*.mzML"))   # all mzML files in the dir
        elif path.is_file():
            path_list = [path]
        else:
            msg = "{} doesn't exist".format(path)
            raise ValueError(msg)
    else:
        msg = (
            "Path must be a string, a Path object pointing to a directory with "
            "mzML files or a list of strings with the path to mzML files."
        )
        raise ValueError(msg)

    for p in path_list:
        # check if all files in the list exists
        if not p.is_file() or (p.suffix != ".mzML"):
            msg = "{} is not a path to a mzML file".format(p)
            raise ValueError(msg)

    path_list = [x.absolute() for x in path_list]
    return path_list


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
        "annotate_isotopologues": dict(),
        "annotate_adducts": dict(),
        "match_features": dict(),
        "fill_missing": dict(),
        "processed_samples": {
            "detect_features": set(),
            "extract_features": set(),
            "describe_features": set(),
            "annotate_isotopologues": set(),
            "annotate_adducts": set(),
            "build_feature_table": set(),
            "match_features": set(),
            "fill_missing": set(),
        },
        "preprocess_steps": {
            "detect_features": False,
            "extract_features": False,
            "describe_features": False,
            "annotate_isotopologues": False,
            "annotate_adducts": False,
            "build_feature_table": True,
            "match_features": False,
            "fill_missing": False,
        }
    }
    return params_dict


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
        roi_dir = p.joinpath(c.ROI_DIR)
        ft_dir = p.joinpath(c.FT_DIR)
        metadata_path = p.joinpath(c.MANAGER_FILENAME)
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


def _lc_feature_data_from_feature_table(feature_table: pd.DataFrame):
    # remove noise
    rm_noise_mask = feature_table[c.LABEL] > -1
    feature_data = feature_table[rm_noise_mask]

    # feature names
    cluster_to_ft = _cluster_to_feature_name(feature_table)

    # compute aggregate statistics for each feature -> feature metadata
    estimators = {"mz": ["mean", "std", "min", "max"],
                  "rt": ["mean", "std", "min", "max"]}
    feature_metadata = feature_data.groupby(c.LABEL).agg(estimators)
    feature_metadata.columns = _flatten_column_multindex(feature_metadata)
    feature_metadata.index = feature_metadata.index.map(cluster_to_ft)
    feature_metadata.index.name = c.FEATURE

    # make data matrix
    data_matrix = feature_data.pivot(index=c.SAMPLE, columns=c.LABEL,
                                     values=c.AREA)
    data_matrix.columns = data_matrix.columns.map(cluster_to_ft)

    return data_matrix, feature_metadata


def _cluster_to_feature_name(feature_table: pd.DataFrame):
    # feature names
    unique_cluster = feature_table[c.LABEL].unique()
    n_cluster = unique_cluster[unique_cluster > -1].size
    max_n_chars_cluster = len(str(n_cluster))
    template = "FT-{{:0{}d}}".format(max_n_chars_cluster)
    cluster_to_ft = {k: template.format(k + 1) for k in range(n_cluster)}
    return cluster_to_ft
