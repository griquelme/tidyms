"""Tools to process LC-MS-based datasets."""


import numpy as np
from collections import Counter
from typing import Any, Optional
from .base import (
    AssayData,
    FeatureExtractor,
    FeatureMatcher,
    SampleData,
    SingleSampleProcessor,
)
from . import _constants as c
from . import peaks
from .correspondence import match_features
from .fileio import MSData_from_file
from .lcms import LCTrace, Peak
from .raw_data_utils import make_roi
from .validation import is_all_positive, ValidatorWithLowerThan, validate


class LCTraceExtractor(SingleSampleProcessor):
    """
    Extracts regions-of-interest (ROI) from raw data represented as m/z traces.

    Traces are created by connecting values across consecutive scans based on the
    closeness in m/z. See the :ref:`user guide <roi-creation>` for a description
    of the algorithm used.

    Attributes
    ----------
    tolerance : positive number, default=0.01
        m/z tolerance to connect values across consecutive scans.
    max_missing : non-negative integer, default=1
        maximum number of consecutive missing values in a trace. If during the
        extension of a trace the number of consecutive values surpasses this
        threshold the trace is flagged as completed.
    min_length : positive integer, default=2
        Completed traces with length lower than this value are discarded. The
        length of a trace is the number of non-NaN elements.
    min_intensity : non-negative number , default=250.0
        At least one element in a completed trace must have an intensity greater
        than this value. Completed traces that do not meet this condition are
        discarded.
    pad: int, default=0
        Pad dummy values at the beginning and at the end of a trace. This produces
        better peak picking results when searching low intensity peaks in a trace.
    smoothing_strength: float or None, default=1.0
        Smooth the intensity of each trace using a Gaussian kernel. The smoothing
        strength is the standard deviation of the gaussian. If ``None``, no smoothing
        is applied.
    targeted_mz : numpy.ndarray or None, default=None
        A list of m/z values to perform a targeted trace creation. If this
        value is provided, only traces with these m/z values will be created.

    See Also
    --------
    lcms.LCTrace : Representation of a ROI using m/z traces.

    """

    def __init__(
        self,
        tolerance: float = 0.01,
        max_missing: int = 1,
        targeted_mz: Optional[np.ndarray] = None,
        min_intensity: float = 250,
        min_length: int = 2,
        pad: int = 0,
    ):
        self.tolerance = tolerance
        self.max_missing = max_missing
        self.targeted_mz = targeted_mz
        self.min_intensity = min_intensity
        self.min_length = min_length
        self.pad = pad

    @staticmethod
    def _check_data(sample_data: SampleData):
        # No checks are necessary for ROI extraction.
        pass

    def _func(self, sample_data: SampleData):
        sample = sample_data.sample
        # TODO: fix this after refactoring MSData
        ms_data = MSData_from_file(sample.path)
        params = self.get_parameters()
        sample_data.roi = make_roi(
            ms_data,
            ms_level=sample.ms_level,
            start_time=sample.start_time,
            end_time=sample.end_time,
            **params,
        )

    def set_default_parameters(self, instrument: str, separation: str):
        """Set the default parameters."""
        defaults: dict[str, Any] = {"max_missing": 1, "pad": 2, "min_intensity": 250}
        instrument_parameters = {c.QTOF: {"tolerance": 0.01}, c.ORBITRAP: {"tolerance": 0.005}}
        separation_parameters = {c.UPLC: {"min_length": 10}, c.HPLC: {"min_length": 20}}
        defaults.update(instrument_parameters[instrument])
        defaults.update(separation_parameters[separation])
        self.set_parameters(defaults)

    @staticmethod
    def _validate_parameters(parameters: dict):
        schema = {
            "tolerance": {"type": "number", "is_positive": True},
            "max_missing": {"type": "integer", "min": 0},
            "targeted_mz": {"nullable": True, "check_with": is_all_positive},
            "min_intensity": {"type": "number", "min": 0.0},
            "min_length": {"type": "integer", "min": 2},
            "pad": {"type": "integer", "min": 0},
            "smoothing_strength": {"type": "number", "nullable": True, "is_positive": True},
        }
        validator = ValidatorWithLowerThan(schema)
        validate(parameters, validator)


class LCFeatureExtractor(FeatureExtractor):
    """
    Detect peaks in LC m/z traces.

    Peaks are stored in the `features` attribute of each LCTrace.

    A complete description can be found :ref:`here <feature-extraction>`.

    """

    def __init__(
        self, filters: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None
    ):
        super().__init__(filters)

    @staticmethod
    def _check_data(sample_data: SampleData):
        pass

    def set_default_parameters(self, instrument: str, separation: str):
        """Set the default parameters."""
        if separation == c.HPLC:
            filters = {"width": (10, 90), "snr": (5, None)}
        else:  # mode = "uplc"
            filters = {"width": (4, 60), "snr": (5, None)}
        defaults = {"filters": filters}
        self.set_parameters(defaults)

    @staticmethod
    def _extract_features_func(roi: LCTrace, **params):
        start, apex, end = peaks.detect_peaks(roi.spint, roi.noise, roi.baseline, **params)
        roi.features = [Peak(s, a, e, roi) for s, a, e in zip(start, apex, end)]

    @staticmethod
    def _validate_parameters(parameters: dict):
        filters = parameters["filters"]
        valid_descriptors = Peak.descriptor_names()
        for descriptor in filters:
            if descriptor not in valid_descriptors:
                msg = f"{descriptor} is not a valid descriptor name."
                raise ValueError(msg)


class LCFeatureMatcher(FeatureMatcher):
    """
    Perform feature correspondence on LC-MS data using a cluster-based approach.

    Features are initially grouped by m/z and Rt similarity using DBSCAN. In
    a second step, these clusters are further processed using a GMM approach,
    obtaining clusters where each sample contributes with only one sample.

    See the :ref:`user guide <ft-correspondence>` for a detailed description of
    the algorithm.

    Attributes
    ----------
    mz_tolerance : float, default=0.01
        m/z tolerance used to group close features. Sets the `eps` parameter in
        the DBSCAN algorithm.
    rt_tolerance : float, default=3.0
        Rt tolerance in seconds used to group close features. Sets the `eps`
        parameter in the DBSCAN algorithm.
    min_fraction : float, default=0.25
        Minimum fraction of samples of a given group in a cluster. If
        `include_groups` is ``None``, the total number of sample is used
        to compute the minimum fraction.
    include_groups : List or None, default=None
        Sample groups used to estimate the minimum cluster size and number of
        chemical species in a cluster.
    max_deviation : float, default=3.0
        The maximum deviation of a feature from a cluster, measured in numbers
        of standard deviations from the cluster.
    n_jobs: int or None, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    silent : bool, default=True
        If False, shows a progress.

    """

    def __init__(
        self,
        mz_tolerance: float = 0.01,
        rt_tolerance: float = 3.0,
        min_fraction: float = 0.25,
        include_groups: Optional[list[str]] = None,
        max_deviation: float = 3.0,
        n_jobs: Optional[int] = None,
        silent: bool = True,
    ):
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.min_fraction = min_fraction
        self.include_groups = include_groups
        self.max_deviation = max_deviation
        self.n_jobs = n_jobs
        self.silent = silent

    def _func(self, data: AssayData) -> dict[int, int]:
        descriptors_names = [c.MZ, c.RT]
        descriptors = data.get_descriptors(descriptors=descriptors_names)
        feature_id = descriptors.pop("id")

        sample_names = descriptors.pop(c.SAMPLE)
        sample_to_code = {name: k for k, name in enumerate(set(sample_names))}
        samples = np.array([sample_to_code[x] for x in sample_names])

        group_names = descriptors.pop(c.CLASS)
        group_to_code = {name: k for k, name in enumerate(set(group_names))}
        groups = np.array([group_to_code[x] for x in group_names])

        X = np.hstack([descriptors[x] for x in descriptors_names]).T

        parameters = self.get_parameters()
        parameters.pop(c.MZ)
        parameters.pop(c.RT)
        parameters["tolerance"] = np.array([self.mz_tolerance, self.rt_tolerance])

        samples_per_group = _count_samples_per_group(data, group_to_code)
        labels = match_features(X, samples, groups, samples_per_group, **parameters)

        feature_id_to_label = {x: y for x, y in zip(feature_id, labels)}

        return feature_id_to_label

    def set_default_parameters(self, instrument: str, separation: str) -> dict:
        """
        Set default values based on instrument type and separation method.

        Parameters
        ----------
        instrument : str
            MS instrument used for the measurements. See

        """
        if instrument == c.QTOF:
            mz_tolerance = 0.01
        elif instrument == c.ORBITRAP:
            mz_tolerance = 0.005
        else:
            msg = f"Valid instruments are: {c.MS_INSTRUMENTS}. Got {instrument}."
            raise ValueError(msg)

        if separation == c.HPLC:
            rt_tolerance = 10
        elif separation == c.UPLC:
            rt_tolerance = 5
        else:
            msg = f"Valid separation modes are: {c.LC_MODES}. Got {separation}."
            raise ValueError(msg)

        defaults = {
            "mz_tolerance": mz_tolerance,
            "rt_tolerance": rt_tolerance,
            "min_fraction": 0.25,
            "max_deviation": 3.0,
            "include_groups": None,
            "n_jobs": None,
            "verbose": False,
        }
        return defaults


def _count_samples_per_group(data: AssayData, group_to_code: dict[str, int]) -> dict[int, int]:
    counter = Counter([group_to_code[x.group] for x in data.get_samples()])
    return dict(counter)
