from .assay_processor import SingleSampleProcessor
from .assay_data import SampleData
from ..raw_data_utils import make_roi
from ..fileio import MSData
from typing import Any, Optional
import numpy as np
from ..validation import is_all_positive
from .. import _constants as c


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
    targeted_mz : numpy.ndarray or None, default=None
        A list of m/z values to perform a targeted trace creation. If this
        value is provided, only traces with these m/z values will be created.

    See Also
    --------
    lcms.LCTrace : Representation of a ROI using m/z traces.

    """

    _validation_schema = {
        "tolerance": {"type": "number", "is_positive": True},
        "max_missing": {"type": "integer", "min": 0},
        "targeted_mz": {"nullable": True, "check_with": is_all_positive},
        "min_intensity": {"type": "number", "min": 0.0},
        "min_length": {"type": "integer", "min": 2},
        "pad": {"type": "integer", "min": 0},
    }

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
        ms_data = MSData(sample.path)
        params = self.get_parameters()
        sample_data.roi = make_roi(
            ms_data,
            ms_level=sample.ms_level,
            start_time=sample.start_time,
            end_time=sample.end_time,
            **params,
        )

    def set_default_parameters(self, instrument: str, separation: str):
        defaults: dict[str, Any] = {"max_missing": 1, "pad": 2, "min_intensity": 250}
        instrument_parameters = {c.QTOF: {"tolerance": 0.01}, c.ORBITRAP: {"tolerance": 0.005}}
        separation_parameters = {c.UPLC: {"min_length": 10}, c.HPLC: {"min_length": 20}}
        defaults.update(instrument_parameters[instrument])
        defaults.update(separation_parameters[separation])
        self.set_parameters(defaults)


class LCFeatureExtractor(SingleSampleProcessor):
    _validation_schema = {}

    def __init__(self):
        pass

    @staticmethod
    def _check_data(sample_data: SampleData):
        pass

    def _func(self, sample_data: SampleData):
        # TODO: remove this after fixing ROIProcessor
        for roi in sample_data.roi:
            roi.extract_features(store_smoothed=True)

    def set_default_parameters(self, instrument: str, separation: str):
        defaults = {}
        self.set_parameters(defaults)