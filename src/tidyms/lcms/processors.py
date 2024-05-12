"""Tools to process LC-MS-based datasets."""

import pydantic
from numpy import zeros_like

from ..algorithms.raw import MakeRoiParameters, make_roi
from ..algorithms.signal import detect_peaks, estimate_baseline, estimate_noise, smooth
from ..annotation.correspondence import FeatureCorrespondenceParameters, match_features
from ..core import constants as c
from ..core.models import FeatureGroupMap, Sample
from ..core.processors import AssayDataInterface, FeatureExtractor, FeatureMatcher, RoiExtractor, RoiTransformer
from ..io import MSData
from .lcms import MZTrace, Peak


class LCTraceExtractor(RoiExtractor, MakeRoiParameters):
    """
    Extracts regions-of-interest (ROI) from raw data represented as m/z traces.

    Traces are created by connecting values across consecutive scans based on the
    closeness in m/z. See the :ref:`user guide <roi-creation>` for a description
    of the algorithm used.

    See Also
    --------
    lcms.MZTrace : Representation of a ROI using m/z traces.
    lcms.MakeRoiParameters : Parameters used by the ROI extraction algorithm

    """

    def _extract_roi_func(self, sample: Sample) -> list[MZTrace]:
        ms_data = MSData(sample.path, reader=sample.reader)
        params = self.model_copy()
        params.ms_level = sample.ms_level
        params.start_time = sample.start_time
        params.end_time = sample.end_time
        return make_roi(ms_data, params)

    def get_default_parameters(self, instrument: c.MSInstrument, separation: c.SeparationMode, polarity: c.Polarity):
        """
        Set the processor default parameters.

        Parameters
        ----------
        instrument : MSInstrument
        separation : SeparationMode
        polarity : Polarity

        """
        params = dict(max_missing=1, pad=2, min_intensity=250.0)

        if instrument == c.MSInstrument.QTOF:
            params["tolerance"] = 0.01
        elif instrument == c.MSInstrument.ORBITRAP:
            params["tolerance"] = 0.005
        else:
            raise NotImplementedError

        if separation == c.SeparationMode.UPLC:
            params["min_length"] = 10
        elif separation == c.SeparationMode.HPLC:
            params["min_length"] = 20
        else:
            raise NotImplementedError
        return params


class LCTraceBaselineEstimator(RoiTransformer):
    """
    Estimate the noise level and baseline in an m/z trace.

    The default values for this filter usually produce good results in all LC
    traces. Do not modify these values unless you know what you are doing.
    See :ref:`here <peak-picking>` for a description of the noise estimation
    and baseline estimation algorithms.

    """

    robust: bool = True
    """If ``True``, use the median absolute deviation as an estimator of the
    noise standard deviation. If ``False``. use the standard deviation."""

    min_slice_size: pydantic.PositiveInt = 200
    """The minimum size of a signal slice for local noise estimation. If the signal
    size is smaller than this value, the noise is estimated using the whole array.
    """

    n_slices: pydantic.PositiveInt = 5
    """Number of slices to create. The size of each slice must be greater than
    `min_slice_size`.
    """

    smoothing_strength: pydantic.PositiveFloat | None = 1.0
    """If specified, apply a temporary gaussian smoothing to the trace intensity.
    This step usually improves baseline estimation.
    """

    min_proba: float = pydantic.Field(default=0.05, le=0.5, gt=0.0)
    """The minimum probability of a signal chunk to be considered as baseline."""

    def _transform_roi(self, roi: MZTrace):
        roi.noise = estimate_noise(
            roi.spint, min_slice_size=self.min_slice_size, n_slices=self.n_slices, robust=self.robust
        )
        if self.smoothing_strength is not None:
            roi.spint = smooth(roi.spint, self.smoothing_strength)
        roi.baseline = estimate_baseline(roi.spint, roi.noise, min_proba=self.min_proba)

    def get_default_parameters(self, instrument: c.MSInstrument, separation: c.SeparationMode, polarity: c.Polarity):
        """
        Set the processor default parameters.

        Parameters
        ----------
        instrument : MSInstrument
        separation : SeparationMode
        polarity : Polarity

        """
        return dict()


class PeakExtractor(FeatureExtractor):
    """
    Extract peaks from LC m/z traces.

    A complete description can be found :ref:`here <feature-extraction>`.

    """

    def get_default_parameters(self, instrument: c.MSInstrument, separation: c.SeparationMode, polarity: c.Polarity):
        """
        Set the processor default parameters.

        Parameters
        ----------
        instrument : MSInstrument
        separation : SeparationMode
        polarity : Polarity

        """
        params = dict()
        if separation == c.SeparationMode.HPLC:
            params["filters"] = {"width": (10, 90), "snr": (5, None)}
        elif separation == c.SeparationMode.UPLC:
            params["filters"] = {"width": (4, 60), "snr": (5, None)}
        else:
            raise NotImplementedError
        return params

    def _extract_features_func(self, roi: MZTrace):
        if roi.noise is None:
            roi.noise = estimate_noise(roi.spint)

        if roi.baseline is None:
            roi.baseline = zeros_like(roi.spint)

        start, apex, end = detect_peaks(roi.spint, roi.noise, roi.baseline)
        return [Peak(start=s, apex=a, end=e, roi=roi) for s, a, e in zip(start, apex, end)]


class LCFeatureMatcher(FeatureMatcher, FeatureCorrespondenceParameters):
    """
    Perform feature correspondence on LC-MS datasets using a cluster-based approach.

    Features are initially grouped by m/z and Rt similarity using DBSCAN. In
    a second step, these clusters are further processed using a GMM approach,
    obtaining clusters where each sample contributes with only one sample.

    See the :ref:`user guide <ft-correspondence>` for a detailed description of
    the algorithm.

    """

    mz_tolerance: pydantic.PositiveFloat = 0.01
    """m/z tolerance used to group close features. Sets the `eps` parameter in the DBSCAN algorithm."""

    rt_tolerance: pydantic.PositiveFloat = 3.0
    """Rt tolerance in seconds used to group close features. Sets the `eps` parameter in the DBSCAN algorithm."""

    def _func(self, data: AssayDataInterface[Peak, MZTrace]) -> list[FeatureGroupMap]:
        tolerance = {"mz": self.mz_tolerance, "rt": self.rt_tolerance}
        descriptors = data.fetch_descriptors(descriptors=list(tolerance))
        return match_features(descriptors, tolerance, self)

    def get_default_parameters(self, instrument: c.MSInstrument, separation: c.SeparationMode, polarity: c.Polarity):
        """
        Set the processor default parameters.

        Parameters
        ----------
        instrument : MSInstrument
        separation : SeparationMode
        polarity : Polarity

        """
        params = {"min_fraction": 0.25, "max_deviation": 3.0, "include_groups": None, "n_jobs": None, "silent": True}

        if instrument == c.MSInstrument.QTOF:
            params["mz_tolerance"] = 0.01
        elif instrument == c.MSInstrument.ORBITRAP:
            params["mz_tolerance"] = 0.005
        else:
            raise NotImplementedError

        if separation == c.SeparationMode.HPLC:
            params["rt_tolerance"] = 10
        elif separation == c.SeparationMode.UPLC:
            params["rt_tolerance"] = 5
        else:
            raise NotImplementedError

        return params
