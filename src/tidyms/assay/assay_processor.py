"""
Tools to process Assay data.

"""
import inspect
import logging as log
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing.pool import Pool
from typing import Callable, Generator, Optional, Sequence, Type
from .assay_data import AssayData, Sample
from .. import _constants as c
from .. import validation as val
from ..utils import get_progress_bar
from ..lcms import Feature, Roi


PROCESSING_ORDER = [
    c.DETECT_FEATURES,
    c.EXTRACT_FEATURES,
    c.MATCH_FEATURES,
    c.MAKE_DATA_MATRIX,
]


class PreprocessingOrderError(ValueError):
    """
    Exception raised when the preprocessing methods are called in the wrong order.

    """

    pass


class AssayProcessor(ABC):
    """
    The base class for Assay processors.

    Contains functionality validate Processor parameters and store parameters
    in the DB managed by AssayData.

    Processor Parameters MUST be defined as key-value parameters and value
    constraints specified in the _validation_schema.

    `_step` must be specified for each schema.

    Attributes
    -----------
    _step : str
        (class attribute). Processing step. Must be one of the values defined
        in `_constants.PREPROCESSING_STEPS`.
    _validation_schema : dict
        (class attribute). Dictionary with a cerberus schema used to validate
        processor parameters.

    """

    _step: str
    _validation_schema: dict
    _roi: Type[Roi]
    _feature: Type[Feature]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        arg_str = ", ".join([f"{k}={v}" for k, v in self.get_parameters().items()])
        return f"{name}({arg_str})"

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""

        init_signature = inspect.signature(cls.__init__)

        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Processors should always specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_parameters(self) -> dict:
        """
        Get the estimator parameters.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        parameters = {k: getattr(self, k) for k in self._get_param_names()}
        validator = val.ValidatorWithLowerThan(self._validation_schema)
        val.validate(parameters, validator)
        return parameters

    def set_parameters(self, parameters: dict):
        """
        Set processor parameters.

        Parameters
        ----------
        parameters : dict

        """
        valid_parameters = self._get_param_names()
        for k, v in parameters.items():
            if k not in valid_parameters:
                msg = f"{k} is not a valid parameter for {self.__class__}"
                raise ValueError(msg)
            setattr(self, k, v)

    def _store_parameters(self, data: AssayData):
        data.set_processing_parameters(self._step, self.get_parameters())

    @abstractmethod
    def process(self, assay_data: AssayData):
        ...

    @staticmethod
    @abstractmethod
    def _func(*args, **kwargs):
        ...

    def _check_step(self, data: AssayData):
        all_samples = data.get_samples()
        step_index = PROCESSING_ORDER.index(self._step)
        if step_index > 0:
            previous_step = PROCESSING_ORDER[step_index - 1]
            processed_samples = data.get_samples(step=previous_step)
            check_okay = set(all_samples) == set(processed_samples)
        else:
            check_okay = True
            previous_step = None

        if not check_okay:
            msg = f"{previous_step} method must be applied on Assay before applying {self._step}."
            raise PreprocessingOrderError(msg)

    @abstractmethod
    def set_default_parameters(self, instrument: str, separation: str):
        ...


class SingleSampleProcessor(AssayProcessor):
    """
    Base class to process samples in an assay independently.

    """

    @abstractmethod
    def _generate_data(self, assay_data: AssayData, sample_list: list[Sample]) -> Generator:
        ...

    @abstractmethod
    def _delete_old_data(self, assay_data: AssayData):
        ...

    @staticmethod
    @abstractmethod
    def _save_results(assay_data: AssayData, sample: Sample, results):
        ...

    def _create_worker(self) -> Callable:
        parameters = self.get_parameters()
        worker = partial(self._func, **parameters)
        return worker

    def process(
        self, assay_data: AssayData, verbose: bool = False, n_jobs: Optional[int] = None
    ):
        if self._roi != assay_data.roi:
            msg = "Incompatible ROI types."
            raise ValueError(msg)

        self._check_step(assay_data)
        self._delete_old_data(assay_data)
        sample_list = assay_data.get_unprocessed_samples(self._step)
        sample_data = self._generate_data(assay_data, sample_list)
        worker = self._create_worker()
        logger = log.getLogger("assay")
        logger.info(f"Processing samples with {self}.")
        with Pool(n_jobs) as pool:
            # issue tasks to the process pool
            iterator = zip(pool.imap_unordered(worker, sample_data), sample_list)
            if verbose:
                progress_bar = get_progress_bar()
                iterator = progress_bar(iterator, total=len(sample_list))
            for roi_list, sample in iterator:
                self._save_results(assay_data, sample, roi_list)
                logger.info(f"Processed sample {sample.id}.")


class RoiExtractor(SingleSampleProcessor):
    """
    Base class to extract ROI from a set of samples in an Assay.

    Attributes
    ----------
    _roi : Roi
        (class attribute) ROI class extracted from samples.

    """

    _roi: Type[Roi]
    _step = c.DETECT_FEATURES

    def _delete_old_data(self, data: AssayData):
        parameters = self.get_parameters()
        db_parameters = data.get_processing_parameters(self._step)
        if (db_parameters is not None) and (parameters != db_parameters):
            data.delete_roi()
            logger = log.getLogger("assay")
            msg = (
                "Some samples where processed with different extraction"
                " parameters. Deleting old ROI data."
            )
            logger.info(msg)

    def _generate_data(
        self, assay_data: AssayData, sample_list: list[Sample]
    ) -> Generator[Sample, None, None]:
        for sample in sample_list:
            yield sample

    @staticmethod
    def _save_results(assay_data: AssayData, sample: Sample, results: Sequence[Roi]):
        assay_data.add_roi_list(results, sample)


class FeatureExtractor(SingleSampleProcessor):
    """
    Base class to extract features from a set of samples in an Assay.

    Attributes
    ----------
    _roi : Roi
        (class attribute) ROI class extracted from samples.
    _feature : Feature
        (class attribute) Feature class extracted from ROIs.

    """

    _roi: Type[Roi]
    _step = c.DETECT_FEATURES

    def _delete_old_data(self, data: AssayData):
        parameters = self.get_parameters()
        db_parameters = data.get_processing_parameters(self._step)
        if (db_parameters is not None) and (parameters != db_parameters):
            logger = log.getLogger("assay")
            data.delete_features()
            msg = (
                "Some samples where processed with different extraction"
                " parameters. Deleting old feature data."
            )
            logger.info(msg)

    def _generate_data(
        self, assay_data: AssayData, sample_list: list[Sample]
    ) -> Generator[Sequence[Roi], None, None]:
        for sample in sample_list:
            yield assay_data.get_roi_list(sample)

    @staticmethod
    def _save_results(assay_data: AssayData, sample: Sample, results: Sequence[Roi]):
        assay_data.add_features(results, sample)
