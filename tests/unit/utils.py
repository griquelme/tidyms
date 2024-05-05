"""Dummy classes for tests."""

from __future__ import annotations

from math import isnan, nan
from pathlib import Path
from random import randint, random
from typing import Sequence

from tidyms.core import constants as c
from tidyms.core import models, processors
from tidyms.core.models import Feature, Roi, IsotopicEnvelope
from tidyms.core.processors import SampleData
from typing_extensions import Self


class ConcreteRoi(models.Roi):
    data: list[float] = list()

    def __eq__(self, other: ConcreteRoi) -> bool:
        is_equal_data = (self.data == other.data) and (self.id == other.id)
        is_equal_n_features = len(self.features) == len(other.features)
        is_equal_features = all(x.equal(y) for x, y in zip(self.features, other.features))  # type: ignore
        return is_equal_data and is_equal_n_features and is_equal_features


class ConcreteFeature(models.AnnotableFeature[ConcreteRoi]):
    data: int
    custom_descriptor: float = nan

    def _set_custom_descriptor(self):
        self.custom_descriptor = 100.0

    def _set_mz(self):
        # make mz depend on data field to test equal method
        if isnan(self.mz):
            self.mz = 100.0 * self.data

    def _set_height(self):
        if isnan(self.height):
            self.height = self.area

    def equal(self, other: ConcreteFeature) -> bool:
        return self.data == other.data and self.annotation == other.annotation

    def compare(self, other: Self) -> float:
        return 1.0

    @staticmethod
    def compute_isotopic_envelope(features: list[ConcreteFeature]) -> IsotopicEnvelope:
        total_area = sum([x.area for x in features])
        mz = [x.mz for x in features]
        p = [x.area / total_area for x in features]
        return IsotopicEnvelope(mz=mz, p=p)


@processors.ProcessorRegistry.register
class DummyRoiExtractor(processors.RoiExtractor):
    param1: int = 10
    param2: str = "default"

    def _extract_roi_func(self, sample: models.Sample) -> list[ConcreteRoi]:
        return [create_dummy_roi() for _ in range(5)]

    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        return dict()


@processors.ProcessorRegistry.register
class DummyRoiTransformer(processors.RoiTransformer):
    param1: float = 10.0
    param2: bool = False

    def _transform_roi(self, roi: Roi):
        pass

    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        return dict()


@processors.ProcessorRegistry.register
class DummyFeatureExtractor(processors.FeatureExtractor):
    param1: int = 2
    param2: str = "default"

    def _extract_features_func(self, roi: ConcreteRoi) -> Sequence[ConcreteFeature]:
        feature_list = list()
        for _ in range(self.param1):
            ann = models.Annotation()
            ft = create_dummy_feature(roi, ann)
            feature_list.append(ft)
        return feature_list

    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        return dict()


@processors.ProcessorRegistry.register
class DummyFeatureTransformer(processors.FeatureTransformer):
    param1: float = 10.0
    param2: bool = False

    def _transform_feature(self, feature: Feature):
        pass

    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        return dict()


def create_dummy_sample(path: Path, suffix: int, group: str = "") -> models.Sample:
    file = path / f"sample-{suffix}.mzML"
    file.touch()
    sample = models.Sample(path=file, id=file.stem, group=group, order=suffix)
    return sample


def create_dummy_roi() -> ConcreteRoi:
    data = [random() for _ in range(5)]
    return ConcreteRoi(data=data)


def create_dummy_feature(roi: ConcreteRoi, annotation: models.Annotation) -> ConcreteFeature:
    data = randint(0, 10)
    return ConcreteFeature(roi=roi, data=data, annotation=annotation, height=float(data))


def add_dummy_features(roi_list: list[ConcreteRoi], n: int):
    label_counter = 0
    for roi in roi_list:
        for _ in range(n):
            annotation = models.Annotation(group=label_counter)
            ft = create_dummy_feature(roi, annotation)
            roi.add_feature(ft)
            label_counter += 1


def get_feature_list(roi_list: list[ConcreteRoi]) -> list[ConcreteFeature]:
    feature_list = list()
    for roi in roi_list:
        feature_list.extend(roi.features)
    return feature_list


def create_dummy_sample_data(
    path: Path, suffix: int, group: str = "", with_roi: bool = False, with_feature: bool = False
) -> SampleData:
    """
    Create a dummy sample data instance.

    Parameters
    ----------
    path : Path
        dummy path pointing to raw data.
    suffix : int
        numeric suffix appended to sample id.
    group : str, optional
        The group where the sample belongs to., by default ""
    with_roi : bool, optional
        If ``True``, adds dummy ROIs to sample, by default False
    with_feature : bool, optional
        If ``True`` add dummy features to ROI, by default False

    Returns
    -------
    SampleData

    """
    sample = create_dummy_sample(path, suffix, group)

    if with_roi:
        roi_list = [create_dummy_roi() for _ in range(5)]
    else:
        roi_list = list()

    if with_roi and with_feature:
        add_dummy_features(roi_list, 5)

    return SampleData(sample=sample, roi=roi_list)
