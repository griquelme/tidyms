"""Dummy classes for tests."""

from __future__ import annotations

from math import nan
from pathlib import Path
from random import randint, random
from typing import Sequence

from tidyms.core import constants as c
from tidyms.core import models, processors
from tidyms.core.models import Feature, Roi


class ConcreteRoi(models.Roi):
    data: list[float] = list()

    def __eq__(self, other: ConcreteRoi) -> bool:
        return (self.data == other.data) and (self.id == other.id)


class ConcreteFeature(models.Feature):
    data: int
    custom_descriptor: float = nan

    def _set_custom_descriptor(self):
        self.custom_descriptor = 100.0

    def _set_mz(self):
        # make mz depend on data field to test equal method
        self.mz = 100.0 * self.data

    def equal(self, other: ConcreteFeature) -> bool:
        return (
            (self.data == other.data)
            and (self.id == other.id)
            and (self.annotation == other.annotation)
        )

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


def create_dummy_feature(
    roi: ConcreteRoi, annotation: models.Annotation
) -> ConcreteFeature:
    data = randint(0, 10)
    return ConcreteFeature(roi=roi, data=data, annotation=annotation)


def add_dummy_features(roi_list: list[ConcreteRoi], n: int):
    label_counter = 0
    for roi in roi_list:
        for _ in range(n):
            annotation = models.Annotation(label=label_counter)
            ft = create_dummy_feature(roi, annotation)
            roi.add_feature(ft)
            label_counter += 1


def get_feature_list(roi_list: list[ConcreteRoi]) -> list[ConcreteFeature]:
    feature_list = list()
    for roi in roi_list:
        feature_list.extend(roi.features)
    return feature_list
