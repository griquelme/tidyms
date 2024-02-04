"""Dummy classes for tests."""

from __future__ import annotations

import json

from pathlib import Path
from random import random, randint
from typing import Sequence

from tidyms.base import assay, base
from tidyms.base import constants as c
from tidyms.base.base import Feature, Roi


class ConcreteRoi(base.Roi):
    def __init__(self, data: list[float], *, id_: int = -1):
        self.data = data
        super().__init__(id_=id_)

    def to_str(self) -> str:
        return json.dumps({"data": self.data, "id": self.id})

    @classmethod
    def from_str(cls, s: str) -> ConcreteRoi:
        d = json.loads(s)
        roi = cls(d["data"], id_=d["id"])
        return roi

    def __eq__(self, other: ConcreteRoi) -> bool:
        return (self.data == other.data) and (self.id == other.id)


class ConcreteFeature(base.Feature):
    def __init__(
        self,
        roi: base.Roi,
        data: int,
        id_: int = -1,
        annotation: base.Annotation | None = None,
    ):
        super().__init__(roi, id_=id_, annotation=annotation)
        self.data = data

    def get_mz(self):
        return 100.0 * self.data

    def get_area(self):
        return 100.0

    def get_height(self) -> float:
        return 100.0

    def get_my_custom_descriptor(self) -> float:
        return 100.0

    def to_str(self) -> str:
        return json.dumps({"data": self.data, "id": self.id})

    @classmethod
    def from_str(
        cls, s: str, roi: base.Roi, annotation: base.Annotation
    ) -> ConcreteFeature:
        d = json.loads(s)
        ft = cls(roi, data=d["data"], id_=d["id"], annotation=annotation)
        ft.roi = roi
        ft.id = d["id"]
        return ft

    def equal(self, other: ConcreteFeature) -> bool:
        return (
            (self.data == other.data)
            and (self.id == other.id)
            and (self.annotation == other.annotation)
        )


class DummyRoiExtractor(assay.BaseRoiExtractor):
    param1: int = 10
    param2: str = "default"

    def _extract_roi_func(self, sample: base.Sample) -> list[ConcreteRoi]:
        return [create_dummy_roi() for _ in range(5)]

    def get_default_parameters(
        self,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        separation: c.SeparationMode = c.SeparationMode.UPLC,
        polarity: c.Polarity = c.Polarity.POSITIVE,
    ):
        return dict()


class DummyRoiTransformer(assay.BaseRoiTransformer):
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


class DummyFeatureExtractor(assay.BaseFeatureExtractor):
    param1: int = 2
    param2: str = "default"

    def _extract_features_func(self, roi: ConcreteRoi) -> Sequence[ConcreteFeature]:
        feature_list = list()
        for _ in range(self.param1):
            ann = base.Annotation()
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


class DummyFeatureTransformer(assay.BaseFeatureTransformer):
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


def create_dummy_sample(path: Path, suffix: int, group: str = "") -> base.Sample:
    file = path / f"sample-{suffix}.mzML"
    file.touch()
    sample = base.Sample(path=file, id=file.stem, group=group, order=suffix)
    return sample


def create_dummy_roi() -> ConcreteRoi:
    data = [random() for _ in range(5)]
    return ConcreteRoi(data)


def create_dummy_feature(
    roi: ConcreteRoi, annotation: base.Annotation
) -> ConcreteFeature:
    data = randint(0, 10)
    return ConcreteFeature(roi, data, annotation=annotation)


def add_dummy_features(roi_list: list[ConcreteRoi], n: int):
    label_counter = 0
    for roi in roi_list:
        for _ in range(n):
            annotation = base.Annotation(label=label_counter)
            ft = create_dummy_feature(roi, annotation)
            roi.add_feature(ft)
            label_counter += 1


def get_feature_list(roi_list: list[ConcreteRoi]) -> list[ConcreteFeature]:
    feature_list = list()
    for roi in roi_list:
        feature_list.extend(roi.features)
    return feature_list
