"""Dummy classes for tests."""

from __future__ import annotations

import json

from tidyms.base import base


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
