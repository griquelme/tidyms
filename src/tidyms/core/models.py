"""
Data models used by TidyMS.

Annotation : Annotation data from a feature.
Feature : A ROI region that may contain chemical species information.
Roi : A Region of Interest extracted from raw data.
Sample : Contains metadata from a measurement.
SampleData : Container class for a Sample and the ROIs detected.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from copy import deepcopy
import json
from math import isnan, nan
from pathlib import Path

from typing import Any, Sequence

import pydantic
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator

from . import validation_utils as validation


class Roi(pydantic.BaseModel):
    """
    Base class for Regions of Interest (ROI) extracted from raw MS data.

    Properties
    ----------
    id : int
        An identifier for the ROI. Used by :class:`tidyms.base.AssayData` to
        persist data.
    features : list[Feature]
        Features extracted from the ROI.

    """

    model_config = pydantic.ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )
    id: int = -1
    features: list[Feature] = list()

    def add_feature(self, feature: Feature):
        self.features.append(feature)

    def remove_feature(self, feature: Feature):
        """Remove feature from feature list."""
        self.features.remove(feature)

    @classmethod
    def from_str(cls: type[Roi], s: str) -> Roi:
        """Load a ROI from a JSON string."""
        return cls(**json.loads(s))

    def to_str(self) -> str:
        """Serialize a ROI into a string."""
        return self.model_dump_json(exclude={"features"})


class Annotation(pydantic.BaseModel):
    """
    Store annotation data of features.

    If information for an attribute is not available, ``-1`` is used.

    Attributes
    ----------
    label : int
        Feature group of features. Groups features from different samples based
        their chemical identity. Used to create a data matrix.
    isotopologue_label : int
        Groups features from the same isotopic envelope.
    isotopologue_index : int
        Position of the feature in an isotopic envelope.
    charge : int
        Charge state.
    """

    label: int = -1
    isotopologue_label: int = -1
    isotopologue_index: int = -1
    charge: int = -1


class Feature(pydantic.BaseModel):
    """
    Base class to represent a feature.

    Attributes
    ----------
    roi: Roi
        ROI where the feature was detected.
    annotation: Annotation
        Annotation data of the feature.
    id: int
        An unique label used by :class:`tidyms.base.AssayData` to persist data.

    """

    id: int = -1
    roi: Roi
    annotation: Annotation = Annotation()
    mz: float = nan
    height: float = nan
    area: float = nan

    def __lt__(self: Feature, other: Feature | float):
        if isinstance(other, float):
            return self.get("mz") < other
        elif isinstance(other, Feature):
            return self.get("mz") < other.get("mz")

    def __le__(self, other: Feature | float):
        if isinstance(other, float):
            return self.get("mz") <= other
        elif isinstance(other, Feature):
            return self.get("mz") <= other.get("mz")

    def __gt__(self, other: Feature | float):
        if isinstance(other, float):
            return self.get("mz") > other
        elif isinstance(other, Feature):
            return self.get("mz") > other.get("mz")

    def __ge__(self, other: Feature | float):
        if isinstance(other, float):
            return self.get("mz") >= other
        elif isinstance(other, Feature):
            return self.get("mz") >= other.get("mz")

    def _set_mz(self):
        """Get the feature m/z."""
        if isnan(self.mz):
            self.mz = 100.0

    def _set_area(self):
        """Get the feature area."""
        if isnan(self.area):
            self.area = 100.0

    def _set_height(self):
        """Get the feature height."""
        if isnan(self.height):
            self.height = 100.0

    def describe(self) -> dict[str, float]:
        """
        Compute all available descriptors for the feature.

        A descriptor is any method that starts with get_.

        """
        descriptors = dict()
        for descriptor in self.descriptor_names():
            descriptors[descriptor] = self.get(descriptor)
        return descriptors

    def to_str(self) -> str:
        """
        Serialize the feature data into a string.

        ROI data must not be serialized. See Peak implementation as an example.

        """
        exclude = self.descriptor_names() | {"roi", "annotation"}
        return self.model_dump_json(exclude=exclude)

    @classmethod
    def from_str(cls, s: str, roi: Roi, annotation: Annotation) -> Feature:
        """
        Create a feature instance from a string.

        Parameters
        ----------
        s : str
            Feature string generated with `to_str`.
        roi : Roi
            ROI where the feature was detected.
        annotation : Annotation

        """
        d = json.loads(s)
        return cls(roi=roi, annotation=annotation, **d)

    def get(self, descriptor: str) -> float:
        """
        Compute descriptor using its name.

        A descriptor is a method that starts with `get_`.

        Returns
        -------
        float
            The value of the requested descriptor

        Raises
        ------
        ValueError
            If an non existent descriptor name is passed.

        """
        try:
            val = getattr(self, descriptor)
            if isnan(val):
                getattr(self, f"_set_{descriptor}")()
                val = getattr(self, descriptor)
            return val
        except AttributeError as e:
            msg = f"{descriptor} is not a valid descriptor."
            raise ValueError(msg) from e

    @classmethod
    @lru_cache
    def descriptor_names(cls) -> set[str]:
        """
        List all descriptor names.

        A descriptor is a method that starts with `get_`.

        Returns
        -------
        list [str]

        """
        # trim _get_ from function name
        return {x[5:] for x in dir(cls) if x.startswith("_set_")}


class AnnotableFeature(Feature, ABC):
    """Base feature with also implements methods for feature annotation."""

    @abstractmethod
    def compare(self, other: "Feature") -> float:
        """
        Compare the similarity between two features.

        Must return a number between 0.0 and 1.0.

        Used by annotation methods to group features.

        """
        ...

    @staticmethod
    @abstractmethod
    def compute_isotopic_envelope(
        feature: Sequence[Feature],
    ) -> tuple[list[float], list[float]]:
        """
        Compute the isotopic envelope from a list of isotopologue features.

        Must return two lists:

        - The sorted m/z values of the envelope.
        - The abundance associated to each isotopologue. Normalized to 1.

        Used by annotation algorithms to annotate isotopologues.

        """
        ...


class Sample(pydantic.BaseModel):
    """
    Sample class to manage iteration over MSData objects.

    Attributes
    ----------
    path : Path or str
        Path to a raw data file.
    id : str
        Sample name.
    ms_level : int, default=1
        MS level of data to use.
    start_time: float, default=0.0
        Minimum acquisition time of MS scans to include. If ``None``, start from the first scan.
    end_time: float or None, default=None
        Maximum acquisition time of MS scans to include. If ``None``, end at the last scan.
    group : str, default=""
        Sample group.
    order : int, default=0
        Measurement order of sample in an assay.
    batch : int, default=0
        Analytical batch of sample in an assay.

    """

    path: Annotated[Path, AfterValidator(validation.is_file)]
    id: str
    ms_level: pydantic.PositiveInt = 1
    start_time: pydantic.NonNegativeFloat = 0.0
    end_time: pydantic.NonNegativeFloat | None = None
    group: str = ""
    order: pydantic.NonNegativeInt = 0
    batch: pydantic.NonNegativeInt = 0

    @pydantic.field_serializer("path")
    def serialize_path(self, path: Path, _info) -> str:
        return str(path)


class SampleData:
    """
    Container class for sample data.

    Attributes
    ----------
    sample : Sample
    copy : bool
        If ``True``, create an independent copy of the data for each processing
        step.

    """

    def __init__(self, sample: Sample, snapshot: bool = False) -> None:
        self.sample = sample
        self._snapshot = snapshot
        self._latest_roi: list[Roi] = list()
        self._roi_snapshots: list[list[Roi]] = list()
        self._roi_snapshots.append(self._latest_roi)
        self._processing_steps: list[ProcessorInformation] = list()
        self._snapshots_id: set[str] = set()

    @property
    def processing_steps(self) -> list[ProcessorInformation]:
        return self._processing_steps

    def get_feature_list(self) -> list[Feature]:
        feature_list = list()
        for roi in self.get_roi_list():
            feature_list.extend(roi.features)
        return feature_list

    def get_feature_list_snapshot(self, step: ProcessorInformation) -> list[Feature]:
        """Update the feature attribute using feature data from ROIs."""
        feature_list = list()
        for roi in self.get_roi_list_snapshot(step):
            feature_list.extend(roi.features)
        return feature_list

    def set_roi_list(self, roi_list: list[Roi], step: ProcessorInformation):

        if step.id in self._snapshots_id:
            msg = f"A snapshot with id={step.id} already exists for sample={self.sample.id}."
            raise ValueError(msg)

        self._latest_roi = roi_list
        self._snapshot_data(step)

    def get_roi_list(self) -> list[Roi]:
        return self._roi_snapshots[-1]

    def get_roi_list_snapshot(self, step: ProcessorInformation) -> list[Roi]:
        """
        Retrieve the list of ROIs stored.

        Parameters
        ----------
        step: str
            The id of processing step applied to the sample. If the SampleData
            instance was created with the `copy` parameter set to ``False``,
            this parameter is ignored.

        Raises
        ------
        ValueError
            If an invalid processing step id is passed.

        """
        if step.id not in self._snapshots_id:
            self._snapshot_data(step)

        if self._snapshot:
            step_index = self._processing_steps.index(step)
            roi_list = self._roi_snapshots[step_index]
        else:
            roi_list = self._latest_roi
        return roi_list

    def delete_empty_roi(self, step: ProcessorInformation):
        """Delete ROI with no features."""
        if step.id not in self._snapshots_id:
            msg = f"No snapshot taken for processor with id={step.id} for sample={self.sample.id}."
            raise ValueError(msg)

        index = self._processing_steps.index(step)
        self._roi_snapshots[index] = [
            x for x in self.get_roi_list_snapshot(step) if x.features
        ]

    def _snapshot_data(self, info: ProcessorInformation):
        if self._snapshot:
            roi_list = [x.model_copy(deep=True) for x in self._latest_roi]
        else:
            roi_list = self._latest_roi

        self._roi_snapshots.append(roi_list)
        self._latest_roi = roi_list
        self._processing_steps.append(info)
        self._snapshots_id.add(info.id)

    def delete_latest_snapshot(self):
        if self._processing_steps:
            self._processing_steps.pop()
            self._roi_snapshots.pop()
            self._latest_roi = self._roi_snapshots[-1]


class ProcessorInformation(pydantic.BaseModel):
    """Stores sample processing step name and parameters"""

    id: str
    pipeline: str | None
    order: int | None
    parameters: dict[str, Any]

    @pydantic.field_serializer("parameters")
    def serialize_parameters(self, parameters: dict[str, Any], _info) -> str:
        return json.dumps(parameters)
