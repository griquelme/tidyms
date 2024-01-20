"""
Abstract base classes used by TidyMS.

Annotation : Stores annotation data from a feature.
Feature : A region associated with a ROI that contains a chemical species.
Roi : A Region of Interest extracted from raw data. Usually a subset of raw data.
Sample : Stores metadata from a measurement.
SampleData : Container class for a Sample and the ROIs detected.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Type

import pydantic
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator

from . import validation_utils as validation


class Roi(ABC):
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

    def __init__(self, *, id_: int = -1):
        self.id = id_
        self.features = list()

    @property
    def features(self) -> list[Feature]:
        return self._features

    @features.setter
    def features(self, value: list[Feature]):
        self._features = value

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    def add_feature(self, feature: Feature):
        self.features.append(feature)

    def remove_feature(self, feature: Feature):
        """Remove feature from feature list."""
        self.features.remove(feature)

    @classmethod
    def from_str(cls: Type[Roi], s: str) -> Roi:
        """Load a ROI from a JSON string."""
        ...

    @abstractmethod
    def to_str(self) -> str:
        """Serialize a ROI into a string."""
        ...


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


class Feature(ABC):
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

    def __init__(
        self, roi: Roi, *, id_: int = -1, annotation: Annotation | None = None
    ):
        self.id = id_
        self.roi = roi
        self._descriptors = dict()
        roi.add_feature(self)
        if annotation is None:
            annotation = Annotation()
        self._annotation = annotation

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def roi(self) -> Roi:
        return self._roi

    @roi.setter
    def roi(self, value: Roi):
        self._roi = value

    @property
    def annotation(self) -> Annotation:
        return self._annotation

    @property
    def mz(self) -> float:
        """Wrapper for get_mz."""
        return self.get_mz()

    @property
    def area(self) -> float:
        """Wrapper fot get_area."""
        return self.get_area()

    @property
    def height(self) -> float:
        """Wrapper for get_height."""
        return self.get_height()

    def __lt__(self: Feature, other: Feature | float):
        if isinstance(other, float):
            return self.mz < other
        elif isinstance(other, Feature):
            return self.mz < other.mz

    def __le__(self, other: Feature | float):
        if isinstance(other, float):
            return self.mz <= other
        elif isinstance(other, Feature):
            return self.mz <= other.mz

    def __gt__(self, other: Feature | float):
        if isinstance(other, float):
            return self.mz > other
        elif isinstance(other, Feature):
            return self.mz > other.mz

    def __ge__(self, other: Feature | float):
        if isinstance(other, float):
            return self.mz >= other
        elif isinstance(other, Feature):
            return self.mz >= other.mz

    @abstractmethod
    def get_mz(self) -> float:
        """Get the feature m/z."""
        ...

    @abstractmethod
    def get_area(self) -> float:
        """Get the feature area."""
        ...

    @abstractmethod
    def get_height(self) -> float:
        """Get the feature height."""
        ...

    def describe(self) -> dict[str, float]:
        """
        Compute all available descriptors for the feature.

        A descriptor is any method that starts with get_.

        """
        descriptors = dict()
        for descriptor in self.descriptor_names():
            descriptors[descriptor] = self.get(descriptor)
        return descriptors

    @abstractmethod
    def to_str(self) -> str:
        """
        Serialize the feature data into a string.

        ROI data must not be serialized. See Peak implementation as an example.

        """
        ...

    @classmethod
    @abstractmethod
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
        ...

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
        value = self._descriptors.get(descriptor)
        if value is not None:
            return value

        try:
            value = getattr(self, f"get_{descriptor}")()
            return self._descriptors.setdefault(descriptor, value)
        except AttributeError as e:
            msg = f"{descriptor} is not a valid descriptor."
            raise ValueError(msg) from e

    @classmethod
    @lru_cache
    def descriptor_names(cls) -> list[str]:
        """
        List all descriptor names.

        A descriptor is a method that starts with `get_`.

        Returns
        -------
        list [str]

        """
        return [x.split("_", 1)[-1] for x in dir(cls) if x.startswith("get_")]


class AnnotableFeature(Feature):
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
    Container class for the associated with a sample.

    Attributes
    ----------
    sample : Sample
    roi : Sequence[Roi] | None

    """

    def __init__(self, sample: Sample, roi: Sequence[Roi] | None = None) -> None:
        self.sample = sample
        if roi is None:
            roi = list()
        self._roi = roi
        self._features: list[Feature] | None = None
        self._descriptors: dict[str, list[float]] | None = None

    @property
    def roi(self) -> Sequence[Roi]:
        """Get the roi in the sample."""
        return self._roi

    @roi.setter
    def roi(self, value: Sequence[Roi]):
        self._roi = value

    def get_feature_list(self) -> list[Feature]:
        """Update the feature attribute using feature data from ROIs."""
        feature_list = list()
        for roi in self.roi:
            if roi.features is not None:
                feature_list.extend(roi.features)
        return feature_list

    def delete_empty_roi(self):
        """Delete ROI with no features."""
        self.roi = [x for x in self.roi if x.features]
