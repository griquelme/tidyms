"""Data models used by TidyMS.

Annotation : Annotation data from a feature.
Feature : A ROI region that may contain chemical species information.
Roi : A Region of Interest extracted from raw data.
Sample : Contains metadata from a measurement.
SampleData : Container class for a Sample and the ROIs detected.

"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from math import isnan, nan
from pathlib import Path
from typing import Any, Sequence

import pydantic
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from . import validation_utils as validation


class Roi(pydantic.BaseModel):
    """
    Base class for Regions of Interest (ROI) extracted from raw MS data.

    Roi inherits from pydantic BaseModel, and support most of its functionality.
    New Rois subclasses are created by inheritance of this class and setting
    data fields using Pydantic's standard approach.

    For Numpy array fields, check out the `NumpyArray` type which provides type
    checking for arrays and efficient serialization/deserialization.

    Two attributes are defined for the Roi class: `id` and `features`. Both
    are managed by other components and should never be set directly by the
    user.

    TODO: add link
    See THIS GUIDE for an example on how to create a new Roi class.

    Attributes
    ----------
    id : int, default=-1
        An identifier for the ROI.
    features : list[Feature], defaults to an empty list.
        Features extracted from the ROI.

    """

    model_config = pydantic.ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )
    id: int = -1
    features: list[Feature] = list()

    def add_feature(self, feature: Feature):
        """Associate a feature with the ROI.

        Parameters
        ----------
        feature : Feature
            The feature to associate with the ROI.

        """
        self.features.append(feature)

    def remove_feature(self, feature: Feature):
        """Remove feature from feature list."""
        self.features.remove(feature)

    @classmethod
    def from_str(cls: type[Roi], s: str) -> Roi:
        """Create a ROI instance from a JSON string.

        Parameters
        ----------
        s : str
            A serialized ROI obtained using the `to_str` method.

        Returns
        -------
        Roi
            A new ROI instance.

        """
        return cls(**json.loads(s))

    def to_str(self) -> str:
        """Serialize a ROI into a string.

        Returns
        -------
        str
            A string serialization of the ROI.

        """
        return self.model_dump_json(exclude={"features"})


class Annotation(pydantic.BaseModel):
    """
    Store annotation data of features.

    If information for an attribute is not available, ``-1`` is used.

    Attributes
    ----------
    label : int, default=-1
        Feature group id. Groups features from different samples based
        their chemical identity. Used to create a data matrix.
    isotopologue_label : int, default=-1
        Groups features from the same isotopic envelope in a sample.
    isotopologue_index : int, default=-1
        Position of the feature in an isotopic envelope.
    charge : int, default=-1
        Feature charge state.
    """

    label: int = -1
    isotopologue_label: int = -1
    isotopologue_index: int = -1
    charge: int = -1


class Feature(pydantic.BaseModel):
    """
    Base class to represent a feature extracted from a ROI.

    Feature inherits from pydantic BaseModel, and support most of its
    functionality. New Feature subclasses are created by inheritance of this
    class and setting data fields using Pydantic's standard approach.

    There are two type of data fields for features: data fields and descriptors.
    Data fields contain information to represent the feature. e.g. the start
    and end position of a chromatographic peak. Descriptors are properties that
    describe the feature. e.g, the peak width or peak area in a chromatographic
    peak. Descriptors are set in the same way as data fields, but two
    additional restriction apply. First, the type of a descriptor must be `float`
    and the default value of the descriptor must be ``nan``. Second, a method
    called `_set_descriptor_name` must be created for each descriptor, which
    computes the corresponding descriptor value and stores it in the
    corresponding instance attribute. As an example:

    .. code-block: python

        from math import nan
        from tidyms.core.models import Feature

        class MyFeature(Feature):
            custom_descriptor: float = nan

            def _set_custom_descriptor(self):
                self.custom_descriptor = 100.0

    Three descriptors are defined by default: mz, area and height. Those
    descriptors must be redefined for each new subclass.

    Finally, three attributes are defined for the Feature class: `id`, `roi` and
    `annotation`. These parameters are managed by other components and should
    never be set directly by the user.

    TODO: add link
    See THIS GUIDE for an example on how to create a new Feature class.

    Attributes
    ----------
    roi: Roi
        the ROI where the feature was detected.
    annotation: Annotation
        Annotation data of the feature.
    id: int
        An identifier for the feature.

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
        """Compute all available descriptors for the feature.

        A descriptor is any method that starts with get_.

        Returns
        -------
        dict[str, float]
            A dictionary that maps descriptor names to descriptor values.

        """
        descriptors = dict()
        for descriptor in self.descriptor_names():
            descriptors[descriptor] = self.get(descriptor)
        return descriptors

    def to_str(self) -> str:
        """Serialize the feature data into a string.

        Returns
        -------
        str
            A string serialization of the feature.

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

        Returns
        -------
        Feature
            A new feature instance.

        """
        d = json.loads(s)
        return cls(roi=roi, annotation=annotation, **d)

    def get(self, descriptor: str) -> float:
        """Compute a descriptor value.

        Parameters
        ----------
        descriptor : str
            The descriptor name.

        Returns
        -------
        float
            The descriptor value.

        Raises
        ------
        ValueError
            If an invalid descriptor name is passed.

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
        """Retrieve the available descriptor names.

        Returns
        -------
        set[str]
            The descriptor names.

        """
        # trim _get_ from function name
        return {x[5:] for x in dir(cls) if x.startswith("_set_")}


class AnnotableFeature(Feature, ABC):
    """
    Abstract feature class which inherits from Feature.

    Provides extra functionality to perform feature annotation.
    Base feature with also implements methods for feature annotation.

    """

    @abstractmethod
    def compare(self, other: Feature) -> float:
        """Compare the similarity between two features.

        Must be a symmetric function that returns a number between 0.0 and 1.0.

        Parameters
        ----------
        other : Feature
            Feature to compare with.

        Returns
        -------
        float
            The similarity between the feature pair.

        """
        ...

    @staticmethod
    @abstractmethod
    def compute_isotopic_envelope(
        feature: Sequence[Feature],
    ) -> tuple[list[float], list[float]]:
        """
        Compute the isotopic envelope from a list of isotopologue features.

        Parameters
        ----------
        features : Sequence[Feature]
            The Collection of features used to compute the envelope.

        Returns
        -------
        mz : list[float]
            The m/z of each feature in the envelope, sorted.
        abundance list[float]
            The abundance of each feature. The total abundance is normalized to 1.

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
        """Serialize path into a string."""
        return str(path)


class SampleData(pydantic.BaseModel):
    """Stores data state during a sample processing pipeline."""

    sample: Sample
    roi: list[Roi] = list()
    processing_info: list[ProcessorInformation] = list()

    def get_features(self) -> list[Feature]:
        """Update the feature attribute using feature data from ROIs."""
        feature_list = list()
        for roi in self.roi:
            feature_list.extend(roi.features)
        return feature_list


class ProcessorInformation(pydantic.BaseModel):
    """Stores sample processing step name and parameters."""

    id: str
    pipeline: str | None
    order: int | None
    parameters: dict[str, Any]

    @pydantic.field_serializer("parameters")
    def serialize_parameters(self, parameters: dict[str, Any], _info) -> str:
        """Serialize parameters field into a JSON string."""
        return json.dumps(parameters)
