"""
Base classes used by TidyMS.

Annotation : Stores annotation data from a feature.
Feature : A region associated with a ROI that contains a chemical species.
Roi : A Region of Interest extracted from raw data. Usually a subset of raw data.
Sample : Stores metadata from a measurement.
SampleData : Container class for a Sample and the ROIs detected.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Type, TypeVar, Union

from .. import _constants as c

# Replace with Self when code is updated to Python 3.11
AnyRoi = TypeVar("AnyRoi", bound="Roi")
AnyFeature = TypeVar("AnyFeature", bound="Feature")


class Roi(ABC):
    """
    Base class for Regions of Interest (ROI) extracted from raw MS data.

    MUST implement the `extract_features` method.
    MUST implement `_deserialize` and `to_string` methods to be used in an Assay.
    MUST implement the class method `get_feature_type` to check feature
    compatibility.


    Attributes
    ----------
    id : int
        An identifier for the ROI. Used by :class:`tidyms.base.AssayData` to
        persist data.
    features : list[Feature]
        Features extracted from the ROI using the `extract_features` method.

    """

    _feature_type: Type["Feature"]

    def __init__(self):
        self.id = -1
        self.features: Optional[list[Feature]] = None

    @abstractmethod
    def extract_features(self, **kwargs) -> list["Feature"]:
        """Extract feature from ROI and stores in the `features` attribute."""
        ...

    @classmethod
    def from_string(cls: Type[AnyRoi], s: str) -> AnyRoi:
        """Load a ROI from a JSON string."""
        d = cls._deserialize(s)
        features = d.pop(c.ROI_FEATURE_LIST)
        roi = cls(**d)
        ft_class = cls._get_feature_type()
        if features is not None:
            roi.features = [ft_class.from_str(x, roi) for x in features]
        return roi

    @classmethod
    @abstractmethod
    def _get_feature_type(cls) -> Type["Feature"]:
        """Get the Feature type to be used with the ROI."""
        ...

    @staticmethod
    @abstractmethod
    def _deserialize(s: str) -> dict:
        """
        Convert a JSON str into a dictionary used to create a ROI instance.

        See MZTrace implementation as an example.
        """
        ...

    @abstractmethod
    def to_string(self) -> str:
        """Serialize a ROI into a string."""
        ...


@dataclass
class Annotation:
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

    def __init__(self, roi: Roi, annotation: Optional[Annotation] = None):
        self.roi = roi
        self._mz = None
        self._area = None
        self._height = None
        if annotation is None:
            annotation = Annotation()
        self.annotation = annotation
        self.id = -1

    @property
    def mz(self) -> float:
        """Wrapper for get_mz."""
        if self._mz is None:
            self._mz = self.get_mz()
        return self._mz

    @property
    def area(self) -> float:
        """Wrapper fot get_area."""
        if self._area is None:
            self._area = self.get_area()
        return self._area

    @property
    def height(self) -> float:
        """Wrapper for get_height."""
        if self._height is None:
            self._height = self.get_height()
        return self._height

    def __lt__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz < other
        elif isinstance(other, Feature):
            return self.mz < other.mz

    def __le__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz <= other
        elif isinstance(other, Feature):
            return self.mz <= other.mz

    def __gt__(self, other: Union["Feature", float]):
        if isinstance(other, float):
            return self.mz > other
        elif isinstance(other, Feature):
            return self.mz > other.mz

    def __ge__(self, other: Union["Feature", float]):
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
        descriptors: dict[str, float] = dict()
        for descriptor in self.descriptor_names():
            descriptors[descriptor] = self.get(descriptor)
        return descriptors

    @abstractmethod
    def compare(self, other: "Feature") -> float:
        """
        Compare the similarity between two features.

        Must return a number between 0.0 and 1.0.

        Used by annotation methods to group features.

        """
        ...

    @abstractmethod
    def to_str(self) -> str:
        """
        Serialize the feature data into a string.

        ROI data must not be serialized. See Peak implementation as an example.

        """
        ...

    @classmethod
    def from_str(cls, s: str, roi: Roi, annotation: Optional[Annotation] = None) -> "Feature":
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
        d = cls._deserialize(s)
        return cls(roi=roi, annotation=annotation, **d)

    @staticmethod
    @abstractmethod
    def _deserialize(s: str) -> dict:
        ...

    @staticmethod
    @abstractmethod
    def compute_isotopic_envelope(
        feature: Sequence["Feature"],
    ) -> Tuple[list[float], list[float]]:
        """
        Compute the isotopic envelope from a list of isotopologue features.

        Must return two list:

        - The sorted m/z values of the envelope.
        - The abundance associated to each isotopologue. Normalized to 1.

        Used by annotation algorithms to annotate isotopologues.

        """
        ...

    def get(self, descriptor: str) -> float:
        """
        Compute descriptor using its name.

        A descriptor is a method that starts with `get_`.

        """
        return self.__getattribute__(f"get_{descriptor}")()

    @classmethod
    def descriptor_names(cls) -> list[str]:
        """
        List all descriptor names.

        A descriptor is a method that starts with `get_`.

        Returns
        -------
        list [str]

        """
        return [x.split("_", 1)[-1] for x in dir(cls) if x.startswith("get_")]


@dataclass
class Sample:
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

    """

    path: Union[Path, str]
    id: str
    ms_level: int = 1
    start_time: float = 0.0
    end_time: Optional[float] = None
    group: str = ""

    def __post_init__(self):
        """Normalize and validate data fields."""
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def to_dict(self) -> dict:
        """Convert data into a dictionary."""
        d = asdict(self)
        d["path"] = str(d["path"])
        return d


class SampleData:
    """
    Container class for the associated with a sample.

    Attributes
    ----------
    sample : Sample
    roi : Optional[Sequence[Roi]]

    """

    def __init__(self, sample: Sample, roi: Optional[Sequence[Roi]] = None) -> None:
        self.sample = sample
        if roi is None:
            roi = list()
        self.roi = roi

    def get_feature_list(self) -> Sequence[Feature]:
        """Create a list of features from features stored in each ROI."""
        feature_list = list()
        for roi in self.roi:
            if roi.features is not None:
                feature_list.extend(roi.features)
        return feature_list
