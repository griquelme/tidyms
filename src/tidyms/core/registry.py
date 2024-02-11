"""Register utilities for ROI, Features and processors."""

from typing import TypeVar

from . import exceptions
from .models import Feature, Roi
from .processors import Processor
from .reader import Reader

FeatureType = TypeVar("FeatureType", bound=Feature)
RoiType = TypeVar("RoiType", bound=Roi)
ProcessorType = TypeVar("ProcessorType", bound=Processor)


_REGISTERED_FEATURES: dict[str, type[Feature]] = dict()
_REGISTERED_PROCESSORS: dict[str, type[Processor]] = dict()
_REGISTERED_ROIS: dict[str, type[Roi]] = dict()
_REGISTERED_READERS: dict[str, type[Reader]] = dict()


def get_roi_type(type_: str) -> type[Roi]:
    """
    Retrieve a ROI type from the registry.

    Parameters
    ----------
    type_ : str
        The name of the ROI type to retrieve.

    Returns
    -------
    type[Roi]

    Raises
    ------
    RoiTypeNotRegistered
        If a non-registered ROI type is requested.

    """
    try:
        return _REGISTERED_ROIS[type_]
    except KeyError as e:
        raise exceptions.RoiTypeNotRegistered(type_) from e


def list_roi_types() -> list[str]:
    """Retrieve the list of registered ROI types."""
    return list(_REGISTERED_ROIS)


def register_roi(roi: type[RoiType]) -> type[RoiType]:
    """
    Register a ROI type into the ROI registry.

    Parameters
    ----------
    roi : type[Roi]

    Returns
    -------
    type[Roi]

    """
    _REGISTERED_ROIS[roi.__name__] = roi
    return roi


def get_feature_type(type_: str) -> type[Feature]:
    """
    Retrieve a Feature type from the registry.

    Parameters
    ----------
    type_ : str
        The name of the Feature type to retrieve.

    Returns
    -------
    type[Roi]

    Raises
    ------
    FeatureTypeNotRegistered
        If a non-registered Feature name is requested

    """
    try:
        return _REGISTERED_FEATURES[type_]
    except KeyError as e:
        raise exceptions.FeatureTypeNotRegistered(type_) from e


def list_feature_types() -> list[str]:
    """Retrieve the list of Feature types."""
    return list(_REGISTERED_FEATURES)


def register_feature(feature: type[FeatureType]) -> type[FeatureType]:
    """
    Register a Feature into the Feature registry.

    Parameters
    ----------
    feature : type[Feature]

    Returns
    -------
    type[Feature]

    """
    _REGISTERED_FEATURES[feature.__name__] = feature
    return feature


def get_processor_type(type_: str) -> type[Processor]:
    """
    Retrieve a Processor type from the registry.

    Parameters
    ----------
    type_ : str
        The name of the Processor to retrieve.

    Returns
    -------
    type[Processor]

    Raises
    ------
    ProcessorTypeNotRegistered
        If a non-registered Processor name is requested

    """
    try:
        return _REGISTERED_PROCESSORS[type_]
    except KeyError as e:
        raise exceptions.ProcessorTypeNotRegistered(type_) from e


def list_processor_types() -> list[str]:
    """Retrieve the list of Feature types."""
    return list(_REGISTERED_PROCESSORS)


def register_processor(processor: type[ProcessorType]) -> type[ProcessorType]:
    """
    Register a Processor into the registry.

    Parameters
    ----------
    processor : type[Processor]

    Returns
    -------
    type[Processor]

    """
    _REGISTERED_PROCESSORS[processor.__name__] = processor
    return processor


def get_reader_type(type_: str) -> type[Reader]:
    """
    Retrieve a Reader type from the registry.

    Parameters
    ----------
    type_ : str
        The name of the Processor to retrieve.

    Returns
    -------
    type[Processor]

    Raises
    ------
    ProcessorTypeNotRegistered
        If a non-registered Processor name is requested

    """
    try:
        type_str = f"{type_.upper()}Reader"
        return _REGISTERED_READERS[type_str]
    except KeyError as e:
        raise exceptions.ReaderNotFound(type_) from e


def list_reader_types() -> list[str]:
    """Retrieve the list of Feature types."""
    return list(_REGISTERED_READERS)


def register_reader(processor: type[ProcessorType]) -> type[ProcessorType]:
    """
    Register a Processor into the registry.

    Parameters
    ----------
    processor : type[Processor]

    Returns
    -------
    type[Processor]

    """
    _REGISTERED_PROCESSORS[processor.__name__] = processor
    return processor
