"""Register utilities for ROI, Features and processors."""

from typing import TypeVar

from . import exceptions
from .models import Feature, Roi
from .processors import Processor

FeatureType = TypeVar("FeatureType", bound=Feature)
RoiType = TypeVar("RoiType", bound=Roi)
ProcessorType = TypeVar("ProcessorType", bound=Processor)


_REGISTERED_FEATURES: dict[str, type[Feature]] = dict()
_REGISTERED_ROIS: dict[str, type[Roi]] = dict()


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



