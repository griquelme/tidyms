"""Maintain a runtime registry of readers."""

from ...core.exceptions import ReaderNotFound
from .interface import Reader

_REGISTERED_READERS: dict[str, type[Reader]] = dict()

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
        raise ReaderNotFound(type_) from e


def list_reader_types() -> list[str]:
    """Retrieve the list of Feature types."""
    return list(_REGISTERED_READERS)
