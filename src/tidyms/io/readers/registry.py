"""Maintain a runtime registry of readers."""


from typing import TypeVar

from ...core.exceptions import ReaderNotFound
from .interface import Reader

ReaderType = TypeVar("ReaderType", bound=Reader)

_REGISTERED_READERS: dict[str, type[Reader]] = dict()
_EXTENSION_TO_READER: dict[str, str] = dict()


def get_reader_from_extension(ext: str) -> type[Reader]:
    """Fetch a MS data reader from a file extension."""
    type_ = _EXTENSION_TO_READER.get(ext)

    if type_ is None:
        msg = f"No reader associated with extension {ext}."
        raise ValueError(msg)

    return get_reader(type_)

def get_reader(type_: str) -> type[Reader]:
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
        return _REGISTERED_READERS[type_]
    except KeyError as e:
        raise ReaderNotFound(type_) from e


def list_reader_types() -> list[str]:
    """Retrieve the list of Feature types."""
    return list(_REGISTERED_READERS)


def register(reader: type[ReaderType]) -> type[ReaderType]:
    """Add a reader to the registry."""
    _REGISTERED_READERS[reader.__name__] = reader
    return reader
