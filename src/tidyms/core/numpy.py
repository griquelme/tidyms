"""Serializable numpy array types."""
from __future__ import annotations

import base64
import json
from typing import Literal, TypeVar

from numpy import float_, frombuffer, int_
from numpy.typing import NDArray
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


def array_to_json_str(arr: NDArray) -> str:
    """
    Serialize a numpy array as a JSON string.

    Parameters
    ----------
    arr : array

    Returns
    -------
    str
        JSON string with the following three fields. `dtype` store the array
        dtype, `shape` contains the array shape and `base64_bytes` stores
        the array data in base64 format.

    """
    d = {
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "base64_bytes": base64.b64encode(arr.tobytes()).decode("utf8"),
    }
    return json.dumps(d)


def json_str_to_array(s: str):
    """
    Decode a string generated with array_to_json_str into a numpy array.

    Parameters
    ----------
    s : str

    Returns
    -------
    numpy.ndarray

    """
    d = json.loads(s)
    dtype = d["dtype"]
    shape = d["shape"]
    data = base64.b64decode(bytes(d["base64_bytes"], "utf8"))
    return frombuffer(data, dtype=dtype).reshape(shape).copy()


def validate_serializable_array(arr: NDArray) -> NDArray:
    """Create an array if a serialized string is provided."""
    if isinstance(arr, str):
        arr = json_str_to_array(arr)
    return arr

FloatDtype = TypeVar("FloatDtype", bound=float_)
IntDtype = TypeVar("IntDtype", bound=int_)


FloatingArray = Annotated[
    NDArray[FloatDtype],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

IntArray = Annotated[
    NDArray[IntDtype],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

FloatArray1D = Annotated[
    NDArray[FloatDtype],
    Literal["N"],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

IntArray1D = Annotated[
    NDArray[IntDtype],
    Literal["N"],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]
