import numpy as np
import pydantic
from tidyms.core.numpy import FloatingArray, IntArray


class NumpyFloatModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    arr: FloatingArray


class NumpyIntModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    arr: IntArray


def test_serialize_empty_float_array():
    expected = np.array([], dtype=float)
    model = NumpyFloatModel(arr=expected)
    actual = NumpyFloatModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)


def test_serialize_1D_float_array():
    expected = np.random.normal(size=100)
    model = NumpyFloatModel(arr=expected)
    actual = NumpyFloatModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)


def test_serialize_2D_float_array():
    expected = np.random.normal(size=(100, 20))
    model = NumpyFloatModel(arr=expected)
    actual = NumpyFloatModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)


def test_serialize_empty_int_array():
    expected = np.array([], dtype=int)
    model = NumpyIntModel(arr=expected)
    actual = NumpyIntModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)


def test_serialize_1D_int_array():
    expected = np.random.randint(low=0, high=100, size=100)
    model = NumpyIntModel(arr=expected)
    actual = NumpyIntModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)


def test_serialize_2D_int_array():
    expected = np.random.randint(low=0, high=100, size=(100, 25))
    model = NumpyIntModel(arr=expected)
    actual = NumpyIntModel(**model.model_dump()).arr
    assert np.array_equal(actual, expected)
