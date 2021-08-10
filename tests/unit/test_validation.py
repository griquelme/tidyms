from tidyms.validation import *
import pytest


@pytest.fixture
def example_validator():
    schema = {
        "positive_number": {"is_positive": True},
        "a": {"lower_than": "b"},
        "b": {"lower_or_equal": "c"},
        "c": {"type": "number"},
        "some_function": {"check_with": is_callable}
    }
    return ValidatorWithLowerThan(schema)


def test_is_positive_positive_number(example_validator):
    params = {"positive_number": 5}
    validate(params, example_validator)
    assert True


def test_is_positive_zero(example_validator):
    params = {"positive_number": 0}
    with pytest.raises(ValueError):
        validate(params, example_validator)


def test_is_positive_negative_number(example_validator):
    params = {"positive_number": -1}
    with pytest.raises(ValueError):
        validate(params, example_validator)


def test_lower_than_valid(example_validator):
    # a must be lower than b
    params = {"a": 5, "b": 6}
    validate(params, example_validator)
    assert True


def test_lower_than_invalid(example_validator):
    # a must be lower than b
    params = {"a": 5, "b": 4}
    with pytest.raises(ValueError):
        validate(params, example_validator)


def test_lower_than_invalid_equal(example_validator):
    # a must be lower than b
    params = {"a": 5, "b": 5}
    with pytest.raises(ValueError):
        validate(params, example_validator)


def test_lower_or_equal_valid(example_validator):
    # a must be lower than b
    params = {"b": 5, "c": 7}
    validate(params, example_validator)
    assert True


def test_lower_or_equal_valid_equal(example_validator):
    # a must be lower than b
    params = {"b": 5, "c": 5}
    validate(params, example_validator)
    assert True


def test_lower_or_equal_invalid(example_validator):
    # a must be lower than b
    params = {"b": 5, "c": 4}
    with pytest.raises(ValueError):
        validate(params, example_validator)


def test_is_callable_valid(example_validator):
    # a must be lower than b
    params = {"some_function": sum}
    validate(params, example_validator)
    assert True


def test_is_callable_invalid(example_validator):
    # a must be lower than b
    params = {"some_function": "invalid_value"}
    with pytest.raises(ValueError):
        validate(params, example_validator)
