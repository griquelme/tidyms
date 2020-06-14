from ms_feature_validation import utils
import numpy as np
import pytest


def test_find_closest_left_border():
    x = np.arange(10)
    y = -1
    ind = utils.find_closest(x, y)
    assert ind == 0


def test_find_closest_right_border():
    x = np.arange(10)
    y = 10
    ind = utils.find_closest(x, y)
    assert ind == (x.size - 1)


def test_find_closest_middle():
    x = np.arange(10)
    y = 4.6
    ind = utils.find_closest(x, y)
    assert ind == 5


def test_find_closest_empty_x():
    x = np.array([])
    y = 10
    with pytest.raises(ValueError):
        utils.find_closest(x, y)


def test_find_closest_empty_y():
    x = np.arange(10)
    y = np.array([])
    with pytest.raises(ValueError):
        utils.find_closest(x, y)


def test_find_closest_multiple_values():
    x = np.arange(100)
    y = np.array([-10, 4.6, 67.1, 101])
    ind = np.array([0, 5, 67, 99], dtype=int)
    result = utils.find_closest(x, y)
    assert np.array_equal(result, ind)
