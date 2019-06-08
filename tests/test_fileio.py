from ms_feature_validation import fileio
import numpy as np


def test_select_roi():
    x = np.linspace(0, 200, num=1000)
    x_roi = np.array([100, 150, 175])
    tolerance = 0.005
    result = np.array([[495, 500], [745, 750], [870, 875]], dtype=np.int64)
    assert np.array_equal(fileio._select_roi(x, x_roi, tolerance), result)