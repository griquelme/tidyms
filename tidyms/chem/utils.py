# -*- coding: utf-8 -*-
"""
functions and classes used in different modules
"""

import numpy as np


def cartesian_product(*args):
    res = None
    for x in args:
        if res is None:
            # initialize cartesian product array
            res = np.array(x)
            res = res.reshape((res.size, 1))
        else:
            x = np.array(x)
            row, col = res.shape
            new_res_shape = (row * x.size, col + 1)
            new_res = np.zeros(shape=new_res_shape, dtype=res.dtype)
            ind = np.repeat(np.arange(row), x.size)
            new_col = np.tile(x, row)
            new_res[:, :col] = res[ind]
            new_res[:, -1] = new_col
            res = new_res
    return res
