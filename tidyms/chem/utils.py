# -*- coding: utf-8 -*-
"""
functions and classes used in different modules
"""

import numpy as np
from typing import List, Optional
from .atoms import EM


def cartesian_product(*v):
    nv = len(v)
    cartesian = np.array(np.meshgrid(*v)).T.reshape(-1, nv)
    return cartesian


def cartesian_product_from_range_list(range_list: List[range]):
    res = None
    for r in range_list:
        if res is None:
            # initialize cartesian product array
            res = np.array(r)
            res = res.reshape((res.size, 1))
        else:
            x = np.array(r)
            row, col = res.shape
            new_res_shape = (row * x.size, col + 1)
            new_res = np.zeros(shape=new_res_shape, dtype=res.dtype)
            ind = np.repeat(np.arange(row), x.size)
            new_col = np.tile(x, row)
            new_res[:, :col] = res[ind]
            new_res[:, -1] = new_col
            res = new_res
    return res


def has_unique_elements(x: np.ndarray) -> bool:
    """
    Return True if all elements are unique.

    Parameters
    ----------
    x: numpy array

    Returns
    -------
    bool
    """
    d = np.diff(x)
    return (d > 0).all()


def convert_mass_tolerance(value: float, input_units: str,
                           output_units: str = "da",
                           reference: Optional[float] = None) -> float:
    """
    Convert mass tolerance between different units.

    Parameters
    ----------
    value: float
    input_units: {"da", "mda", "ppm"}
    output_units: {"da", "mda", "ppm"}
    reference: float, optional
        reference mass value in da, used for conversion to/from ppm
    Returns
    -------
    output: float
    """
    converted = convert_to_da(value, input_units, reference=reference)
    converted = convert_from_da(converted, output_units, reference=reference)
    return converted


def convert_to_da(value: float, input_units: str,
                  reference: Optional[float] = None) -> float:
    """aux function to convert_mass_tolerance"""
    if input_units == "da":
        output = value
    elif input_units == "mda":
        output = value / 1000
    elif input_units == "ppm":
        output = value * reference / 1e6
    else:
        msg = "Valid input units are `da`, `mda` or `ppm`"
        raise ValueError(msg)
    return output


def convert_from_da(value: float, output_units: str,
                    reference: Optional[float] = None) -> float:
    """aux function to convert_mass_tolerance"""
    if output_units == "da":
        output = value
    elif output_units == "mda":
        output = value * 1000
    elif output_units == "ppm":
        output = value * 1e6 / reference
    else:
        msg = "Valid output units are `da`, `mda` or `ppm`"
        raise ValueError(msg)
    return output


def mz_to_mass(mz: np.ndarray, charge: int):
    if charge != 0:
        mass = mz * abs(charge) + charge * EM
    else:
        mass = mz
    return mass


def mass_to_mz(mass: np.ndarray, charge: int):
    if charge != 0:
        mz = (mass - charge * EM) / abs(charge)
    else:
        mz = mass
    return mz
