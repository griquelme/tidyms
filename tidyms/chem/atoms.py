"""
Tools for working with Isotopes and Elements.

Objects
-------

- Isotope
- Element

Constants
---------

- PTABLE: a dict with element symbols, Elements key-value pairs.
- Z_TO_SYMBOL: a dict with atomic number, symbol key-value pairs.
- EM: Mass of the electron.

Exceptions
----------

- InvalidIsotope

"""
import json
import numpy as np
import os.path
import string
from typing import Dict, Final, Tuple


EM = 0.00054858     # electron mass


class Isotope:
    """
    Representation of an Isotope.

    Attributes
    ----------
    z: int
        atomic number
    n: int
        neutron number
    a: int
        mass number
    m: float
        exact mass.
    defect: float
        difference between Exact mass and Mass number.
    abundance: float
        relative abundance of the isotope.

    """
    __slots__ = ("z", "n", "a", "m", "defect", "abundance")

    def __init__(
        self,
        z: int,
        n: int,
        a: int,
        m: float,
        defect: float,
        abundance: float
    ):
        self.z = z
        self.n = n
        self.a = a
        self.m = m
        self.defect = defect
        self.abundance = abundance

    def __str__(self):
        return "{}{}".format(self.a, self.get_symbol())

    def __repr__(self):
        return "Isotope({})".format(str(self))

    def get_symbol(self) -> str:
        return Z_TO_SYMBOL[self.z]

    def is_most_abundant(self) -> bool:
        symbol = self.get_symbol()
        return PTABLE[symbol].nominal_mass == self.a

    def get_element(self) -> "Element":
        return PTABLE[self.get_symbol()]


class Element(object):
    """
    A representation of a chemical element

    Attributes
    ----------
    name : str
    symbol : str
    isotopes : Dict[int, Isotope]
        Mapping from mass number to an isotope
    z : int
    nominal_mass : int
        Mass number of the most abundant isotope
    monoisotopic_mass : float
        Exact mass of the most abundant isotope
    mass_defect : float
        Mass defect of the most abundant isotope

    """

    def __init__(
        self,
        symbol: str,
        name: str,
        isotopes: Dict[int, Isotope]
    ):
        _validate_element_params(symbol, name, isotopes)
        self.name = name
        self.symbol = symbol
        self.isotopes = isotopes
        self.z = list(isotopes.values())[0].z
        self.nominal_mass = _get_nominal_mass(isotopes)
        self.monoisotopic_mass = _get_monoisotopic_mass(isotopes)
        self.mass_defect = self.monoisotopic_mass - self.nominal_mass

    def __repr__(self):
        return "Element(" + self.symbol + ")"

    def __str__(self):  # pragma: no cover
        return self.symbol

    def get_abundances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the Mass number, exact mass and abundance of each Isotope.

        Returns
        -------
        mass_number: array[int]
            Mass number of each isotope.
        exact_mass: array[float]
            Exact mass of each isotope.
        abundance: array[float]
            Natural abundance of each isotope.
        """
        n_isotopes = len(self.isotopes)
        mass_number = np.zeros(n_isotopes, dtype=int)
        exact_mass = np.zeros(n_isotopes, dtype=float)
        abundance = np.zeros(n_isotopes, dtype=float)
        for k, isotope in enumerate(self.isotopes.values()):
            mass_number[k] = isotope.a
            exact_mass[k] = isotope.m
            abundance[k] = isotope.abundance
        return mass_number, exact_mass, abundance

    def get_most_abundant_isotope(self) -> Isotope:
        return self.isotopes[self.nominal_mass]


def _get_nominal_mass(isotopes: Dict[int, Isotope]) -> int:
    nominal_mass, max_abundance = 0, 0
    for isotope in isotopes.values():
        if isotope.abundance > max_abundance:
            max_abundance = isotope.abundance
            nominal_mass = isotope.a
    return nominal_mass


def _get_monoisotopic_mass(isotopes: Dict[int, Isotope]) -> float:
    monoisotopic_mass, max_abundance = 0, 0
    for isotope in isotopes.values():
        if isotope.abundance > max_abundance:
            max_abundance = isotope.abundance
            monoisotopic_mass = isotope.m
    return monoisotopic_mass


def _validate_element_params(
    symbol: str,
    name: str,
    isotopes: Dict[int, Isotope]
) -> None:
    if not isinstance(symbol, str):
        msg = "symbol must be a string"
        raise TypeError(msg)
    if not isinstance(name, str):
        msg = "name must be a string"
        raise TypeError(msg)

    z = isotopes[list(isotopes.keys())[0]].z
    total_abundance = 0
    for isotope in isotopes.values():
        if isotope.z != z:
            msg = "Atomic number must be the same for all isotopes."
            raise ValueError(msg)
        total_abundance += isotope.abundance
    if not np.isclose(total_abundance, 1):
        msg = "the sum of the abundance of each isotope should be 1"
        raise ValueError(msg)


def _make_periodic_table():
    this_dir, _ = os.path.split(__file__)
    elements_path = os.path.join(this_dir, "elements.json")
    with open(elements_path, "r") as fin:
        element_data = json.load(fin)

    isotopes_path = os.path.join(this_dir, "isotopes.json")
    with open(isotopes_path, "r") as fin:
        isotope_data = json.load(fin)

    periodic_table = dict()
    for element in isotope_data:
        element_isotopes = isotope_data[element]
        isotopes = {x["a"]: Isotope(**x) for x in element_isotopes}
        name = element_data[element]
        periodic_table[element] = Element(element, name, isotopes)
    return periodic_table


def _make_z_to_symbol_dictionary():
    periodic_table_elements = PTABLE.values()
    return {x.z: x.symbol for x in periodic_table_elements}


PTABLE: Final[Dict[str, Element]] = _make_periodic_table()
Z_TO_SYMBOL: Final[Dict[int, str]] = _make_z_to_symbol_dictionary()


def find_isotope(s: str) -> Isotope:
    ind = 0
    try:
        while s[ind] in string.digits:
            ind += 1
        a = s[:ind]
        symbol = s[ind:]
        a = int(a) if a else PTABLE[symbol].nominal_mass
        isotope = PTABLE[symbol].isotopes[a]
        return isotope
    except (IndexError, KeyError):
        msg = "{} is not a valid isotope or element symbol".format(s)
        raise InvalidIsotope(msg)


class InvalidIsotope(ValueError):
    pass
