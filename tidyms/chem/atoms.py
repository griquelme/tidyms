"""
Tools for working with Isotopes and Elements.

Objects
-------
- Element
- Isotope
- PeriodicTable

Constants
---------
- EM: Mass of the electron.

Exceptions
----------
- InvalidIsotope

"""
import json
import numpy as np
import os.path
from string import digits
from typing import Dict, Final, Tuple, Union


EM: Final[float] = 0.00054858  # electron mass


class Isotope:
    """
    Representation of an Isotope.

    Attributes
    ----------
    z: int
        Atomic number
    n: int
        Neutron number
    a: int
        Mass number
    m: float
        Exact mass.
    defect: float
        Difference between the exact mass and mass number.
    abundance: float
        Relative abundance of the isotope.

    """

    __slots__ = ("z", "n", "a", "m", "defect", "abundance")

    def __init__(self, z: int, a: int, m: float, abundance: float):
        self.z = z
        self.n = a - z
        self.a = a
        self.m = m
        self.defect = m - a
        self.abundance = abundance

    def __str__(self):
        return "{}{}".format(self.a, self.get_symbol())

    def __repr__(self):
        return "Isotope({})".format(str(self))

    def get_element(self) -> "Element":
        return PeriodicTable().get_element(self.z)

    def get_symbol(self) -> str:
        return self.get_element().symbol


class Element(object):
    """
    Representation of a chemical element.

    Attributes
    ----------
    name : str
        Element name.
    symbol : str
        Element symbol
    isotopes : Dict[int, Isotope]
        Mapping from mass number to an isotope
    z : int
        Atomic number.
    nominal_mass : int
        Mass number of the most abundant isotope

    """

    def __init__(self, symbol: str, name: str, isotopes: Dict[int, Isotope]):
        self.name = name
        self.symbol = symbol
        self.isotopes = isotopes
        monoisotope = self.get_monoisotope()
        self.z = monoisotope.z
        self.nominal_mass = monoisotope.a
        self.monoisotopic_mass = monoisotope.m
        self.mass_defect = self.monoisotopic_mass - self.nominal_mass

    def __repr__(self):
        return "Element({})".format(self.symbol)

    def __str__(self):  # pragma: no cover
        return self.symbol

    def get_abundances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the Mass number, exact mass and abundance of each Isotope.

        Returns
        -------
        m: array[int]
            Mass number of each isotope.
        M: array[float]
            Exact mass of each isotope.
        p: array[float]
            Abundance of each isotope.

        """
        isotopes = list(self.isotopes.values())
        m = np.array([x.a for x in isotopes], dtype=int)
        M = np.array([x.m for x in isotopes])
        p = np.array([x.abundance for x in isotopes])
        return m, M, p

    def get_mmi(self) -> Isotope:
        """
        Returns the isotope with the lowest atomic mass.

        """
        return min(self.isotopes.values(), key=lambda x: x.a)

    def get_monoisotope(self) -> Isotope:
        """
        Returns the most abundant isotope.

        """
        return max(self.isotopes.values(), key=lambda x: x.abundance)


def PeriodicTable():
    """
    Reference the PeriodicTable object.

    Examples
    --------
    >>> import tidyms as ms
    >>> ptable = ms.chem.PeriodicTable()

    """
    if _PeriodicTable.instance is None:
        _PeriodicTable.instance = _PeriodicTable()
    return _PeriodicTable.instance


class _PeriodicTable:
    """
    Periodic Table representation. Contains element and isotope information.

    Methods
    -------
    get_element
    get_isotope

    """

    instance = None

    def __init__(self):
        self._symbol_to_element = _make_periodic_table()
        self._z_to_element = {v.z: v for v in self._symbol_to_element.values()}
        self._za_to_isotope = dict()
        self._str_to_isotope = dict()
        for el_str in self._symbol_to_element:
            el = self._symbol_to_element[el_str]
            for isotope in el.isotopes.values():
                self._za_to_isotope[(isotope.z, isotope.a)] = isotope
                self._str_to_isotope[str(isotope.a) + el_str] = isotope

    def get_element(self, element: Union[str, int]) -> Element:
        """
        Returns an Element object using its symbol or atomic number.

        Parameters
        ----------
        element : str or int
            element symbol or atomic number.

        Returns
        -------
        Element

        Examples
        --------
        >>> import tidyms as ms
        >>> ptable = ms.chem.PeriodicTable()
        >>> h = ptable.get_element("H")
        >>> c = ptable.get_element(6)

        """
        if isinstance(element, int):
            element = self._z_to_element[element]
        else:
            element = self._symbol_to_element[element]
        return element

    def get_isotope(self, x: str, copy: bool = False) -> Isotope:
        """
        Returns an isotope object from a string representation.

        Parameters
        ----------
        x : str
            A string representation of an isotope. If only the symbol is
            provided in the string, the monoisotope is returned.
        copy : bool
            If True creates a new Isotope object.

        Returns
        -------
        Isotope

        Examples
        --------
        >>> import tidyms as ms
        >>> ptable = ms.chem.PeriodicTable()
        >>> d = ptable.get_isotope("2H")
        >>> cl35 = ptable.get_isotope("Cl")

        """
        try:
            if x[0] in digits:
                isotope = self._str_to_isotope[x]
            else:
                isotope = self.get_element(x).get_monoisotope()
            if copy:
                isotope = Isotope(isotope.z, isotope.a, isotope.m, isotope.abundance)
            return isotope
        except KeyError:
            msg = "{} is not a valid input.".format(x)
            raise InvalidIsotope(msg)


def _make_periodic_table() -> Dict[str, Element]:
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


class InvalidIsotope(ValueError):
    pass
