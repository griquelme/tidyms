"""
Chemistry
=========

Provides:

1. A Formula object to compute the exact mass and isotopic distribution of molecular formulas.
2. A PeriodicTable with element and isotope information.
3. A formula generator object to search molecular formulas based on exact mass values.
4. An EnvelopeScorer that scores the similarity between experimental and theoretical isotopic envelopes.

Objects
-------
- PeriodicTable
- Formula
- FormulaGenerator
- EnvelopeScorer

Constants
---------
- EM : electron mass

"""

from ._formula_generator import FormulaGenerator
from .envelope_tools import EnvelopeScorer, EnvelopeValidator
from .formula import Formula
from .envelope_finder import EnvelopeFinder
from .mmi_finder import MMIFinder
from .atoms import EM, PeriodicTable
