"""
Chemistry
=========

Provides
    1. A Formula object to compute the exact mass and isotopic distribution of
        molecular formulas.
    2. A "Periodic Table" with elements and isotopes information.
    3. A formula generator object to search molecular formulas based on exact
        mass values.
    4. An isotopic envelope scorer that scores the similarity between
        experimental and theoretical isotopic distributions.
"""

from .formula_generator import FormulaGenerator
from .isotope_scorer import IsotopeScorer, EnvelopeValidator
from .formula import Formula
from .envelope_finder import EnvelopeFinder
from .mmi_finder import MMIFinder
from .atoms import PTABLE, EM
from . import _isotope_distributions
from . import mmi_finder
