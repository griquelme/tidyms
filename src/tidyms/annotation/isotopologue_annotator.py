"""Processor class used for isotopologue annotation of features detected in a sample."""

from typing import Optional
from . import create_annotation_tools, annotate
from ..base.base import SampleData
from ..base.assay import BaseSampleProcessor
from ..chem import get_chnops_bounds
from ..validation import ValidatorWithLowerThan, validate


class IsotopologueAnnotator(BaseSampleProcessor):
    """
    Annotates isotopologues in a sample.

    Groups isotopologue features. Each group is assigned an unique label, and a
    charge state. Each feature in a group is assigned an unique index that
    determines the position in the envelope.

    Annotations are stored to the `annotation` attribute of each feature.

    Attributes
    ----------
    bounds : Dict or None, default=None
        A dictionary of expected elements to minimum and maximum formula coefficients.
        If ``None``, bounds are obtained using :func:`tidyms.chem.get_chnops_bound`
        with ``2000`` as parameter.
    max_mass : float
        Maximum exact mass of the features.
    max_charge : int
        Maximum charge of the features. Use negative values for negative polarity.
    max_length : int
        Maximum length of the envelopes.
    min_M_tol : float
        Minimum mass tolerance used during search. isotopologues with abundance
        equal to 1 use this value. Isotopologues with abundance equal to 0 use
        `max_M_tol`. For values in between, a weighted tolerance is used based
        on the abundance.
    max_M_tol : float
    p_tol : float
        Abundance tolerance.
    min_similarity : float
        Minimum cosine similarity between a pair of features
    min_p : float
        Minimum abundance of isotopes to include in candidate search.

    """

    def __init__(
        self,
        bounds: Optional[dict[str, tuple[int, int]]] = None,
        max_mass: float = 2000.0,
        max_charge: int = 3,
        max_length: int = 10,
        min_M_tol: float = 0.005,
        max_M_tol: float = 0.01,
        p_tol: float = 0.05,
        min_similarity: float = 0.9,
        min_p: float = 0.005,
    ):
        if bounds is None:
            bounds = get_chnops_bounds(2000)
        self.bounds = bounds
        self.max_mass = max_mass
        self.max_charge = max_charge
        self.max_length = max_length
        self.min_M_tol = min_M_tol
        self.max_M_tol = max_M_tol
        self.p_tol = p_tol
        self.min_similarity = min_similarity
        self.min_p = min_p

        parameters = self.get_parameters()
        tools = create_annotation_tools(**parameters)
        self._mmi_finder = tools[0]
        self._envelope_finder = tools[1]
        self._envelope_validator = tools[2]

    def _func(self, sample_data: SampleData):
        annotate(
            sample_data.get_feature_list_snapshot(),
            self._mmi_finder,
            self._envelope_finder,
            self._envelope_validator,
        )

    @staticmethod
    def _check_data(sample_data: SampleData):
        pass

    def set_default_parameters(self, instrument: str, separation: str):
        """
        Set parameters using instrument type and separation method information.

        Parameters
        ----------
        instrument : str
            MS instrument type.
        separation : str
            Analytical separation method.

        """
        # TODO : set defaults using orbitrap and qtof
        bounds = get_chnops_bounds(2000)
        defaults = {
            "bounds": bounds,
            "max_mass": 2000.0,
            "max_charge": 3,
            "max_length": 10,
            "min_M_tol": 0.005,
            "max_M_tol": 0.01,
            "p_tol": 0.05,
            "min_similarity": 0.9,
            "min_p": 0.005,
        }
        return defaults

    @staticmethod
    def _validate_parameters(parameters: dict):
        schema = {
            "bounds": {
                "type": "dict",
                "keyrules": {"type": "string"},
                "valuesrules": {
                    "type": "tuple",
                    "items": [
                        {"type": "number", "nullable": True},
                        {"type": "number", "nullable": True},
                    ],
                },
            },
            "max_mass": {"type": "number", "is_positive": True},
            "max_charge": {"type": "integer"},
            "max_length": {"type": "integer", "min": 2},
            "min_M_tol": {"type": "number", "is_positive": True},
            "max_M_tol": {"type": "number", "is_positive": True},
            "p_tol": {"type": "number", "is_positive": True, "max": 0.2},
            "min_similarity": {"type": "number", "min": 0.5, "max": 1.0},
            "min_p": {"type": "number", "min": 0.0, "max": 0.1},
        }
        validator = ValidatorWithLowerThan(schema)
        validate(parameters, validator)
