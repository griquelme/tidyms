"""Processor class for sample isotopologue annotation."""

import pydantic

from ..chem import get_chnops_bounds
from ..core.constants import MSInstrument, Polarity, SeparationMode
from ..core.models import AnnotableFeatureType
from ..core.processors import BaseIsotopologueAnnotator
from .annotation import annotate, create_annotation_tools


class IsotopologueAnnotator(BaseIsotopologueAnnotator):
    """
    Annotates isotopologues in a sample.

    Groups isotopologue features. Each group is assigned an unique label, and a
    charge state. Each feature in a group is assigned an unique index that
    determines the position in the envelope.

    Annotations are stored to the `annotation` attribute of each feature.
    """

    bounds: dict[str, tuple[int, int]] | None = None
    """A dictionary of expected elements to minimum and maximum formula coefficients. If ``None``, bounds are
    obtained using :func:`tidyms.chem.get_chnops_bound` with ``2000`` as parameter.
    """

    max_mass: pydantic.PositiveFloat = 2000.0
    """Maximum exact mass of the features"""

    max_charge: int = 3
    """Maximum charge of the features. Use negative values for negative polarity."""

    max_length: int = 10
    """Maximum length of the envelopes."""

    min_M_tol: pydantic.PositiveFloat = 0.005
    """Minimum mass tolerance used during search. isotopologues with abundance equal to 1 use this value.
    Isotopologues with abundance equal to 0 use `max_M_tol`. For values in between, a weighted tolerance
    is used based on the abundance.
    """

    max_M_tol: pydantic.PositiveFloat = 0.01
    """Maximum mass tolerance used during search"""

    p_tol: float
    """Abundance tolerance"""

    min_similarity: float = pydantic.Field(default=0.9, ge=0.5, le=1.0)
    """Minimum cosine similarity between a pair of features"""

    min_p: float = pydantic.Field(default=0.005, ge=0.0, le=0.1)
    """Minimum abundance of isotopes to include in candidate search."""

    def _func(self, features: list[AnnotableFeatureType]):
        parameters = self.get_parameters()
        tools = create_annotation_tools(**parameters)
        annotate(features, *tools)

    def get_default_parameters(self, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        """
        Set the processor default parameters.

        Parameters
        ----------
        instrument : MSInstrument
        separation : SeparationMode
        polarity : Polarity

        """
        defaults = {
            "bounds": get_chnops_bounds(2000),
            "max_mass": 2000.0,
            "max_charge": 3 if polarity == Polarity.POSITIVE else -3,
            "max_length": 10,
            "min_M_tol": 0.005,
            "max_M_tol": 0.01,
            "p_tol": 0.05,
            "min_similarity": 0.9,
            "min_p": 0.005,
        }
        return defaults
