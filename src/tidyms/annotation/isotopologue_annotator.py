"""Processor class for sample isotopologue annotation."""

from ..chem import get_chnops_bounds
from ..core.constants import MSInstrument, Polarity, SeparationMode
from ..core.models import AnnotableFeatureType
from ..core.processors import BaseIsotopologueAnnotator
from .annotation import AnnotatorParameters, annotate, create_annotation_tools


class IsotopologueAnnotator(BaseIsotopologueAnnotator, AnnotatorParameters):
    """
    Annotates isotopologues in a sample.

    Groups isotopologue features. Each group is assigned an unique label, and a
    charge state. Each feature in a group is assigned an unique index that
    determines the position in the envelope.

    Annotations are stored to the `annotation` attribute of each feature.
    """

    bounds: dict[str, tuple[int, int]] = dict()
    """A dictionary of expected elements to minimum and maximum formula coefficients. If ``None``, bounds are
    obtained using :func:`tidyms.chem.get_chnops_bound` with ``2000`` as parameter.
    """

    def _annotate_func(self, features: list[AnnotableFeatureType]):
        tools = create_annotation_tools(self)
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
