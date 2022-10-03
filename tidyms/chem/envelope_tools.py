"""
Scores sum formula candidates using the isotopic envelope.

"""
import numpy as np
from scipy.special import erfc
from typing import Optional, Dict, Tuple, Callable, Union, Generator
from .atoms import PeriodicTable
from ._formula_generator import FormulaGenerator, FormulaCoefficientBounds, FormulaCoefficients
from ._envelope_utils import combine_envelopes, make_envelope_arrays
from .. import _constants as c
from .. import validation


class _EnvelopeGenerator:
    """
    Base class to generate envelopes from a list of molecular formulas.


    """
    def __init__(
        self,
        formula_generator: FormulaGenerator,
        max_length: int = 10,
        custom_abundances: Optional[dict] = None
    ):
        r"""
        Generates isotopic envelopes from molecular formulas

        Parameters
        ----------
        formula_generator : FormulaGenerator
        max_length : int, 10
            Length of the generated envelopes.
        custom_abundances : dict, optional
            Provides custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.

        """
        self.formula_generator = formula_generator
        self.max_length = max_length
        self._pos_env = make_formula_coefficients_envelopes(
            formula_generator.bounds,
            formula_generator.pos,
            max_length,
            p=custom_abundances
        )

        self._neg_env = make_formula_coefficients_envelopes(
            formula_generator.bounds,
            formula_generator.neg,
            max_length,
            p=custom_abundances)

        c12 = PeriodicTable().get_isotope("12C")
        if c12 in formula_generator.bounds.bounds:
            nc_min, nc_max = formula_generator.bounds.bounds.get(c12)
        else:
            nc_min, nc_max = 0, 0
        if custom_abundances is None:
            p = None
        else:
            p = custom_abundances.get(c12.get_symbol())
        Mc, pc = make_envelope_arrays(c12, nc_min, nc_max, max_length, p=p)
        self._c_env = CoefficientEnvelope(Mc, pc)

        self._query: Optional[_EnvelopeQuery] = None
        self.results: Optional[CoefficientEnvelope] = None

    def generate_envelopes(
        self,
        M: np.ndarray,
        p: np.ndarray,
        tolerance: float
    ):
        """
        Computes isotopic envelopes for formula candidates generated using the
        MMI mass from the envelope.

        Parameters
        ----------
        M: array
            Exact mass of the envelope.
        p: array
            Envelope abundances.
        tolerance : float
            Mass tolerance to generate formulas.

        """
        if M.size > self.max_length:
            msg = "`max_length` ({}) is lower than the query length ({})"
            raise ValueError(msg)
        M = M[:self.max_length]
        p = p[:self.max_length]
        p = p / np.sum(p)
        self._query = _EnvelopeQuery(M, p)
        self.formula_generator.generate_formulas(self._query.get_mmi_mass(), tolerance)
        self.results = _find_result_envelopes(
            self.formula_generator, self._pos_env, self._neg_env, self._c_env
        )

    def filter(self, min_M_tol: float, max_M_tol: float, p_tol: float, k: int):
        r"""
        Filters values from the k-th envelope that are outside the specified bounds.

        Parameters
        ----------
        min_M_tol : float
            Mass values lower than this value are filtered.
        max_M_tol : float
            Mass values greater than this value are filtered.
        p_tol : float
            Abundance tolerance
        k: int

        Notes
        -----
        Envelopes are filtered based on the following inequality. For each i-th
        peak the m/z tolerance is defined as follows:

        .. math::

            t_{i} = t + (T - t)(1 - y_{i})

        where :math:`t_{i}` is the mass tolerance for the i-th peak, t is the
        `min_mz_tolerance`, T is the `max_mz_tolerance` and :math:`y_[i]` is
        the abundance of the i-th value. Using this tolerance, an interval is
        built for the query mass, and candidates outside this interval are
        removed. This approach accounts for greater m/z errors for lower
        intensity peaks in the envelope.

        """
        if self.results is None:
            msg = "Envelopes must be generated first using the generate method."
            raise ValueError(msg)
        else:
            M_min, M_max = self._query.get_mass_bounds(min_M_tol, max_M_tol)
            query_size = M_min.size
            if k <= query_size:
                Mk_min = M_min[k]
                Mk_max = M_max[k]
                pk_min = max(0, self._query.p[k] - p_tol)
                pk_max = self._query.p[k] + p_tol
                self.results.filter(k, Mk_min, Mk_max, pk_min, pk_max)
            else:
                msg = "`k` must be lower than the length of the envelope"
                raise ValueError(msg)


class EnvelopeValidator(_EnvelopeGenerator):

    def __init__(
        self,
        formula_generator: FormulaGenerator,
        instrument: str = "qtof",
        max_length: int = 10,
        p_tol: Optional[float] = None,
        min_M_tol: Optional[float] = None,
        max_M_tol: Optional[float] = None,
        custom_abundances: Optional[dict] = None
    ):
        r"""

        Parameters
        ----------
        formula_generator : FormulaGenerator
        instrument : {"qtof", "orbitrap"}
            Sets default values for `p_tol`, `min_M_tol` and `max_M_tol`. If
            ``qtof`` is used, the values are set to ``0.05``, ``0.005`` and
            ``0.01`` respectively. If ``orbitrap`` is used, the values are set
            to ``0.05``, ``0.0025`` and ``0.005``. These values can be
            overwritten by setting their value in each parameter.
        max_length : int, default=10
            Maximum length of the envelopes.
        min_M_tol : float or None, default=None
            Exact mass tolerance for high abundance isotopologues. If ``None``,
            the parameter is set based on the `mode` value. See the notes
            for an explanation of how this value is used.
        max_M_tol : float or None, default=None
            Exact mass tolerance for low abundance isotopologues.  If ``None``,
            the parameter is set based on the `mode` value. See the notes
            for an explanation of how this value is used.
        p_tol : float or None, default=None
            tolerance threshold to include in the abundance results
        custom_abundances : dict, optional
            Provides custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.

        Notes
        -----
        Envelope validation is performed as follows:

        1.  For a query envelope mass and abundance `Mq`and `pq`, all formulas
            compatibles with the MMI are computed (see FormulaGenerator).
        2.  For each i-th pair of `Mq` and `pq`, a mass tolerance and abundance
            tolerance is defined as follows:

            .. math::
                dM_{i} = dM^{\textrm{max}} * pq_{i} + dM^{\textrm{min}} (1 - pq_{i})
            Where :math:`dM^{\textrm{max}}` is `min_M_tol`, :math:`dM^{\textrm{min}}`
            is `max_M_tol` and :math:`pq_{i}` is the i-th query abundance.
            Using the mass tolerance and abundance tolerance, candidates with
            mass or abundance values outside this interval are removed.
        3.  The candidates that remains define a mass and abundance window for
            the i + 1 elements of `Mq` and `pq`. If the values fall inside the
            window, the i + 1 elements are validated and the procedure is repeated
            until all isotopologues are validated or until an invalid isotopologue
            is found.

        """
        super(EnvelopeValidator, self).__init__(
            formula_generator, max_length, custom_abundances)
        if instrument in c.MS_INSTRUMENTS:
            params = _get_envelope_validator_params(instrument)
            user_params = {
                "p_tol": p_tol,
                "min_M_tol": min_M_tol,
                "max_M_tol": max_M_tol
            }
            user_params = {k: v for k, v in user_params.items() if v is not None}
            params.update(user_params)
            schema = _get_envelope_validator_schema()
            validator = validation.ValidatorWithLowerThan(schema)
            validation.validate(params, validator)
        else:
            msg = "Valid instruments are one of {}. Got {}"
            raise ValueError(msg.format(c.MS_INSTRUMENTS, instrument))
        self.p_tol = params["p_tol"]
        self.min_M_tol = params["min_M_tol"]
        self.max_M_tol = params["max_M_tol"]

    def _find_bounds(self, k: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Find exact mass and abundance bounds for an envelope based on formulas
        compatible with the MMI mass.

        Returns
        -------
        min_mass : array
            minimum valid mass for each peak
        max_mass : array
            maximum valid mass for each peak
        min_abundance : array
            minimum abundance for each peak
        min_abundance : array
            maximum abundance for each peak

        """
        self.filter(self.min_M_tol, self.max_M_tol, self.p_tol, k)

        M_tol = self._query.get_mass_tolerance(self.min_M_tol, self.max_M_tol)
        M_tol = M_tol[k]
        if self.results.get_envelope_size():

            # abundance bounds
            pk = self.results.get_pk(k)
            min_p = max(pk.min() - self.p_tol, 0.0)
            max_p = min(pk.max() + self.p_tol, 1.0)

            # exact mass bounds
            Mk = self.results.get_Mk(k)
            min_M = Mk.min() - M_tol
            max_M = Mk.max() + M_tol

            return min_M, max_M, min_p, max_p
        else:
            return None

    def validate(self, M: np.ndarray, p: np.ndarray) -> int:
        size = M.size
        length = 0
        tolerance = p[0] * self.min_M_tol + (1 - p[0]) * self.max_M_tol
        self.generate_envelopes(M, p, tolerance)
        for k in range(size):
            bounds = self._find_bounds(k)
            if bounds is None:
                break
            else:
                min_Mk, max_Mk, min_pk, max_pk = bounds
                valid = (
                    (M[k] >= min_Mk) and (M[k] <= max_Mk) and
                    (p[k] >= min_pk) and (p[k] <= max_pk)
                )
                if valid:
                    length += 1
                else:
                    break
        return length


class EnvelopeScorer(_EnvelopeGenerator):

    def __init__(
        self,
        formula_generator: FormulaGenerator,
        scorer: Union[str, Callable] = "qtof",
        max_length: int = 10,
        custom_abundances: Optional[dict] = None,
        **kwargs
    ):
        """

        Parameters
        ----------
        formula_generator : FormulaGenerator
        scorer : {"qtof", "orbitrap"} or Callable, default="qtof"
            If the ``"qtof"`` or ``"orbitrap"`` are used, the `score_envelopes` function
            is used. For ``"qtof"``, the values of `min_sigma_p`, `max_sigma_p`,
            `min_sigma_M` and `max_sigma_M` are set to 0.05, 0.05, 0.005 and
            0.01 respectively. For ``"orbitrap"`` the defaults are 0.05, 0.05,
            0.001 and 0.005. These defaults can be overwritten by passing
            them as key values arguments.

            A custom scoring function can be passed with the following
            signature:

            .. code-block:: python

                def score(M, p, Mq, pq, **kwargs):
                    pass

            where `M` and `p` are arrays of the formula candidates exact mass
            and abundances and `Mq` and `pq` are the query mass and query abundance.
        max_length : int, 10
            Length of the generated envelopes.
        custom_abundances : dict, optional
            Provides custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.

        Other Parameters
        ----------------
        kwargs :
            Optional parameter to pass into the scoring function.

        """
        super(EnvelopeScorer, self).__init__(
            formula_generator, max_length, custom_abundances)

        if scorer in c.MS_INSTRUMENTS:
            self.scorer = score_envelope
            self.scorer_params = _get_envelope_scorer_params(scorer)
            self.scorer_params.update(kwargs)
            schema = _get_envelope_scorer_schema()
            validator = validation.ValidatorWithLowerThan(schema)
            validation.validate(self.scorer_params, validator)
        else:
            self.scorer = scorer
            self.scorer_params = kwargs
        self.scores = None

    def score(self, M: np.ndarray, p: np.ndarray, tol: float):
        """
        Scores the isotopic envelope. The results can be recovered using the
        `get_top_results` method.

        """
        self.generate_envelopes(M, p, tol)
        n_results = self.results.get_envelope_size()
        scores = np.zeros(n_results)

        for i, (Mi, pi) in enumerate(self.results.iterate_rows()):
            scores[i] = self.scorer(
                Mi, pi, self._query.M, self._query.p, **self.scorer_params
            )
        self.scores = scores

    def get_top_results(self, n: Optional[int] = 10):
        """
        Return the scores for each formula candidate.

        Parameters
        ----------
        n: int, default
            number of first n results to return. If ``None``, return all formula candidates.

        Returns
        -------
        coefficients : array
            Formula coefficients. Each row is a formula candidate, each column
            is an element.
        elements : array
            The corresponding element to each column of `coefficients`.
        scores : array
            The corresponding score to each row of `coefficients`.

        """
        coefficients, elements, _ = self.formula_generator.results_to_array()

        # sort coefficients using the score and keep the first n values
        top_n_index = np.argsort(self.scores)
        if n is not None:
            top_n_index = top_n_index[:(-n - 1): -1]

        scores = self.scores[top_n_index]
        coefficients = coefficients[top_n_index]
        return coefficients, elements, scores


class CoefficientEnvelope:

    def __init__(self, M: np.ndarray, p: np.ndarray):
        self.M = M
        self.p = p
        self._filter: Optional[np.ndarray] = None

    def iterate_rows(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if self._filter is None:
            ind = np.arange(self.get_envelope_size())
        else:
            ind = self._filter

        for i in ind:
            yield self.M[i], self.p[i]

    def get_envelope_size(self) -> int:
        if self._filter is None:
            size = self.M.shape[0]
        else:
            size = self._filter.size
        return size

    def get_Mk(self, k):
        if self._filter is None:
            res = self.M[:, k]
        else:
            res = self.M[self._filter, k]
        return res

    def get_pk(self, k):
        if self._filter is None:
            res = self.p[:, k]
        else:
            res = self.p[self._filter, k]
        return res

    def filter(self, k: int, M_min: float, M_max: float, p_min: float, p_max: float):
        Mk = self.get_Mk(k)
        if self._filter is None:
            self._filter = np.arange(self.M.shape[0])
        if M_min > 0:
            valid_M = (Mk >= M_min) & (Mk <= M_max)
            self._filter = self._filter[valid_M]
        if p_min > 0:
            pk = self.get_pk(k)
            valid_p = (pk >= p_min) & (pk <= p_max)
            self._filter = self._filter[valid_p]


class _EnvelopeQuery:
    """
    Container class for Envelope Queries

    Attributes
    ----------
    M : array
        Exact mass of the envelope
    p : array
        Abundance of the envelope, normalized to 1.

    """

    def __init__(self, M: np.ndarray, p: np.ndarray):
        self.M = M
        self.p = p

    def get_mmi_mass(self):
        return self.M[0]

    def get_mass_tolerance(self, min_tol: float, max_tol: float):
        mass_slope = max_tol - min_tol
        mass_intercept = min_tol
        return mass_intercept + (1 - self.p) * mass_slope

    def get_mass_bounds(self, min_tol: float, max_tol: float):
        mass_tolerance = self.get_mass_tolerance(min_tol, max_tol)
        min_mass = np.maximum(self.M - mass_tolerance, 0.0)
        max_mass = self.M + mass_tolerance
        return min_mass, max_mass


def score_envelope(
    M: np.ndarray,
    p: np.ndarray,
    Mq: np.ndarray,
    pq: np.ndarray,
    min_sigma_M,
    max_sigma_M,
    min_sigma_p,
    max_sigma_p,
):
    r"""
    Scores the similarity between two isotopes.
    Parameters
    ----------
    M : array
        Theoretical mass values.
    p : array
        Theoretical abundances.
    Mq : array
        Query Mass values
    pq : array
        Query abundances.

    min_sigma_M : float
        Minimum mass standard deviation
    max_sigma_M : float
        Maximum mass standard deviation
    min_sigma_p : float
        Minimum abundance standard deviation.
    max_sigma_p : float
        Maximum abundance standard deviation.

    Returns
    -------
    score : float
        Number between 0 and 1. Higher values are related with similar envelopes.

    Notes
    -----
    The query envelope is compared against the theoretical envelope assuming
    a likelihood approach, similar to the described in [1]_. It is assumed
    that the theoretical mass and abundance is a normal random variable,
    with mean values defined by `M` and `p` and standard deviation computed as
    follows:

    .. math::

        \sigma_{M,i} = p_{i} \sigma_{M}^{\textrm{max}} + (1 - p_{i}) \sigma_{M}^{\textrm{min}}

    Where :math:`\sigma_{M,i}` is the standard deviation for the i-th element
    of `M`, :math:`p_{i}` is the i-th element of `p`, :math:`\sigma_{M}^{\textrm{max}}`
    is `max_sigma_M` and :math:`\sigma_{M}^{\textrm{min}}` is `min_sigma_M`. An
    analogous computation is done to compute the standard deviation for each
    abundance. Using this values, the likelihood of generating the values `Mq` and
    `pq` from `M` and `p` is computed using the error function.

    References
    ----------
    ..  [1] Sebastian Böcker, Matthias C. Letzel, Zsuzsanna Lipták, Anton
        Pervukhin, SIRIUS: decomposing isotope patterns for metabolite
        identification, Bioinformatics, Volume 25, Issue 2, 15 January 2009,
        Pages 218–224, https://doi.org/10.1093/bioinformatics/btn603

    """
    mz_sigma = max_sigma_M + (min_sigma_M - max_sigma_M) * pq
    sp_sigma = max_sigma_p + (min_sigma_p - max_sigma_p) * pq
    M = M[: Mq.size]
    p = p[: Mq.size]
    # normalize again the candidate intensity to 1
    p = p / p.sum()

    # corrects overestimation of the first peak area. This is done computing
    # an offset factor to subtract to the first peak. This correction is applied
    # only if the offset is positive. The offset value is computed in a way to
    # satisfy two conditions: the abundance of the first peak is equal to the
    # abundance of the candidate peak and the total area is normalized to one.
    # offset = (spq[0] - sp[0]) / (1 - sp[0])
    # offset = max(0, offset)
    norm = (pq[0] - 1) / (p[0] - 1)
    # spq = spq / (1 - offset)
    if norm < 1:
        pq = pq / norm
        pq[0] = p[0]

    # add a max offset parameter

    Mq = Mq + M[0] - Mq[0]
    dmz = np.abs(M - Mq) / (np.sqrt(2) * mz_sigma)
    dmz = dmz[pq > 0]
    dsp = np.abs(p - pq) / (np.sqrt(2) * sp_sigma)
    score = erfc(dmz).prod() * erfc(dsp).prod()
    return score


def make_formula_coefficients_envelopes(
    bounds: FormulaCoefficientBounds,
    coefficients: FormulaCoefficients,
    max_length: int,
    p: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Computes the isotopic envelopes for coefficient formulas.


    """
    if p is None:
        p = dict()

    # initialize envelopes
    rows = coefficients.coefficients.shape[0]
    M_arr = np.zeros((rows, max_length))
    p_arr = np.zeros((rows, max_length))
    p_arr[:, 0] = 1

    for k, isotope in enumerate(coefficients.isotopes):
        if isotope in bounds.bounds:
            lb, ub = bounds.bounds[isotope]
            symbol = isotope.get_symbol()
            tmp_abundance = p.get(symbol)
            Mi, pi = make_envelope_arrays(
                isotope, lb, ub, max_length, p=tmp_abundance)
            # lower corrects indices in cases when 0 is not the lower bound
            Mk = Mi[coefficients.coefficients[:, k] - lb, :]
            pk = pi[coefficients.coefficients[:, k] - lb, :]
            M_arr, p_arr = combine_envelopes(M_arr, p_arr, Mk, pk)
    envelope = CoefficientEnvelope(M_arr, p_arr)
    return envelope


def _find_result_envelopes(
    fg: FormulaGenerator,
    pos_env: CoefficientEnvelope,
    neg_env: CoefficientEnvelope,
    c_env: CoefficientEnvelope
) -> CoefficientEnvelope:
    shape = (fg.n_results, pos_env.M.shape[1])
    M = np.zeros(shape, dtype=float)
    p = np.zeros(shape, dtype=float)
    start = 0
    for k, (kp_index, kn_index, kc_index) in fg.results.items():
        k_size = kp_index.size
        if k_size > 0:
            end = start + k_size
            Mk, pk = _make_results_envelope_aux(
                kp_index, kn_index, kc_index, pos_env, neg_env, c_env
            )
            M[start:end] = Mk
            p[start:end] = pk
            start = end
    envelopes = CoefficientEnvelope(M, p)
    return envelopes


def _make_results_envelope_aux(
        p_index: np.ndarray,
        n_index: np.ndarray,
        c_index: np.ndarray,
        p_env: CoefficientEnvelope,
        n_env: CoefficientEnvelope,
        c_env: CoefficientEnvelope
) -> Tuple[np.ndarray, np.ndarray]:
    # combine positive and negative envelopes
    M, p = combine_envelopes(
        p_env.M[p_index], p_env.p[p_index], n_env.M[n_index], n_env.p[n_index])

    # combine with 12C envelopes
    M, p = combine_envelopes(M, p, c_env.M[c_index], c_env.p[c_index])

    return M, p


def _get_envelope_scorer_params(mode: str):
    res = {"min_sigma_p": 0.05, "max_sigma_p": 0.05}
    if mode == "qtof":
        res["min_sigma_M"] = 0.005
        res["max_sigma_M"] = 0.01
    elif mode == "orbitrap":
        res["min_sigma_M"] = 0.001
        res["max_sigma_M"] = 0.005
    else:
        msg = "valid modes are `qtof` and `orbitrap`"
        raise ValueError(msg)
    return res


def _get_envelope_validator_params(instrument: str) -> Dict[str, float]:
    p_tol = 0.05
    if instrument == c.QTOF:
        min_M_tol = 0.005
        max_M_tol = 0.01
    elif instrument == c.ORBITRAP:
        min_M_tol = 0.0025
        max_M_tol = 0.005
    else:
        msg = "invalid `instrument` value. Expected one of: {}"
        msg = msg.format(", ".join(c.MS_INSTRUMENTS))
        raise ValueError(msg)
    params = {"p_tol": p_tol, "min_M_tol": min_M_tol, "max_M_tol": max_M_tol}
    return params


def _get_envelope_validator_schema():
    schema = {
        "p_tol": {
            "type": "number",
            "is_positive": True
        },
        "min_M_tol": {
            "type": "number",
            "is_positive": True,
            "lower_or_equal": "max_M_tol"
        },
        "max_M_tol": {
            "type": "number"
        }
    }
    return schema


def _get_envelope_scorer_schema():
    schema = {
        "min_sigma_p": {
            "type": "number",
            "is_positive": True,
            "lower_or_equal": "max_sigma_p"
        },
        "max_sigma_p": {
            "type": "number",
        },
        "min_sigma_M": {
            "type": "number",
            "is_positive": True,
            "lower_or_equal": "max_sigma_M"
        },
        "max_sigma_M": {
            "type": "number",
        }
    }
    return schema
