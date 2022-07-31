"""
Scores sum formula candidates using the isotopic envelope.
"""
import warnings
from collections import namedtuple
import numpy as np
from typing import Optional, Dict, Tuple, Callable, Union
from scipy.special import erfc
from .formula_generator import FormulaGenerator
from .atoms import find_isotope
from ._isotope_distributions import make_coeff_abundances
from ._isotope_distributions import _make_element_abundance_array
from . _isotope_distributions import _combine_array_abundances
from .utils import mz_to_mass, mass_to_mz
abundance_dict_type = Optional[Dict[str, Tuple[int, int]]]
scorer_type = Optional[Union[Callable, str]]
IsotopicEnvelope = namedtuple("IsotopicEnvelope",
                              ["nominal", "exact", "abundance"])


class _IsotopeQuery:
    """
    passes mass and abundance values to a IsotopeScorer

    Attributes
    ----------
    mz: array
        A sorted array of m/z values. Each element match a nominal mass
        increment of 1. A value of zero in an element indicates a missing m/z
        value. The input is cropped to match the max length parameter.
    abundance : array
        Matching abundances for each m/z value, normalized to total abundance
        of 1.
    charge: int
        The charge of the species.
    monoisotopic_index: int
        Index of the monoisotopic mass, used to generate candidate formulas.
    """
    def __init__(self, mz: np.ndarray, abundance: np.ndarray, charge: int = 0,
                 monoisotopic_index: int = 0, length: int = 10):

        # compute nominal mass increments for each isotopologue
        isotope_index = mz - mz[0]
        if charge != 0:
            isotope_index = isotope_index * abs(charge)
        isotope_index = np.round(isotope_index).astype(int)

        # maximum mass increment
        max_isotope = isotope_index[-1]
        query_size = max(max_isotope + 1, length)

        self.mz = np.zeros(query_size)
        self.mz[isotope_index] = mz
        self.abundance = np.zeros(query_size)
        self.abundance[isotope_index] = abundance
        self.charge = charge
        self.monoisotopic_index = isotope_index[monoisotopic_index]

        if length <= max_isotope:
            msg = "mz and abundance was truncated to match `length` {}"
            msg = msg.format(length)
            warnings.warn(msg, UserWarning)
            self.mz = self.mz[:length]
            self.abundance = self.abundance[:length]

        self.abundance = self.abundance / self.abundance.sum()

    def get_mass(self):
        """
        computes the mass of the based on m/z and charge.
        """
        return mz_to_mass(self.mz, self.charge)

    def get_mass_bounds(self, min_mz_tolerance: float = 0.005,
                        max_mz_tolerance: float = 0.01):
        r"""
        Computes m/z bounds to filter formula candidates.

        An interval is built centered on each m/z value ans using the following
        equation:

        .. math::

            \epsilon_{i} = tol_{min} * Q * y_{i} + tol_{max} * q * (1 - y_{i})
            mz_{min, i} = mz_{i} - \epsilon_{i}
            mz_{max, i} = mz_{i} + \epsilon_{i}

        where :math:`y_{i}` is the abundance of the i-th isotope, q is the
        charge of the species.

        """
        if self.charge == 0:
            charge = 1
        else:
            charge = abs(self.charge)
        mass_slope = (max_mz_tolerance - min_mz_tolerance) * charge
        mass_intercept = min_mz_tolerance * charge
        mass_tolerance = mass_intercept + (1 - self.abundance) * mass_slope
        mass = self.get_mass()
        min_mass = mass - mass_tolerance
        min_mass[mass == 0] = 0
        max_mass = mass + mass_tolerance
        max_mass[mass == 0] = np.inf
        return min_mass, max_mass

    def get_monoisotopic_mass(self):
        mass = self.get_mass()
        return mass[self.monoisotopic_index]


class IsotopeScorer:
    """
    find compatible formulas and creates an score based on the theoretical
    isotopic envelope of formula candidates.
    """

    def __init__(self, fg: FormulaGenerator, scorer: scorer_type = "qtof",
                 min_abundance: float = 0.01, max_length: int = 10,
                 custom_abundances: Optional[dict] = None,
                 min_p: float = 1e-10, scorer_params: Optional[dict] = None):
        r"""
        find compatible formulas and creates an score based on the theoretical
        isotopic envelope of formula candidates.

        Parameters
        ----------
        fg : FormulaGenerator
        scorer : {"qtof", "orbitrap"} or Callable
            Function used to calculate the score. The scorer prototype must be
            as follows:

            scorer(mz_t, sp_t, mz_q, sp_q, **scorer_params) -> float.

            Where mz_t is an array of theoretical mz, sp_t are its related
            intensities, and mz_q and sp_q are the mz and intensities of the
            query. A perfect score should be equal to one and bad scores should
            be close to zero.
            When using `qtof` or `orbitrap`, the scorer function is quite
            similar to the one described in [1]. See the Notes for a detailed
            description of the function.
        min_abundance : float
            Minimum abundance of a isotopic peak to be considered
        max_length : int, 10
            Maximum number of isotopologues to consider for the scoring.
        custom_abundances : dict, optional
            Used to provide custom elemental abundances. A mapping from element
            symbols str to an abundance array. The abundance array must have
            the same size that the natural abundance and its sum must be equal
            to one. For example, for "C", an alternative abundance can be
            array([0.15, 0.85]) for isotopes with nominal mass 12 and 13.
        min_p : float, 1e-10
            Minimum value of an abundance. This value is used to remove errors
            in small values due to floating point rounding in the fft based
            convolution used to compute the isotopic abundance. Usually this
            value should not be changed.
        scorer_params: dict, optional
            Optional parameters to pass to the scorer function.

        Notes
        -----

        The scorer models the m/z and abundance values of the candidates as
        the product of gaussian distributions. Using this distribution, the
        score :math:`S` can be interpreted as the probability of generating the
        values observed in the query:

        .. math::

                S = \prod_{j=1}^{n}\textrm{erfc}(\frac{M_{j} - m_{j}} \\
                {\sqrt{2}\sigma_{m, j}}\textrm{erfc}(\frac{I_{j} - i_{j}} \\
                {\sqrt{2}\sigma_{i, j}}

        where :math:`M_{j}` and :math:`m_{j}` are exact mass of the j-th peak of
        the candidate and the query respectively; :math:`I_{j}` and
        :math:`i_{j}` are the abundance of the j-th peak. The terms
        :math:`\sigma{m, j}` and :math:`\sigma{i, j}` are computed using the
        abundance values of the query and reduces the penalty for peaks with
        low intensity:

        .. math::

        \sigma_{m, j} = \sigma_{m, max} + \\
        (\sigma_{m, min} - \sigma_{m, max}) i_{j}

        \sigma_{i, j} = \sigma_{i, max} + \\
        (\sigma_{i, min} - \sigma_{i, max}) i_{j}

        :math:`\sigma{m, min}` is the standard deviation in m/z for peaks with
        abundance close to one, :math:`\sigma{m, max}`is the standard
        deviation in m/z for peaks with abundance close to zero,
        :math:`\sigma{i, min}`is the standard deviation in abundance for peaks
        with abundance close to one and :math:`\sigma{i, max}`is the standard
        deviation in abundance for peaks with abundance close to zero.
        This parameters can be passed to the scorer function through the
        `scorer_params` parameter (see the documentation of the `score_isotope`
        function). For `qtof`, the min and max m/z standard deviations are
        0.005 and 0.01. In the case of `orbitrap` they are 0.001 and 0.005. In
        both cases the standard deviation for abundance is 0.05.

        """
        self.formula_generator = fg
        self.min_abundance = min_abundance
        self.max_length = max_length
        self.min_p = min_p

        if scorer in ["qtof", "orbitrap"]:
            self.scorer = score_isotope
            self.scorer_params = _get_scorer_params(scorer)
        else:
            self.scorer = scorer

        if scorer_params is None:
            self.scorer_params = dict()

        self._coefficient_envelopes = \
            _make_isotopic_envelopes(fg, length=max_length, min_p=min_p,
                                     abundances=custom_abundances)
        self._query = None
        self.envelopes = None
        self.scores = None
        self._valid_index = None

    def generate_envelopes(self, mz: np.ndarray, sp: np.ndarray,
                           charge: int = 0, monoisotopic_index: int = 0):
        """
        Generate isotopic envelopes using formula candidates.

        Parameters
        ----------
        mz: array
            Array of sorted m/z values. Must include values from ONE isotopic
            envelope.
        sp: array
            Isotopologue abundances, normalized to 1.
        charge: charge of the species, if 0, assumes a neutral mass.
        monoisotopic_index : int
            position in `mz` of the monoisotopic mass, 0 by default.

        """
        # query = _IsotopeQuery(mz, sp, charge=charge, length=self.max_length,
        #                       monoisotopic_index=monoisotopic_index,)
        self._query = _IsotopeQuery(mz, sp, charge=charge,
                                    length=self.max_length,
                                    monoisotopic_index=monoisotopic_index)
        self.scores = None
        self._valid_index = None
        monoisotopic_mass = self._query.get_monoisotopic_mass()
        self.formula_generator.generate_formulas(monoisotopic_mass)
        res = _merge_results(self.formula_generator.n_results,
                             self.formula_generator._results,
                             self._coefficient_envelopes, self.min_p)
        self.envelopes = IsotopicEnvelope(*res)

    def filter_envelopes(self, min_mz_tolerance: float = 0.005,
                         max_mz_tolerance: float = 0.01,
                         abundance_tolerance: float = 0.05):
        r"""
        Remove candidates that aren't inside the specified m/z and abundance
        tolerance. Using this method before trying to score candidates greatly
        reduces the computing time.

        Parameters
        ----------
        min_mz_tolerance : float
        max_mz_tolerance : float
        abundance_tolerance : float

        Notes
        -----
        Envelopes are filter based on the following inequality. For each i-th
        peak a m/z tolerance is defined as follows:

        .. math::

            t_{i} = t + (T - t)(1 - y_{i})

        where :math:`t_{i}` is the mass tolerance for the i-th peak, t is the
        `min_mz_tolerance`, T is the `max_mz_tolerance` and :math:`y_[i]` is
        the abundance of the i-th value. Using this tolerance, an interval is
        built for the query mass, and candidates outside this interval are
        removed. This approach accounts for greater m/z errors for lower
        intensity peaks in the envelope.

        """
        if self.envelopes is None:
            msg = "envelopes must be generated  with the generate_envelopes " \
                  "method to find valid envelopes"
            raise ValueError(msg)
        else:
            n_rows, n_cols = self.envelopes.exact.shape
            valid_index = np.arange(n_rows)
            min_mass, max_mass = self._query.get_mass_bounds(min_mz_tolerance,
                                                             max_mz_tolerance)
            envelope_exact = self.envelopes.exact
            query_size = min_mass.size
            envelope_abundance = self.envelopes.abundance

            for k in range(query_size):
                exact_k = envelope_exact[valid_index, k]
                if min_mass[k] > 0:
                    valid_mass = ((exact_k >= min_mass[k]) &
                                  (exact_k <= max_mass[k]))
                    valid_index = valid_index[valid_mass]
                max_abundance_k = self._query.abundance[k] + abundance_tolerance
                min_abundance_k = self._query.abundance[k] - abundance_tolerance
                min_abundance_k = max(0, min_abundance_k)
                if min_abundance_k > 0:
                    env_abundance_k = envelope_abundance[valid_index, k]
                    valid_abundance = ((env_abundance_k >= min_abundance_k) &
                                       (env_abundance_k <= max_abundance_k))
                    valid_index = valid_index[valid_abundance]

            self._valid_index = valid_index

    def find_valid_bounds(self, min_mz_tolerance: float = 0.005,
                          max_mz_tolerance: float = 0.01,
                          abundance_tolerance: float = 0.05,
                          return_mz: bool = True):
        """
        Find m/z and abundance bounds for each isotopologue based on formulas
        compatible with the monoisotopic mass. This can be used to test
        if an isotopic envelope is valid.

        Parameters
        ----------
        min_mz_tolerance : float
            m/z tolerance for high abundance isotopologues.
        max_mz_tolerance : float
            m/z tolerance for low abundance isotopologues.
        abundance_tolerance : float
            tolerance threshold to include in the abundance results
        return_mz : bool
            If True, return the mass values as m/z. Else returns mass values.

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
        if self._valid_index is None:
            self.filter_envelopes(min_mz_tolerance=min_mz_tolerance,
                                  max_mz_tolerance=max_mz_tolerance,
                                  abundance_tolerance=abundance_tolerance)
        min_mass, max_mass = self._query.get_mass_bounds(min_mz_tolerance,
                                                         max_mz_tolerance)
        if self._valid_index.size > 0:
            abundance = self.envelopes.abundance[self._valid_index]
            min_abundance = abundance.min(axis=0) - abundance_tolerance
            min_abundance[min_abundance < 0] = 0
            max_abundance = abundance.max(axis=0) + abundance_tolerance
            min_abundance = min_abundance[:min_mass.size]
            max_abundance[max_abundance > 1] = 1
            max_abundance = max_abundance[:min_mass.size]
            if return_mz:
                min_mass = mass_to_mz(min_mass, self._query.charge)
                max_mass = mass_to_mz(max_mass, self._query.charge)
            return min_mass, max_mass, min_abundance, max_abundance
        else:
            return None

    def score(self):
        """
        Scores the isotopic envelope. The results can be recovered using the
        `get_top_results` method.

        """
        if self._query is None:
            msg = "candidate envelopes must be generated with the " \
                  "generate_envelopes method before scoring"
            raise ValueError(msg)

        mass, abundance = self._query.get_mass(), self._query.abundance

        if self._valid_index is None:
            n_results = self.formula_generator.n_results
            ind = range(n_results)
        else:
            n_results = self._valid_index.size
            ind = self._valid_index
        scores = np.zeros(n_results)

        for k, k_ind in enumerate(ind):
            scores[k] = self.scorer(self.envelopes.exact[k_ind],
                                    self.envelopes.abundance[k_ind],
                                    mass, abundance, **self.scorer_params)
        self.scores = scores

    def get_top_results(self, n=10):
        """
        Return the scores for each formula candidate.

        Parameters
        ----------
        n: int
            number of first n results to return. If None, return all formula
            candidates.

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

        # remove formulas filtered by m/z
        if self._valid_index is not None:
            coefficients = coefficients[self._valid_index]

        # sort coefficients using the score and keep the first n values
        top_n_index = np.argsort(self.scores)
        if n is not None:
            top_n_index = top_n_index[:(-n - 1):-1]

        scores = self.scores[top_n_index]
        coefficients = coefficients[top_n_index]
        return coefficients, elements, scores

    def reset_valid_index(self):
        self._valid_index = None

    def crop_envelopes(self, size: int = 1):
        abundance = self.envelopes.abundance
        abundance[:, -size:] = 0.0
        abundance /= abundance.sum(axis=1)[:, np.newaxis]


def _make_isotopic_envelopes(fg: FormulaGenerator, length: int = 10,
                             min_p: float = 1e-10,
                             abundances: abundance_dict_type = None):

    if abundances is None:
        abundances = dict()

    # positive envelopes
    p_isotopes = fg.pos.isotopes
    p_coefficient = fg.pos.coefficients
    has_positive = p_coefficient.max() > 0
    if has_positive:
        p_bounds = [fg.bounds[x] for x in p_isotopes]
        p_envelopes = make_coeff_abundances(p_bounds, p_coefficient, p_isotopes,
                                            length, abundances=abundances,
                                            min_p=min_p)
        p_envelopes = IsotopicEnvelope(*p_envelopes)
    else:
        p_envelopes = _make_empty_envelope(length)

    n_isotopes = fg.neg.isotopes
    n_coefficient = fg.neg.coefficients
    has_negative = n_coefficient.max() > 0
    if has_negative:
        n_bounds = [fg.bounds[x] for x in n_isotopes]
        n_envelopes = make_coeff_abundances(n_bounds, n_coefficient, n_isotopes,
                                            length, abundances=abundances,
                                            min_p=min_p)
        n_envelopes = IsotopicEnvelope(*n_envelopes)
    else:
        n_envelopes = _make_empty_envelope(length)

    if fg.has_carbon:
        c12 = find_isotope("12C")
        c_min, c_max = fg.bounds[c12]
        c_abundance = abundances.get("C")
        c_envelopes = _make_element_abundance_array(c12, c_min, c_max, length,
                                                    abundance=c_abundance)
        c_envelopes = IsotopicEnvelope(*c_envelopes)
    else:
        c_envelopes = _make_empty_envelope(length)
    return p_envelopes, n_envelopes, c_envelopes


def _make_empty_envelope(max_length: int):
    shape = (1, max_length)
    nominal = np.zeros(shape, dtype=int)
    exact = np.zeros(shape, dtype=float)
    abundance = np.zeros(shape, dtype=float)
    abundance[0, 0] = 1.0
    return IsotopicEnvelope(nominal, exact, abundance)


def _make_results_envelope(index, envelopes: Tuple[IsotopicEnvelope],
                           min_p):
    shape = (index[0].size, envelopes[0].nominal.shape[1])
    nominal = np.zeros(shape, dtype=int)
    exact = np.zeros(shape, dtype=float)
    abundance = np.zeros(shape, dtype=float)
    abundance[:, 0] = 1
    for env, ind in zip(envelopes, index):
        nominal, exact, abundance = \
            _combine_array_abundances(nominal, exact, abundance,
                                      env.nominal[ind], env.exact[ind],
                                      env.abundance[ind],
                                      min_p=min_p)
    return nominal, exact, abundance


def _merge_results(n_results, results, envelopes, min_p):
    shape = (n_results, envelopes[0].nominal.shape[1])
    nominal = np.zeros(shape, dtype=int)
    exact = np.zeros(shape, dtype=float)
    abundance = np.zeros(shape, dtype=float)
    start = 0
    for k, index in results.items():
        # check that there are results
        if index[0].size > 0:
            end = start + index[0].size
            tmp_nom, tmp_ex, tmp_ab = _make_results_envelope(index, envelopes,
                                                             min_p)
            nominal[start:end] = tmp_nom
            exact[start:end] = tmp_ex
            abundance[start:end] = tmp_ab
            start = end
    return nominal, exact, abundance


def score_isotope(mz: np.ndarray, sp: np.ndarray, mzq: np.ndarray,
                  spq: np.ndarray, min_sigma_mz: float = 0.005,
                  max_sigma_mz: float = 0.01, min_sigma_sp: float = 0.05,
                  max_sigma_sp: float = 0.05):
    mz_sigma = max_sigma_mz + (min_sigma_mz - max_sigma_mz) * spq
    sp_sigma = max_sigma_sp + (min_sigma_sp - max_sigma_sp) * spq
    mz = mz[:mzq.size]
    sp = sp[:mzq.size]
    # normalize again the candidate intensity to 1
    sp = sp / sp.sum()

    # corrects overestimation of the first peak area. This is done computing
    # an offset factor to subtract to the first peak. This correction is applied
    # only if the offset is positive. The offset value is computed in a way to
    # satisfy two conditions: the abundance of the first peak is equal to the
    # abundance of the candidate peak and the total area is normalized to one.
    # offset = (spq[0] - sp[0]) / (1 - sp[0])
    # offset = max(0, offset)
    norm = (spq[0] - 1) / (sp[0] - 1)
    # spq = spq / (1 - offset)
    if norm < 1:
        spq = spq / norm
        spq[0] = sp[0]

    # add a max offset parameter

    mzq = mzq + mz[0] - mzq[0]
    dmz = np.abs(mz - mzq) / (np.sqrt(2) * mz_sigma)
    dmz = dmz[spq > 0]
    dsp = np.abs(sp - spq) / (np.sqrt(2) * sp_sigma)
    score = erfc(dmz).prod() * erfc(dsp).prod()
    return score


def _get_scorer_params(mode: str):
    res = {"min_sigma_sp": 0.05, "max_sigma_sp": 0.05}
    if mode == "qtof":
        res["min_sigma_mz"] = 0.005
        res["max_sigma_mz"] = 0.01
    elif mode == "orbitrap":
        res["min_sigma_mz"] = 0.001
        res["max_sigma_mz"] = 0.005
    else:
        msg = "valid modes are `qtof` and `orbitrap`"
        raise ValueError(msg)
    return res
