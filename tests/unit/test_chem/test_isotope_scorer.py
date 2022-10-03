import pytest
import numpy as np
from tidyms.chem import EnvelopeScorer, EnvelopeValidator
from tidyms.chem import Formula
from tidyms.chem import FormulaGenerator


@pytest.fixture
def formula_generator():
    return FormulaGenerator.from_hmdb(500)


formula_str_list = ["C11H12N2O2", "C6H12O6", "C27H46O", "CO2", "HCOOH"]


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer(f_str, formula_generator):
    # test that the best scoring candidate has the same molecular formula
    f = Formula(f_str)
    max_length = 5
    chnops_scorer = EnvelopeScorer(formula_generator, max_length=max_length)
    M, p = f.get_isotopic_envelope(max_length)
    tolerance = 0.005
    chnops_scorer.score(M, p, tolerance)
    coeff, isotopes, score = chnops_scorer.get_top_results(5)
    expected_coeff = [f.composition[x] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_length_gt_scorer_max_length(f_str, formula_generator):
    # test that the best scoring candidate has the same molecular formula
    f = Formula(f_str)
    max_length = 3
    chnops_scorer = EnvelopeScorer(formula_generator, max_length=max_length)
    M, p = f.get_isotopic_envelope(max_length + 1)
    tolerance = 0.005

    with pytest.raises(ValueError):
        chnops_scorer.score(M, p, tolerance)
        coeff, isotopes, score = chnops_scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_find_valid_bounds(f_str, formula_generator):
    max_length = 5
    validator = EnvelopeValidator(formula_generator, max_length=max_length)
    f = Formula(f_str)
    M, p = f.get_isotopic_envelope(max_length)
    tolerance = 0.005
    validator.generate_envelopes(M, p, tolerance)
    for k in range(M.size):
        min_mz, max_mz, min_ab, max_ab = validator._find_bounds(k)
        assert np.all(min_mz < M[k])
        assert np.all(max_mz > M[k])
        assert np.all(min_ab < p[k])
        assert np.all(max_ab > p[k])


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_custom_scorer(f_str, formula_generator):

    def cosine_scorer(mz1, ab1, mz2, ab2, **scorer_params):
        n1 = np.linalg.norm(ab1)
        n2 = np.linalg.norm(ab2)
        norm = np.linalg.norm(ab1 - ab2)
        cosine = norm / (n1 * n2)
        return 1 - cosine

    f = Formula(f_str)
    max_length = 5
    M, p = f.get_isotopic_envelope(max_length)

    envelope_scorer = EnvelopeScorer(
        formula_generator, scorer=cosine_scorer, max_length=max_length)
    tolerance = 0.005
    envelope_scorer.score(M, p, tolerance)
    coeff, isotopes, score = envelope_scorer.get_top_results(5)
    expected_coeff = [f.composition[x] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.fixture
def positive_elements_scorer():
    bounds = {"C": (0, 10), "H": (0, 10), "N": (0, 10)}
    max_mass = 300
    tolerance = 0.001
    fg = FormulaGenerator(bounds, max_mass, tolerance)
    return EnvelopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["C2H3N", "N2H4", "C3N3H3"])
def test_scorer_positive_defect_elements_only(f_str, positive_elements_scorer):
    f = Formula(f_str)
    max_length = positive_elements_scorer.max_length
    M, p = f.get_isotopic_envelope(max_length)
    tolerance = 0.005
    positive_elements_scorer.score(M, p, tolerance)
    coeff, isotopes, score = positive_elements_scorer.get_top_results(5)
    expected_coeff = [f.composition[x] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.fixture
def negative_elements_scorer():
    elements = {"C": (0, 10), "O": (0, 10), "S": (0, 10)}
    max_mass = 300
    fg = FormulaGenerator(elements, max_mass)
    return EnvelopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["CS2", "C2OS2", "C3SO"])
def test_scorer_negative_defect_elements_only(f_str, negative_elements_scorer):
    f = Formula(f_str)
    max_length = negative_elements_scorer.max_length
    M, p = f.get_isotopic_envelope(max_length)
    tolerance = 0.001
    negative_elements_scorer.score(M, p, tolerance)
    coeff, isotopes, score = negative_elements_scorer.get_top_results(5)
    expected_coeff = [f.composition[x] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.fixture
def no_carbon_scorer():
    elements = {"H": (0, 10), "O": (0, 5), "S": (0, 5), "P": (0, 5)}
    fg = FormulaGenerator(elements)
    return EnvelopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["H2O", "H3PO4", "H2SO4"])
def test_scorer_no_carbon(f_str, no_carbon_scorer):
    f = Formula(f_str)
    max_length = no_carbon_scorer.max_length
    M, p = f.get_isotopic_envelope(max_length)
    tolerance = 0.005
    no_carbon_scorer.score(M, p, tolerance)
    coeff, isotopes, score = no_carbon_scorer.get_top_results(5)
    expected_coeff = [f.composition[x] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


def test_EnvelopeScorer_invalid_scorer(formula_generator):
    with pytest.raises(ValueError):
        EnvelopeScorer(formula_generator, scorer="bad-scorer")


def test_EnvelopeValidator_invalid_instrument(formula_generator):
    with pytest.raises(ValueError):
        EnvelopeValidator(formula_generator, instrument="bad-scorer")


@pytest.mark.parametrize("f_str", formula_str_list)
def test_EnvelopeValidator_validate(formula_generator, f_str):
    max_length = 5
    validator = EnvelopeValidator(formula_generator, max_length=max_length)
    f = Formula(f_str)
    M, p = f.get_isotopic_envelope(max_length)
    validated_length = validator.validate(M, p)
    assert validated_length == max_length


def test_EnvelopeValidator_validate_invalid_envelope(formula_generator):
    max_length = 5
    validator = EnvelopeValidator(formula_generator, max_length=max_length)
    f = Formula("C2H8B")
    M, p = f.get_isotopic_envelope(max_length)
    validated_length = validator.validate(M, p)
    expected_length = 0
    assert validated_length == expected_length