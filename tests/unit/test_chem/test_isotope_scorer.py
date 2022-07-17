import pytest
import numpy as np
from tidyms.chem import isotope_scorer as isc
from tidyms.chem import Formula
from tidyms.chem import FormulaGenerator
from tidyms.chem.atoms import find_isotope


@pytest.fixture
def formula_generator():
    return FormulaGenerator.from_hmdb(1000, "qtof")


@pytest.fixture
def chnops_scorer(formula_generator):
    return isc.IsotopeScorer(formula_generator, max_length=5)

formula_str_list = ["C11H12N2O2", "C6H12O6", "C27H46O", "CO2", "HCOOH"]

@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer(f_str, chnops_scorer):
    # test that the best scoring candidate has the same molecular formula
    f = Formula(f_str)
    length = chnops_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)

    chnops_scorer.generate_envelopes(exact, abundance)
    chnops_scorer.score()
    coeff, isotopes, score = chnops_scorer.get_top_results(5)

    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_length_gt_scorer_max_length(f_str, chnops_scorer):
    # test that the best scoring candidate has the same molecular formula
    f = Formula(f_str)
    length = chnops_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length + 1)

    with pytest.warns(UserWarning):
        chnops_scorer.generate_envelopes(exact, abundance)
        chnops_scorer.score()
        coeff, isotopes, score = chnops_scorer.get_top_results(5)
        expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_mz_filter(f_str, chnops_scorer):
    # test that the best scoring candidate has the same molecular formula
    f = Formula(f_str)
    length = chnops_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)

    chnops_scorer.generate_envelopes(exact, abundance)
    chnops_scorer.filter_envelopes()
    chnops_scorer.score()
    coeff, isotopes, score = chnops_scorer.get_top_results(5)

    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


charged_formula_str_list = ["[C11H12N2O2]+", "[C6H12O6]2-", "[C27H46O]3+",
                            "[CO2]+", "[HCOOH]3-"]


@pytest.mark.parametrize("f_str", charged_formula_str_list)
def test_isotope_scorer_charged_species(f_str, chnops_scorer):
    f = Formula(f_str)
    length = chnops_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)
    charge = f.charge
    mz = exact / abs(charge)
    chnops_scorer.generate_envelopes(mz, abundance, charge=charge)
    chnops_scorer.filter_envelopes()
    chnops_scorer.score()
    coeff, isotopes, score = chnops_scorer.get_top_results(5)

    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.mark.parametrize("f_str", charged_formula_str_list)
def test_isotope_scorer_find_valid_bounds(f_str, chnops_scorer):
    f = Formula(f_str)
    length = chnops_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)
    charge = f.charge
    mz = exact / abs(charge)
    chnops_scorer.generate_envelopes(mz, abundance, charge=charge)
    min_mz, max_mz, min_ab, max_ab = chnops_scorer.find_valid_bounds()
    assert (min_mz < mz).all()
    assert (max_mz > mz).all()
    assert (min_ab < abundance).all()
    assert (max_ab > abundance).all()


@pytest.mark.parametrize("f_str", formula_str_list)
def test_isotope_scorer_custom_scorer(f_str, formula_generator):

    def cosine_scorer(mz1, ab1, mz2, ab2, **scorer_params):
        n1 = np.linalg.norm(ab1)
        n2 = np.linalg.norm(ab2)
        norm = np.linalg.norm(ab1 - ab2)
        cosine = norm / (n1 * n2)
        return 1 - cosine

    f = Formula(f_str)
    length = 5
    _, exact, abundance = f.get_isotopic_envelope(length)

    isotope_scorer = isc.IsotopeScorer(formula_generator, scorer=cosine_scorer,
                                       max_length=length)
    isotope_scorer.generate_envelopes(exact, abundance)
    isotope_scorer.filter_envelopes()
    isotope_scorer.score()
    coeff, isotopes, score = isotope_scorer.get_top_results(5)
    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


def test_isotope_scorer_error_score_before_generate_envelopes(chnops_scorer):
    f_str = "C20H40O2"
    f = Formula(f_str)
    length = 5
    _, exact, abundance = f.get_isotopic_envelope(length)
    with pytest.raises(ValueError):
        chnops_scorer.score()


def test_isotope_scorer_error_filter_before_generate_envelopes(chnops_scorer):
    f_str = "C20H40O2"
    f = Formula(f_str)
    length = 5
    _, exact, abundance = f.get_isotopic_envelope(length)
    with pytest.raises(ValueError):
        chnops_scorer.filter_envelopes()


@pytest.fixture
def positive_elements_scorer():
    elements = ["C", "H", "N"]
    max_mass = 300
    tolerance = 0.001
    fg = FormulaGenerator(elements, max_mass, tolerance)
    return isc.IsotopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["C2H3N", "N2H4", "C3N3H3"])
def test_scorer_positive_defect_elements_only(f_str, positive_elements_scorer):
    f = Formula(f_str)
    length = positive_elements_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)
    positive_elements_scorer.generate_envelopes(exact, abundance)
    positive_elements_scorer.filter_envelopes()
    positive_elements_scorer.score()
    coeff, isotopes, score = positive_elements_scorer.get_top_results(5)
    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.fixture
def negative_elements_scorer():
    elements = ["C", "O", "S"]
    max_mass = 300
    tolerance = 0.001
    fg = FormulaGenerator(elements, max_mass, tolerance)
    return isc.IsotopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["CS2", "C2OS2", "C3SO"])
def test_scorer_negative_defect_elements_only(f_str, negative_elements_scorer):
    f = Formula(f_str)
    length = negative_elements_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)
    negative_elements_scorer.generate_envelopes(exact, abundance)
    negative_elements_scorer.filter_envelopes()
    negative_elements_scorer.score()
    coeff, isotopes, score = negative_elements_scorer.get_top_results(5)
    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


@pytest.fixture
def no_carbon_scorer():
    elements = ["H", "O", "S", "P"]
    max_mass = 300
    tolerance = 0.001
    fg = FormulaGenerator(elements, max_mass, tolerance)
    return isc.IsotopeScorer(fg, max_length=5)


@pytest.mark.parametrize("f_str", ["H2O", "H3PO4", "H2SO4"])
def test_scorer_no_carbon(f_str, no_carbon_scorer):
    f = Formula(f_str)
    length = no_carbon_scorer.max_length
    _, exact, abundance = f.get_isotopic_envelope(length)
    no_carbon_scorer.generate_envelopes(exact, abundance)
    no_carbon_scorer.filter_envelopes()
    no_carbon_scorer.score()
    coeff, isotopes, score = no_carbon_scorer.get_top_results(5)
    expected_coeff = [f.composition[find_isotope(x)] for x in isotopes]
    assert np.array_equal(expected_coeff, coeff[0])


# TODO: test scorer using orbitrap params
