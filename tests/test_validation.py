from tidyms.validation import *
import pytest


@pytest.fixture
def blank_corrector_params():
    params = {"process_classes": ["a", "b", "c"],
              "corrector_classes": ["d"],
              "mode": "loq",
              "factor": 3}
    return params


@pytest.fixture
def prevalence_filter_params():
    params = {"process_classes": ["a", "b", "c"],
              "lb": 0,
              "ub": 0.9,
              "intraclass": True}
    return params


@pytest.fixture
def dratio_filter_params():
    params = {"lb": 0,
              "ub": 0.3,
              "robust": True}
    return params


@pytest.fixture
def batch_corrector_params():
    params = {"n_min": 5,
              "frac": None,
              "interpolator": "splines",
              "threshold": 0}
    return params


########################################
#   ~ validate function ~
########################################

def test_validate(blank_corrector_params):
    validate(blank_corrector_params, blankCorrectorValidator)
    assert True


def test_validate_invalid_param(blank_corrector_params):
    blank_corrector_params["mode"] = 4
    with pytest.raises(ValueError):
        validate(blank_corrector_params, blankCorrectorValidator)


def test_validate_multiple_invalid_param(blank_corrector_params):
    blank_corrector_params["mode"] = 4
    blank_corrector_params["factor"] = 0
    with pytest.raises(ValueError):
        validate(blank_corrector_params, blankCorrectorValidator)


########################################
#   ~ Blank Corrector Validator ~
########################################

def test_blank_corrector_validator_valid_input(blank_corrector_params):
    assert blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_process_classes_none(blank_corrector_params):
    blank_corrector_params["process_classes"] = None
    assert blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_bad_process_classes(blank_corrector_params):
    blank_corrector_params["process_classes"] = ["a", 2, "c"]
    assert not blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_mode_callable(blank_corrector_params):
    blank_corrector_params["mode"] = sum
    assert blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_bad_mode(blank_corrector_params):
    blank_corrector_params["mode"] = "loc"
    assert not blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_bad_factor(blank_corrector_params):
    blank_corrector_params["factor"] = "loc"
    assert not blankCorrectorValidator.validate(blank_corrector_params)


def test_blank_corrector_validator_no_pos_factor(blank_corrector_params):
    blank_corrector_params["factor"] = 0
    assert not blankCorrectorValidator.validate(blank_corrector_params)


########################################
#   ~ Prevalence Filter Validator ~
########################################

def test_prevalence_filter(prevalence_filter_params):
    assert prevalenceFilterValidator.validate(prevalence_filter_params)


def test_prevalence_filter_lb_greater_than_1(prevalence_filter_params):
    prevalence_filter_params["lb"] = 2
    assert not prevalenceFilterValidator.validate(prevalence_filter_params)


def test_prevalence_filter_negative_lb(prevalence_filter_params):
    prevalence_filter_params["lb"] = -0.1
    assert not prevalenceFilterValidator.validate(prevalence_filter_params)


def test_prevalence_filter_lb_greater_than_ub(prevalence_filter_params):
    prevalence_filter_params["lb"] = 0.5
    prevalence_filter_params["ub"] = 0.4
    assert not prevalenceFilterValidator.validate(prevalence_filter_params)


def test_prevalence_filter_intraclass_not_bool(prevalence_filter_params):
    prevalence_filter_params["intraclass"] = 1
    assert not prevalenceFilterValidator.validate(prevalence_filter_params)


def test_prevalence_filter_negative_threshold(prevalence_filter_params):
    prevalence_filter_params["threshold"] = -1
    assert not prevalenceFilterValidator.validate(prevalence_filter_params)


########################################
#   ~ D-Ratio Filter Validator ~
########################################

def test_dratio_filter(dratio_filter_params):
    assert dRatioFilterValidator.validate(dratio_filter_params)


def test_dratio_filter_robust_not_bool(dratio_filter_params):
    dratio_filter_params["robust"] = 4
    assert not dRatioFilterValidator.validate(dratio_filter_params)


########################################
#   ~ Batch Corrector Validator ~
########################################

def test_batch_corrector(batch_corrector_params):
    assert batchCorrectorValidator.validate(batch_corrector_params)


def test_batch_corrector_n_min_lower_than_4(batch_corrector_params):
    batch_corrector_params["n_min"] = 3
    assert not batchCorrectorValidator.validate(batch_corrector_params)


def test_batch_corrector_not_integer_n_min(batch_corrector_params):
    batch_corrector_params["n_min"] = 4.2
    assert not batchCorrectorValidator.validate(batch_corrector_params)


def test_batch_corrector_negative_frac(batch_corrector_params):
    batch_corrector_params["frac"] = -1
    assert not batchCorrectorValidator.validate(batch_corrector_params)


def test_batch_corrector_frac_greater_than_1(batch_corrector_params):
    batch_corrector_params["frac"] = 1.2
    assert not batchCorrectorValidator.validate(batch_corrector_params)


def test_batch_corrector_bad_interpolator(batch_corrector_params):
    batch_corrector_params["interpolator"] = "cubic"
    assert not batchCorrectorValidator.validate(batch_corrector_params)
