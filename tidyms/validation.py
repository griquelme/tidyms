"""
Validation functions for Filter, Pipelines and DataContainer
"""


import warnings
from os import getcwd
from os.path import isdir
from os.path import join
import cerberus

# TODO: validate DataContainer using cerberus


# functions used by check_with
def is_callable(field, value, error):
    if not hasattr(value, "__call__"):
        msg = "Must be a string or callable"
        error(field, msg)


def is_all_positive(field, value, error):
    if value is not None:
        cond = (value > 0).all()
        if not cond:
            msg = "All of the values must be positive"
            error(field, msg)
# -----------------------------


def validate(params: dict, validator: cerberus.Validator) -> None:
    """
    Function used to validate parameters.

    Parameters
    ----------
    params: dict
    validator: cerberus.Validator

    Returns
    -------
    None

    Raises
    ------
    ValueError: if any of the parameters are invalid.
    """
    pass_validation = validator.validate(params)
    if not pass_validation:
        msg = ""
        for field, e_msgs in validator.errors.items():
            for e_msg in e_msgs:
                msg += "{}: {}\n".format(field, e_msg)
        raise ValueError(msg)


class ValidatorWithLowerThan(cerberus.Validator):
    def _validate_lower_than(self, other, field, value):
        """
        Tests if a value is lower than the value of other field.

        The rule's arguments are validated against this schema:
        {"type": "string"}
        """
        if (other not in self.document) or (self.document[other] is None):
            return False
        if value >= self.document[other]:
            msg = "{} must be lower than {}".format(field, other)
            self._error(field, msg)

    def _validate_lower_or_equal(self, other, field, value):
        """
        Tests if a value is lower than the value of other field.

        The rule's arguments are validated against this schema:
        {"type": "string"}
        """
        if (other not in self.document) or (self.document[other] is None):
            return False
        if value > self.document[other]:
            msg = "{}, must be lower or equal than {}".format(field, other)
            self._error(field, msg)

    def _validate_is_positive(self, is_positive, field, value):
        """
        Tests if a value is positive

        The rule's arguments are validated against this schema:
        {"type": "boolean"}
        """
        if is_positive and (value is not None) and (value <= 0):
            msg = "Must be a positive number"
            self._error(field, msg)


def validate_data_matrix(df):
    """
    Check type and values for data matrix.

    Check
    Parameters
    ----------
    df : pd.DataFrame.

    Returns
    -------

    """
    float_columns = df.dtypes == float
    # check that all columns have float values
    if not float_columns.all():
        no_float_columns = df.columns[~float_columns]
        msg = "Some columns have non float values: {}."
        msg = msg.format(", ".join(no_float_columns))
        raise TypeError(msg)
    # check that all elements are non negative
    if (df < 0).sum().sum():
        raise ValueError("Data matrix has negative elements")
    # check nan
    if (df.isna()).sum().sum():
        warnings.warn("Data matrix has NANs. These values should be imputed "
                      "before further analysis")


def validate_feature_metadata(df):
    """
    Check type and values of rt and mz
    Parameters
    ----------
    df

    Returns
    -------
    """
    # rt, mz and class information check
    if "mz" not in df.columns:
        raise KeyError("mz values are required for all features")
    if "rt" not in df.columns:
        raise KeyError("rt values are required for all features")
    # rt, mz and class and data matrix check:
    if (df["mz"] < 0).any():
        raise ValueError("mz values should be greater than zero")
    if (df["rt"] < 0).any():
        raise ValueError("mz values should be greater than zero")


def validate_sample_metadata(df):
    """
    Check sample class.
    Parameters
    ----------
    df

    Returns
    -------
    """
    if "class" not in df.columns:
        raise KeyError("class information are required for all samples")


def validate_data_container(data_matrix, feature_definitions,
                            sample_info, data_path):
    validate_data_matrix(data_matrix)
    validate_feature_metadata(feature_definitions)
    validate_sample_metadata(sample_info)
    samples_equal = data_matrix.index.equals(sample_info.index)
    if not samples_equal:
        msg = "Samples names should be equal in data matrix and sample info."
        raise ValueError(msg)
    features_equal = data_matrix.columns.equals(feature_definitions.index)
    if not features_equal:
        msg = "feature names should be equal in data matrix and feature " \
              "definitions."
        raise ValueError(msg)
    # path validation
    if (data_path is not None) and not isdir(data_path):
        full_path = join(getcwd(), data_path)
        msg = "{} not found".format(full_path)
        raise FileNotFoundError(msg)


def validate_blank_corrector_params(params):
    schema = {
        "corrector_classes": {"type": "list", "nullable": True,
                              "schema": {"type": "string"}},
        "process_classes": {"type": "list", "nullable": True,
                            "schema": {"type": "string"}},
        "mode": {"anyof": [{"type": "string",
                            "allowed": ["lod", "loq", "mean", "max"]},
                           {"check_with": is_callable}]},
        "factor": {"type": "number", "is_positive": True},
        "process_blanks": {"type": "boolean"}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_prevalence_filter_params(params):
    schema = {
        "lb": {"type": "number", "min": 0, "max": 1, "lower_or_equal": "ub"},
        "ub": {"type": "number", "min": 0, "max": 1},
        "threshold": {"type": "number", "min": 0},
        "intraclass": {"type": "boolean"},
        "process_classes": {"type": "list", "nullable": True,
                            "schema": {"type": "string"}}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_dratio_filter_params(params):
    schema = {
        "robust": {"type": "boolean"},
        "lb": {"type": "number", "min": 0, "max": 1, "lower_or_equal": "ub"},
        "ub": {"type": "number", "min": 0, "max": 1}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_variation_filter_params(params):
    schema = {
        "robust": {"type": "boolean"},
        "lb": {"type": "number", "min": 0, "max": 1, "lower_or_equal": "ub"},
        "ub": {"type": "number", "min": 0, "max": 1},
        "intraclass": {"type": "boolean"},
        "process_classes": {"type": "list", "nullable": True,
                            "schema": {"type": "string"}}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_batch_corrector_params(params):
    schema = {
        "min_qc_dr": {"type": "number", "is_positive": True, "max": 1},
        "frac": {"type": "number", "is_positive": True, "max": 1,
                 "nullable": True},
        "n_qc": {"type": "integer", "nullable": True, "is_positive": True},
        "interpolator": {"type": "string", "allowed": ["splines", "linear"]},
        "process_qc": {"type": "boolean"},
        "threshold": {"type": "number", "min": 0},
        "corrector_classes": {"type": "list", "nullable": True,
                              "schema": {"type": "string"}},
        "process_classes": {"type": "list", "nullable": True,
                            "schema": {"type": "string"}}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_make_roi_params(n_spectra, params):
    schema = {"tolerance": {"type": "number", "is_positive": True},
              "max_missing": {"type": "integer", "min": 0},
              "targeted_mz": {"nullable": True, "check_with": is_all_positive},
              "start": {"type": "integer", "nullable": True, "min": 0,
                        "lower_than": "end"},
              "end": {"type": "integer", "nullable": True,
                      "max": n_spectra - 1},
              "multiple_match": {"allowed": ["closest", "reduce"]},
              "mz_reduce": {"any_of": [{"allowed": ["mean"]},
                                       {"check_with": is_callable}]},
              "sp_reduce": {"any_of": [{"allowed": ["mean", "sum"]},
                                       {"check_with": is_callable}]},
              "mode": {"allowed": ["hplc", "uplc"]}}
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_cwt_peak_picking_params(params):

    estimators = {"anyof": [{"type": "string",
                             "allowed": ["default", "cwt"]},
                            {"type": "dict",
                             "schema": get_peak_picking_estimators_schema()}
                            ]}

    schema = {"max_distance": {"type": "number", "is_positive": True},
              "min_length": {"type": "integer", "is_positive": True},
              "gap_threshold": {"type": "integer", "min": 0},
              "snr": {"type": "number", "is_positive": True},
              "min_width": {"type": "number", "is_positive": True,
                            "lower_than": "max_width"},
              "max_width": {"type": "number", "is_positive": True},
              "estimators": estimators
              }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def get_peak_picking_estimators_schema():
    schema = {"baseline": {"check_with": is_callable},
              "noise": {"check_with": is_callable},
              "width": {"check_with": is_callable},
              "area": {"check_with": is_callable},
              "loc": {"check_with": is_callable}}
    return schema


def validate_peak_picking_estimators(params):
    schema = get_peak_picking_estimators_schema()
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


# validator for make_chromatograms

def validate_make_chromatograms_params(n_spectra, params):
    schema = {"mz": {"empty": False},
              "window": {"type": "number", "is_positive": True},
              "accumulator": {"type": "string", "allowed": ["mean", "sum"]},
              "start": {"type": "integer", "lower_than": "end", "min": 0},
              "end": {"type": "integer", "max": n_spectra}
              }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_accumulate_spectra_params(n_spectra, params):
    schema = {"start": {"type": "integer", "min": 0, "lower_than": "end"},
              "end": {"type": "integer", "max": n_spectra - 1,
                      "lower_or_equal": "subtract_right"},
              "subtract_left": {"type": "integer", "lower_or_equal": "start",
                                "min": 0},
              "subtract_right": {"type": "integer", "max": n_spectra - 1},
              "kind": {"type": "string"}
              }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)
