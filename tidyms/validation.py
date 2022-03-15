"""
Validation functions for Filter, Pipelines and DataContainer
"""


import warnings
from functools import wraps
from inspect import getfullargspec
import cerberus
from typing import Callable
import numpy as np


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


def validate(params: dict, validator: cerberus.Validator) -> dict:
    """
    Function used to validate parameters.

    Parameters
    ----------
    params: dict
    validator: cerberus.Validator

    Returns
    -------
    dict: Validated and normalized parameters

    Raises
    ------
    ValueError: if any of the parameters are invalid.
    """
    normalized = validator.normalized(params)
    if not validator.validate(normalized):
        msg = ""
        for field, e_msgs in validator.errors.items():
            for e_msg in e_msgs:
                msg += "{}: {}\n".format(field, e_msg)
        raise ValueError(msg)
    return normalized


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
    # check that all elements are non-negative
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
                            sample_info):
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
        "robust": {"type": "boolean"},
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


def validate_dilution_filter_params(params):
    schema = {
        "min_corr": {"type": "number", "min": 0, "max": 1},
        "plim": {"type": "number", "min": 0, "max": 1},
        "mode": {"allowed": ["ols", "spearman"]}
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
                            "schema": {"type": "string"}},
        "method": {"allowed": ["additive", "multiplicative"], "type": "string"}
    }
    validator = ValidatorWithLowerThan(schema)
    validate(params, validator)


def validate_descriptors(params):
    schema = {"type": "dict", "keysrules": {"type": "string"},
              "valuesrules": {"check_with": is_callable}}
    if params is not None:
        validator = ValidatorWithLowerThan(schema)
        validate(params, validator)


def validate_filters(params):
    schema = {"type": "dict", "keysrules": {"type": "string"},
              "valuesrules": {"type": "list",
                              "items": [{'type': 'number', "nullable": True},
                                        {'type': 'number', "nullable": True}]
                              }
              }
    if params is not None:
        validator = ValidatorWithLowerThan(schema)
        validate(params, validator)


def validate_detect_peaks_params(params):
    noise_schema = {"min_slice_size": {"is_positive": True, "type": "integer"},
                    "n_slices": {"is_positive": True, "type": "integer"}
                    }
    baseline_schema = {"min_proba": {"is_positive": True, "max": 1.0,
                                     "type": "number"}
                       }
    schema = \
        {"noise_params": {"type": "dict",
                          "schema": noise_schema,
                          "nullable": True
                          },
         "baseline_params": {"type": "dict",
                             "schema": baseline_schema,
                             "nullable": True
                             },
         "smoothing_strength": {"type": "number",
                                "nullable": True,
                                "is_positive": True}
         }
    if params is not None:
        validator = ValidatorWithLowerThan(schema)
        validate(params, validator)


def spectra_iterator_schema(ms_data):
    n_spectra = ms_data.get_n_spectra()
    schema = {
        "ms_level": {
            "type": "integer",
            "min": 1,
        },
        "start": {
            "type": "integer",
            "min": 0,
            "lower_than": "end",
        },
        "end": {
            "type": "integer",
            "max": n_spectra,
            "default": n_spectra
        },
        "start_time": {
            "type": "number",
            "min": 0.0,
            "lower_than": "end_time",
        },
        "end_time": {
            "type": "number",
            "nullable": True,
        }
    }

    defaults = spectra_iterator_defaults(ms_data)
    set_defaults(schema, defaults)
    return schema


def spectra_iterator_defaults(ms_data):
    return dict()


def make_chromatogram_schema(ms_data) -> dict:

    n_spectra = ms_data.get_n_spectra()

    schema = {
        "mz": {
            "empty": False,
            "coerce": np.array,
        },
        "window": {
            "type": "number",
            "is_positive": True,
            "nullable": True
        },
        "accumulator": {
            "type": "string",
            "allowed": ["mean", "sum"]
        },
        "ms_level": {
            "type": "integer",
            "min": 1,
        },
        "start": {
            "type": "integer",
            "min": 0,
            "lower_than": "end",
        },
        "end": {
            "type": "integer",
            "max": n_spectra,
            "nullable": True,
            "default": n_spectra
        },
        "start_time": {
            "type": "number",
            "min": 0.0,
            "lower_than": "end_time",
        },
        "end_time": {
            "type": "number",
            "nullable": True,
        }
    }

    defaults = make_chromatogram_defaults(ms_data)
    set_defaults(schema, defaults)
    return schema


def make_chromatogram_defaults(ms_data):
    instrument = ms_data.instrument

    if instrument == "qtof":
        window = 0.05
    else:   # orbitrap
        window = 0.01

    defaults = {
        "window": {"default": window}
    }
    return defaults


def make_roi_schema(ms_data):
    n_spectra = ms_data.get_n_spectra()
    schema = {
        "tolerance": {
            "type": "number",
            "is_positive": True,
        },
        "max_missing": {
            "type": "integer",
            "min": 0,
        },
        "targeted_mz": {
            "nullable": True,
            "check_with": is_all_positive
        },
        "multiple_match": {
            "allowed": ["closest", "reduce"]
        },
        "mz_reduce": {
            "anyof": [
                {"allowed": ["mean"]},
                {"check_with": is_callable}
            ]
        },
        "sp_reduce": {
            "anyof": [
                {"allowed": ["mean", "sum"]},
                {"check_with": is_callable}
            ]
        },
        "min_intensity": {
            "type": "number",
            "min": 0.0,
            "nullable": True
        },
        "min_length": {
            "type": "integer",
            "min": 1,
            "nullable": True,
        },
        "pad": {
            "type": "integer",
            "min": 0,
            "nullable": True,
        },
        "ms_level": {
            "type": "integer",
            "min": 1,
        },
        "start": {
            "type": "integer",
            "min": 0,
            "lower_than": "end",
        },
        "end": {
            "type": "integer",
            "max": n_spectra,
            "nullable": True,
            "default": n_spectra
        },
        "start_time": {
            "type": "number",
            "min": 0.0,
            "lower_than": "end_time",
        },
        "end_time": {
            "type": "number",
            "nullable": True,
        },
        "min_snr": {
            "type": "number",
            "is_positive": True,
        },
        "min_distance": {
            "type": "number",
            "is_positive": True,
        }
    }
    defaults = make_roi_defaults(ms_data)
    set_defaults(schema, defaults)
    return schema


def make_roi_defaults(ms_data):
    separation = ms_data.separation
    instrument = ms_data.instrument

    if instrument == "qtof":
        tolerance = 0.01
    else:   # orbitrap
        tolerance = 0.005

    if separation == "uplc":
        min_length = 10
    else:   # hplc
        min_length = 20

    defaults = {
        "pad": {
            "default": 2
        },
        "max_missing": {
            "default": 1
        },
        "min_length": {
            "default": min_length
        },
        "tolerance": {
            "default": tolerance
        },
    }

    return defaults


def accumulate_spectra_schema(ms_data):

    n_spectra = ms_data.get_n_spectra()

    schema = {
        "start": {
            "type": "integer",
            "min": 0,
            "lower_than": "end"
        },
        "end": {
            "type": "integer",
            "max": n_spectra,
            "lower_or_equal": "subtract_right"
        },
        "subtract_left": {
            "type": "integer",
            "lower_or_equal": "start",
            "min": 0,
            "nullable": True,
            "default_setter": lambda doc: doc["start"]
        },
        "subtract_right": {
            "type": "integer",
            "max": n_spectra,
            "nullable": True,
            "default_setter": lambda doc: doc["end"]
        },
        "kind": {
            "type": "string"
        },
        "ms_level": {
            "type": "integer",
            "min": 1
        }
    }
    # no default functions is necessary, defaults can be obtained from
    # start and end
    return schema


def set_defaults(schema: dict, defaults: dict):
    for k, v in schema.items():
        if k in defaults:
            if "anyof" in v:
                # TODO: add code for anyof case
                pass
            else:
                v.update(defaults[k])


def validated_ms_data(schema_getter: Callable):
    """
    Validates input for MSData methods and set default values based on
    the `instrument` and `separation` attributes.

    Parameters
    ----------
    schema_getter : callable
        Function that returns the schema used for validation. The function must
        take the MSData object as the only argument.

    Returns
    -------

    """

    def validator_wrapper(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # build a dictionary with input
            func_argspec = getfullargspec(func)
            func_argnames = func_argspec.args[1:]    # we don't need self
            params = dict(zip(func_argnames, args[1:]))
            params.update(kwargs)
            ms_data = args[0]
            # get schema and defaults
            schema = schema_getter(ms_data)
            # validate input
            validator = ValidatorWithLowerThan(schema)
            validated = validate(params, validator)
            return func(ms_data, **validated)
        return func_wrapper
    return validator_wrapper
