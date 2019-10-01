"""
Validation functions for Filter, Pipelines and DataContainer
"""


import warnings
from os import getcwd
from os.path import isdir
from os.path import join


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


def validate_feature_definitions(df):
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


def validate_sample_information(df):
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
    # if "batch" not in df.columns:
    #     warnings.warn("No batch information available. Batch and order "
    #                   "information is required for batch and interbatch "
    #                   "correction.")
    # if "order" not in df.columns:
    #     warnings.warn("No order information available. Batch and order "
    #                   "information is required for batch and interbatch "
    #                   "correction.")


def validate_data_container(data_matrix, feature_definitions,
                            sample_info, data_path):
    validate_data_matrix(data_matrix)
    validate_feature_definitions(feature_definitions)
    validate_sample_information(sample_info)
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
    if not all([isinstance(x, str) for x in params["corrector_classes"]]):
        msg = "`corrector_classes` should be a list of strings"
        raise TypeError(msg)
    if not all([isinstance(x, str) for x in params["process_classes"]]):
        msg = "`process_classes` should be a list of strings"
        raise TypeError(msg)
    blank_relation_check = (isinstance(params["lb"], (float, int)) and
                            (params["lb"] <= 1) and
                            (params["lb"] >= 0) and
                            (params["lb"] <= params["ub"]))
    if not blank_relation_check:
        msg = "`blank_relation` should be an scalar between 0 and 1."
        raise ValueError(msg)
    if not params["mode"] in ["mean", "max"]:
        msg = "`mode` should be `max` or `mean`."
        raise ValueError(msg)


