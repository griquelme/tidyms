import pandas as pd


def blank_correction(data, blanks, mode, blank_relation):
    corrector = {"max": lambda x: x.max(), "mean": lambda x: x.mean()}
    correction = corrector[mode](blanks) * blank_relation
    corrected = data.subtract(correction)
    corrected[corrected < 0] = 0
    return corrected


def prevalence_filter(data, lb, ub):
    """
    Return columns with relative counts outside the [lb, ub] interval.

    Parameters
    ----------
    data : pandas.DataFrame
    lb : float.
         Relative lower bound of prevalence.
    ub : float.
         Relative upper bound of prevalence.
    Returns
    -------
    outside_bounds_columns: pandas.Index
    """
    nlb, nub = lb * data.shape[0], ub * data.shape[0]
    column_counts = (data > 0).sum()
    bounds = (column_counts < nlb) | (column_counts > nub)
    outside_bounds_columns = column_counts[bounds].index
    return outside_bounds_columns


def variation_filter(data, lb, ub, robust):
    """
    Return columns with variation outside the [lb, ub] interval.
    Parameters
    ----------
    data : pandas.DataFrame
    lb : float
         lower bound of variation
    ub : float
         upper bound of variation
    robust : bool
             if True uses iqr as a metric of variation. if False uses cv
    Returns
    -------
    remove_features : pandas.Index
    """

    variation = iqr if robust else cv
    data_variation = variation(data)
    bounds = (data_variation < lb) | (data_variation > ub)
    outside_bounds_columns = data_variation[bounds].index
    return outside_bounds_columns


def iqr(df):
    return (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)


def cv(df):
    return df.std() / df.mean()
