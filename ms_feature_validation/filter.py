import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess


def blank_correction(data, blanks, mode, blank_relation):
    """
    Correct data using blanks.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to correct.
    blanks : pandas.DataFrame
        Blank Samples.
    mode : {'mean', 'max'}, optional
    blank_relation: int.
        Number of times to substract the correction.

    Returns
    -------
    corrected : pandas.DataFrame
        Data with applied correction

    Notes
    -----

    Blank correction is applied for each feature in the following way:

    .. math:: X_{corrected} = X_{uncorrected} - blank_relation * mode(X_{blank})
    """
    corrector = {"max": lambda x: x.max(axis=0),
                 "mean": lambda x: x.mean(axis=0)}
    correction = corrector[mode](blanks) * blank_relation
    corrected = data - correction
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


def loess_correction(corrector_order, corrector, samples_order, samples):
    """
    Correct instrument response drift using LOESS regression [1].

    Parameters
    ----------
    corrector_order : Iterable[int]
        Order of corrector samples.
    corrector: pandas.DataFrame
    samples_order: Iterable[int]
        Order of samples
    samples: pandas.DataFrame

    Returns
    -------
    corrected: pandas.DataFrame

    References.
    -----
    .. [1] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography coupled
    to mass spectrometry", Nature Protocols volume 6, pages 1060–1083 (2011).

    """
    corrected = pd.DataFrame()
    for features in samples.columns:
        corrector_loess = lowess(corrector[features], corrector_order,
                                 return_sorted=False)
        spline = CubicSpline(corrector_order,
                             corrector_loess / corrector[features])
        correction_factor = spline(samples_order)
        corrected[features] = correction_factor * samples[features]
    return corrected

