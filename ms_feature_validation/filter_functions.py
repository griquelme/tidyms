"""
Functions to correct and filter data matrix from LC-MS Metabolomics data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Callable, Union, Optional


def input_na(df: pd.DataFrame, classes: pd.Series, mode: str) -> pd.DataFrame:
    """
    Fill missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
    classes: ps.Series
    mode : {'zero', 'mean', 'min'}
    
    Returns
    -------
    filled : pd.DataFrame    
    """
    if mode == "zero":
        return df.fillna(0)
    elif mode == "mean":
        return (df.groupby(classes)
                .apply(lambda x: x.fillna(x.mean()))
                .droplevel(0))
    elif mode == "min":
        return (df.groupby(classes)
                .apply(lambda x: x.fillna(x.min()))
                .droplevel(0))
    else:
        msg = "mode should be `zero`, `mean` or `min`"
        raise ValueError(msg)


def replicate_averager(data: pd.DataFrame, sample_id: pd.Series,
                       classes: pd.Series,
                       process_classes: List[str]) -> pd.DataFrame:
    """
    Group samples by id and computes the average.

    Parameters
    ----------
    data: pd.DataFrame
    sample_id: pd.Series
    classes: pd.Series
    process_classes: list[str]
    Returns
    -------
    pd.DataFrame
    """
    include_samples = classes[classes.isin(process_classes)].index
    exclude_samples = classes[~classes.isin(process_classes)].index
    mapper = sample_id[include_samples].drop_duplicates()
    mapper = pd.Series(data=mapper.index, index=mapper.values)
    included_data = data.loc[include_samples, :]
    excluded_data = data.loc[exclude_samples, :]
    averaged_data = (included_data.groupby(sample_id[include_samples])
                     .mean())
    averaged_data.index = averaged_data.index.map(mapper)
    result = pd.concat((averaged_data, excluded_data)).sort_index()
    return result


def blank_correction(df: pd.DataFrame, classes: pd.Series,
                     corrector_classes: List[str], process_classes: List[str],
                     mode: Union[str, Callable] = "mean") -> pd.DataFrame:
    """
    Correct samples using blanks.

    Parameters
    ----------
    df: pandas.DataFrame
        Data to correct.
    classes: pandas.Series
        Samples class labels.
    corrector_classes: list[str]
        Classes to be used as blanks.
    process_classes: list[str]
        Classes to be used as samples
    mode: {'mean', 'max', 'lod', 'loq'} or function

    Returns
    -------
    corrected : pandas.DataFrame
        Data with applied correction

    Notes
    -----

    Blank correction is applied for each feature in the following way:

    .. math:: X_{corrected} = X_{uncorrected} - blank_relation * mode(X_{blank})
    """
    corrector = {"max": lambda x: x.max(),
                 "mean": lambda x: x.mean(),
                 "lod": lambda x: x.mean() + 3 * x.std(),
                 "loq": lambda x: x.mean() + 10 * x.std()}
    if hasattr(mode, "__call__"):
        corrector = mode
    else:
        corrector = corrector[mode]
    samples = df[classes.isin(process_classes)]
    blanks = df[classes.isin(corrector_classes)]
    correction = corrector(blanks)
    corrected = samples - correction
    corrected[corrected < 0] = 0
    df[classes.isin(process_classes)] = corrected
    return df


# def prevalence_filter(data, classes, process_classes,
#                       lb, ub, intraclass, threshold):
#     """
#     Return columns with relative counts outside the [lb, ub] interval.
#
#     Parameters
#     ----------
#     data : pandas.DataFrame
#     classes : pandas.Series
#         sample class labels
#     process_classes : list[str]
#         classes included in the filter
#     lb : float.
#         Relative lower bound of prevalence.
#     ub : float.
#         Relative upper bound of prevalence.
#     intraclass : bool
#         if True computes prevalence for each class separately and return
#         columns
#         that are outside bounds in all of `include_classes`.
#     threshold : float
#         minimum value to define prevalence
#     Returns
#     -------
#     outside_bounds_columns: pandas.Index
#     """
#     # selects process_classes
#     classes = classes[classes.isin(process_classes)]
#     data = data.loc[classes.index, :]
#     # pipeline for prevalence filter
#     is_outside_bounds = ((data > threshold)
#                          .pipe(grouper(classes, intraclass))
#                          .sum()
#                          .pipe(normalizer(classes, intraclass))
#                          .pipe(bounds_checker(lb, ub, intraclass)))
#     outside_bounds_columns = is_outside_bounds[is_outside_bounds].index
#     return outside_bounds_columns
#
#
# def variation_filter(data, classes, process_classes, lb, ub,
#                      intraclass, robust):
#     """
#     Return columns with variation outside the [lb, ub] interval.
#
#     Parameters
#     ----------
#     data : pandas.DataFrame
#     classes : pandas.Series
#         class labels of samples.
#     process_classes : list[str]
#         classes included in the filter
#     lb : float
#         lower bound of variation
#     ub : float
#         upper bound of variation
#     intraclass : bool
#          if True computes prevalence for each class separately and return
#          columns that are outside bounds in all of `include_classes`.
#     robust : bool
#         if True uses iqr as a metric of variation. if False uses cv
#     Returns
#     -------
#     remove_features : pandas.Index
#     """
#     # selects process_classes
#     classes = classes[classes.isin(process_classes)]
#     data = data.loc[classes.index, :]
#     # pipeline for variation filter
#     is_outside_bound = (data.pipe(grouper(classes, intraclass))
#                         .pipe(variation(robust))
#                         .pipe(bounds_checker(lb, ub, intraclass)))
#     outside_bounds_columns = is_outside_bound[is_outside_bound].index
#     return outside_bounds_columns
#
#
# def normalizer(classes, intraclass):
#     """
#     function to normalize data using samples count or classes count.
#
#     Parameters
#     ----------
#     classes : pd.Series
#         Class label of samples
#     intraclass : bool
#         wether to normalize using total number of samples, or class count.
#
#     Returns
#     -------
#     normalizer : function
#     """
#     class_counts = classes.value_counts()
#     n = class_counts.sum()
#     return lambda x: x.divide(class_counts, axis=0) if intraclass else x / n
#
#
# def grouper(classes, intraclass):
#     return lambda x: x.groupby(classes) if intraclass else x
#
#
# def bounds_checker(lb, ub, intraclass):
#     if intraclass:
#         return lambda x: ((x < lb) | (x > ub)).all()
#     else:
#         return lambda x: ((x < lb) | (x > ub))
#
#
# def cv(df):
#     res = df.std() / df.mean()
#     res = res.fillna(0)
#     return res
#
#
# def iqr(df):
#     res = (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)
#     res = res.fillna(0)
#     return res
#
#
# def variation(robust):
#     if robust:
#         return iqr
#     else:
#         return cv


def cspline_correction(x, y, xq, yq):
    sp = CubicSpline(x, y)
    y_coeff = sp(xq)
    y_corrected = (yq / y_coeff)
    y_corrected *= yq.mean() / y_corrected.mean()   # scaling to yq mean
    return y_corrected


def loess_correction(x, y, xq, yq, **kwargs):
    y_loess = lowess(y, x, return_sorted=False, **kwargs)
    y_corrected = cspline_correction(x, y_loess, xq, yq)
    return y_corrected


def batch_correction(df: pd.DataFrame, run_order: pd.Series, classes: pd.Series,
                     corrector_classes: List[str], process_classes: List[str],
                     mode: str, **kwargs) -> pd.DataFrame:
    """
    Correct instrument response drift using LOESS regression [1].

    Parameters
    ----------
    df : pandas.DataFrame
    run_order : pandas.Series
        run order of samples
    classes : pandas.Series
        class label for samples
    corrector_classes : str
        label of corrector class
    process_classes: list[str]
        samples to correct
    mode: {'loess', 'splines'}
    kwargs: optional arguments to pass to loess corrector.

    Returns
    -------
    corrected: pandas.DataFrame

    References.
    -----
    .. [1] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography coupled
    to mass spectrometry", Nature Protocols volume 6, pages 1060–1083 (2011).

    """
    corrector = {"loess": loess_correction, "splines": cspline_correction}
    corrector = corrector[mode]
    corrector_class_mask = classes.isin(corrector_classes)
    corrector_run = run_order[corrector_class_mask]
    sample_classes_mask = classes.isin(process_classes)
    sample_run = run_order[sample_classes_mask]
    corrected = df.apply(lambda x: corrector(corrector_run,
                                             x[corrector_class_mask],
                                             sample_run,
                                             x[sample_classes_mask],
                                             **kwargs))
    return corrected


def _remove_invalid_features(corrector_df: pd.DataFrame,
                             process_df: pd.DataFrame,
                             n_min_samples: Optional[int] = 4) -> pd.Index:
    """
    Remove features that can't be corrected in a batch correction. Features that
    can't be corrected are those that have samples with an order that  cannot
    be interpolated using the corrector samples.

    Parameters
    ----------
    corrector_order: pd.Series
        Run order of corrector samples
    corrector_df: pd.DataFrame
        Data matrix of corrector samples
    process_order: pd.Series
        run order of process samples
    process_df: pd.DataFrame
        Data matrix of corrector samples.
    n_min_samples: int
        Mininum number of samples detected in corrector. Features with a lower
        number of samples are removed.

    Returns
    -------
    Index with valid features.
    """

    min_corr = corrector_df.idxmin()
    max_corr = corrector_df.idxmax()
    min_process = process_df.idxmin()
    max_process = process_df.idxmax()
    n_samples = (process_df > 0).count()
    valid_order = ((min_corr < min_process) &
                   (max_corr > max_process) &
                   (n_samples >= n_min_samples))
    valid_order = valid_order.index
    return valid_order


def coov_loess(x_order: pd.Series, x: pd.Series,
               frac: Optional[float] = None) -> tuple:
    """
    Helper function for batch_correction. Computes loess correction with LOOCV.

    Parameters
    ----------
    x_order: pd.Series
        Run order
    x: pd.Series
        Feature intensities
    frac: float, optional
        fraction of sample to use in LOESS correction. If None, determines the
        best value using LOOCV.
    Returns
    -------
    frac: float.
        Best frac found by LOOCV
    corrected: pd.Series
        LOESS corrected data
    """
    if frac is None:
        # valid frac values, from 4/N to 1 samples.
        frac_list = [x / x_order.size for x in range(4, x_order.size + 1)]
        rms = np.inf
        best_frac = 0
        for frac in frac_list:
            curr_rms = 0
            for x_loocv in x_order.index[1:-1]:
                y_temp = x.drop(x_loocv)
                x_temp = x_order.drop(x_loocv)
                y_loess = lowess(y_temp, x_temp, return_sorted=False, frac=frac)
                interp = CubicSpline(x_temp, y_loess)
                curr_rms += (x_order[x_loocv] - interp(x_loocv)) ** 2
            if rms > curr_rms:
                best_frac = frac
                rms = curr_rms
        frac = best_frac

    return  frac, lowess(x, x_order, return_sorted=False, frac=frac)


def batch_correction2(df: pd.DataFrame, run_order: pd.Series,
                      classes: pd.Series, corrector_classes: List[str],
                      process_classes: List[str],
                      mode: str, **kwargs) -> pd.DataFrame:
    """
    Correct instrument response drift using LOESS regression [1, 2].

    Parameters
    ----------
    df : pandas.DataFrame
    run_order : pandas.Series
        run order of samples
    classes : pandas.Series
        class label for samples
    corrector_classes : str
        label of corrector class
    process_classes: list[str]
        samples to correct
    mode: {'loess', 'splines'}
    kwargs: optional arguments to pass to loess corrector.

    Returns
    -------
    corrected: pandas.DataFrame

    References.
    -----
    .. [1] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
    serum and plasma using gas chromatography and liquid chromatography coupled
    to mass spectrometry", Nature Protocols volume 6, pages 1060–1083 (2011).
    .. [2] D Broadhurst *et al*, "Guidelines and considerations for the use of
    system suitability and quality control samples in mass spectrometry assays
    applied in untargeted clinical metabolomic studies.", Metabolomics,
    2018;14(6):72. doi: 10.1007/s11306-018-1367-3

    Notes
    -----
    The correction is applied as described by Broadhurst in [2]. Using QC
    samples a correction is generated for each feature in the following way:
    The signal of a Quality control can be described in terms of three
    components: a mean value, a systematic bias f and error.

    .. math::
        m_{i} = \bar{m_{i}} + f(t) + \epsilon

    f(t) is estimated after mean substraction using Locally weighted scatterplot
    smoothing (LOESS). The optimal fraction of samples for each local
    regression is found using LOOCV.
    """

    # splitting data into corrector and process
    df = df.sort_values(run_order).set_index(run_order)
    corrector_df = df[classes.isin(corrector_classes)]
    # corrector_order = run_order[classes.isin(corrector_classes)]
    process_df = df[classes.isin(process_classes)]
    # process_order = run_order[classes.isin(process_classes)]

    # removing features that cannot be interpolated
    valid_samples = _remove_invalid_features(corrector_df, process_df)
    corrector_df = corrector_df.loc[:, valid_samples]
    process_df = process_df.loc[:, valid_samples]

    # LOESS correction for corrector samples
    loess_func = lambda x: coov_loess(corrector_order, x)
    corrector_df -= corrector_df.mean()
    corrector_df =  corrector_df.apply(loess_func).set_index(corrector_order)

    # interpolation of f(t) to correct process samples
    f = pd.DataFrame(data=(np.ones_like(process_df.values) * np.nan),
                     index=process_order)
    f = f.fillna()


    # if mode == "loess":
        # corrector_df.apply(lambda x: lowess(order))


def interbatch_correction(df: pd.DataFrame, batch: pd.Series,
                          run_order: pd.Series, classes: pd.Series,
                          corrector_classes: List[str],
                          process_classes: List[str],
                          mode: str, **kwargs) -> pd.DataFrame:
    """
    Apply batch correction to several batches. Interbatch correction is achieved
    using a scaling factor obtained from the mean values in corrector samples.

    Parameters
    ----------
    df : pd.DataFrame
    batch : pd.Series
        batch number for a given sample
    run_order : pd.Series
    classes : pd.Series
        Class labels for samples
    corrector_classes : list[str]
        class used to correct samples
    process_classes : list[str]
        class labels to correct.
    mode : {"loess", "splines"}
    kwargs : optional arguments to pass to loess corrector

    Returns
    -------
    corrected: pandas.DataFrame
    """
    corrector_samples_mask = classes.isin([corrector_classes])
    scaling_factor = df[corrector_samples_mask].mean()

    def corrector_helper(x: pd.DataFrame):
        corrected_batch = batch_correction(x, run_order.loc[x.index], classes,
                                           corrector_classes, process_classes,
                                           mode, **kwargs)
        corrected_batch = corrected_batch.divide(corrected_batch.mean())
        return corrected_batch

    corrected = (df.groupby(batch)
                 .apply(corrector_helper)
                 * scaling_factor)
    corrected.index = corrected.index.droplevel("batch")
    return corrected


def get_outside_bounds_index(data: Union[pd.Series, pd.DataFrame], lb: float,
                             ub: float) -> pd.Index:
    """
    return index of columns with values outside bounds.
    Parameters
    ----------
    data: pd.Series or pd.DataFrame
    lb: float
        lower bound
    ub: float
        upper bound
    Returns
    -------

    """
    result = ((data < lb) | (data > ub))
    if isinstance(data, pd.DataFrame):
        result = result.all()
    if result.empty:
        return pd.Index([])
    else:
        return result[result].index
