"""
Functions to correct and filter data matrix from LC-MS Metabolomics data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Callable, Union, Optional, Tuple


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


def average_replicates(data: pd.DataFrame, sample_id: pd.Series,
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


def correct_blanks(df: pd.DataFrame, classes: pd.Series,
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
#
#
# def cspline_correction(x, y, xq, yq):
#     sp = CubicSpline(x, y)
#     y_coeff = sp(xq)
#     y_corrected = (yq / y_coeff)
#     y_corrected *= yq.mean() / y_corrected.mean()   # scaling to yq mean
#     return y_corrected
#
#
# def loess_correction(x, y, xq, yq, **kwargs):
#     y_loess = lowess(y, x, return_sorted=False, **kwargs)
#     y_corrected = cspline_correction(x, y_loess, xq, yq)
#     return y_corrected
#
#
# def batch_correction(df: pd.DataFrame, run_order: pd.Series, classes: pd.Series,
#                      corrector_classes: List[str], process_classes: List[str],
#                      mode: str, **kwargs) -> pd.DataFrame:
#     """
#     Correct instrument response drift using LOESS regression [1].
#
#     Parameters
#     ----------
#     df : pandas.DataFrame
#     run_order : pandas.Series
#         run order of samples
#     classes : pandas.Series
#         class label for samples
#     corrector_classes : str
#         label of corrector class
#     process_classes: list[str]
#         samples to correct
#     mode: {'loess', 'splines'}
#     kwargs: optional arguments to pass to loess corrector.
#
#     Returns
#     -------
#     corrected: pandas.DataFrame
#
#     References.
#     -----
#     .. [1] W B Dunn *et al*, "Procedures for large-scale metabolic profiling of
#     serum and plasma using gas chromatography and liquid chromatography coupled
#     to mass spectrometry", Nature Protocols volume 6, pages 1060–1083 (2011).
#
#     """
#     corrector = {"loess": loess_correction, "splines": cspline_correction}
#     corrector = corrector[mode]
#     corrector_class_mask = classes.isin(corrector_classes)
#     corrector_run = run_order[corrector_class_mask]
#     sample_classes_mask = classes.isin(process_classes)
#     sample_run = run_order[sample_classes_mask]
#     corrected = df.apply(lambda x: corrector(corrector_run,
#                                              x[corrector_class_mask],
#                                              sample_run,
#                                              x[sample_classes_mask],
#                                              **kwargs))
#     return corrected


def _select_valid_features(corrector_df: pd.DataFrame,
                           process_df: pd.DataFrame,
                           n_min_samples: Optional[int] = 4) -> pd.Index:
    """
    Remove features that can't be corrected in a batch correction. Features that
    can't be corrected are those that have samples with an order that  cannot
    be interpolated using the corrector samples.

    Parameters
    ----------
    corrector_df: pd.DataFrame
        Data matrix for corrector samples, with run order as indices.
    process_df: pd.DataFrame
        Data matrix for process samples, with run order as indices.
    n_min_samples: int
        Minimum number of samples detected in corrector. Features with a lower
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


def _coov_loess(x: pd.Series,
                frac: Optional[float] = None) -> tuple:
    """
    Helper function for batch_correction. Computes loess correction with LOOCV.

    Parameters
    ----------
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
        # valid frac values, from 4/N to 1/N, where N is the number of corrector
        # samples.
        frac_list = [k / x.size for k in range(4, x.size + 1)]
        rms = np.inf    # initial value for root mean square error
        best_frac = 1
        for frac in frac_list:
            curr_rms = 0
            for x_loocv in x.index[1:-1]:
                y_temp = x.drop(x_loocv)
                x_temp = y_temp.index
                y_loess = lowess(y_temp, x_temp, return_sorted=False, frac=frac)
                interp = CubicSpline(x_temp, y_loess)
                curr_rms += (x[x_loocv] - interp(x_loocv)) ** 2
            if rms > curr_rms:
                best_frac = frac
                rms = curr_rms
        frac = best_frac
    return lowess(x.values, x.index, return_sorted=False, frac=frac)


def _generate_batches(df: pd.DataFrame, run_order: pd.Series, batch: pd.Series,
                      classes: pd.Series, corrector_classes: List[str],
                      process_classes: List[str]):
    batch_order = (pd.concat((batch, run_order), axis=1)
                   .sort_values(["batch", "order"]))
    grouped = batch_order.groupby("batch")
    for n_batch, group in grouped:
        df_batch = df.loc[group.index, :]
        classes_batch = classes[group.index]
        process_df = df_batch.loc[classes_batch.isin(process_classes), :]
        corrector_df = df_batch.loc[classes_batch.isin(corrector_classes), :]
        process_order = run_order[process_df.index]
        corrector_order = run_order[corrector_df.index]
        batch_order = (run_order[corrector_df.index.union(process_df.index)]
                       .sort_values())
        corrector_df = corrector_df.set_index(corrector_order).sort_index()
        process_df = process_df.set_index(process_order).sort_index()
        yield corrector_df, process_df, batch_order


def batch_correction2(df: pd.DataFrame, run_order: pd.Series, batch: pd.Series,
                      classes: pd.Series, corrector_classes: List[str],
                      process_classes: List[str],
                      min_prevalence: Union[float, int] = 1,
                      frac: Optional[float] = None
                      ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Correct instrument response drift using LOESS regression [1, 2].

    Parameters
    ----------
    df : pandas.DataFrame
    run_order : pandas.Series
        run order of samples
    batch: pandas.Series
        batch number of samples
    classes : pandas.Series
        class label for samples
    corrector_classes : str
        label of corrector class
    process_classes: list[str]
        samples to correct
    min_prevalence: Union[float, int]
        Minimum number of batches where a feature is corrected. Features
        below this level are selected in invalid_features.
    frac: float, optional.
        fraction of samples used to build local regression.

    Returns
    -------
    corrected: pandas.DataFrame
        corrected data
    invalid_features: pandas.Series
        features that where corrected in a number of batches lower than
        min_batch_prevalence.

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

    f(t) is estimated after mean subtraction using Locally weighted scatterplot
    smoothing (LOESS). The optimal fraction of samples for each local
    regression is found using LOOCV.
    """

    # splitting data into corrector and process
    # df = df.sort_values(run_order)
    grand_mean = 0
    n_corrector_samples = 0
    batch_prevalence = pd.Series(data=0, index=df.columns)
    batches = _generate_batches(df, run_order, batch, classes,
                                corrector_classes, process_classes)
    n_batch = 0
    corrected = df.copy()
    for corrector_df, process_df, batch_order in batches:
        n_batch += 1
        n_corrector_samples += corrector_df.shape[0]
        valid_features = _select_valid_features(corrector_df, process_df)
        batch_prevalence[valid_features] += 1
        corrector_df = corrector_df.loc[:, valid_features]

        # LOESS correction for corrector samples
        def loess_function(x):
            return _coov_loess(x, frac=frac)

        f = (corrector_df - corrector_df.mean()).apply(loess_function)
        batch_mean = (corrector_df - f).mean()
        grand_mean += batch_mean * corrector_df.shape[0]

        # interpolation of f(t) to correct process samples
        n_qc = corrector_df.shape[0]
        degree = min(n_qc - 1, 3)

        f_interp = pd.DataFrame(data=np.nan,
                                index=batch_order,
                                columns=valid_features)
        f_interp.loc[corrector_df.index, :] = f
        f_interp = f_interp.interpolate(axis=0, method="spline", order=degree)
        f_interp += batch_mean
        f_interp.loc[f_interp.index.difference(process_df.index), :] = 0

        # revert index to sample index
        # ind = pd.Series(data=batch_order.index, index=batch_order)
        f_interp.index = batch_order.index
        corrected.loc[f_interp.index, :] -= f_interp

    # normalizing data to global mean
    grand_mean /= n_corrector_samples
    corrected.loc[classes.isin(process_classes), :] += grand_mean

    # find features that are corrected in a low number of batches
    batch_prevalence /= n_batch
    invalid_features = batch_prevalence < min_prevalence
    invalid_features = invalid_features[invalid_features].index
    return corrected, invalid_features


# def interbatch_correction(df: pd.DataFrame, batch: pd.Series,
#                           run_order: pd.Series, classes: pd.Series,
#                           corrector_classes: List[str],
#                           process_classes: List[str],
#                           mode: str, **kwargs) -> pd.DataFrame:
#     """
#     Apply batch correction to several batches. Interbatch correction is achieved
#     using a scaling factor obtained from the mean values in corrector samples.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#     batch : pd.Series
#         batch number for a given sample
#     run_order : pd.Series
#     classes : pd.Series
#         Class labels for samples
#     corrector_classes : list[str]
#         class used to correct samples
#     process_classes : list[str]
#         class labels to correct.
#     mode : {"loess", "splines"}
#     kwargs : optional arguments to pass to loess corrector
#
#     Returns
#     -------
#     corrected: pandas.DataFrame
#     """
#     corrector_samples_mask = classes.isin([corrector_classes])
#     scaling_factor = df[corrector_samples_mask].mean()
#
#     def corrector_helper(x: pd.DataFrame):
#         corrected_batch = batch_correction(x, run_order.loc[x.index], classes,
#                                            corrector_classes, process_classes,
#                                            mode, **kwargs)
#         corrected_batch = corrected_batch.divide(corrected_batch.mean())
#         return corrected_batch
#
#     corrected = (df.groupby(batch)
#                  .apply(corrector_helper)
#                  * scaling_factor)
#     corrected.index = corrected.index.droplevel("batch")
#     return corrected


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
