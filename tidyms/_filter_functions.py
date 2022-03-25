"""
Functions to correct and filter data matrix from LC-MS Metabolomics data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Callable, Union, Optional
from .utils import mad
from ._names import *


# def input_na(df: pd.DataFrame, classes: pd.Series, mode: str) -> pd.DataFrame:
#     """
#     Fill missing values.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#     classes: ps.Series
#     mode : {'zero', 'mean', 'min'}
#
#     Returns
#     -------
#     filled : pd.DataFrame
#     """
#     if mode == "zero":
#         return df.fillna(0)
#     elif mode == "mean":
#         return (df.groupby(classes)
#                 .apply(lambda x: x.fillna(x.mean()))
#                 .droplevel(0))
#     elif mode == "min":
#         return (df.groupby(classes)
#                 .apply(lambda x: x.fillna(x.min()))
#                 .droplevel(0))
#     else:
#         msg = "mode should be `zero`, `mean` or `min`"
#         raise ValueError(msg)


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
                   factor: float = 1.0, robust: bool = True,
                   mode: Union[str, Callable] = "mean",
                   process_blanks: bool = False) -> pd.DataFrame:
    """
    Correct samples using blanks.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to correct.
    classes : pandas.Series
        Samples class labels.
    corrector_classes : list[str]
        Classes to be used as blanks.
    process_classes : list[str]
        Classes to be used as samples
    process_blanks : bool
        If True apply blank correction to corrector classes.
    factor : float
        factor used to convert low values to zero (see notes)
    mode : {'mean', 'max', 'lod', 'loq'} or function

    Returns
    -------
    corrected : pandas.DataFrame
        Data with applied correction

    """
    if robust:
        mean_func = lambda x: x.median()
        std_func = mad
    else:
        mean_func = lambda x: x.mean()
        std_func = lambda x: x.std()

    corrector = {"max": lambda x: x.max(),
                 "mean": lambda x: mean_func(x),
                 "lod": lambda x: mean_func(x) + 3 * std_func(x),
                 "loq": lambda x: mean_func(x) + 10 * std_func(x)}
    if hasattr(mode, "__call__"):
        corrector = mode
    else:
        corrector = corrector[mode]
    samples = df[classes.isin(process_classes)]
    blanks = df[classes.isin(corrector_classes)]

    correction = corrector(blanks)
    corrected = samples - correction
    corrected[(samples - factor * correction) < 0] = 0
    df[classes.isin(process_classes)] = corrected
    if process_blanks:
        corrected_blanks = blanks - correction
        corrected_blanks[(blanks - factor * correction) < 0] = 0
        df[classes.isin(corrector_classes)] = corrected_blanks
    return df


def _loocv_loess(x: pd.Series, y: pd.Series, interpolator: Callable,
                 frac: Optional[float] = None) -> tuple:
    """
    Helper function for batch_correction. Computes loess correction with LOOCV.

    Parameters
    ----------
    x: pd.Series
    y: pd.Series
    frac: float, optional
        fraction of sample to use in LOESS correction. If None, determines the
        best value using LOOCV.
    interpolator = callable
        interpolator function used to predict new values.
    Returns
    -------
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
            for loocv_index in x.index[1:-1]:
                y_temp = y.drop(loocv_index)
                x_temp = x.drop(loocv_index)
                y_loess = lowess(y_temp, x_temp, return_sorted=False, frac=frac)
                interp = interpolator(x_temp, y_loess)
                curr_rms += (y[loocv_index] - interp(x[loocv_index])) ** 2
            if rms > curr_rms:
                best_frac = frac
                rms = curr_rms
        frac = best_frac
    return lowess(y, x, return_sorted=False, frac=frac)


def _generate_batches(df: pd.DataFrame, run_order: pd.Series, batch: pd.Series,
                      classes: pd.Series, corrector_classes: List[str],
                      process_classes: List[str]):
    batch_order = (pd.concat((batch, run_order), axis=1)
                   .sort_values([_sample_batch, _sample_order]))
    grouped = batch_order.groupby(_sample_batch)
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


def batch_ext(order: pd.Series, batch: pd.Series, classes: pd.Series,
              class_list: List[str], ext: str) -> pd.Series:
    """
    get minimum/maximum order of samples of classes in class_list. Auxiliary
    function to be used with BatchChecker / FeatureCheckerBatchCorrection
    Parameters
    ----------
    order: pandas.Series
        run order
    batch: pandas.Series
        batch number
    classes: pandas.Series
        sample classes
    class_list: list[str]
        classes to be considered
    ext: {"min", "max"}
        Search for the min/max order in each batch.

    Returns
    -------
    pd.Series with the corresponding min/max order with batch as index.
    """
    func = {"min": lambda x: x.min(), "max": lambda x: x.max()}
    func = func[ext]

    ext_order = (order
                 .groupby([classes, batch])
                 .apply(func)
                 .reset_index()
                 .groupby(classes.name)
                 .filter(lambda x: x.name in class_list)
                 .groupby(batch.name)
                 .apply(func)[order.name])
    return ext_order


def check_qc_prevalence(data_matrix: pd.DataFrame,
                        batch: pd.Series, classes: pd.Series,
                        qc_classes: List[str], sample_classes: List[str],
                        threshold: float = 0,
                        min_qc_dr: float = 0.9) -> pd.Index:
    """
    Remove features with low detection rate in the QC samples. Also check that
    each feature is detected in the first and last block (this step is necessary
    interpolate the bias contribution to biological samples).

    Aux function to use in the BatchCorrector Pipeline.

    Parameters
    ----------
    data_matrix: DataFrame
    batch: Series
    classes: Series
    qc_classes: List[str]
    sample_classes: List[str]
    threshold: float
    min_qc_dr: float

    Returns
    -------
    index of invalid features
    """
    invalid_features = pd.Index([])
    for batch_number, batch_class in classes.groupby(batch):
        block_type, block_number = \
            make_sample_blocks(batch_class, qc_classes, sample_classes)
        qc_blocks = block_number[block_type == 0]
        block_prevalence = (data_matrix.loc[qc_blocks.index]
                            .groupby(qc_blocks)
                            .apply(lambda x: (x > threshold).any()))

        # check start block
        start_block_mask = block_prevalence.loc[qc_blocks.iloc[0]]
        tmp_rm = data_matrix.columns[~start_block_mask]
        invalid_features = invalid_features.union(tmp_rm)

        # check end block
        end_block_mask = block_prevalence.loc[qc_blocks.iloc[-1]]
        tmp_rm = data_matrix.columns[~end_block_mask]
        invalid_features = invalid_features.union(tmp_rm)

        # check qc prevalence
        n_blocks = qc_blocks.unique().size
        qc_prevalence = block_prevalence.sum() / n_blocks
        batch_min_qc_dr = max(4 / n_blocks, min_qc_dr)
        tmp_rm = data_matrix.columns[qc_prevalence < batch_min_qc_dr]
        invalid_features = invalid_features.union(tmp_rm)

    return invalid_features


def loess_interp(ft_data: pd.Series, order: pd.Series, qc_index: pd.Index,
                 sample_index: pd.Index, frac: float, interpolator: Callable,
                 n_qc: Optional[int] = None, method: str = "multiplicative"
                 ) -> pd.Series:
    """
    Applies LOESS-correction interpolation on a feature. Auxiliary function of
    batch_corrector_func

    Parameters
    ----------
    ft_data: pd.Series
        Feature intensity
    order: pd.Series
    qc_index: pd.Index
    sample_index: pd.Index
    frac: float
    interpolator: Callable
    n_qc: int, optional
    Number of QCs involved in mean calculation. If None, all QCs are involved.
    method : {"multiplicative", "additive"}
        Method used to model variation.
    Returns
    -------
    pd.Series
    """
    if n_qc is None:
        n_qc = qc_index.size

    qc_median = ft_data[qc_index[:n_qc]].median()
    # if there are several 0 values the median may be 0, this prevent against
    # such cases
    if np.isclose(qc_median, 0.0):
        qc_median = ft_data[qc_index[:n_qc]].mean()

    qc_loess = _loocv_loess(order[qc_index], ft_data[qc_index], interpolator,
                            frac=frac)
    interp = interpolator(order[qc_index], qc_loess)

    if method == "additive":
        bias = interp(order[sample_index]) - qc_median
        ft_data[sample_index] -= bias
    elif method == "multiplicative":
        qc_smooth = interp(order[sample_index])
        # smoothed values close to zero can become negative. As we are using
        # a multiplicative correction, this prevents changing signs while
        # applying the correction
        if qc_median <= 0.0:
            factor = 0 * qc_smooth
        else:
            qc_smooth[qc_smooth <= 0] = qc_median
            factor = qc_median / qc_smooth
        ft_data[sample_index] *= factor
    else:
        msg = "Valid methods are `additive` or `multiplicative`."
        raise ValueError(msg)
    return ft_data


def batch_corrector_func(df_batch: pd.DataFrame, order: pd.Series,
                         classes: pd.Series, frac: float,
                         interpolator: Callable, qc_classes: List[str],
                         sample_classes: List[str],
                         n_qc: Optional[int] = None,
                         method: str = "multiplicative") -> pd.DataFrame:
    """
    Applies LOESS correction - interpolation on a single batch. Auxiliary
    function of interbatch_correction.

    Parameters
    ----------
    df_batch: pandas.DataFrame
    order: pandas.Series
    classes: pandas.Series
    frac: float
    interpolator: Callable
    qc_classes: list[str]
    sample_classes: list[str]
    n_qc: int, optional
    Number of QCs involved in mean calculation. If None, all QCs are involved.
    method : {"multiplicative", "additive"}
        Method used to model variation.

    Returns
    -------
    pandas.DataFrame
    """
    qc_index = classes.isin(qc_classes)
    qc_index = qc_index[qc_index].index
    sample_index = classes.isin(sample_classes)
    sample_index = sample_index[sample_index].index
    df_batch.loc[sample_index, :] = \
        (df_batch.apply(loess_interp,
                        args=(order, qc_index, sample_index, frac,
                              interpolator),
                        n_qc=n_qc, method=method))
    return df_batch


def interbatch_correction(df: pd.DataFrame, order: pd.Series, batch: pd.Series,
                          classes: pd.Series, corrector_classes: List[str],
                          process_classes: List[str],
                          frac: Optional[float] = None,
                          interpolator: Optional[str] = "splines",
                          n_qc: Optional[int] = None,
                          process_qc: bool = True,
                          method: str = "multiplicative"
                          ) -> pd.DataFrame:
    r"""
    Correct instrument response drift using LOESS regression [1, 2]
    and center each batch to a common mean.

    Parameters
    ----------
    df : pandas.DataFrame
    order : pandas.Series
        run order of samples
    batch: pandas.Series
        batch number of samples
    classes : pandas.Series
        class label for samples
    corrector_classes : str
       label of corrector class
    process_classes: list[str]
       samples to correct
    frac: float, optional.
       fraction of samples used to build local regression.If None, finds the
       best value using LOOCV.
    interpolator: {"linear", "splines"}
        Type of interpolator to use.
    n_qc: int, optional
        Number of QCs involved in mean calculation. If None, all QCs are
        involved.
    process_qc : bool
        If True, applies correction to QC samples.
    method : {"multiplicative", "additive"}
        Method used to model variation.

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

    f(t) is estimated after mean subtraction using Locally weighted scatter plot
    smoothing (LOESS). The optimal fraction of samples for each local
    regression is found using LOOCV.

    Mean centering is performed subtracting a batch mean and adding a grand
    mean.
    """
    interp_func = {"splines": CubicSpline, "linear": interp1d}
    interp_func = interp_func[interpolator]

    if process_qc:
        # add QC classes to process classes
        process_classes = corrector_classes + process_classes
        process_classes = list(set(process_classes))

    def corrector_helper(df_group):
        return batch_corrector_func(df_group, order[df_group.index],
                                    classes[df_group.index], frac,
                                    interp_func, corrector_classes,
                                    process_classes, n_qc=n_qc,
                                    method=method)

    # intra batch correction
    corrected = df.groupby(batch).apply(corrector_helper)

    # inter batch mean alignment
    def batch_mean_func(df_group):
        batch_mean = (df_group[classes[df_group.index].isin(corrector_classes)]
                      .mean())
        is_process_sample = classes[df_group.index].isin(process_classes)
        if method == "additive":
            df_group[is_process_sample] -= batch_mean
        else:
            batch_mean[batch_mean <= 0] = np.nan
            df_group[is_process_sample] /= batch_mean
            df_group.fillna(0)
        return df_group

    global_median = corrected[classes.isin(corrector_classes)].median()
    corrected = corrected.groupby(batch).apply(batch_mean_func)
    process_mask = classes.isin(process_classes)
    if method == "additive":
        corrected.loc[process_mask, :] += global_median
    else:
        corrected.loc[process_mask, :] *= global_median
    corrected[corrected < 0] = 0

    return corrected


def make_sample_blocks(classes: pd.Series, corrector_classes: List[str],
                       process_classes: List[str]):
    """
    groups samples into blocks of consecutive samples of the same type
    aux function in BatchCorrector pipeline.

    each class is assigned to each one of three possible sample blocks:
    0 if the sample is mapped as QC, 1 if the sample is mapped as a
    sample, and 2 otherwise.

    Each block is assigned an unique number.
    """
    class_to_block_type = dict()
    for c in classes.unique():
        if c in corrector_classes:
            class_to_block_type[c] = 0
        elif c in process_classes:
            class_to_block_type[c] = 1
        else:
            class_to_block_type[c] = 2
    block_type = classes.map(class_to_block_type)
    block_number = (block_type.diff().fillna(0) != 0).cumsum()
    return block_type, block_number
