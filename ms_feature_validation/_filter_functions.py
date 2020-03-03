"""
Functions to correct and filter data matrix from LC-MS Metabolomics data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
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


def _coov_loess(x: pd.Series, y: pd.Series, interpolator: Callable,
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
    get minimum/maximum order of samples of classes in class_list. Auxiliar
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
        classes to be considererd
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


def check_qc_prevalence(data_matrix: pd.DataFrame, order: pd.Series,
                        batch: pd.Series, classes: pd.Series,
                        qc_classes: List[str], sample_classes: List[str],
                        threshold: float = 0, min_n_qc: int = 4) -> pd.Index:
    """
    Check prevalence in the QC samples. This step is necessary interpolate
    the bias contribution to biological samples.

    Parameters
    ----------
    data_matrix
    order
    batch
    classes
    qc_classes
    sample_classes
    threshold
    min_n_qc

    Returns
    -------

    """
    min_qc_order = batch_ext(order, batch, classes, qc_classes, "min")
    min_sample_order = batch_ext(order, batch, classes,
                                 sample_classes, "min")
    max_qc_order = batch_ext(order, batch, classes, qc_classes, "max")
    max_sample_order = batch_ext(order, batch, classes,
                                 sample_classes, "max")
    batches = batch[classes.isin(qc_classes)].unique()
    valid_features = data_matrix.columns

    for k_batch in batches:
        # feature check is done for each batch in three parts:
        # | start block | middle block      | end block |
        #   q   q         ssss q ssss q ssss  q  q
        #  where q is a qc sample and s is a biological sample
        # in the start block, a feature is valid if is detected
        # in at least one sample of the block
        # in the middle block, a feature is valid if the number
        # of qc samples where the feature was detected is greater
        # than the total number of qc samples in the block minus the
        # n_missing parameter
        # in the end block the same strategy applied in the start
        # block is used.
        # A feature is considered valid only if is valid in the totallity
        # of the batches.

        # start block check
        start_block_qc_samples = (order[(order >= min_qc_order[k_batch])
                                        & (order < min_sample_order[k_batch])
                                        & classes.isin(qc_classes)]
                                  .index)
        start_block_valid_features = \
            (data_matrix.loc[start_block_qc_samples] > threshold).any()
        start_block_valid_features = \
            start_block_valid_features[start_block_valid_features].index
        valid_features = valid_features.intersection(start_block_valid_features)

        # middle block check
        middle_block_qc_samples = (order[(order > min_sample_order[k_batch])
                                         & (order < max_sample_order[k_batch])
                                         & classes.isin(qc_classes)]
                                   .index)
        midd_block_valid_features = ((data_matrix
                                      .loc[middle_block_qc_samples] > threshold)
                                     .sum() >= min_n_qc)
        midd_block_valid_features = \
            midd_block_valid_features[midd_block_valid_features].index

        valid_features = valid_features.intersection(
            midd_block_valid_features)

        # end block check
        end_block_qc_samples = (order[(order > max_sample_order[k_batch])
                                      & (order <= max_qc_order[k_batch])
                                      & classes.isin(qc_classes)]
                                .index)
        end_block_valid_features = \
            (data_matrix.loc[end_block_qc_samples] > threshold).any()
        end_block_valid_features = end_block_valid_features[
            end_block_valid_features].index
        valid_features = valid_features.intersection(end_block_valid_features)

    invalid_features = data_matrix.columns.difference(valid_features)
    return invalid_features


def loess_interp(ft_data: pd.Series, order: pd.Series, qc_index: pd.Index,
                 sample_index: pd.Index, frac: float, interpolator: Callable):
    """
    Applies LOESS-correction interpolation on a feature. Auxilliary function of
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

    Returns
    -------
    pd.Series
    """
    qc_loess = _coov_loess(order[qc_index], ft_data[qc_index], interpolator,
                           frac=frac)
    interp = interpolator(order[qc_index], qc_loess)
    ft_data[sample_index] = interp(order[sample_index])
    return ft_data


def batch_corrector_func(df_batch: pd.DataFrame, order: pd.Series,
                         classes: pd.Series, frac: float,
                         interpolator: Callable, qc_classes: List[str],
                         sample_classes: List[str]) -> pd.DataFrame:
    """
    Applies LOESS correction - interpolation on a single batch. Auxilliary
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

    Returns
    -------
    pandas.DataFrame
    """
    qc_index = classes.isin(qc_classes)
    qc_index = qc_index[qc_index].index
    sample_index = classes.isin(sample_classes)
    sample_index = sample_index[sample_index].index
    df_batch.loc[sample_index, :] = \
        (df_batch.apply(lambda x: loess_interp(x, order, qc_index, sample_index,
                                               frac, interpolator)))
    return df_batch


def interbatch_correction(df: pd.DataFrame, order: pd.Series, batch: pd.Series,
                          classes: pd.Series, corrector_classes: List[str],
                          process_classes: List[str],
                          frac: Optional[float] = None,
                          interpolator: Optional[str] = "splines"
                          ) -> pd.DataFrame:
    """
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

    Mean centering is performed subtracting a batch mean and adding a grand
    mean.
    """
    interp_func = {"splines": CubicSpline, "linear": interp1d}
    interp_func = interp_func[interpolator]

    def corrector_helper(df_group):
        return batch_corrector_func(df_group, order[df_group.index],
                                    classes[df_group.index], frac,
                                    interp_func, corrector_classes,
                                    process_classes)

    # intra batch correction
    corrected = df.groupby(batch).apply(corrector_helper)
    # inter batch mean alignment
    global_mean = corrected.mean()
    corrected = corrected.groupby(batch).apply(lambda x: x - x.mean())
    corrected = corrected + global_mean
    corrected[corrected < 0] = 0
    return corrected
