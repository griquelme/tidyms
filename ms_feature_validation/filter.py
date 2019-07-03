import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess


def blank_correction(data, classes, blank_classes, sample_classes, mode, blank_relation):
    """
    Correct samples using blanks.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to correct.
    classes : pandas.Series
        Samples class labels.
    blank_classes : list[str]
        Classes to be used as blanks.
    sample_classes : list[str]
        Classes to be used as samples
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
    samples = data[classes.isin(sample_classes)]
    blanks = data[classes.isin(blank_classes)]
    correction = corrector[mode](blanks) * blank_relation
    corrected = samples - correction
    corrected[corrected < 0] = 0
    data[classes.isin(sample_classes)] = corrected
    return data


def prevalence_filter(data, classes, include_classes, lb, ub, intraclass):
    """
    Return columns with relative counts outside the [lb, ub] interval.

    Parameters
    ----------
    data : pandas.DataFrame
    classes : pandas.Series
        sample class labels
    include_classes : list[str]
        classes included in the filter
    lb : float.
        Relative lower bound of prevalence.
    ub : float.
        Relative upper bound of prevalence.
    intraclass : bool
        if True computes prevalence for each class separately and return columns
        that are outside bounds in all of `include_classes`.
    Returns
    -------
    outside_bounds_columns: pandas.Index
    """
    # selects include_classes
    classes = classes[classes.isin(include_classes)]
    data = data.loc[classes.index, :]
    # pipeline for prevalence filter
    is_outside_bounds = ((data > 0)
                         .pipe(grouper(classes, intraclass))
                         .sum()
                         .pipe(normalizer(classes, intraclass))
                         .pipe(bounds_checker(lb, ub, intraclass)))
    outside_bounds_columns = is_outside_bounds[is_outside_bounds].index
    return outside_bounds_columns


def variation_filter(data, classes, include_classes, lb, ub,
                     intraclass, robust):
    """
    Return columns with variation outside the [lb, ub] interval.

    Parameters
    ----------
    data : pandas.DataFrame
    classes : pandas.Series
        class labels of samples.
    include_classes : list[str]
        classes included in the filter
    lb : float
        lower bound of variation
    ub : float
        upper bound of variation
    intraclass : bool
         if True computes prevalence for each class separately and return
         columns that are outside bounds in all of `include_classes`.
    robust : bool
        if True uses iqr as a metric of variation. if False uses cv
    Returns
    -------
    remove_features : pandas.Index
    """

    # selects include_classes
    classes = classes[classes.isin(include_classes)]
    data = data.loc[classes.index, :]
    # pipeline for variation filter
    is_outside_bound = (data.pipe(grouper(classes, intraclass))
                        .pipe(variation(robust))
                        .pipe(bounds_checker(lb, ub, intraclass)))
    outside_bounds_columns = is_outside_bound[is_outside_bound].index
    return outside_bounds_columns


def normalizer(classes, intraclass):
    class_counts = classes.value_counts()
    n = class_counts.sum()
    return lambda x: x.divide(class_counts, axis=0) if intraclass else x / n


def grouper(classes, intraclass):
    return lambda x: x.groupby(classes) if intraclass else x


def bounds_checker(lb, ub, intraclass):
    if intraclass:
        return lambda x: ((x < lb) | (x > ub)).any()
    else:
        return lambda x: ((x < lb) | (x > ub))


def variation(robust):
    if robust:
        return lambda x: (x.quantile(0.75) - x.quantile(0.25)) / x.quantile(0.5)
    else:
        return lambda x: x.std() / x.mean()


def cspline_correction(x, y, xq):
    sp = CubicSpline(x, y)
    yq = sp(xq)
    return yq


def loess_correction(x, y, xq, **kwargs):
    y_loess = lowess(y, x, return_sorted=False, **kwargs)
    yq = cspline_correction(x, y_loess, xq)
    return yq


def batch_correction(data, run_order, classes, corrector_class, sample_classes,
                     mode, **kwargs):
    """
    Correct instrument response drift using LOESS regression [1].

    Parameters
    ----------
    data : pandas.DataFrame
    run_order : pandas.Series
        run order of samples
    classes : pandas.Series
        class label for samples
    corrector_class : str
        label of corrector class
    sample_classes: list[str]
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
    corrector_class_mask = classes.isin([corrector_class])
    corrector_run = run_order[corrector_class_mask]
    sample_classes_mask = classes.isin(sample_classes)
    sample_run = run_order[sample_classes_mask]
    corrected = data.apply(lambda x: corrector(corrector_run,
                                               x[corrector_class_mask],
                                               sample_run,
                                               **kwargs))
    return corrected


def interbatch_correction(data, batch, run_order, classes, corrector_class,
                          sample_classes, mode, **kwargs):
    """
    Apply batch correction to several batches. Interbatch correction is achieved
    using a scaling factor obtained from the mean values in corrector samples.

    Parameters
    ----------
    data : pd.DataFrame
    batch : pd.Series
        batch number for a given sample
    run_order : pd.Series
    classes : pd.Series
        Class labels for samples
    corrector_class : str
        class used to correct samples
    sample_classes : list[str]
        class labels to correct.
    mode : {"loess", "splines"}
    kwargs : optional arguments to pass to loess corrector

    Returns
    -------
    corrected: pandas.DataFrame
    """
    corrector_samples_mask = classes.isin([corrector_class])
    scaling_factor = data[corrector_samples_mask].mean()

    def corrector_helper(x):
        corrected_batch = batch_correction(x, run_order.loc[x.index], classes,
                                           corrector_class, sample_classes,
                                           mode, **kwargs)
        return corrected_batch

    corrected = (data.groupby(batch)
                 .apply(corrector_helper)
                 * scaling_factor)
    return corrected
