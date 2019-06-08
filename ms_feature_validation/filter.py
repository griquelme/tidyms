import pandas as pd


def get_blank_correction(data_container, blank_classes, correction_type, blank_relation):
    blanks = data_container.select_classes(blank_classes)
    # corrector function
    corrector = {"max": lambda x: x.max(),
                 "mean": lambda x: x.mean()}
    correction = corrector[correction_type](blanks.data_matrix) * blank_relation
    return correction


def blank_correction(data_container, blank_classes, correction_type, blank_relation):
    correction = get_blank_correction(data_container, blank_classes, correction_type, blank_relation)
    data_container.data_matrix = data_container.data_matrix.subtract(correction, axis="columns")
    data_container.data_matrix[data_container.data_matrix < 0] = 0


def prevalence_filter(data_container, include_classes, lb, ub):
    """
    Return features  with prevalence outside the [lb, ub] interval in each of include_classes.

    Parameters
    ----------
    data_container : DataContainer
    include_classes : Iterable[str]
                      classes selected to evaluate prevalence
    lb : float.
         lower bound of prevalence
    ub : float.
         upper bound of prevalence
    Returns
    -------
    remove_features: pandas.Index
    """
    remove_features = pd.Index([])
    for class_group, group in data_container.group_by_class():
        if class_group in include_classes:
            lb_group, ub_group = lb * group.shape[0], ub * group.shape[0]
            feature_counts = (group > 0).sum()
            outside_bounds_features = feature_counts[feature_counts < lb_group | feature_counts > ub_group].index
            remove_features = remove_features.union(outside_bounds_features)
    return remove_features


def variation_filter(data_container, include_classes, lb, ub, robust):
    """
    Return features with variation outside the [lb, ub] interval in each of include_classes.
    Parameters
    ----------
    data_container : DataContainer
    include_classes : Iterable[str]
                      classes selected to evaluate variation
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

    if robust:
        variation = iqr
    else:
        variation = cv
    remove_columns = pd.Index([])
    for class_group, group in data_container.group_by_class():
        if class_group in include_classes:
            group_variation = variation(group)
            outside_bounds_features = group_variation[(group_variation < lb) | (group_variation > ub)].index
            remove_columns = remove_columns.union(outside_bounds_features)
    return remove_columns


def iqr(df):
    return (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)


def cv(df):
    return df.std() / df.mean()
