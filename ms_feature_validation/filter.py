import pandas as pd


def blank_correction(data, blanks, mode, blank_relation):
    corrector = {"max": lambda x: x.max(), "mean": lambda x: x.mean()}
    correction = corrector[mode](blanks) * blank_relation
    corrected =  data.subtract(correction)
    corrected[corrected < 0] = 0
    return corrected


def prevalence_filter(data_container, include_classes, lb, ub):
    """
    Return features  with prevalence outside the [lb, ub] interval in each of
    included classes.

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
            bounds = (feature_counts < lb_group) | (feature_counts > ub_group)
            outside_bounds_features = feature_counts[bounds].index
            remove_features = remove_features.union(outside_bounds_features)
    return remove_features


def variation_filter(data_container, include_classes, lb, ub, robust):
    """
    Return features with variation outside the [lb, ub] interval in each of
    include_classes.
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
            bounds = (group_variation < lb) | (group_variation > ub)
            outside_bounds_features = group_variation[bounds].index
            remove_columns = remove_columns.union(outside_bounds_features)
    return remove_columns


def iqr(df):
    return (df.quantile(0.75) - df.quantile(0.25)) / df.quantile(0.5)


def cv(df):
    return df.std() / df.mean()


def corrector_map(data_container, mapper):
    """
    Generates tuples of DataContainers one used to generate a correction and
    other to be corrected.
    Parameters
    ----------
    data_container : DataContainer
    mapper: dict[str: list[str]]
            dictionary of

    Yields
    ------
    corrector_dc, to_correct_dc
    """
    for k, v in mapper.items():
        if isinstance(k, str):
            k = [k]
        else:
            k = list(k)
        if isinstance(v, str):
            v = [v]
        else:
            v = list(v)
        corrector_dc = data_container.select_classes(list(k))
        to_correct_dc = data_container.select_classes(list(v))
        yield corrector_dc.index, to_correct_dc.index
#TODO: corregir esto porque tiene que apuntar al data container original para que lacorreccion se aplique