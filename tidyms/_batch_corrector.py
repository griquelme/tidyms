import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from joblib import Parallel, delayed
from typing import List, Optional, Tuple
from tqdm.notebook import tqdm
from functools import partial

MIN_LOESS_SIZE = 4      # the minimum number of points necessary to use LOESS


def find_invalid_samples(
    sample_metadata: pd.DataFrame,
    sample_class: List[str],
    qc_class: List[str]
) -> pd.Index:
    """
    Finds samples that cannot be corrected using LOESS batch correction.

    Parameters
    ----------
    sample_metadata : DataFrame
        sample metadata from a DataContainer
    sample_class : List[str]
        Classes where the correction is going to be applied.
    qc_class : List[str]
        Classes used to estimate the correction factor.

    Returns
    -------
    invalid_samples = pd.Index

    """
    invalid_samples = pd.Index([])
    for name, gdf in sample_metadata.groupby("batch"):
        qc_mask = gdf["class"].isin(qc_class)
        sample_mask = gdf["class"].isin(sample_class)
        qc_order = gdf.loc[qc_mask, "order"]
        sample_order = gdf.loc[sample_mask, "order"]    # type: pd.Series
        min_qc_order = qc_order.min()
        max_qc_order = qc_order.max()
        n_qc = qc_order.size
        if n_qc >= MIN_LOESS_SIZE:
            mask = (
                    (sample_order < min_qc_order) |
                    (sample_order > max_qc_order)
            )
            rm_index = mask[mask].index
        else:
            rm_index = gdf.index
        invalid_samples = invalid_samples.union(rm_index)
    return invalid_samples


def find_invalid_features(
    data_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    sample_class: List[str],
    qc_class: List[str],
    threshold: float,
    min_detection_rate: float
) -> pd.Index:
    """
    Remove features detected in a low number of QC samples.

    Parameters
    ----------
    data_matrix : DataFrame
        data matrix from a DataContainer
    sample_metadata : DataFrame
        sample metadata from a DataContainer
    sample_class : List[str]
        Classes where the correction is going to be applied.
    qc_class : List[str]
        Classes used to estimate the correction factor.
    threshold : positive number
        Minimum value to consider a feature detected.
    min_detection_rate : number between 0 and 1
        Minimum fraction of samples where a feature was detected

    Returns
    -------
    invalid_features : pd.Index

    """
    invalid_features = pd.Index([])
    for name, gdf in sample_metadata.groupby("batch"):
        qc_mask = gdf["class"].isin(qc_class)
        sample_mask = gdf["class"].isin(sample_class)
        qc_order = gdf.loc[qc_mask, "order"]    # type: pd.Series
        sample_order = gdf.loc[sample_mask, "order"]
        min_sample_order = sample_order.min()
        max_sample_order = sample_order.max()

        # first block: qc samples before any study samples
        first_block_samples = qc_order < min_sample_order
        first_block_samples = first_block_samples[first_block_samples].index
        first_block_n = data_matrix.loc[first_block_samples, :]
        first_block_n = (first_block_n > threshold).sum()

        # middle block: qc sample measured between study samples
        middle_block_samples = ((qc_order > min_sample_order) |
                                (qc_order < max_sample_order))
        middle_block_samples = middle_block_samples[middle_block_samples].index
        middle_block_n = data_matrix.loc[middle_block_samples, :]
        middle_block_n = (middle_block_n > threshold).sum()

        # last block: qc samples after all study samples
        last_block_samples = qc_order > max_sample_order
        last_block_samples = last_block_samples[last_block_samples].index
        last_block_n = data_matrix.loc[last_block_samples, :]
        last_block_n = (last_block_n > threshold).sum()

        total = first_block_n + middle_block_n + last_block_n
        valid_blocks = (
            (first_block_n > 0) &
            (middle_block_n > 0) &
            (last_block_n > 0)
        )    # type: pd.Series
        valid_n = (total >= MIN_LOESS_SIZE)
        n_rows = qc_order.size
        valid_dr = (total / n_rows) >= min_detection_rate   # type: pd.Series
        invalid_mask = ~(valid_n & valid_blocks & valid_dr)
        rm_features = invalid_mask[invalid_mask].index
        invalid_features = invalid_features.union(rm_features)
    return invalid_features


def correct_batches(
    data_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    sample_class: List[str],
    qc_class: List[str],
    threshold: float = 0.0,
    frac: Optional[float] = None,
    first_n: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Inter-batch correction using LOESS smoothing.

    Parameters
    ----------
    data_matrix : DataFrame
        data matrix from a DataContainer
    sample_metadata : DataFrame
        sample metadata from a DataContainer
    sample_class : List[str]
        Classes where the correction is going to be applied.
    qc_class : List[str]
        Classes used to estimate the correction factor.
    threshold : positive number
        Minimum value to consider a feature detected. Features in QC samples
        above this value are used to compute the correction factor.
    frac : Number between 0 and 1.0 or None, default=None
        frac value passed to LOESS function. If None, value is optimized for
        each feature using cross validation.
    first_n : int or None, default=None
        If specified, computes the mean value using the firs n qc samples in
        each batch. If ``None``, uses all qc samples in a batch to estimate a
        mean.
    n_jobs: int or None, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    verbose : bool, default=True
        If ``True``, displays a progress bar.

    Returns
    -------
    corrected: DataFrame
        Data matrix with the correction applied.

    """
    # data matrix is split into columns, and each column is split according to
    # the number of batches.
    iterator = _split_data_matrix(
        data_matrix,
        sample_metadata,
        sample_class,
        qc_class,
        threshold
    )

    if verbose:
        n_ft = data_matrix.shape[1]
        n_batch = sample_metadata["batch"].unique().size
        desc = "Correcting {} features in {} batches".format(n_ft, n_batch)
        total = _get_tqdm_total(data_matrix, sample_metadata)
        iterator = tqdm(iterator, total=total, desc=desc)

    # intra-batch correction
    corrector_func = partial(_correct_intra_batch, first_n=first_n, frac=frac)
    func = delayed(corrector_func)
    data = Parallel(n_jobs=n_jobs)(func(x) for x in iterator)
    data_corrected = _rebuild_data_matrix(data_matrix.shape, data)
    data_corrected = pd.DataFrame(
        data=data_corrected,
        index=data_matrix.index,
        columns=data_matrix.columns
    )
    # inter-batch correction
    data_corrected = _inter_batch_correction(
        data_corrected, sample_metadata, qc_class
    )

    return data_corrected


class _LoessCorrector(BaseEstimator, RegressorMixin):
    """
    Intra-batch corrector implementation using sklearn.

    Parameters
    ----------
    frac : number between 0 and 1, default=0.66
        Fraction of samples used for local regressions

    """
    def __init__(self, frac: float = 0.66):
        """
        Constructor function.

        """

        self.frac = frac
        self.interpolator_ = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        x = X.flatten()
        y_fit = lowess(
            y, x, frac=self.frac, is_sorted=True, return_sorted=False
        )
        fill = (y[0], y[-1])
        self.interpolator_ = interp1d(
            x, y_fit, fill_value=fill, bounds_error=False)
        return self

    def predict(self, X):
        xf = X.flatten()
        x_interp = self.interpolator_(xf)
        return x_interp


def _split_data_matrix(
        data_matrix: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        sample_class: List[str],
        qc_class: List[str],
        threshold: float
):
    """
    Yields chunks of the data matrix and the order array associated with the
    data. Aux function to correct_batches.

    Parameters
    ----------
    data_matrix : DataFrame
    sample_metadata : DataFrame
    sample_class : List[str]
        classes where the correction will be applied
    qc_class : List[str]
        classes used to create the correction
    threshold : positive number
        Minimum value to consider a feature detected.
    Yields
    -------
    row_start : int
        index of row where x starts.
    column : int
        index of the column of x.
    x : np.ndarray.
        Fragment of the data matrix.
    qc_index : np.ndarray
        indices in x where qcs are located.
    sample_index : np.ndarray
        indices in x where samples are located.

    """
    # classes used to train the corrector and classes to be processed
    corrector_class = qc_class
    process_class = qc_class + sample_class

    X = data_matrix.to_numpy()
    n_sample, n_ft = X.shape
    grouped = sample_metadata.groupby("batch")
    start_index = 0
    for name, g in grouped:
        # corrector index
        index = pd.Series(np.arange(g.shape[0]), index=g.index)
        order = g.order.to_numpy()[:, np.newaxis]
        train_index = index[g["class"].isin(corrector_class)].to_numpy()
        # processor index
        predict_index = index[g["class"].isin(process_class)].to_numpy()
        for column in range(n_ft):
            x = X[index + start_index, column]
            ft_train_index = train_index[x[train_index] >= threshold]
            yield start_index, column, order, x, ft_train_index, predict_index
        start_index += g.shape[0]


def _correct_intra_batch(
        args: Tuple,
        frac: Optional[float] = None,
        first_n: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply LOESS correction on features in a batch. Aux function to
    correct_batches.

    Parameters
    ----------
    args : Tuple
        output yield from _split_data_matrix
    frac : number or None, default=None
    first_n: int or None, default=None

    Returns
    -------
    x : np.ndarray
    index : np.ndarray
    column : int

    """
    start_index, column, order, x, train_index, predict_index = args

    # train and predict values
    x_train = order[train_index]
    y_train = x[train_index]
    x_predict = order[predict_index]
    y_predict = x[predict_index]

    # cross validation is bypassed if a frac value is specified
    if (frac is None) or (x_train.size > MIN_LOESS_SIZE):
        corrector = _LoessCorrector()
        cv = _get_cv(train_index)
        grid_params = _get_param_grid_loess_corrector(train_index)
        scoring = "neg_mean_squared_error"
        grid = GridSearchCV(corrector, grid_params, cv=cv, scoring=scoring)
        grid.fit(x_train, y_train)
        corrector.set_params(**grid.best_params_)
    else:
        frac = 1.0
        corrector = _LoessCorrector(frac=frac)
    corrector.fit(x_train, y_train)

    # compute the mean value in qc, used as `true` value for scaling
    if first_n:
        x_mean = x[train_index[:first_n]].mean()
    else:
        x_mean = x[train_index].mean()

    # compute corrected values
    x_qc = corrector.predict(x_predict)
    factor = np.zeros_like(x_qc)
    # correct nans in zero values and negative values generated during LOESS
    is_positive = x_qc > 0
    factor = np.divide(x_mean, x_qc, out=factor, where=is_positive)
    corrected = y_predict * factor
    x[predict_index] = corrected

    index = np.arange(x.size) + start_index
    return x, index, column


def _rebuild_data_matrix(shape, data: List[Tuple]) -> np.ndarray:
    """
    Rebuilds the data matrix from the processed output. Aux function to\
    correct_batches.

    Parameters
    ----------
    shape : tuple
        shape of the corrected data matrix
    data : list
        corrected data.
    Returns
    -------
    X : np.ndarray

    """
    Xr = np.zeros(shape=shape, dtype=float)
    for x, index, column in data:
        Xr[index, column] = x
    return Xr


def _inter_batch_correction(
        data_matrix: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        qc_class: List[str]
) -> pd.DataFrame:
    """
    corrects the mean in each batch to a common mean. Aux function to
    correct_batches.

    Parameters
    ----------
    data_matrix : pd.DataFrame
    sample_metadata : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    qc_mask = sample_metadata["class"].isin(qc_class)
    qc_dm = data_matrix[qc_mask]
    batch = sample_metadata["batch"]
    qc_batch = batch[qc_mask]
    n_batches = batch.unique().size
    if n_batches == 1:
        res = data_matrix
    else:
        factor = _get_inter_batch_correction_factor(qc_dm, qc_batch)
        res = data_matrix.groupby(batch).apply(lambda x: x * factor[x.name])
    return res


def _get_inter_batch_correction_factor(
        data_matrix: pd.DataFrame,
        batch: pd.Series
) -> pd.DataFrame:
    """
    Estimates a correction factor for inter-batch correction. Aux function to
    _inter_batch_correction.

    """
    inter_batch_mean = data_matrix.mean()
    batch_mean = data_matrix.groupby(batch).mean()
    factor = inter_batch_mean / batch_mean
    factor[~np.isfinite(factor)] = 0
    factor = factor.T
    return factor


def _get_cv(train_index):
    """
    Select a cross validator according to the number of train samples. Aux
    function to _correct_intra_batch.

    """
    if train_index.size > 15:
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
    else:
        cv = LeaveOneOut()
    return cv


def _get_param_grid_loess_corrector(train_index: np.ndarray) -> dict:
    """
    Builds a parameter grid for GridSearchCV.

    """
    n = train_index.size
    min_frac = min(MIN_LOESS_SIZE / n, 1.0)
    # Limits the number of points in the grid to at most 5
    if n < 9:
        frac = np.arange(MIN_LOESS_SIZE, n + 1) / n
    else:
        n_points = 5
        frac = np.linspace(min_frac, 1.0, n_points)

    grid_params = {"frac": frac}
    return grid_params


def _get_tqdm_total(
        data_matrix: pd.DataFrame, sample_metadata: pd.DataFrame) -> int:
    """
    Computes the number of items to compute a percentage in the progress bar.

    """
    _, n_ft = data_matrix.shape
    n_batch = sample_metadata["batch"].unique().size
    total = n_batch * n_ft
    return total
