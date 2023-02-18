import numpy as np
import pandas as pd
from typing import List, Optional
from .container import DataContainer


def simulate_dataset(population: dict, mean_dict: dict, cov_dict: dict,
                     feature_mz: np.ndarray, feature_rt: np.ndarray,
                     blank_contribution: Optional[np.ndarray] = None,
                     noise_dict: Optional[dict] = None,
                     qc_template: bool = True, qc_bracket_size: int = 5,
                     triple_qc: bool = True, prepend_blank: int = 0,
                     append_blank: int = 0, batch_size: int = 25
                     ) -> DataContainer:
    """
    Creates DataContainer with simulated data

    Parameters
    ----------
    population: dict[str, int]
        A dictionary where the keys are the available classes in
        the experiment and the values are the number of samples
        from each class.
    mean_dict : dict
        a dictionary where each key is a class and the value is a 1D array with
        the mean values of each feature in the class
    cov_dict : dict
        a dictionary where each key is a class and the value is an array with
        the covariance matrix for the features in each class. A 2D array must
        be used to create correlated data. a 1D array where each element is
        the variance of the features can be used to create uncorrelated data.
    feature_mz : array
        m/z values assigned to each feature.
    feature_rt : array
        Rt values assigned to each feature.
    noise_dict : dict
        a dictionary where each key is a class and the value is a 1D array
        with the standard deviation of an additive noise level for each feature.
    blank_contribution : array, optional
        an with the mean value for each feature in the blanks.
    qc_template: bool
        If True, includes a template of QC samples surrounding
        the study samples.
    qc_bracket_size: int
        The number of study samples between QCs samples.
    triple_qc: bool
        If True adds three consecutive QC samples before the study samples
    prepend_blank: int
        number of blank samples to prepend to the batch
    append_blank: int`
        number of blank samples to append to the batch
    batch_size : int
        Number of study samples per analytical batch.

    Returns
    -------
    DataContainer

    """

    sm = _make_sample_list(population, qc_template, qc_bracket_size,
                           triple_qc, prepend_blank, append_blank, batch_size)
    dm = _make_data_matrix(population, sm, mean_dict, cov_dict,
                           blank_contribution, noise_dict)
    fm = np.vstack((feature_mz, feature_rt)).T
    fm = pd.DataFrame(data=fm, columns=["mz", "rt"], index=dm.columns)
    mapping = {"sample": list(population)}
    if qc_template:
        mapping["qc"] = ["QC"]
    if max(prepend_blank, append_blank) > 0:
        mapping["blank"] = ["blank"]
    return DataContainer(dm, fm, sm, mapping)


def _make_sample_list(population: dict, qc_template: bool = True,
                      qc_bracket_size: int = 5, triple_qc: bool = True,
                      prepend_blank: int = 0, append_blank: int = 0,
                      batch_size: int = 25):
    """
    creates a simulated sample list for an untargeted metabolomic assay.

    Parameters
    ----------
    population: dict[str, int]
        A dictionary where the keys are the available classes in
        the experiment and the values are the number of samples
        from each class.
    qc_template: bool
        If True, includes a template of QC samples surrounding
        the study samples.
    qc_bracket_size: int
        The number of study samples between QCs samples.
    triple_qc: bool
        If True adds three consecutive QC samples before the study samples
    prepend_blank: int
        number of blank samples to prepend to the batch
    append_blank: int`
        number of blank samples to append to the batch
    batch_size : int
        Number of study samples per analytical batch.

    """
    n_study_samples = sum(population.values())

    prepend_blank = ["blank"] * prepend_blank
    append_blank = ["blank"] * append_blank

    # append QCs before and after study samples
    if qc_template:
        if triple_qc:
            n_append = 3
        else:
            n_append = 1
        prepend_blank = prepend_blank + ["QC"] * n_append
        append_blank = ["QC"] * n_append + append_blank
    else:
        # setting this bracket size will avoid including QC samples
        qc_bracket_size = n_study_samples + 1

    # randomize classes and make QC brackets
    classes = _make_sample_classes(population, batch_size,
                                   qc_bracket_size, prepend_blank, append_blank)
    n_prepend = len(prepend_blank)
    n_append = len(append_blank)
    batch_number = _make_batch_number(batch_size, n_study_samples, n_prepend,
                                      n_append, qc_bracket_size)
    order = list(range(1, len(batch_number) + 1))
    sample_name = _make_sample_name(len(batch_number))
    sample_list = {"id": sample_name, "class": classes, "order": order,
                   "batch": batch_number}
    sample_list = pd.DataFrame(data=sample_list, index=sample_name)
    sample_list.index.name = "samples"
    return sample_list


def _make_data_matrix(population: dict, sample_list: pd.DataFrame,
                      mean_dict: dict, cov_dict: dict,
                      blank_contribution: Optional[np.ndarray] = None,
                      noise_dict: Optional[dict] = None):
    """
    Creates a simulated data matrix

    Parameters
    ----------
    population: dict
        A dictionary where each key is a sample class and the value is the
        number of samples.
    sample_list : DataFrame
        sample list created with _make_sample_list
    mean_dict : dict
        a dictionary where each key is a class and the value is a 1D array with
        the mean values of each feature in the class
    cov_dict : dict
        a dictionary where each key is a class and the value is an array with
        the covariance matrix for the features in each class. A 2D array must
        be used to create correlated data. a 1D array where each element is
        the variance of the features can be used to create uncorrelated data.
    noise_dict : dict
        a dictionary where each key is a class and the value is a 1D array
        with the standard deviation of an additive noise level for each feature.
    blank_contribution : array, optional
        an with the mean value for each feature in the blanks.

    Returns
    -------

    """
    data_mats = dict()
    for k_class, k_obs in population.items():
        mean = mean_dict[k_class]
        cov = cov_dict[k_class]
        mat = generate_correlated_data(k_obs, mean, cov)
        data_mats[k_class] = mat

    if (sample_list["class"] == "blank").any():
        if blank_contribution is None:
            blank_contribution = np.zeros_like(mean)
        data_mats["blank"] = \
            _generate_blank_data(sample_list, blank_contribution)

    if (sample_list["class"] == "QC").any():
        data_mats["QC"] = _generate_qc_data(population, sample_list,
                                            mean_dict)

    if noise_dict is not None:
        for c, x in data_mats.items():
            data_mats[c] = _add_noise(x, noise_dict[c])

    data = _concat_data(data_mats, sample_list)
    ft_names = _generate_feature_names(data.shape[1])
    data_matrix = pd.DataFrame(data=data, index=sample_list.index,
                               columns=ft_names)
    return data_matrix


def _make_sample_classes(sample_distribution, batch_size, qc_bracket_size,
                         prepend, append):
    n_study_samples = sum(sample_distribution.values())
    # randomize classes and make QC brackets
    classes = list()
    for k, v in sample_distribution.items():
        classes.extend([k] * v)

    permutation_index = np.random.permutation(n_study_samples)
    qb, rb = divmod(n_study_samples, batch_size)
    n_batches = qb + bool(rb)
    res = list()
    start = 0
    end = min(batch_size, n_study_samples)
    for k_batches in range(n_batches):
        res += prepend
        res += _make_batch_sample_block(classes,
                                        permutation_index[start:end],
                                        qc_bracket_size)
        res += append
        start = end
        end = min(start + batch_size, n_study_samples)
    return res


def _make_batch_number(batch_size, n_study_samples, n_prepend, n_append,
                       qc_bracket_size):
    n_batch, n_study_samples_last_batch = divmod(n_study_samples, batch_size)
    # adds an extra batch for the last batch
    n_batch += int(n_study_samples_last_batch > 0)
    n_qc_batch = _n_qc_per_batch(batch_size, qc_bracket_size)
    n_samples_per_batch = batch_size + n_prepend + n_append + n_qc_batch
    batch_number = list()
    for k_batch in range(1, n_batch):
        batch_number.extend([k_batch] * n_samples_per_batch)
    if n_study_samples_last_batch:

        n_qc_last_batch = _n_qc_per_batch(n_study_samples_last_batch,
                                          qc_bracket_size)
        n_samples_last_batch = (n_study_samples_last_batch + n_prepend +
                                n_append + n_qc_last_batch)
        batch_number.extend([n_batch] * n_samples_last_batch)
    return batch_number


def _make_sample_name(n_samples: int):
    pad_length = len(str(n_samples))
    template_str = "Sample-{{:0{}n}}".format(pad_length)
    sample_name = [template_str.format(x) for x in range(1, n_samples + 1)]
    return sample_name


def generate_correlated_data(n_obs: int, mean: np.array,
                             cov: np.array) -> np.array:
    """
    Generates a data matrix with normal correlated columns

    Parameters
    ----------
    n_obs : int
        number of observations in the data matrix
    mean : vector with length `n_vars`
        mean vector of the variables.
    cov : vector with length `n_vars` or matrix with shape [n_vars, n_vars]
        covariance matrix of the variables. If `cov` is a vector, then
        a diagonal covariance matrix is used.

    Returns
    -------
    X : array with shape [n_obs, n_vars]
        a matrix with rows of observations and columns of variables

    """
    if len(cov.shape) == 1:
        cov = np.diag(cov)

    if not _is_valid_covariance_matrix(cov):
        msg = "`cov` must be symmetric and positive definite."
        raise ValueError(msg)

    if cov.shape[1] != mean.size:
        msg = "Shape mismatch between the mean and the covariance matrix"
        raise ValueError(msg)

    L = np.linalg.cholesky(cov)
    n_vars = mean.size
    X = np.random.normal(size=(n_obs, n_vars))
    X = np.matmul(X, L.T) + mean
    return X


def _generate_blank_data(sample_list, blank_contribution):
    n_blank = (sample_list["class"] == "blank").sum()
    blank_data = np.tile(blank_contribution, [n_blank, 1])
    return blank_data


def _generate_qc_data(sample_population, sample_list, mean_dict):
    n_qc = (sample_list["class"] == "QC").sum()
    n_samples = sum(sample_population.values())
    mean = sum([n * mean_dict[c] for c, n in sample_population.items()])
    mean = np.array(mean) / n_samples
    qc_data = np.tile(mean, [n_qc, 1])
    return qc_data


def _is_valid_covariance_matrix(cov: np.array):
    """Checks if a matrix is symmetric and positive definite"""
    is_sym = _is_symmetric(cov)
    is_def_pos = (np.linalg.eigvals(cov) > 0).all()
    return is_sym and is_def_pos


def _is_symmetric(x: np.array) -> bool:
    """Check if a matrix is symmetric"""
    return (x == x.T).all()


def _generate_random_covariance_matrix(n: int) -> np.array:
    """Creates a random symmetric, positive definite matrix"""
    x = np.random.normal(size=(n, n))
    x = (x + x.T) / 2 + np.diag(np.abs(x).sum(axis=1)) * 2
    return x


def _make_batch_sample_block(classes, batch_perm_index,
                             qc_bracket_size):
    classes_rand = list()
    for k, ind in enumerate(batch_perm_index):

        if ((k % qc_bracket_size) == 0) and k > 0:
            classes_rand.append("QC")
        classes_rand.append(classes[ind])
    return classes_rand


def _n_qc_per_batch(batch_size, qc_bracket_size):
    q, r = divmod(batch_size, qc_bracket_size)
    if q > 0 and r == 0:
        q -= 1
    return q


def _add_noise(x, noise):
    xn = x + np.random.normal(size=x.shape[1], scale=noise)
    xn[xn < 0] = 0
    return xn


def _concat_data(data_mats, sample_list):
    classes = sample_list["class"]
    classes_unique = np.sort(classes.unique())
    mat = [data_mats[x] for x in classes_unique]
    mat = np.vstack(mat)
    # sort in sample list order
    sorted_index = np.argsort(classes).values
    data = np.zeros_like(mat)
    data[sorted_index, :] = mat
    return data


def _generate_feature_names(n_ft):
    pad_length = len(str(n_ft))
    template_str = "FT-{{:0{}n}}".format(pad_length)
    ft_name = [template_str.format(x) for x in range(1, n_ft + 1)]
    return ft_name
