from tidyms import utils
import numpy as np
import pandas as pd
import pytest
import string
import os


abc = list(string.ascii_lowercase)


@pytest.fixture
def random_df():
    data = np.random.normal(size=(200, 50), loc=10)
    data = pd.DataFrame(data)
    return data


@pytest.fixture
def single_row_df():
    data = pd.DataFrame(columns=range(10))
    data.loc[0] = np.arange(10)
    return data


@pytest.fixture
def empty_df():
    data = pd.DataFrame(data=[])
    return data


def create_random_file_name():
    name = "".join(list(np.random.choice(abc, 10)))
    ext = "".join(list(np.random.choice(abc, 3)))
    file_name = name + "." + ext
    return file_name


def test_normalize_sum(random_df):
    data = random_df
    normalized = utils.normalize(data, "sum")

    # check max index
    didx = data.idxmax(axis=1)
    nidx = normalized.idxmax(axis=1)
    assert (didx == nidx).all()
    # check that each row sums 1
    assert np.isclose(normalized.sum(axis=1).values, 1).all()


def test_normalize_max(random_df):
    data = random_df
    normalized = utils.normalize(data, "max")
    # check max index
    didx = data.idxmax(axis=1)
    nidx = normalized.idxmax(axis=1)
    assert (didx == nidx).all()
    # check max equals to 1
    assert np.isclose(normalized.max(axis=1).values, 1).all()


def test_normalize_euclidean(random_df):
    data = random_df
    normalized = utils.normalize(data, "euclidean")
    norm = normalized.apply(lambda x: np.linalg.norm(x), axis=1)
    assert np.isclose(norm, 1).all()


def test_normalize_feature(random_df):
    data = random_df
    ft = data.columns[25]
    normalized = utils.normalize(data, "feature", ft)
    assert np.isclose(normalized[ft], 1).all()


def test_normalize_invalid_mode(random_df):
    data = random_df
    with pytest.raises(ValueError):
        utils.normalize(data, "invalid_mode")


def test_scale_autoscaling(random_df):
    data = random_df
    scaled = utils.scale(data, "autoscaling")
    assert np.isclose(scaled.mean(), 0).all()
    assert np.isclose(scaled.var(), 1).all()


def test_scale_rescaling():
    data = np.random.normal(size=(200, 50), loc=10)
    data = pd.DataFrame(data)
    scaled = utils.scale(data, "rescaling")
    assert np.isclose(scaled.min(), 0).all()
    assert np.isclose(scaled.max(), 1).all()


def test_scale_paretto(random_df):
    data = random_df
    scaled = utils.scale(data, "pareto")
    assert np.isclose(scaled.mean(), 0).all()
    # in pareto scaling the std of x is equal to the variance of x scaled
    assert np.isclose(scaled.var(), data.std()).all()


def test_scale_invalid_mode(random_df):
    data = random_df
    with pytest.raises(ValueError):
        utils.scale(data, "invalid_mode")


def test_transform_log(random_df):
    data = random_df
    transformed = utils.transform(data, "log")
    assert np.isclose(transformed, np.log10(data.values)).all()


def test_transform_power(random_df):
    data = random_df
    transformed = utils.transform(data, "power")
    assert np.isclose(transformed, np.sqrt(data.values)).all()


def test_transform_invalid_mode(random_df):
    data = random_df
    with pytest.raises(ValueError):
        utils.transform(data, "invalid_mode")


def test_find_closest_left_border():
    x = np.arange(10)
    y = -1
    ind = utils.find_closest(x, y)
    assert ind == 0


def test_find_closest_right_border():
    x = np.arange(10)
    y = 10
    ind = utils.find_closest(x, y)
    assert ind == (x.size - 1)


def test_find_closest_middle():
    x = np.arange(10)
    y = 4.6
    ind = utils.find_closest(x, y)
    assert ind == 5


def test_find_closest_empty_x():
    x = np.array([])
    y = 10
    with pytest.raises(ValueError):
        utils.find_closest(x, y)


def test_find_closest_empty_y():
    x = np.arange(10)
    y = np.array([])
    res = utils.find_closest(x, y)
    assert res.size == 0


def test_find_closest_multiple_values():
    x = np.arange(100)
    y = np.array([-10, 4.6, 67.1, 101])
    ind = np.array([0, 5, 67, 99], dtype=int)
    result = utils.find_closest(x, y)
    assert np.array_equal(result, ind)


def test_sample_to_path(tmpdir):

    # create random names and then create empty files for a subset of them
    random_file_names = [create_random_file_name() for _ in range(100)]
    sample_names = [os.path.splitext(x)[0] for x in random_file_names]
    available = np.random.choice(random_file_names, 30)
    for names in available:
        path = os.path.join(tmpdir, names)
        open(path, "a").close()

    sample_to_path = utils.sample_to_path(sample_names, tmpdir)
    for names in available:
        path = os.path.join(tmpdir, names)
        fname = os.path.splitext(names)[0]
        assert sample_to_path[fname] == path


def test_cv_single_row_df(single_row_df):
    data = single_row_df
    # with one row the std is nan
    assert utils.cv(data).isna().all()


def test_cv_valid_data(random_df):
    data = random_df
    # fill missing values to zero to compare closeness of the result
    cv = utils.cv(data, fill_value=0)
    cv_test = data.std() / data.mean()
    cv_test = cv_test.fillna(0)
    assert np.isclose(cv, cv_test).all()


def test_cv_all_zeros():
    data = np.zeros((100, 200))
    data = pd.DataFrame(data)
    cv = utils.cv(data)
    assert cv.isna().all()


def test_cv_fill_na():
    fill = 1
    data = np.zeros((100, 200))
    data = pd.DataFrame(data)
    cv = utils.cv(data, fill_value=fill)
    assert (cv == fill).all()


def test_cv_series():
    data = pd.Series(np.random.normal(size=100, loc=1))
    cv = utils.cv(data)
    assert np.isclose(cv, data.std() / data.mean())


def test_cv_series_fill_na():
    fill = 1
    data = pd.Series(np.zeros(100))
    cv = utils.cv(data, fill_value=fill)
    assert np.isclose(cv, fill)


def test_robust_cv_single_row_df(single_row_df):
    data = single_row_df
    # nans are converted to inf, all values should be inf
    assert utils.robust_cv(data).isna().all()


def test_robust_cv_valid_data(random_df):
    data = random_df
    cv = utils.robust_cv(data, fill_value=0)
    cv_test = utils.mad(data) / data.median()
    cv_test = cv_test.fillna(0)
    assert np.isclose(cv, cv_test).all()


def test_robust_cv_all_zeros():
    data = np.zeros((100, 200))
    data = pd.DataFrame(data)
    cv = utils.robust_cv(data)
    assert cv.isna().all()


def test_robust_cv_fill_na():
    fill = 1
    data = np.zeros((100, 200))
    data = pd.DataFrame(data)
    cv = utils.robust_cv(data, fill_value=fill)
    assert (cv == fill).all()


def test_robust_cv_series():
    data = pd.Series(np.random.normal(size=100, loc=1))
    cv = utils.robust_cv(data)
    test_cv = utils.median_abs_deviation(data, scale="normal") / data.median()
    assert np.isclose(cv, test_cv)


def test_robust_cv_series_fill_na():
    fill = 1
    data = pd.Series(np.zeros(100))
    cv = utils.robust_cv(data, fill_value=fill)
    assert np.isclose(cv, fill)


def test_sd_ratio(random_df):
    df1 = random_df
    df2 = df1.copy() * 2
    ratio = df1.std() / df2.std()
    ratio = ratio.fillna(0)
    test_ratio = utils.sd_ratio(df1, df2, fill_value=0)
    assert np.isclose(ratio, test_ratio).all()


def test_sd_ratio_robust(random_df):
    df1 = random_df
    df2 = df1.copy() * 2
    ratio = utils.mad(df1) / utils.mad(df2)
    ratio = ratio.fillna(0)
    test_ratio = utils.sd_ratio(df1, df2, robust=True, fill_value=0)
    assert np.isclose(ratio, test_ratio).all()


def test_find_closest_unsorted_single_value():
    n = 100
    x = np.random.normal(size=n)
    # select three random points
    random_index = np.random.choice(n)
    xq = x[random_index]
    closest_index = utils.find_closest(x, xq, is_sorted=False)
    assert np.equal(random_index, closest_index).all()


def test_find_closest_unsorted_multiple_values():
    n = 100
    x = np.random.normal(size=n)
    # select three random points
    random_index = np.random.choice(n, 3)
    xq = x[random_index]
    closest_index = utils.find_closest(x, xq, is_sorted=False)
    assert np.equal(random_index, closest_index).all()


def test_get_filename(tmpdir):
    filename = create_random_file_name()
    fullpath = os.path.join(tmpdir, filename)
    open(fullpath, 'a').close()
    name = os.path.splitext(filename)[0]
    assert name == utils.get_filename(fullpath)


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_array1d_to_str_conversion_array_dtype_float(size):
    x = np.random.normal(size=size)
    x_str = utils.array1d_to_str(x)
    x_from_str = utils.str_to_array1d(x_str)
    assert np.array_equal(x, x_from_str)


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_array1d_to_str_conversion_arange(size):
    x = np.arange(size)
    x_str = utils.array1d_to_str(x)
    x_from_str = utils.str_to_array1d(x_str)
    assert np.array_equal(x, x_from_str)


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_array1d_to_str_conversion_array_dtype_int(size):
    x = np.random.normal(size=size).astype(int)
    x_str = utils.array1d_to_str(x)
    x_from_str = utils.str_to_array1d(x_str)
    assert np.array_equal(x, x_from_str)


def test_array_to_str_conversion_empty_float_array():
    x = np.array([], dtype=float)
    x_str = utils.array1d_to_str(x)
    x_from_str = utils.str_to_array1d(x_str)
    assert np.array_equal(x, x_from_str)


def test_array_to_str_conversion_empty_int_array():
    x = np.array([], dtype=int)
    x_str = utils.array1d_to_str(x)
    x_from_str = utils.str_to_array1d(x_str)
    assert np.array_equal(x, x_from_str)
