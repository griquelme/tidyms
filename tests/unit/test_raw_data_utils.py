from tidyms.fileio import MSData_simulated
from tidyms.lcms import Chromatogram
from tidyms.raw_data_utils import _match_mz
import pytest
import numpy as np
import tidyms as ms


mz_list = np.array([200, 250, 300, 420, 450])


@pytest.fixture
def sim_ms_data():
    mz = np.array(mz_list)
    rt = np.linspace(0, 100, 100)

    # simulated features params
    mz_params = np.array([mz_list, [3, 10, 5, 31, 22]])
    mz_params = mz_params.T
    rt_params = np.array([[30, 40, 60, 80, 80], [1, 2, 2, 3, 3], [1, 1, 1, 1, 1]])
    rt_params = rt_params.T

    noise_level = 0.1
    sim_exp = MSData_simulated(mz, rt, mz_params, rt_params, noise=noise_level)
    return sim_exp


def test_make_roi(sim_ms_data):
    roi_list = ms.make_roi(sim_ms_data, tolerance=0.005, max_missing=0, min_length=1)
    assert len(roi_list) == sim_ms_data._reader.mz_params.shape[0]


def test_make_roi_targeted_mz(sim_ms_data):
    # the first three m/z values generated by simulated experiment are used
    targeted_mz = sim_ms_data._reader.mz_params[:, 0][:3]
    roi_list = ms.make_roi(
        sim_ms_data,
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        min_intensity=0,
        targeted_mz=targeted_mz,
    )
    assert len(roi_list) == targeted_mz.size


def test_make_roi_min_intensity(sim_ms_data):
    min_intensity = 15
    roi_list = ms.make_roi(
        sim_ms_data,
        tolerance=0.005,
        max_missing=0,
        min_length=1,
        min_intensity=min_intensity,
    )
    # only two roi should have intensities greater than 15
    assert len(roi_list) == 2


def test_make_roi_multiple_match_closest(sim_ms_data):
    roi_list = ms.make_roi(
        sim_ms_data, tolerance=0.005, max_missing=0, min_length=1, multiple_match="closest"
    )
    assert len(roi_list) == sim_ms_data._reader.mz_params.shape[0]


def test_make_roi_multiple_match_reduce_merge(sim_ms_data):
    # set a tolerance such that two mz values are merged
    # test is done in targeted mode to force a multiple match by removing
    # one of the mz values
    targeted_mz = sim_ms_data._reader.mz_params[:, 0]
    targeted_mz = np.delete(targeted_mz, 3)
    tolerance = 31
    roi_list = ms.make_roi(
        sim_ms_data, tolerance=tolerance, max_missing=0, min_length=1, targeted_mz=targeted_mz
    )
    assert len(roi_list) == (sim_ms_data._reader.mz_params.shape[0] - 1)


def test_make_roi_invalid_multiple_match(sim_ms_data):
    with pytest.raises(ValueError):
        ms.make_roi(
            sim_ms_data,
            tolerance=0.005,
            max_missing=0,
            min_length=0,
            multiple_match="invalid-value",
        )


# # test accumulate spectra


def test_accumulate_spectra_centroid(sim_ms_data):
    start_time = sim_ms_data._reader.rt.min()
    end_time = sim_ms_data._reader.rt.max()
    sp = ms.accumulate_spectra(sim_ms_data, start_time=start_time, end_time=end_time)
    assert sp.mz.size == sim_ms_data._reader.mz_params.shape[0]


def test_accumulate_spectra_centroid_subtract_left(sim_ms_data):
    start_time = sim_ms_data._reader.rt[70]
    end_time = sim_ms_data._reader.rt[90]
    subtract_left_time = sim_ms_data._reader.rt[20]
    sp = ms.accumulate_spectra(
        sim_ms_data,
        start_time=start_time,
        end_time=end_time,
        subtract_left_time=subtract_left_time,
    )
    # only two peaks at rt 80 should be present
    assert sp.mz.size == 2


# test make_chromatogram


def test_make_chromatograms(sim_ms_data):
    # test that the chromatograms generated are valid

    # create chromatograms
    n_sp = sim_ms_data.get_n_spectra()
    n_mz = sim_ms_data._reader.mz_params.shape[0]
    rt = np.zeros(n_sp)
    chromatogram = np.zeros((n_mz, n_sp))
    for scan, sp in sim_ms_data.get_spectra_iterator():
        sp = sim_ms_data.get_spectrum(scan)
        rt[scan] = sp.time
        chromatogram[:, scan] = sp.spint

    expected_chromatograms = [Chromatogram(rt, x) for x in chromatogram]
    test_chromatograms = ms.make_chromatograms(sim_ms_data, mz_list)
    assert len(test_chromatograms) == len(expected_chromatograms)
    for ec, tc in zip(expected_chromatograms, test_chromatograms):
        assert np.array_equal(ec.time, tc.time)
        assert np.array_equal(ec.spint, tc.spint)


def test_make_chromatograms_accumulator_mean(sim_ms_data):
    ms.make_chromatograms(sim_ms_data, mz_list, accumulator="mean")
    assert True


def test_make_tic(sim_ms_data):
    ms.make_tic(sim_ms_data, kind="tic")
    assert True


def test_make_tic_bpi(sim_ms_data):
    ms.make_tic(sim_ms_data, kind="bpi")
    assert True


# Test _RoiMaker


def test_TempRoi_creation():
    temp_roi = ms.raw_data_utils._TempRoi()
    assert len(temp_roi.mz) == 0
    assert len(temp_roi.spint) == 0
    assert len(temp_roi.scan) == 0


def test_TempRoi_append():
    mz, sp, scan = 150.0, 200.0, 10
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(mz, sp, scan)

    # check append
    assert temp_roi.mz[-1] == mz
    assert temp_roi.spint[-1] == sp
    assert temp_roi.scan[-1] == scan


def test_TempRoi_clear():
    mz, sp, scan = 150.0, 200.0, 10
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(mz, sp, scan)
    temp_roi.clear()

    # check empty ROI
    assert len(temp_roi.mz) == 0
    assert len(temp_roi.spint) == 0
    assert len(temp_roi.scan) == 0


def test_TempRoi_pad():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 2)
    temp_roi.append(1, 1, 3)
    scans = np.array([0, 1, 2, 3, 4, 5])
    temp_roi.pad(2, scans)
    assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1, np.nan, np.nan]
    assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1, np.nan, np.nan]
    assert list(temp_roi.scan) == list(scans)


def test_TempRoi_pad_crop_left_pad():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 1)
    temp_roi.append(1, 1, 2)
    scans = np.array([0, 1, 2, 3, 4, 5])
    temp_roi.pad(2, scans)
    # 0 is the minimum scan number, only one element should be padded to the
    # left
    assert list(temp_roi.mz) == [np.nan, 1, 1, np.nan, np.nan]
    assert list(temp_roi.spint) == [np.nan, 1, 1, np.nan, np.nan]
    assert list(temp_roi.scan) == list([0, 1, 2, 3, 4])


def test_TempRoi_pad_no_left_pad():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 0)
    temp_roi.append(1, 1, 1)
    scans = np.arange(6)
    temp_roi.pad(2, scans)
    # 0 is the minimum scan number, no elements should be padded to the left
    assert list(temp_roi.mz) == [1, 1, np.nan, np.nan]
    assert list(temp_roi.spint) == [1, 1, np.nan, np.nan]
    assert list(temp_roi.scan) == list([0, 1, 2, 3])


def test_TempRoi_pad_crop_right_pad():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 3)
    temp_roi.append(1, 1, 4)
    scans = np.arange(6)
    temp_roi.pad(2, scans)
    # 5 is the maximum scan number, right pad should add only one element
    assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1, np.nan]
    assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1, np.nan]
    assert list(temp_roi.scan) == list([1, 2, 3, 4, 5])


def test_TempRoi_pad_no_right_pad():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 4)
    temp_roi.append(1, 1, 5)
    scans = np.arange(5)
    temp_roi.pad(2, scans)
    # 5 is the maximum scan number, no elements should be padded to the right
    assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1]
    assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1]
    assert list(temp_roi.scan) == list([2, 3, 4, 5])


def test_TempRoi_convert_to_roi_lc_roi():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 2)
    temp_roi.append(1, 1, 4)
    temp_roi.append(1, 1, 5)
    scans = np.arange(6)
    time = np.arange(6)
    roi = temp_roi.convert_to_roi(time, scans, "uplc")

    assert np.array_equal(roi.scan, [2, 3, 4, 5])
    assert np.allclose(roi.time, [2, 3, 4, 5])
    assert np.allclose(roi.mz, [1, np.nan, 1, 1], equal_nan=True)
    assert np.allclose(roi.spint, [1, np.nan, 1, 1], equal_nan=True)


def test_TempRoi_convert_to_roi_di_roi():
    temp_roi = ms.raw_data_utils._TempRoi()
    temp_roi.append(1, 1, 2)
    temp_roi.append(1, 1, 4)
    temp_roi.append(1, 1, 5)
    scans = np.arange(6)
    time = np.arange(6)
    with pytest.raises(NotImplementedError):
        # TODO: FIX after implementing DI ROI creation
        temp_roi.convert_to_roi(time, scans, "di")


def test_RoiList_creation():
    ms.raw_data_utils._TempRoiList()
    assert True


def test_RoiList_insert_empty_RoiList():
    roi_list = ms.raw_data_utils._TempRoiList()
    n = 10
    scan = 1
    mz = np.arange(n)
    spint = np.arange(n)
    roi_list.insert(mz, spint, scan)

    # check tracking values
    assert np.array_equal(mz, roi_list.mz_mean)
    assert np.array_equal(mz, roi_list.mz_sum)
    assert np.array_equal(spint, roi_list.max_int)
    assert (roi_list.length == 1).all()
    assert (roi_list.missing_count == 0).all()

    for r, r_mz, r_sp in zip(roi_list.roi, mz, spint):
        assert len(r.mz) == 1
        assert len(r.spint) == 1
        assert len(r.scan) == 1
        assert r.mz[-1] == r_mz
        assert r.spint[-1] == r_sp
        assert r.scan[-1] == scan


def test_RoiList_two_consecutive_insert():
    roi_list = ms.raw_data_utils._TempRoiList()
    # first insert
    scan1 = 1
    mz1 = np.array([0, 1, 2, 3, 4, 6, 7, 9])  # 5 and 8 are missing
    spint1 = mz1.copy()
    roi_list.insert(mz1, spint1, scan1)

    # second insert
    scan2 = 2
    mz2 = np.array([5, 8])
    spint2 = mz2.copy()
    roi_list.insert(mz2, spint2, scan2)

    # check tracking values
    expected_mz = np.arange(10)
    expected_missing_count = np.zeros_like(roi_list.missing_count)
    expected_missing_count[mz1] = 1
    assert np.allclose(roi_list.mz_mean, expected_mz)
    assert np.allclose(roi_list.mz_sum, expected_mz)
    assert np.allclose(roi_list.max_int, expected_mz)
    assert (roi_list.length == 1).all()
    assert (roi_list.missing_count == 0).all()
    assert (np.diff(roi_list.mz_mean) >= 0).all()  # check is sorted

    # check roi values
    # values for mz and spint should be 0, 1, 2, 3, ...
    for expected, r in enumerate(roi_list.roi):
        assert r.mz[-1] == expected
        assert r.spint[-1] == expected
        if expected in [5, 8]:
            assert r.scan[-1] == scan2
        else:
            assert r.scan[-1] == scan1
        assert len(r.mz) == 1
        assert len(r.spint) == 1
        assert len(r.scan) == 1


def test_RoiList_extend_update_means_true():
    roi_list = ms.raw_data_utils._TempRoiList(update_mean=True)
    scan1 = 1
    mz1 = np.array([0, 1, 2, 3, 4], dtype=float)
    spint1 = mz1.copy()
    roi_list.insert(mz1, spint1, scan1)

    mz_extend = np.array([2, 2], dtype=float)
    sp_extend = np.array([10, 10], dtype=float)
    extend_index = np.array([1, 3])
    scan_extend = 2
    roi_list.extend(mz_extend, sp_extend, scan_extend, extend_index)

    # length increases only for extended roi
    expected_length = np.array([1, 2, 1, 2, 1])
    assert np.array_equal(roi_list.length, expected_length)
    # if update_mea is True, mz_sum and mz_mean are updated for extended roi
    expected_mz_sum = mz1.copy()
    expected_mz_sum[extend_index] += mz_extend
    assert np.allclose(roi_list.mz_sum, expected_mz_sum)
    expected_mz_mean = expected_mz_sum.copy()
    expected_mz_mean[extend_index] /= 2
    assert np.allclose(roi_list.mz_mean, expected_mz_mean)
    # max int is updated for extended array
    expected_max_int = spint1.copy()
    expected_max_int[extend_index] = sp_extend
    assert np.allclose(roi_list.max_int, expected_max_int)
    # missing count increases for non-extended roi and is set to zero for
    # extended roi
    expected_missing_count = np.array([1, 0, 1, 0, 1])
    assert np.array_equal(roi_list.missing_count, expected_missing_count)

    # check values in ROIs
    for k, mz, sp in zip(extend_index, mz_extend, sp_extend):
        r = roi_list.roi[k]
        assert len(r.mz) == 2
        assert r.mz[-1] == mz
        assert r.spint[-1] == sp
        assert r.scan[-1] == scan_extend

    for k in range(len(roi_list.roi)):
        r = roi_list.roi[k]
        if k not in extend_index:
            assert len(r.mz) == 1
            assert r.mz[-1] == mz1[k]
            assert r.spint[-1] == spint1[k]
            assert r.scan[-1] == scan1


def test_RoiList_extend_update_means_false():
    roi_list = ms.raw_data_utils._TempRoiList(update_mean=False)
    scan1 = 1
    mz1 = np.array([0, 1, 2, 3, 4], dtype=float)
    spint1 = mz1.copy()
    roi_list.insert(mz1, spint1, scan1)

    mz_extend = np.array([2, 2], dtype=float)
    sp_extend = np.array([10, 10], dtype=float)
    extend_index = np.array([1, 3])
    scan_extend = 2
    roi_list.extend(mz_extend, sp_extend, scan_extend, extend_index)

    # length increases only for extended roi
    expected_length = np.array([1, 2, 1, 2, 1])
    assert np.array_equal(roi_list.length, expected_length)
    # if update_mea is False, mz_sum is ignored and mz_mean remains the same
    # after insert.
    expected_mz_mean = mz1
    assert np.allclose(roi_list.mz_mean, expected_mz_mean)
    # max int is updated for extended array
    expected_max_int = spint1.copy()
    expected_max_int[extend_index] = sp_extend
    assert np.allclose(roi_list.max_int, expected_max_int)
    # missing count increases for non-extended roi and is set to zero for
    # extended roi
    expected_missing_count = np.array([1, 0, 1, 0, 1])
    assert np.array_equal(roi_list.missing_count, expected_missing_count)

    # check values in ROIs
    for k, mz, sp in zip(extend_index, mz_extend, sp_extend):
        r = roi_list.roi[k]
        assert len(r.mz) == 2
        assert r.mz[-1] == mz
        assert r.spint[-1] == sp
        assert r.scan[-1] == scan_extend

    for k in range(len(roi_list.roi)):
        r = roi_list.roi[k]
        if k not in extend_index:
            assert len(r.mz) == 1
            assert r.mz[-1] == mz1[k]
            assert r.spint[-1] == spint1[k]
            assert r.scan[-1] == scan1


def test_RoiList_clear():
    roi_list = ms.raw_data_utils._TempRoiList(update_mean=True)
    scan1 = 1
    mz1 = np.array([0, 1, 2, 3, 4], dtype=float)
    spint1 = mz1.copy()
    roi_list.insert(mz1, spint1, scan1)
    clear_index = np.array([1, 3])
    roi_list.clear(clear_index)
    # cleared roi length is set to zero
    expected_length = np.array([1, 0, 1, 0, 1])
    assert np.array_equal(roi_list.length, expected_length)
    # cleared roi missing count is set to zero
    assert (roi_list.missing_count == 0).all()
    # max int is set to zero
    expected_max_int = spint1.copy()
    expected_max_int[clear_index] = 0
    assert np.allclose(roi_list.max_int, expected_max_int)
    # mz sum is set to zero
    expected_mz_sum = mz1.copy()
    expected_mz_sum[clear_index] = 0
    assert np.allclose(roi_list.mz_sum, expected_mz_sum)
    # mz mean is not modified by clear
    assert np.allclose(roi_list.mz_mean, mz1)
    for k, r in enumerate(roi_list.roi):
        if k in clear_index:
            assert len(r.mz) == 0
            assert len(r.spint) == 0
            assert len(r.scan) == 0
        else:
            assert len(r.mz) == 1
            assert len(r.spint) == 1
            assert len(r.scan) == 1


def test_filter_invalid_mz_all_valid_mz():
    valid_mz = np.arange(10).astype(float)
    tol = 0.1
    mz = np.array([5.0, 6.0, 7.0])
    spint = np.ones_like(mz)
    mz_filtered, spint_filtered = ms.raw_data_utils._filter_invalid_mz(
        valid_mz, mz, spint, tol
    )
    assert mz_filtered.size == spint_filtered.size


def test_filter_invalid_mz_one_invalid_mz():
    valid_mz = np.arange(10).astype(float)
    tol = 0.1
    mz = np.array([5.0, 6.2, 7.0])
    spint = np.ones_like(mz)
    mz_filtered, spint_filtered = ms.raw_data_utils._filter_invalid_mz(
        valid_mz, mz, spint, tol
    )
    assert spint_filtered.size == 2
    assert np.allclose(mz_filtered, [5.0, 7.0])


def test_filter_invalid_mz_all_invalid_mz():
    valid_mz = np.arange(10).astype(float)
    tol = 0.1
    mz = np.array([5.2, 6.2, 7.2])
    spint = np.ones_like(mz)
    mz_filtered, spint_filtered = ms.raw_data_utils._filter_invalid_mz(
        valid_mz, mz, spint, tol
    )
    assert mz_filtered.size == 0
    assert spint_filtered.size == 0


def test_filter_invalid_mz_empty_array():
    valid_mz = np.arange(10).astype(float)
    tol = 0.1
    mz = np.array([])
    spint = np.array([])
    mz_filtered, spint_filtered = ms.raw_data_utils._filter_invalid_mz(
        valid_mz, mz, spint, tol
    )
    assert mz_filtered.size == 0
    assert spint_filtered.size == 0


def test_RoiProcessor_creation():
    mz_seed = np.arange(10)
    max_missing = 1
    min_length = 10
    tolerance = 0.01
    min_intensity = 100
    multiple_match = "reduce"
    mz_reduce = np.mean
    sp_reduce = np.sum
    processor = ms.raw_data_utils._RoiMaker(
        mz_seed,
        max_missing,
        min_length,
        min_intensity,
        tolerance,
        multiple_match,
        mz_reduce,
        sp_reduce,
    )
    assert np.array_equal(processor.mz_filter, mz_seed)
    assert processor.max_missing == max_missing
    assert processor.min_length == min_length
    assert processor.min_intensity == min_intensity
    assert processor.tolerance == tolerance
    assert processor.multiple_match == multiple_match
    assert processor.mz_reduce == mz_reduce
    assert processor.sp_reduce == sp_reduce


@pytest.fixture
def roi_processor():
    mz_seed = None
    max_missing = 1
    min_length = None
    tolerance = 0.01
    min_intensity = None
    multiple_match = "reduce"
    mz_reduce = np.mean
    sp_reduce = np.sum
    return ms.raw_data_utils._RoiMaker(
        mz_seed,
        max_missing,
        min_length,
        min_intensity,
        tolerance,
        multiple_match,
        mz_reduce,
        sp_reduce,
    )


def test_RoiProcessor_feed_spectrum_empty_processor_no_mz_filter(roi_processor):
    n = 5
    mz = np.arange(n)
    sp = np.arange(n)
    scan = 1
    roi_processor.feed_spectrum(mz, sp, scan)

    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, mz)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, sp)
    assert (roi_processor.tmp_roi_list.length == 1).all()
    assert (roi_processor.tmp_roi_list.missing_count == 0).all()


def test_RoiProcessor_feed_spectrum_empty_processor_mz_filter(roi_processor):
    mz_filter = np.array([0, 1, 2, 3])
    roi_processor.mz_filter = mz_filter
    n = 5
    mz = np.arange(n)
    sp = np.arange(n)
    scan = 1
    roi_processor.feed_spectrum(mz, sp, scan)

    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, mz_filter)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, mz_filter)
    assert (roi_processor.tmp_roi_list.length == 1).all()
    assert (roi_processor.tmp_roi_list.missing_count == 0).all()


def test_RoiProcessor_feed_spectrum_no_mz_filter(roi_processor):
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)

    mz2 = np.array([3, 4, 5, 6])
    sp2 = mz2.copy()
    scan2 = 2
    roi_processor.feed_spectrum(mz2, sp2, scan2)

    expected_mz_mean = np.arange(7)
    expected_max_int = expected_mz_mean
    expected_missing_count = np.array([1, 1, 1, 0, 0, 0, 0])
    expected_length = np.array([1, 1, 1, 2, 2, 1, 1])

    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, expected_mz_mean)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, expected_max_int)
    assert np.array_equal(roi_processor.tmp_roi_list.length, expected_length)
    assert np.array_equal(roi_processor.tmp_roi_list.missing_count, expected_missing_count)


def test_RoiProcessor_feed_spectrum_mz_filter(roi_processor):
    mz_filter = np.array([1, 2, 3, 4])
    roi_processor.mz_filter = mz_filter

    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)

    mz2 = np.array([3, 4, 5, 6])
    sp2 = mz2.copy()
    scan2 = 2
    roi_processor.feed_spectrum(mz2, sp2, scan2)

    expected_mz_mean = np.array([1, 2, 3, 4])
    expected_max_int = expected_mz_mean
    expected_missing_count = np.array([1, 1, 0, 0])
    expected_length = np.array([1, 1, 2, 2])

    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, expected_mz_mean)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, expected_max_int)
    assert np.array_equal(roi_processor.tmp_roi_list.length, expected_length)
    assert np.array_equal(roi_processor.tmp_roi_list.missing_count, expected_missing_count)


def test_RoiProcessor_clear_completed_roi_min_intensity_None_min_length_None(roi_processor):
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)

    mz2 = np.array([3, 4, 5, 6])
    sp2 = mz2.copy()
    scan2 = 2
    roi_processor.feed_spectrum(mz2, sp2, scan2)
    roi_processor.feed_spectrum(mz2, sp2, scan2 + 1)

    roi_processor.clear_completed_roi()

    expected_mz_mean = np.arange(7)
    expected_max_int = np.array([0, 0, 0, 3, 4, 5, 6])
    expected_missing_count = np.array([0, 0, 0, 0, 0, 0, 0])
    expected_length = np.array([0, 0, 0, 3, 3, 2, 2])
    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, expected_mz_mean)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, expected_max_int)
    assert np.array_equal(roi_processor.tmp_roi_list.length, expected_length)
    assert np.array_equal(roi_processor.tmp_roi_list.missing_count, expected_missing_count)
    assert len(roi_processor.valid_roi) == 3


def test_RoiProcessor_clear_completed_roi_min_intensity_int_min_length_None(roi_processor):
    roi_processor.min_intensity = 2
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)

    mz2 = np.array([3, 4, 5, 6])
    sp2 = mz2.copy()
    scan2 = 2
    roi_processor.feed_spectrum(mz2, sp2, scan2)
    roi_processor.feed_spectrum(mz2, sp2, scan2 + 1)

    roi_processor.clear_completed_roi()

    expected_mz_mean = np.arange(7)
    expected_max_int = np.array([0, 0, 0, 3, 4, 5, 6])
    expected_missing_count = np.array([0, 0, 0, 0, 0, 0, 0])
    expected_length = np.array([0, 0, 0, 3, 3, 2, 2])
    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, expected_mz_mean)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, expected_max_int)
    assert np.array_equal(roi_processor.tmp_roi_list.length, expected_length)
    assert np.array_equal(roi_processor.tmp_roi_list.missing_count, expected_missing_count)
    # using a min intensity filter of 2, only the ROI generated from mz1 = 2,
    # sp2 = 2 should be valid
    assert len(roi_processor.valid_roi) == 1
    roi = roi_processor.valid_roi[0]
    assert np.allclose(roi.mz[0], 2.0)
    assert np.allclose(roi.spint[0], 2.0)
    assert np.allclose(roi.scan[0], 1)


def test_RoiProcessor_clear_completed_roi_min_intensity_int_min_length_int(roi_processor):
    roi_processor.min_intensity = 2
    roi_processor.min_length = 2
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)

    mz2 = np.array([3, 4, 5, 6])
    sp2 = mz2.copy()
    scan2 = 2
    roi_processor.feed_spectrum(mz2, sp2, scan2)
    roi_processor.feed_spectrum(mz2, sp2, scan2 + 1)

    roi_processor.clear_completed_roi()

    expected_mz_mean = np.arange(7)
    expected_max_int = np.array([0, 0, 0, 3, 4, 5, 6])
    expected_missing_count = np.array([0, 0, 0, 0, 0, 0, 0])
    expected_length = np.array([0, 0, 0, 3, 3, 2, 2])
    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, expected_mz_mean)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, expected_max_int)
    assert np.array_equal(roi_processor.tmp_roi_list.length, expected_length)
    assert np.array_equal(roi_processor.tmp_roi_list.missing_count, expected_missing_count)
    # using a min length filter of 2, all cleared ROI are invalid
    assert len(roi_processor.valid_roi) == 0


def test_RoiProcessor_flag_as_completed(roi_processor):
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)
    roi_processor.flag_as_completed()
    roi_processor.clear_completed_roi()

    # all ROI should be cleared
    assert np.allclose(roi_processor.tmp_roi_list.mz_mean, mz1)
    assert np.allclose(roi_processor.tmp_roi_list.max_int, np.zeros_like(mz1))
    assert (roi_processor.tmp_roi_list.length == 0).all()
    assert (roi_processor.tmp_roi_list.missing_count == 0).all()


def test_RoiProcessor_tmp_roi_to_roi(roi_processor: ms.raw_data_utils._RoiMaker):
    n = 5
    mz1 = np.arange(n)
    sp1 = np.arange(n)
    scan1 = 1
    roi_processor.feed_spectrum(mz1, sp1, scan1)
    roi_processor.feed_spectrum(mz1, sp1, scan1 + 1)
    roi_processor.flag_as_completed()
    roi_processor.clear_completed_roi()
    valid_scan = np.arange(n)
    time = np.arange(n)
    pad = 2
    separation = "uplc"
    smoothing_strength = None
    roi_list = roi_processor.tmp_roi_to_roi(
        valid_scan, time, pad, smoothing_strength, separation
    )
    assert len(roi_list) == 5
    for r in roi_list:
        assert np.isnan(r.mz).sum() == 0
        assert np.isnan(r.spint).sum() == 0
        assert np.allclose(r.scan, [0, 1, 2, 3, 4])
        assert np.allclose(r.time, np.arange(5).astype(float))


def test_match_mz_no_multiple_matches():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([40, 51, 78, 91, 99, 130, 150])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([0, 2, 4], dtype=int)
    mz2_match_index = np.array([1, 4, 6], dtype=int)
    mz2_no_match_index = np.array([0, 2, 3, 5], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = _match_mz(
        mz1, mz2, sp2, tolerance, mode, np.mean, np.mean
    )
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_no_matches():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([40, 53, 78, 91, 97, 130, 154])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([], dtype=int)
    mz2_match_index = np.array([], dtype=int)
    mz2_no_match_index = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = _match_mz(
        mz1, mz2, sp2, tolerance, mode, np.mean, np.mean
    )
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_all_match():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([51, 77, 99, 126, 150])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    mz1_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
    mz2_no_match_index = np.array([], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = _match_mz(
        mz1, mz2, sp2, tolerance, mode, np.mean, np.mean
    )
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_multiple_matches_mode_closest():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    # in `closest` mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
    mz2_match_index = np.array([0, 4, 6, 7], dtype=int)
    mz2_no_match_index = np.array([1, 2, 3, 5, 8], dtype=int)
    mode = "closest"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = _match_mz(
        mz1, mz2, sp2, tolerance, mode, np.mean, np.mean
    )
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.array_equal(mz2[mz2_match_index], mz2_match)
    assert np.array_equal(sp2[mz2_match_index], sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_multiple_matches_mode_reduce():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150], dtype=float)
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151], dtype=float)
    sp2 = np.array([100] * mz2.size, dtype=float)
    # expected values for match/no match indices
    # in `closest` mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
    mz2_no_match_index = np.array([2], dtype=int)
    expected_mz2_match = [50.0, 100.0, 126.0, 150.5]
    expected_sp2_match = [200, 300, 100, 200]
    mode = "reduce"
    test_mz1_index, mz2_match, sp2_match, mz2_no_match, sp2_no_match = _match_mz(
        mz1, mz2, sp2, tolerance, mode, np.mean, np.sum
    )
    # test match index
    assert np.array_equal(mz1_match_index, test_mz1_index)
    # test match mz and sp values
    assert np.allclose(mz2_match, expected_mz2_match)
    assert np.allclose(sp2_match, expected_sp2_match)
    # test no match mz and sp values
    assert np.array_equal(mz2[mz2_no_match_index], mz2_no_match)
    assert np.array_equal(sp2[mz2_no_match_index], sp2_no_match)


def test_match_mz_invalid_mode():
    tolerance = 2
    mz1 = np.array([50, 75, 100, 125, 150])
    mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
    sp2 = np.array([100] * mz2.size)
    # expected values for match/no match indices
    # in closest mode, argmin is used to select the closest value. If more
    # than one value has the same difference, the first one in the array is
    # going to be selected.
    mode = "invalid-mode"
    with pytest.raises(ValueError):
        _match_mz(mz1, mz2, sp2, tolerance, mode, np.mean, np.mean)
