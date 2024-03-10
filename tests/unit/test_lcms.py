from math import isclose

import numpy as np
import pydantic
import pytest
from tidyms.lcms import MZTrace, Peak


class TestMZTrace:

    trace_size = 20
    nan_index_list = [0, trace_size // 2, trace_size -1]

    @pytest.fixture
    def trace(self) -> MZTrace:
        scans = np.arange(self.trace_size)
        time = np.linspace(0, 20, self.trace_size)
        spint = time.copy()
        mz = np.ones_like(time)
        return MZTrace(scan=scans, mz=mz, time=time, spint=spint)

    def test_noise_with_negative_values_raise_error(self, trace):
        noise = np.ones_like(trace.time)
        noise[self.trace_size // 2] = -0.005
        with pytest.raises(pydantic.ValidationError):
            trace.noise = noise

    def test_baseline_higher_than_intensity_raises_error(self, trace: MZTrace):
        baseline = trace.spint.copy()
        baseline[10] += 0.01
        with pytest.raises(pydantic.ValidationError):
            trace.baseline = baseline

    @pytest.mark.parametrize("nan_index", nan_index_list)
    def test_fill_nan(self, trace, nan_index):
        trace.mz[nan_index] = np.nan
        trace.spint[nan_index] = np.nan
        assert np.any(np.isnan(trace.mz))
        assert np.any(np.isnan(trace.spint))
        trace.fill_nan()
        assert np.all(~np.isnan(trace.mz))
        assert np.all(~np.isnan(trace.spint))

    def test_serialize_deserialize_no_baseline(self, trace: MZTrace):
        expected = trace
        actual = MZTrace.from_str(trace.to_str())
        assert actual == expected

    def test_serialize_deserialize_with_baseline_and_noise(self, trace: MZTrace):
        expected = trace
        expected.noise = np.ones_like(trace.time)
        expected.baseline = trace.spint.copy()
        actual = MZTrace.from_str(trace.to_str())
        assert actual == expected


class TestPeak:

    trace_size = 201
    peak_start = 90
    peak_apex = 100
    peak_end = 111
    peak_mz = 200.0
    peak_height = 10.0
    noise_level = 2.0
    baseline_height = 1.0

    @pytest.fixture
    def trace(self) -> MZTrace:
        """Use a square pulse as a LC-MS peak for unit testing."""
        scans = np.arange(self.trace_size)
        time = np.linspace(0, self.trace_size - 1, self.trace_size)

        mz = np.ones_like(time) * self.peak_mz
        signal = np.zeros_like(time)
        signal[self.peak_start:self.peak_end] += self.peak_height
        baseline = np.ones_like(time) * self.baseline_height
        trace_int = signal + baseline
        noise = np.ones_like(time) * self.noise_level
        return MZTrace(scan=scans, mz=mz, time=time, spint=trace_int, noise=noise, baseline=baseline)

    @pytest.fixture
    def peak(self, trace: MZTrace) -> Peak:
        return Peak(start=self.peak_start, apex=self.peak_apex, end=self.peak_end, roi=trace)

    @pytest.mark.parametrize(
        "start,apex,end", [(0, 0, 10), (10, 5, 12), (5, 10, 10), (5, 10, 8), (0, 10, trace_size)]
    )
    def test_invalid_peak_definition_raises_error(self, trace, start, apex, end):
        with pytest.raises(pydantic.ValidationError):
            Peak(start=start, apex=apex, end=end, roi=trace)

    def test_serialize_deserialize(self, peak):
        expected = peak
        actual = Peak.from_str(expected.to_str(), expected.roi, expected.annotation)
        assert actual == expected

    def test_describe(self, peak):
        for k, v in peak.describe().items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_get_mz_ok(self, peak):
        assert isclose(peak.get("mz"), self.peak_mz)

    def test_get_mz_when_intensity_is_equal_to_baseline_returns_nan(self, peak):
        peak.roi.baseline = peak.roi.spint
        assert np.isnan(peak.get("mz"))

    def test_get_rt_start_ok(self, peak):
        assert isclose(peak.get("rt_start"), peak.roi.time[peak.start])

    def test_get_rt_end_ok(self, peak):
        assert isclose(peak.get("rt_end"), peak.roi.time[peak.end - 1])

    def test_get_rt_ok(self, peak):
        assert isclose(peak.get("rt"), peak.roi.time[peak.apex])

    def test_get_rt_when_intensity_is_equal_to_baseline_returns_nan(self, peak):
        peak.roi.baseline = peak.roi.spint
        assert np.isnan(peak.get("rt"))

    def test_get_height(self, peak):
        assert isclose(peak.get("height"), self.peak_height)

    def test_get_height_no_baseline(self, peak):
        peak.roi.baseline = None
        assert isclose(peak.get("height"), self.peak_height + self.baseline_height)

    def test_get_area_ok(self, peak):
        expected_area = (peak.roi.time[self.peak_end - 1] - peak.roi.time[self.peak_start]) * self.peak_height
        assert isclose(expected_area, peak.get("area"))

    def test_get_area_ok_no_baseline(self, peak):
        expected_area = (
            (peak.roi.time[self.peak_end - 1] - peak.roi.time[self.peak_start])
            * (self.peak_height + self.baseline_height)
        )
        peak.roi.baseline = None
        assert isclose(expected_area, peak.get("area"))

    def test_get_area_when_intensity_is_equal_to_baseline_returns_zero(self, peak):
        peak.roi.baseline = peak.roi.spint
        assert isclose(peak.get("area"), 0.0)

    def test_get_snr_ok(self, peak):
        assert isclose(peak.get("snr"), self.peak_height / self.noise_level)

    def test_get_snr_no_baseline_ok(self, peak):
        peak.roi.baseline = None
        assert peak.get("snr") > 0.0

    def test_get_snr_baseline_equal_to_intensity_returns_zero(self, peak):
        peak.roi.baseline = peak.roi.spint
        assert isclose(peak.get("snr"), 0.0)

    def test_get_snr_noise_level_equal_to_zero_returns_nan(self, peak):
        peak.roi.noise = np.zeros_like(peak.roi.time)
        assert np.isnan(peak.get("snr"))

    def test_get_snr_no_noise_level_returns_nan(self, peak):
        peak.roi.noise = None
        assert np.isnan(peak.get("snr"))

    def test_get_width_ok(self, peak):
        assert peak.get("width") > 0
        assert peak.get("width") <= peak.get("extension")

    def test_get_width_no_baseline_ok(self, peak):
        peak.roi.baseline = None
        assert peak.get("width") > 0
        assert peak.get("width") <= peak.get("extension")

    def test_get_width_baseline_equal_to_intensity_return_nan(self, peak):
        peak.roi.baseline = peak.roi.spint
        assert np.isnan(peak.get("width"))

    def test_compare_with_itself_returns_1(self, peak: Peak):
        actual = peak.compare(peak)
        expected = 1.0
        assert isclose(actual, expected)

    def test_compare_non_overlapping_peaks_return_0(self, peak: Peak):
        length = peak.end - peak.start + 1
        other = peak.model_copy()
        other.end += length
        other.apex += length
        other.start += length
        expected = 0.0
        assert isclose(peak.compare(other), expected)
        assert isclose(other.compare(peak), expected)


    @pytest.mark.parametrize("shift", [0, 1, 2, 3, 4, 5])
    def test_compare_is_symmetric(self, peak: Peak, shift):
        other = peak.model_copy()
        other.end += -shift
        other.start += shift
        similarity = peak.compare(other)
        assert isclose(similarity, other.compare(peak))
        assert 0.0 <= similarity
        assert (similarity < 1.0) or isclose(similarity, 1.0)

    @pytest.mark.parametrize("shift", [0, 1, 2, 3, 4, 5])
    def test_compare_is_in_valid_range(self, peak: Peak, shift):
        other = peak.model_copy()
        other.end += -shift
        other.start += shift
        similarity = peak.compare(other)
        assert 0.0 <= similarity
        assert (similarity < 1.0) or isclose(similarity, 1.0)
