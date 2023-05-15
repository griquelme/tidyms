import numpy as np
import tidyms as ms


def test_get_fill_area_no_peaks_detected(monkeypatch):
    time = np.arange(100)
    spint = np.ones_like(time)
    chromatogram = ms.Chromatogram(time, spint)
    rt = 50
    rt_std = 10
    n_dev = 1

    def mock_extract_features(self, **kwargs):
        self.features = list()

    monkeypatch.setattr(ms.Chromatogram, "extract_features", mock_extract_features)

    area = ms.fill_missing._get_fill_area(chromatogram, rt, rt_std, n_dev)
    assert area is None


def test_get_fill_area_peak_detected_outside_valid_range(monkeypatch):
    time = np.arange(100)
    spint = np.ones_like(time)
    noise = np.zeros_like(time)
    chromatogram = ms.Chromatogram(time, spint)
    chromatogram.noise = noise
    chromatogram.baseline = noise
    rt = 50
    rt_std = 10
    n_dev = 1

    def mock_extract_features(self, **kwargs):
        self.features = [ms.lcms.Peak(70, 75, 80, self)]

    monkeypatch.setattr(ms.Chromatogram, "extract_features", mock_extract_features)

    area = ms.fill_missing._get_fill_area(chromatogram, rt, rt_std, n_dev)
    assert area is None


def test_get_fill_area_peak_detected_inside_valid_range(monkeypatch):
    time = np.arange(100)
    spint = np.ones_like(time)
    chromatogram = ms.Chromatogram(time, spint)
    chromatogram.baseline = np.zeros_like(time)
    rt = 50
    rt_std = 10
    n_dev = 1
    test_peak = ms.lcms.Peak(50, 55, 60, chromatogram)
    expected_area = test_peak.get_area()

    def mock_extract_features(self, **kwargs):
        self.features = [test_peak]

    monkeypatch.setattr(ms.Chromatogram, "extract_features", mock_extract_features)

    area = ms.fill_missing._get_fill_area(chromatogram, rt, rt_std, n_dev)
    assert np.isclose(area, expected_area)


def test_get_fill_area_multiple_valid_peaks_choose_closest(monkeypatch):
    time = np.arange(100)
    spint = np.ones_like(time)
    chromatogram = ms.Chromatogram(time, spint)
    chromatogram.baseline = np.zeros_like(time)
    rt = 50
    rt_std = 10
    n_dev = 1
    valid_peak = ms.lcms.Peak(45, 50, 52, chromatogram)
    detected_peaks = [valid_peak, ms.lcms.Peak(55, 60, 65, chromatogram)]
    expected_area = valid_peak.get_area()

    def mock_extract_features(self, **kwargs):
        self.features = detected_peaks

    monkeypatch.setattr(ms.Chromatogram, "extract_features", mock_extract_features)

    area = ms.fill_missing._get_fill_area(chromatogram, rt, rt_std, n_dev)
    assert np.isclose(area, expected_area)
