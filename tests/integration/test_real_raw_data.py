"""Test lcms and fileio functionality with real data."""

import pytest
from tidyms.core.models import Chromatogram, MSSpectrum
from tidyms.io import MSData
from tidyms.io.datasets import get_dataset_file


class TestMZMLReader:
    @pytest.fixture(scope="class")
    def data_centroid(self) -> MSData:
        filename = "centroid-data-zlib-indexed-compressed.mzML"
        dataset = "test-raw-data"
        path = get_dataset_file(dataset, filename)
        return MSData(path)

    def test_get_n_spectra(self, data_centroid):
        data_centroid.get_n_spectra()

    def test_get_n_chromatograms(self, data_centroid):
        data_centroid.get_n_chromatograms()

    def test_get_spectrum(self, data_centroid: MSData):
        sp = data_centroid.get_spectrum(0)
        assert isinstance(sp, MSSpectrum)

    def test_get_all_spectra(self, data_centroid: MSData):
        expected_n_spectra = data_centroid.get_n_spectra()
        all_spectra = [data_centroid.get_spectrum(x) for x in range(expected_n_spectra)]

        assert len(all_spectra) == expected_n_spectra
        assert all(isinstance(x, MSSpectrum) for x in all_spectra)

    def test_get_chromatogram(self, data_centroid: MSData):
        chromatogram = data_centroid.get_chromatogram(0)
        assert isinstance(chromatogram, Chromatogram)

    def test_get_all_chromatograms(self, data_centroid: MSData):
        expected_n_chromatograms = data_centroid.get_n_chromatograms()
        all_spectra = [data_centroid.get_chromatogram(x) for x in range(expected_n_chromatograms)]

        assert len(all_spectra) == expected_n_chromatograms
        assert all(isinstance(x, Chromatogram) for x in all_spectra)
