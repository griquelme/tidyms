"""General purpose raw data reader."""

from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import IO, Generator

from .constants import MSDataMode
from .models import Chromatogram, MSSpectrum
from .reader import Reader
from .registry import get_reader_type


class MSData:
    """
    Provide access to raw MS data.

    Data is read from disk in a lazy manner and cached in memory.

    Parameters
    ----------
    src : pathlib.Path or IO
        Raw data source.
    reader : Reader or None, default=None
        The Reader class to read raw data. If ``None``, the reader is inferred
        using the file extension.
    centroid : MSDataMode, default=MSDataMode.CENTROID
        The mode in which the data is stored.
    cache : int, default=-1
        The maximum size of the cache, in bytes. The cache will store spectrum
        data until it surpasses this value. At this point, old entries will be
        deleted from the cache. If set ``-1``, the cache can grow indefinitely.

    """

    def __init__(
        self,
        src: pathlib.Path | IO,
        reader: Reader | None = None,
        centroid: MSDataMode = MSDataMode.CENTROID,
        cache: int = -1
    ):

        if isinstance(src, pathlib.Path):
            ext = src.suffix
            reader_type = get_reader_type(ext)
            reader = reader_type(src)
        elif reader is None:
                msg = "Reader must be specified for file objects."
                raise ValueError(msg)
        self._reader = reader
        self._centroid = centroid
        self._cache = MSDataCache(max_size=cache)
        self._n_spectra: int | None = None
        self._n_chromatogram: int | None = None

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatograms stored in the source."""
        if self._n_chromatogram is None:
            self._n_chromatogram = self._reader.get_n_chromatograms()
        return self._n_chromatogram

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra stored in the source."""
        if self._n_spectra is None:
            self._n_spectra = self._reader.get_n_spectra()
        return self._n_spectra

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram by index."""
        return self._reader.get_chromatogram(index)

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum by index."""
        spectrum = self._cache.get(index)
        if spectrum is None:
            spectrum = self._reader.get_spectrum(index)
            spectrum.centroid = self._centroid == MSDataMode.CENTROID
            self._cache.add(spectrum)
        return spectrum

    def iterate(
        self,
        ms_level: int = 1,
        start: int = 0,
        end: int | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Generator[MSSpectrum, None, None]:
        """
        Iterate over all spectra in the data.

        Generated spectra are sorted by index.

        Parameters
        ----------
        ms_level : int = 1
            Skip spectra without this MS level.
        start : int = 0
            Start iterating at this index.
        end : int or None, default=None
            Stop iteration at this index value.
        start_time: float = 0.0
            Skip spectra with time lower than this value.
        end_time: float or None, default=None
            Skip spectra with time greater than this value.

        """
        n_spectra = self.get_n_spectra()
        if end is None:
            end = n_spectra
        elif end > n_spectra:
            end = n_spectra

        for k in range(start, end):
            sp = self.get_spectrum(k)
            if (ms_level == sp.ms_level) and (start_time <= sp.time):
                if (end_time is None) or (end_time > sp.time):
                    yield sp


class MSDataCache:
    """
    Cache spectra data to avoid reading from disk.

    Old entries are deleted if the cache grows larger than total data size in
    bytes. The maximum size of the cache is defined by `max_size`. If set to
    ``-1``, the cache can grow indefinitely.

    """

    def __init__(self, max_size: int = -1):
         self.cache: OrderedDict[int, MSSpectrum] = OrderedDict()
         self.size = 0
         self.max_size = max_size

    def add(self, spectrum: MSSpectrum) -> None:
        """Store a spectrum."""
        self.cache[spectrum.index] = spectrum
        self.size += _get_spectrum_size(spectrum)
        self.clean()

    def get(self, index: int) -> MSSpectrum | None:
        """Retrieve a spectrum from the cache. If not found, returns ``None``."""
        spectrum = self.cache.get(index)
        if isinstance(spectrum, MSSpectrum):
             self.cache.move_to_end(index)
        return spectrum

    def clean(self) -> None:
        """Delete entries until the cache size is lower than max_size."""
        if self.max_size > -1:
            while self.size > self.max_size:
                _, spectrum = self.cache.popitem(last=False)
                self.size -= _get_spectrum_size(spectrum)


def _get_spectrum_size(spectrum: MSSpectrum) -> int:
     return spectrum.int.nbytes + spectrum.mz.nbytes
