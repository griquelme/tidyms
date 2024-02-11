"""Reader interface specification."""

import pathlib
from typing import IO, Protocol

from .models import Chromatogram, MSSpectrum


class Reader(Protocol):
    """Reader interface for raw data."""

    def __init__(self, src: pathlib.Path | IO):
        ...

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram from file."""
        ...

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from file."""
        ...

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatogram."""
        ...

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra."""
        ...
