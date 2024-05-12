"""Utilities to simulate LC-MS data.

Provides:

SimulatedLCMSDataReader
    A Reader that can be plugged into MSData to generate simulated LC-MS data files.
SimulatedLCMSSampleFactory
    A pydantic model that creates simulated sample data configuration.

"""

from __future__ import annotations

import pathlib
from io import StringIO
from typing import BinaryIO, TextIO

import numpy as np
import pydantic
from typing_extensions import Self

from ..chem import EM, Formula
from ..core.models import Chromatogram, MSSpectrum
from ..core.numpy import FloatArray1D
from ..io.readers.registry import register


@register()
class SimulatedLCMSDataReader:
    """Read simulated LC-MS data files."""

    def __init__(self, src: pathlib.Path | TextIO | BinaryIO) -> None:
        if isinstance(src, StringIO):
            self.sample = SimulatedLCMSSample.model_validate_json(src.read())
        elif isinstance(src, pathlib.Path):
            self.sample = SimulatedLCMSSample.from_json(src)
        else:
            raise NotImplementedError
        self.spectrum_factory = MSSpectrumFactory(self.sample)

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram from file."""
        raise NotImplementedError

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from file."""
        return self.spectrum_factory.create(index)

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatograms."""
        raise NotImplementedError

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra."""
        return self.sample.config.n_scans


class SimulatedLCMSDataConfiguration(pydantic.BaseModel):
    """Store configuration of a simulated LC-MS sample."""

    grid: MZGridSpecification | None = None
    """The m/z grid specification. If not specified, a grid is created using
    features m/z
    """

    mz_noise: pydantic.NonNegativeFloat = 0.001
    """Additive noise added to m/z in each scan"""

    amp_noise: pydantic.NonNegativeFloat = 10.0

    mz_width: pydantic.PositiveFloat = 0.005
    """The peak width in the m/z domain"""

    n_scans: pydantic.PositiveInt = 500
    """The number of scans in the sample"""

    time_resolution: pydantic.PositiveFloat = 1.0
    """The time spacing between scans"""

    min_signal_intensity: pydantic.PositiveFloat | None = None
    """If specified, elements in a spectrum with values lower than this parameter are removed"""

    signal_noise: pydantic.PositiveFloat = 10.0
    """Additive noise added to the spectral intensity in each scan"""

    ms_level: pydantic.PositiveInt = 1
    """The spectra MS level"""


class MZGridSpecification(pydantic.BaseModel):
    """Store specification to create an m/z grid."""

    low: pydantic.PositiveFloat = 100.0
    """The minimum value in the grid"""

    high: pydantic.PositiveFloat = 1200.0
    """The maximum value in the grid"""

    size: pydantic.PositiveInt = 10000
    """The number of elements in the grid"""

    def create(self) -> FloatArray1D:
        """Create a m/z grid."""
        return np.linspace(self.low, self.high, self.size)


class SimulatedLCMSSampleFactory(pydantic.BaseModel):
    """Utility that creates simulated data samples."""

    config: SimulatedLCMSDataConfiguration = SimulatedLCMSDataConfiguration()
    """The sample configuration used to simulate data."""

    adducts: list[SimulatedLCMSAdductSpecification] = list()
    """the list of adducts to include in the simulated sample."""

    def make(self):
        """Create a new simulated sample model."""
        features: list[SimulatedLCMSFeature] = list()
        for adduct in self.adducts:
            features.extend(adduct.create_features())
        features = sorted(features, key=lambda x: x.mz)
        return SimulatedLCMSSample(config=self.config, features=features)

    def make_file(self, path: pathlib.Path):
        """Create a new simulated sample file."""
        self.make().to_json(path)

    def make_fileobj(self):
        """Create a text file object."""
        return StringIO(self.make().model_dump_json())


class SimulatedLCMSSample(pydantic.BaseModel):
    """Create simulated LC-MS data files."""

    config: SimulatedLCMSDataConfiguration
    """The sample configuration used to simulate data."""

    features: list[SimulatedLCMSFeature] = list()
    """the list of features in the sample."""

    def make_grid(self) -> FloatArray1D:
        """Create a grid from features m/z values."""
        if self.config.grid is None:
            grid = np.array(sorted([x.mz for x in self.features]))
        else:
            grid = self.config.grid.create()
        return grid

    @classmethod
    def from_json(cls, path: pathlib.Path) -> Self:
        """Create a new instance from a JSON file."""
        with path.open("rt") as f:
            model_json = f.read()
        return cls.model_validate_json(model_json)

    def to_json(self, path: pathlib.Path):
        """Store the model as a JSON file."""
        with path.open("wt") as f:
            f.write(self.model_dump_json())


class SimulatedLCMSAdductSpecification(pydantic.BaseModel):
    """Define an adduct in a simulated sample."""

    formula: str
    """The adduct formula"""

    n_isotopologues: pydantic.PositiveInt = 5
    """The number of isotopologues to simulate."""

    rt_mean: pydantic.PositiveFloat
    """The adduct retention time"""

    rt_noise: pydantic.PositiveFloat | None = None
    """The standard deviation of a gaussian distribution used to define an additive
    retention time noise."""

    rt_width: pydantic.PositiveFloat = 3.0
    """The adduct peak width"""

    base_intensity: pydantic.PositiveFloat = 100.0
    """The base adduct intensity."""

    def create_features(self) -> list[SimulatedLCMSFeature]:
        """Create a list of simulated features to simulate a sample."""
        formula = Formula(self.formula)
        M, p = formula.get_isotopic_envelope(n=self.n_isotopologues)

        if self.rt_noise is None:
            rt_noise = 0.0
        else:
            rt_noise = np.random.normal(scale=self.rt_noise)

        rt = self.rt_mean + rt_noise

        mz = (M - formula.charge * EM) / abs(formula.charge)
        intensity = self.base_intensity * p

        features = list()
        for k in range(self.n_isotopologues):
            ft = SimulatedLCMSFeature(mz=mz[k].item(), rt=rt, int=intensity[k], width=self.rt_width)
            features.append(ft)
        return features


class SimulatedLCMSFeature(pydantic.BaseModel):
    """Store a simulated LC-MS peak information."""

    mz: pydantic.PositiveFloat
    """The feature m/z."""

    rt: pydantic.PositiveFloat
    """The feature retention time."""

    int: pydantic.PositiveFloat
    """the feature intensity."""

    width: pydantic.PositiveFloat
    """The peak width in the time domain"""


class MSSpectrumFactory:
    """Store m/z grid data."""

    def __init__(self, sample: SimulatedLCMSSample) -> None:
        self.sample = sample
        self.grid = sample.make_grid()
        self._is_centroid = sample.config.grid is not None
        scan = sample.config.n_scans
        self._random_seeds = np.random.choice(scan * 10, scan)

    def create(self, scan: int) -> MSSpectrum:
        """Create a ms spectrum instance."""
        assert scan < self.sample.config.n_scans, "`scan` must be lower than the sample `n_scans` parameter."

        mz = self._compute_mz(scan)
        sp = self._compute_intensity(mz, scan)

        if self.sample.config.min_signal_intensity is not None:
            mask = sp >= self.sample.config.min_signal_intensity
            mz = mz[mask]
            sp = sp[mask]

        time = self.sample.config.time_resolution * scan

        return MSSpectrum(
            index=scan, mz=mz, int=sp, ms_level=self.sample.config.ms_level, centroid=self._is_centroid, time=time
        )

    def _compute_mz(self, scan: int) -> FloatArray1D:
        # use the same random seed for a given scan for reproducibility
        seed = self._random_seeds[scan]
        np.random.seed(seed)

        # add random noise to the m/z grid
        noise_level = self.sample.config.mz_noise

        if noise_level > 0.0:
            noise = np.random.normal(size=self.grid.size, scale=noise_level)
            mz = self.grid + noise
        else:
            mz = self.grid.copy()

        return mz

    def _compute_intensity(self, mz: FloatArray1D, scan: int):
        time = self.sample.config.time_resolution * scan
        intensity = np.zeros_like(mz)
        mz_width = self.sample.config.mz_width
        for ft in self.sample.features:
            amp = ft.int * np.power(np.e, -0.5 * ((time - ft.rt) / ft.width) ** 2)
            intensity += amp * np.power(np.e, -0.5 * ((mz - ft.mz) / mz_width) ** 2)

        if self.sample.config.amp_noise > 0.0:
            intensity += np.random.normal(size=intensity.size, scale=self.sample.config.amp_noise)
            intensity[intensity < 0] = 0.0

        return intensity
