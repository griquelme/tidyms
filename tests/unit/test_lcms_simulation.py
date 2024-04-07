import pathlib

import numpy as np
import pytest
from tidyms.core.models import MSSpectrum
from tidyms.io.msdata import MSData
from tidyms.io.readers.registry import get_reader
from tidyms.lcms import simulation
from tidyms.utils import find_closest


@pytest.fixture
def factory():
    config = simulation.SimulatedLCMSDataConfiguration(min_signal_intensity=1.0, n_scans=40)
    formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
    rt_list = [10.0, 20.0, 30.0]
    int_list = [100.0, 200.0, 300.0]
    adduct_list = list()
    for rt, spint, formula in zip(rt_list, int_list, formula_list):
        adduct = simulation.SimulatedLCMSAdductSpecification(
            formula=formula, rt_mean=rt, base_intensity=spint, n_isotopologues=1
        )
        adduct_list.append(adduct)
    return simulation.SimulatedLCMSSampleFactory(config=config, adducts=adduct_list)

@pytest.fixture
def factory_with_grid(factory: simulation.SimulatedLCMSSampleFactory):
    factory.config.grid = simulation.MZGridSpecification()
    return factory

class TestSimulatedLCMSSample:
    def test_make_grid_no_spec_ok(self, factory: simulation.SimulatedLCMSSampleFactory):
        sample = factory.make()
        grid = sample.make_grid()
        assert grid.size == len(sample.features)
        assert np.all(np.diff(grid) > 0.0)

    def test_make_grid_with_spec(self, factory_with_grid: simulation.SimulatedLCMSSampleFactory):
        sample = factory_with_grid.make()
        grid = sample.make_grid()
        assert sample.config.grid is not None
        assert grid.size == sample.config.grid.size
        assert np.all(np.diff(grid) > 0.0)

    def test_make_grid_no_features_return_empty_array(self):
        factory = simulation.SimulatedLCMSSampleFactory()
        sample = factory.make()
        grid = sample.make_grid()
        assert grid.size == 0


class TestSimulatedMSData:

    @pytest.fixture
    def data(self, factory: simulation.SimulatedLCMSSampleFactory, tmp_path: pathlib.Path):
        reader = get_reader("SimulatedLCMSDataReader")
        file = tmp_path / "data.json"
        factory.make_file(file)
        return MSData(file, reader=reader)

    def test_get_spectrum(self, data: MSData):
        index = 0
        sp = data.get_spectrum(index)
        assert isinstance(sp, MSSpectrum)
        assert sp.index == index

    def test_get_spectrum_negative_index_raises_value_error(self, data: MSData):
        with pytest.raises(ValueError):
            data.get_spectrum(-1)

    def test_get_spectrum_index_greater_than_size_raises_error(self, data: MSData):
        with pytest.raises(ValueError):
            index = 100000000
            data.get_spectrum(index)

    def test_spectrum_sanity(self, data: MSData):
        assert isinstance(data._reader, simulation.SimulatedLCMSDataReader)
        check_cond = False
        for sp in data.iterate():
            for feature in data._reader.sample.features:
                if np.isclose(sp.time, feature.rt):
                    closest = find_closest(sp.mz, feature.mz)
                    assert np.allclose(sp.mz[closest], feature.mz)
                    check_cond = True
        assert check_cond




