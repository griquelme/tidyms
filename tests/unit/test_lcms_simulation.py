import numpy as np
import pytest
from tidyms.core.models import MSSpectrum
from tidyms.io.msdata import MSData
from tidyms.lcms import simulation
from tidyms.utils import find_closest


@pytest.fixture
def factory_with_grid(lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
    lcms_sample_factory.config.grid = simulation.MZGridSpecification()
    return lcms_sample_factory

class TestSimulatedLCMSSample:
    def test_make_grid_no_spec_ok(self, lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
        sample = lcms_sample_factory.make()
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
    def data(self, lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
        reader = simulation.SimulatedLCMSDataReader
        return MSData(lcms_sample_factory.make_fileobj(), reader=reader)

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

    def test_multiple_calls_to_get_spectrum_retrieves_the_same_instance(self, data: MSData):
        index = 10
        sp1 = data.get_spectrum(index)
        sp2 = data.get_spectrum(index)
        assert sp1 is sp2

