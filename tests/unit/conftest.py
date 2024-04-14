import pytest
from tidyms.lcms import simulation


@pytest.fixture
def lcms_sample_factory():
    config = simulation.SimulatedLCMSDataConfiguration(min_signal_intensity=1.0, n_scans=40)
    formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
    rt_list = [10.0, 20.0, 30.0]
    int_list = [100.0, 200.0, 300.0]
    adduct_list = list()
    for rt, spint, formula in zip(rt_list, int_list, formula_list):
        adduct = simulation.SimulatedLCMSAdductSpecification(
            formula=formula, rt_mean=rt, base_intensity=spint, n_isotopologues=2
        )
        adduct_list.append(adduct)
    return simulation.SimulatedLCMSSampleFactory(config=config, adducts=adduct_list)
