import pathlib

import pytest
from tidyms.annotation import annotation
from tidyms.core.models import Sample
from tidyms.core.processors import ProcessingPipeline, SampleData
from tidyms.lcms import LCTraceBaselineEstimator, LCTraceExtractor, MZTrace, Peak, PeakExtractor, simulation


@pytest.fixture
def annotation_tools_params():
    bounds = {
        "C": (0, 50),
        "H": (0, 100),
        "O": (0, 20),
        "N": (0, 20),
        "Cl": (0, 2),
        "B": (0, 1),
    }
    params = {
        "bounds": bounds,
        "max_mass": 2500,
        "max_length": 10,
        "max_charge": 3,
        "min_M_tol": 0.005,
        "max_M_tol": 0.01,
        "p_tol": 0.05,
        "min_similarity": 0.9,
        "min_p": 0.01,
    }
    return params


def test__annotate_empty_feature_list(annotation_tools_params):
    tools = annotation.create_annotation_tools(**annotation_tools_params)
    feature_list = list()
    annotation.annotate(feature_list, *tools)


@pytest.fixture
def peak_list(tmp_path: pathlib.Path) -> list[Peak]:
    # load simulated reader to make it to MSData
    from tidyms.lcms.simulation import SimulatedLCMSDataReader  # noqa

    formulas = ["[C10H20O2]-", "[C10H20BO3]-", "[C20H40BO5]2-", "[C18H19N2O3]-", "[C18H20N2O3Cl]-", "[C10H20Cl]-"]
    rt_list = [50, 75, 150, 200, 200, 175]
    amp_list = [10000, 20000, 30000, 25000, 25000, 20000]

    adducts = list()
    for formula, rt, amp in zip(formulas, rt_list, amp_list):
        adduct = simulation.SimulatedLCMSAdductSpecification(
            formula=formula, rt_mean=rt, base_intensity=amp, rt_width=3.0, n_isotopologues=4
        )
        adducts.append(adduct)

    sample_config = simulation.SimulatedLCMSDataConfiguration(n_scans=300)

    factory = simulation.SimulatedLCMSSampleFactory(config=sample_config, adducts=adducts)
    path = tmp_path / "sample.json"
    factory.make_file(path)

    sample = Sample(id="simulated_sample", path=path, reader="SimulatedLCMSDataReader")
    data: SampleData[Peak, MZTrace] = SampleData(sample=sample)

    pipe = ProcessingPipeline(
        id="annotation_pipeline",
        processors=[
            LCTraceExtractor(id="roi_extractor"),
            LCTraceBaselineEstimator(id="baseline_estimator"),
            PeakExtractor(id="peak_extractor"),
        ],
    )
    pipe.process(data)

    return data.get_features()


def test_annotate(peak_list: list[Peak], annotation_tools_params):
    tools = annotation.create_annotation_tools(**annotation_tools_params)
    annotation.annotate(peak_list, *tools)

    # group features by isotopologue label.
    annotation_check = dict()
    for peak in peak_list:
        group_list = annotation_check.setdefault(peak.annotation.isotopologue_label, list())
        group_list.append(peak)
    if -1 in annotation_check:
        annotation_check.pop(-1)
    assert len(annotation_check) == 6
    for v in annotation_check.values():
        assert len(v) == 4  # features where generated with 4 isotopologues.
