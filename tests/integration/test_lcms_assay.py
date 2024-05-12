import pathlib

import pytest
from tidyms.core.exceptions import SampleIdAlreadyInAssay, SampleWithRepeatedOrder
from tidyms.core.models import Sample
from tidyms.core.processors import SampleData
from tidyms.lcms import simulation
from tidyms.lcms.assay import create_lcms_assay, create_sample_pipeline
from tidyms.lcms.processors import LCFeatureMatcher


@pytest.fixture
def sample_factory():
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

    return simulation.SimulatedLCMSSampleFactory(config=sample_config, adducts=adducts)


class TestProcessingPipeline:
    @pytest.fixture
    def data(self, sample_factory: simulation.SimulatedLCMSSampleFactory, tmp_path: pathlib.Path):
        sample_id = "sample"
        path = tmp_path / f"{sample_id}.json"
        sample_factory.make_file(path)
        sample = Sample(id=sample_id, path=path, reader="SimulatedLCMSDataReader", order=0)
        return SampleData(sample=sample)

    def test_process_ok(self, data: SampleData):
        pipeline = create_sample_pipeline(annotate_isotopologues=False)
        pipeline.process(data)

        assert data.roi
        assert data.get_features()

    def test_process_with_isotopologue_annotation_ok(self, data: SampleData):
        pipeline = create_sample_pipeline(annotate_isotopologues=True)
        pipeline.set_default_parameters()
        pipeline.process(data)

        assert data.roi
        assert data.get_features()


class TestAssay:
    def test_assay_creation(self):
        create_lcms_assay()

    @pytest.fixture
    def samples(self, sample_factory: simulation.SimulatedLCMSSampleFactory, tmp_path: pathlib.Path):
        n_samples = 3
        samples = list()
        for k in range(n_samples):
            sample_id = f"sample-{k}"
            path = tmp_path / f"{sample_id}.json"
            sample_factory.make_file(path)
            sample = Sample(id=sample_id, path=path, reader="SimulatedLCMSDataReader", order=k)
            samples.append(sample)
        return samples

    def test_add_samples_to_assay_ok(self, samples: list[Sample]):
        assay = create_lcms_assay(annotate_isotopologues=False)
        assay.set_default_parameters()
        assay.add_samples(*samples)
        unprocessed_samples = assay.list_unprocessed_samples()
        assert all(x.id in unprocessed_samples for x in samples)
        assert not assay.list_processed_samples()

    def test_add_samples_with_repeated_id_raises_error(self, samples: list[Sample]):
        assay = create_lcms_assay(annotate_isotopologues=False)
        assay.add_samples(samples[0])
        with pytest.raises(SampleIdAlreadyInAssay):
            assay.add_samples(samples[0])

    def test_add_samples_with_repeated_order_raises_error(self, samples: list[Sample]):
        assay = create_lcms_assay(annotate_isotopologues=False)
        sample1 = samples[0]
        sample2 = sample1.model_copy()
        sample2.id = "new-sample"

        assay.add_samples(sample1)
        with pytest.raises(SampleWithRepeatedOrder):
            assay.add_samples(sample2)

    def test_process_samples(self, samples: list[Sample]):
        assay = create_lcms_assay(annotate_isotopologues=False)
        assay.set_default_parameters()
        assay.add_samples(*samples)
        assay.process_samples(max_workers=4)
        processed_samples = assay.list_processed_samples()
        assert all(x.id in processed_samples for x in samples)
        assert not assay.list_unprocessed_samples()

    def test_match_features(self, samples: list[Sample]):
        assay = create_lcms_assay(annotate_isotopologues=False)
        assay.set_default_parameters()
        assay.add_samples(*samples)
        assay.process_samples(max_workers=4)

        data = assay.data
        matcher = LCFeatureMatcher()
        matcher.process(data)
