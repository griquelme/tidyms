import tidyms as ms
import pytest
from pathlib import Path


@pytest.fixture
def lc_sample() -> ms.assay.SampleData:
    tidyms_path = ms.fileio.get_tidyms_path()
    file_path = Path(tidyms_path) / "test-nist-raw-data/NZ_20200227_039.mzML"
    sample = ms.assay.Sample(file_path, file_path.stem)
    return ms.assay.SampleData(sample)


def test_lc_single_sample_pipeline(lc_sample: ms.assay.SampleData):
    processing_steps = [
        ("ROI extractor", ms.assay.lcms_assay.LCTraceExtractor()),
        ("Feature extractor", ms.assay.lcms_assay.LCFeatureExtractor()),
    ]
    pipeline = ms.assay.ProcessingPipeline(processing_steps)
    pipeline.set_default_parameters("qtof", "uplc")
    params = {"ROI extractor": {"min_intensity": 5000.0}}
    pipeline.set_parameters(params)
    pipeline.process(lc_sample)
    assert len(lc_sample.roi) > 0
    assert len(lc_sample.get_feature_list()) > 0
