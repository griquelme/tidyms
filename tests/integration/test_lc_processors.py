import tidyms as ms
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def lc_sample() -> ms.base.SampleData:
    tidyms_path = ms.fileio.get_tidyms_path()
    file_path = Path(tidyms_path) / "test-nist-raw-data/NZ_20200227_039.mzML"
    sample = ms.base.Sample(file_path, file_path.stem)
    return ms.base.SampleData(sample)


@pytest.fixture
def pipeline():
    processing_steps = [
        ("ROI extractor", ms.lcms_assay.LCTraceExtractor()),
        ("Feature extractor", ms.lcms_assay.LCFeatureExtractor()),
    ]
    pipeline = ms.base.ProcessingPipeline(processing_steps)
    pipeline.set_default_parameters("qtof", "uplc")
    params = {"ROI extractor": {"min_intensity": 5000.0}}
    pipeline.set_parameters(params)
    return pipeline


def test_Assay_add_sample(pipeline: ms.base.ProcessingPipeline, tmp_path: Path):
    tidyms_path = ms.fileio.get_tidyms_path()
    file_path = Path(tidyms_path) / "test-nist-raw-data/NZ_20200227_039.mzML"
    sample = ms.base.Sample(file_path, file_path.stem)
    data_path = tmp_path / "assay-data"
    assay = ms.base.Assay(data_path, pipeline, ms.lcms.LCTrace, ms.lcms.Peak)
    assay.add_samples([sample])


def test_lc_process_sample(pipeline: ms.base.ProcessingPipeline, tmp_path: Path):
    tidyms_path = ms.fileio.get_tidyms_path()
    file_path = Path(tidyms_path) / "test-nist-raw-data/NZ_20200227_039.mzML"
    sample = ms.base.Sample(file_path, file_path.stem)
    data_path = tmp_path / "assay-data"
    assay = ms.base.Assay(data_path, pipeline, ms.lcms.LCTrace, ms.lcms.Peak)
    assay.add_samples([sample])
    assay.process_samples(n_jobs=None)
