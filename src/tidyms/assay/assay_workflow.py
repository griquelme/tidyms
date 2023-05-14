from .assay_processor import SingleSampleProcessor
from .assay_data import AssayData, SampleData
from typing import Optional, Sequence
import logging
from multiprocessing.pool import Pool


# class IncompatibleProcessorsError(ValueError):
#     pass


# class InvalidAssayPipeline(ValueError):
#     pass


# def check_compatibility(proc1: SingleSampleProcessor, proc2: SingleSampleProcessor):
#     proc1_types = get_type_hints(proc1._func)
#     proc2_types = get_type_hints(proc2._func)
#     proc1_out = proc1_types.get("return")
#     proc2_in = proc2_types.get("data")
#     if proc1_out is None:
#         logger = logging.getLogger("assay")
#         msg = (
#             f"{proc1} does not specify return type in `_func` method."
#             f" Cannot check compatibility with {proc2}"
#         )
#         logger.warn(msg)
#     elif proc2_in is None:
#         logger = logging.getLogger("assay")
#         msg = (
#             f"{proc2} does not specify data type in `_func` method."
#             f" Cannot check compatibility with {proc1}"
#         )
#         logger.warn(msg)
#     elif proc1_out != proc2_in:
#         msg = (
#             f"The output of {proc1} is not compatible with the input of {proc2}."
#             f" Expected {proc1_out} for {proc2}, got {proc2_in}."
#         )
#         raise IncompatibleProcessorsError(msg)


# def check_single_sample_pipeline(pipeline: Sequence[SingleSampleProcessor]):
#     # at least 2 steps in the pipeline:
#     # ROI extraction, feature extraction
#     assert len(pipeline) >= 2

#     if not isinstance(pipeline[0], RoiExtractor):
#         e_msg = f"The first step of the pipeline must be a {RoiExtractor} instance."
#         raise TypeError(e_msg)

#     for proc1, proc2 in zip(pipeline, pipeline[1:]):
#         check_compatibility(proc1, proc2)


class AssayWorkflow:
    def __init__(self, steps: Sequence[tuple[str, SingleSampleProcessor]]):
        # check_single_sample_pipeline(single_sample_pipeline)
        self.processors = [x for _, x in steps]
        self._names = [x for x, _ in steps]
        self._name_to_processor = {x: y for x, y in steps}
        if len(self.processors) > len(self._name_to_processor):
            msg = "Processor names must be unique."
            ValueError(msg)

    def get_parameters(self):
        parameters = list()
        for name in self._names:
            processor = self._name_to_processor[name]
            params = processor.get_parameters()
            parameters.append((name, params))
        return parameters

    def set_parameters(self):
        pass

    def process(self, sample_data: SampleData):
        for processor in self.processors:
            processor.process(sample_data)


def apply_single_sample_workflow(
    workflow: AssayWorkflow, data: AssayData, n_jobs: Optional[int] = None
):
    for processor in workflow.processors:
        with Pool(n_jobs) as pool:
            sample_list = data.get_samples()
            iterator = zip(pool.imap_unordered(pipeline, sample_list), sample_list)
            logger = logging.getLogger("assay")
            for roi_list, sample in iterator:
                self._save_results(assay_data, sample, roi_list)
                logger.info(f"Processed sample {sample.id}.")
