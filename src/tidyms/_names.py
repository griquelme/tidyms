# variables used to name sample information columns

_raw_path = "raw path"
_sample_class = "class"
_sample_id = "id"
_sample_batch = "batch"
_sample_order = "order"
_sample_type = "type"
_sample_dilution = "dilution"
_study_sample_type = "sample"
_qc_sample_type = "qc"
_dilution_qc_type = "dqc"
_blank_sample_type = "blank"
_suitability_sample_type = "suitability"
_zero_type = "zero"
SAMPLE_TYPES = [_study_sample_type, _qc_sample_type, _blank_sample_type,
                _suitability_sample_type, _zero_type, _dilution_qc_type]

__all__ = ["_raw_path", "_sample_class", "_sample_id", "_sample_batch",
           "_sample_order", "_sample_type", "_sample_dilution",
           "_study_sample_type", "_qc_sample_type", "_dilution_qc_type",
           "_blank_sample_type", "_suitability_sample_type", "_zero_type",
           "SAMPLE_TYPES"]