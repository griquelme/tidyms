# variables used to name sample information columns

_raw_path = "raw path"
_sample_class = "class"
_sample_id = "id"
_sample_batch = "batch"
_sample_order = "order"
_sample_type = "type"
_study_sample_type = "sample"
_qc_sample_type = "qc"
_blank_sample_type = "blank"
_suitability_sample_type = "suitability"
_zero_type = "zero"
SAMPLE_TYPES = [_study_sample_type, _qc_sample_type, _blank_sample_type,
                _suitability_sample_type, _zero_type]

__all__ = ["_raw_path", "_sample_class", "_sample_id", "_sample_batch",
           "_sample_order", "_sample_type","_study_sample_type",
           "_qc_sample_type", "_blank_sample_type", "_suitability_sample_type",
           "_zero_type", "SAMPLE_TYPES"]
