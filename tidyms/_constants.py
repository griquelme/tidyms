from typing import Final, List

# TODO: merge with _names.py
# separation modes
HPLC: Final[str] = "hplc"
UPLC: Final[str] = "uplc"
LC_MODES: Final[List[str]] = [UPLC, HPLC]
SEPARATION_MODES: Final[List[str]] = LC_MODES + []

# instruments
QTOF: Final[str] = "qtof"
ORBITRAP: Final[str] = "orbitrap"
MS_INSTRUMENTS: Final[List[str]] = [QTOF, ORBITRAP]

# MS mode
CENTROID: Final[str] = "centroid"
PROFILE: Final[str] = "profile"
MS_MODES: Final[List[str]] = [CENTROID, PROFILE]

# Data loading
MEMORY: Final[str] = "memory"
INFILE: Final[str] = "file"
SIMULATED: Final[str] = "simulated"
DATA_LOAD_MODES: Final[List[str]] = [MEMORY, INFILE, SIMULATED]
DEFAULT_DATA_LOAD_MODE = INFILE

# feature descriptors
FEATURE: Final[str] = "feature"
MZ: Final[str] = "mz"
RT_START: Final[str] = "rt start"
RT_END: Final[str] = "rt end"
RT: Final[str] = "rt"
RT_STD: Final[str] = "rt std"
AREA: Final[str] = "area"
WIDTH: Final[str] = "width"
HEIGHT: Final[str] = "height"
SNR: Final[str] = "snr"
MZ_STD: Final[str] = "mz_std"
ROI_INDEX: Final[str] = "roi_index"
FT_INDEX: Final[str] = "ft_index"
MERGED: Final[str] = "merged"

# sample metadata
SAMPLE: Final[str] = "sample_"
CLASS: Final[str] = "class_"
ORDER: Final[str] = "order_"
BATCH: Final[str] = "batch_"
LABEL: Final[str] = "cluster_"

# assay file and dir names
ROI_DIR: Final[str] = "roi"
FT_DIR: Final[str] = "feature"
MANAGER_FILENAME: Final[str] = "metadata.pickle"
FT_TABLE_FILENAME: Final[str] = "feature-table.pickle"
DATA_MATRIX_FILENAME: Final[str] = "data-matrix.pickle"

# preprocessing steps
DETECT_FEATURES: Final[str] = "detect_features"
EXTRACT_FEATURES: Final[str] = "extract_features"
DESCRIBE_FEATURES: Final[str] = "describe_features"
ANNOTATE_ISOTOPOLOGUES: Final[str] = "annotate_isotopologues"
ANNOTATE_ADDUCTS: Final[str] = "annotate_adducts"
BUILD_FEATURE_TABLE: Final[str] = "build_feature_table"
MATCH_FEATURES: Final[str] = "match_features"
MAKE_DATA_MATRIX: Final[str] = "make_data_matrix"
FILL_MISSING: Final[str] = "fill_missing"

PREPROCESSING_STEPS: Final[List[str]] = [
    DETECT_FEATURES,
    EXTRACT_FEATURES,
    DESCRIBE_FEATURES,
    ANNOTATE_ISOTOPOLOGUES,
    ANNOTATE_ADDUCTS,
    BUILD_FEATURE_TABLE,
    MATCH_FEATURES,
    MAKE_DATA_MATRIX,
    FILL_MISSING
]
