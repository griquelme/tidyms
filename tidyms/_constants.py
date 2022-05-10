from typing import Final, List

# TODO: merge with _names.py
# separation modes
HPLC: Final[str] = "hplc"
UPLC: Final[str] = "uplc"
LC_MODES: Final[List[str]] = [UPLC, HPLC]

# instruments
QTOF: Final[str] = "qtof"
ORBITRAP: Final[str] = "orbitrap"
MS_INSTRUMENTS: Final[List[str]] = [QTOF, ORBITRAP]

# feature descriptors
FEATURE: Final[str] = "feature"
MZ: Final[str] = "mz"
RT: Final[str] = "rt"
AREA: Final[str] = "area"
WIDTH: Final[str] = "width"
HEIGHT: Final[str] = "height"
SNR: Final[str] = "snr"
MZ_STD: Final[str] = "mz_std"
ROI_INDEX: Final[str] = "roi_index"
FT_INDEX: Final[str] = "ft_index"

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
