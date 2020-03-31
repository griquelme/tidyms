class MissingMappingInformation(ValueError):
    "error raised when an empty sample type is used from a mapping"
    pass

class MissingValueError(ValueError):
    "error raise whren a DataContainer's data matrix has missing values"