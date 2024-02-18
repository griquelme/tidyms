"""TidyMS custom exceptions."""


class FeatureTypeNotRegistered(ValueError):
    """"Exception raised when trying to fetch a non-registered Feature type."""


class IncompatibleProcessorStatus(ValueError):
    """Exception raised when the sample data is not compatible with the expected status."""


class InvalidPipelineConfiguration(ValueError):
    """Exception raised when a processing pipeline has an invalid configuration."""


class ProcessorNotFound(ValueError):
    """Exception raised when a processor is not found in a processing pipeline."""


class ProcessorTypeNotRegistered(ValueError):
    """Exception raised when trying to fetch a non-registered Processor type."""


class ReaderNotFound(ValueError):
    """Exception raised when a reader is not found for a specific format."""


class RoiTypeNotRegistered(ValueError):
    """Exception raised when trying to fetch a non-registered ROI type."""


class SampleAlreadyInAssay(ValueError):
    """Exception raised when trying to add a sample with an existing id."""


class SampleDataNotFound(ValueError):
    """Exception raised when trying to retrieve data from a non-existing sample."""
