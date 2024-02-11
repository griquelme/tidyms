"""Exceptions raised by tidyms."""


class FeatureTypeNotRegistered(ValueError):
    """"Exception raised when trying to fetch a non-registered Feature type."""


class IncompatibleProcessorStatus(ValueError):
    """Exception raised when the sample data is not compatible with the expected status."""


class InvalidPipelineConfiguration(ValueError):
    """Exception raised when a processing pipeline has an invalid configuration."""


class ProcessorNotFound(ValueError):
    """Exception raised when a processor is not found in a processing pipeline."""


class ProcessorTypeNotRegistered(ValueError):
    """"Exception raised when trying to fetch a non-registered Processor type."""


class RoiTypeNotRegistered(ValueError):
    """"Exception raised when trying to fetch a non-registered ROI type."""
