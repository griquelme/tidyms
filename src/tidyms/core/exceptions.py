"""Exceptions raised by tidyms."""


class IncompatibleProcessorStatus(ValueError):
    """Exception raised when the sample data is not compatible with the expected status."""

    pass


class InvalidProcessingPipeline(ValueError):
    """Exception raised when a processing pipeline has an invalid configuration."""
