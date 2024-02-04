"""Helper functions for pydantic validation."""

from pathlib import Path


def is_file(p: Path) -> Path:
    """Check if a file exists."""
    assert p.is_file(), f"{p} is not a valid file."
    return p
