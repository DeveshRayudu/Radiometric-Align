"""
raster_align

Aligns a target satellite raster to a source raster and produces a
per-pixel homogeneity (CV) comparison restricted to their common
valid-data footprint.

Public API:
    from raster_align import run_pipeline
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
__version__ = "1.0.0"
