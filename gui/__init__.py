"""
gui

PyQt5 desktop front-end for raster_align. Purely a presentation/control
layer -- it calls into raster_align.run_pipeline (the same processing
backend the CLI uses) and never duplicates or reimplements any of its
logic.
"""
from gui.image_viewer import ImageViewerPanel
