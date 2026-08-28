"""
raster_align.config

Shared constants used across the package -- and across both entry points
(CLI and GUI) that call into it. Anything both need to agree on (default
output location, how a run's folder gets named/timestamped, etc.) belongs
here rather than being duplicated/hardcoded in cli.py and the GUI.
"""

import os

DEFAULT_NUM_THREADS = max(1, os.cpu_count() or 4)

DEFAULT_MIN_OVERLAP_PCT = 5.0

DEFAULT_OUTPUT_DIR = "outputs"

RUN_DIR_PREFIX = "run"
