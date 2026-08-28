"""
raster_align.raster_io

Writing arrays out as tiled/compressed GeoTIFFs with sane block sizes.
"""

from os.path import join

import numpy as np
import rasterio
from rasterio.enums import ColorInterp


def write_raster(output_dir, filename, array, profile, dtype=None, nodata_value=None):
    """Write an array as a tiled, DEFLATE-compressed GeoTIFF with auto-sized blocks."""
    out_path = join(output_dir, filename)
    array = np.asarray(array)
    out_dtype = dtype or array.dtype

    if np.issubdtype(array.dtype, np.floating) and nodata_value is not None:
        array = np.where(np.isfinite(array), array, nodata_value)
    array = array.astype(out_dtype)

    height, width = array.shape[-2], array.shape[-1]
    predictor = 3 if np.issubdtype(np.dtype(out_dtype), np.floating) else 2

    profile = profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    block = 256
    if height >= 16 and width >= 16:
        bx = min(block, (width // 16) * 16 or 16)
        by = min(block, (height // 16) * 16 or 16)
        profile.update(tiled=True, blockxsize=bx, blockysize=by, compress="deflate", predictor=predictor)
    else:
        profile.update(tiled=False, compress="deflate", predictor=predictor)

    if nodata_value is not None:
        profile.update(nodata=nodata_value)

    if array.ndim == 2:
        profile.update(count=1, dtype=out_dtype)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array, 1)
    else:
        profile.update(count=array.shape[0], dtype=out_dtype)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array)
            if array.shape[0] == 3 and out_dtype == np.uint8:
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
