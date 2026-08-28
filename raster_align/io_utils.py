"""
raster_align.io_utils

Reading + normalizing source rasters, band statistics, and metadata CSV
export.

Sentinel-2 / Landsat metadata resolution at load time
-------------------------------------------------------
When read_and_prepare() loads a .jp2 file (Sentinel-2) or a .tif with an
adjacent *_MTL.txt (Landsat), it calls apply_sentinel_correction() /
apply_landsat_correction() from radiometric_correction.py. These ONLY
locate and resolve the sensor's calibration metadata (offsets, quant
value, mult/add coefficients, sun elevation) at this stage -- they do
NOT convert DN to reflectance here, and the array returned (`arr`) is
still raw DN. Resolving metadata early lets a missing/invalid metadata
file surface as an error immediately at load time rather than deep into
the pipeline.

The actual DN -> reflectance conversion for every sensor (Resourcesat-2A,
Sentinel-2, Landsat) happens exactly once, in
radiometric_correction.compute_common_reflectance() -- called from
pipeline.py step "1", immediately after read_and_prepare() returns and
BEFORE common-grid construction/any resampling -- so all downstream
stages (resampling, ECC alignment, homogeneity, composite, outputs, plot)
operate on reflectance,
computed uniformly for every sensor in one place.
"""

from os.path import basename, dirname, abspath, splitext

import numpy as np
import rasterio


def crs_to_epsg_lenient(crs):
    """
    Identify a CRS's EPSG code the same way QGIS/GDAL effectively do,
    which is more lenient than rasterio/pyproj's default.

    rasterio's plain `crs.to_epsg()` only accepts matches at its default
    confidence_threshold=70 (a full-precision comparison against the
    EPSG database). Some GeoTIFFs -- e.g. LISS-3/4 products from NRSC's
    processing chain -- embed a WGS84 ellipsoid with a very slightly
    rounded inverse flattening (e.g. 298.25722293287 instead of the
    textbook 298.257223563), which is close enough for QGIS to still
    display "EPSG:32644" but just under rasterio's strict default
    threshold, so `to_epsg()` returns None and callers fall back to
    printing the full WKT instead of the short code.

    This tries the strict default first (most certain), and only if
    that fails, retries at a much lower confidence_threshold=20 (still
    requires the projection parameters, axes, and units to match --
    just tolerates tiny ellipsoid-definition rounding differences).
    Returns None if no match is found even at the lenient threshold.
    """
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg is not None:
        return epsg
    return crs.to_epsg(confidence_threshold=20)


def read_and_prepare(path, nodata_override=None, fallback_nodata=0.0):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        header_nodata = src.nodata
        info = {
            "image_dir": dirname(abspath(path)),
            "filename": basename(abspath(path)),
            "filepath": abspath(path),
            "driver": src.driver,
            "width_px": src.width,
            "height_px": src.height,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "epsg": crs_to_epsg_lenient(src.crs),
            "transform": src.transform,
            "res": src.res,
            "bounds": src.bounds,
            "nodata": header_nodata,
            "compression": src.compression.name if src.compression else None,
            "interleave": src.profile.get("interleave"),
            "tags": src.tags(),
            "profile": src.profile.copy(),
        }

    if nodata_override is not None:
        nodata = nodata_override
        if header_nodata is not None and nodata_override != header_nodata:
            print(f"[Nodata] {info['filename']}: overriding header nodata={header_nodata} "
                  f"with {nodata_override}.")
    elif header_nodata is not None:
        nodata = header_nodata
    elif fallback_nodata is not None:
        nodata = fallback_nodata
        print(f"[Nodata] {info['filename']}: no nodata tag in header and no override given -- "
              f"FALLING BACK to nodata={fallback_nodata} (LISS dead-zone/off-swath fill "
              f"convention). If {fallback_nodata} is legitimate real data in this file, "
              f"pass --src-nodata none / --target-nodata none to disable this fallback.")
    else:
        nodata = None
        print(f"[WARNING] {info['filename']}: no nodata value in header, override, or fallback -- "
              f"any fill/dead-zone pixels in this file will be treated as REAL data.")

    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
        info["nodata"] = nodata

    ext = splitext(info["filename"])[1].lower()

    if ext == ".jp2":
        from .radiometric_correction import apply_sentinel_correction
        try:
            arr, _s2_calib = apply_sentinel_correction(arr, info)
            info["radiometric_correction"] = "sentinel2_metadata_resolved (conversion happens in pipeline step 1, right after read_and_prepare())"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Radiometric] WARNING: Sentinel-2 metadata resolution failed for "
                f"'{info['filename']}': {exc}. "
                f"compute_common_reflectance() will fail later unless this is fixed."
            )
            info["radiometric_correction"] = "none (sentinel2 metadata resolution failed)"

    elif ext in (".tif", ".tiff"):
        from .radiometric_correction import _find_landsat_mtl, apply_landsat_correction
        if _find_landsat_mtl(info["image_dir"]) is not None:
            try:
                arr, _ = apply_landsat_correction(arr, info)
                info["radiometric_correction"] = "landsat_metadata_resolved (conversion happens in pipeline step 1, right after read_and_prepare())"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[Radiometric] WARNING: Landsat metadata resolution failed for "
                    f"'{info['filename']}': {exc}. "
                    f"compute_common_reflectance() will fail later unless this is fixed."
                )
                info["radiometric_correction"] = "none (landsat metadata resolution failed)"

    return arr, info


def compute_band_stats(arr):
    stats = []
    for band in arr:
        mask = np.isfinite(band)
        if not mask.any():
            stats.append((None, None, None, None))
            continue
        vals = band[mask]
        stats.append((float(vals.min()), float(vals.max()), float(vals.mean()), float(vals.std())))
    return stats


def build_metadata_row(info, stats):
    row = {
        "filename": info["filename"], "filepath": info["filepath"], "driver": info["driver"],
        "width_px": info["width_px"], "height_px": info["height_px"], "band_count": info["band_count"],
        "dtype": info["dtype"], "crs": str(info["crs"]), "epsg": info["epsg"],
        "origin_x": info["transform"].c, "origin_y": info["transform"].f,
        "pixel_width": info["transform"].a, "pixel_height": info["transform"].e,
        "bound_left": info["bounds"].left, "bound_right": info["bounds"].right,
        "bound_bottom": info["bounds"].bottom, "bound_top": info["bounds"].top,
        "nodata": info["nodata"], "compression": info["compression"], "interleave": info["interleave"],
    }
    for i, (mn, mx, mean, std) in enumerate(stats, start=1):
        row[f"band{i}_min"], row[f"band{i}_max"] = mn, mx
        row[f"band{i}_mean"], row[f"band{i}_std"] = mean, std
    for k, v in info["tags"].items():
        row[f"tag_{k}"] = v
    return row

