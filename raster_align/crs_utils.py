"""
raster_align.crs_utils

CRS equality checks, CRS validation, and bounds reprojection helpers.
"""

from rasterio.warp import transform_bounds


def crs_equal(crs1, crs2):
    """
    CRS equality check.

    - None on either side -> not equal (missing georeferencing must be
      caught by validate_crs(), not silently short-circuited here).
    - Otherwise delegates to rasterio's CRS.__eq__, which internally
      compares via EPSG codes when both sides have one, and falls back
      to a full WKT/proj-string comparison when they don't.
    """
    if crs1 is None or crs2 is None:
        return False

    return crs1 == crs2


def validate_crs(crs, label):
    """
    Fail fast and clearly if `crs` isn't usable, instead of letting a
    missing/malformed CRS silently propagate into rasterio.warp.reproject
    or transform_bounds, where it surfaces as an opaque low-level error
    with no indication of which raster or field caused it.
    """
    if crs is None:
        raise ValueError(
            f"[CRS] {label} raster has no CRS defined in its header -- "
            f"cannot align without georeferencing. Assign a CRS to the "
            f"file (e.g. with rasterio.open(..., 'r+').crs = ... or "
            f"gdal_edit.py -a_srs) and re-run."
        )
    try:
        crs.to_wkt()
    except Exception as exc:
        raise ValueError(f"[CRS] {label} raster's CRS is malformed: {exc}") from exc


def crs_units_compatible(crs1, crs2):
    """
    True unless one CRS is geographic (degree units) and the other is
    projected (linear units, typically meters). Comparing raw pixel-size
    numbers between the two -- as _pick_reference_info()/compute_common_grid()
    in grid.py do to decide which raster is "coarser"/"finer" -- is
    meaningless once you cross that boundary (e.g. 0.0001 vs 10 says
    nothing about which pixel is physically larger).
    """
    geo1 = bool(getattr(crs1, "is_geographic", False))
    geo2 = bool(getattr(crs2, "is_geographic", False))
    return geo1 == geo2


def bounds_in_crs(bounds, src_crs, dst_crs):
    if crs_equal(src_crs, dst_crs):
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    return transform_bounds(src_crs, dst_crs, bounds.left, bounds.bottom, bounds.right, bounds.top)


def compute_overlap_bounds(bounds_src, crs_src, bounds_tgt, crs_tgt):
    """
    Reprojects target's bounds into src's CRS (both are lat/lon or
    projected extents taken straight from each file's own georeferencing)
    and intersects them with src's bounds.

    Returns (left, bottom, right, top) of the overlap region, or None if
    the two rasters share no common ground at all -- i.e. there is no
    physical area for alignment/resampling to operate on.
    """
    tgt_left, tgt_bottom, tgt_right, tgt_top = bounds_in_crs(bounds_tgt, crs_tgt, crs_src)

    left = max(bounds_src.left, tgt_left)
    bottom = max(bounds_src.bottom, tgt_bottom)
    right = min(bounds_src.right, tgt_right)
    top = min(bounds_src.top, tgt_top)

    if left >= right or bottom >= top:
        return None
    return left, bottom, right, top


def compute_overlap_percentage(overlap_bounds, bounds_src, crs_src, bounds_tgt, crs_tgt):
    """
    Expresses the overlap area as a percentage of the SMALLER of the two
    rasters' footprints (src vs target, both measured in src's CRS).

    Using the smaller footprint as the denominator -- rather than src's
    area alone -- means a small target draped over a large src (or vice
    versa) is still judged on how much of *itself* actually lands on
    shared ground, instead of being reported as "tiny overlap %" just
    because the other raster is huge.

    Returns 0.0 if overlap_bounds is None (no intersection at all).
    """
    if overlap_bounds is None: 
        return 0.0

    ov_left, ov_bottom, ov_right, ov_top = overlap_bounds
    overlap_area = (ov_right - ov_left) * (ov_top - ov_bottom)

    src_area = (bounds_src.right - bounds_src.left) * (bounds_src.top - bounds_src.bottom)

    tgt_left, tgt_bottom, tgt_right, tgt_top = bounds_in_crs(bounds_tgt, crs_tgt, crs_src)
    tgt_area = (tgt_right - tgt_left) * (tgt_top - tgt_bottom)

    smaller_area = min(src_area, tgt_area)
    if smaller_area <= 0:
        return 0.0

    return (overlap_area / smaller_area) * 100.0
