"""
raster_align.cropping

Cropping arrays down to the true common valid-data bounding box, and
(for the final output crop) down to the true common valid-data *polygon*
so that rotated/irregular footprints don't leave nodata "black triangle"
corners inside an otherwise-rectangular crop.
"""

import numpy as np
from rasterio import Affine
from rasterio.features import shapes as _rio_shapes, rasterize as _rio_rasterize
from shapely.geometry import shape as _shp_shape, mapping as _shp_mapping
from shapely.ops import unary_union as _shp_unary_union


def bbox_of_mask(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def crop_to_bbox(transform, r0, r1, c0, c1, *arrays):
    new_transform = transform * Affine.translation(c0, r0)
    cropped = [a[..., r0:r1, c0:c1] for a in arrays]
    return new_transform, cropped


def crop_to_valid_extent(arr, transform):
    """
    Crop a native array (and its transform) down to the bounding box of its
    own valid (finite, in any band) pixels.

    Purpose: some source rasters carry a large nodata margin of their own
    (unrelated to the common-grid's pad_px padding, which is added later in
    grid.py). Reprojecting the full raster means warping all of that dead
    space too. Trimming it here first shrinks what reproject() has to
    process, without touching the destination/common-grid extent at all --
    so it has no effect on the ECC correction margin.

    Returns (arr, transform) unchanged if there's nothing valid to bound, or
    if the raster is already valid edge-to-edge (no margin to trim).
    """
    valid = np.isfinite(arr).any(axis=0)
    bbox = bbox_of_mask(valid)
    if bbox is None:
        return arr, transform
    r0, r1, c0, c1 = bbox
    if r0 == 0 and c0 == 0 and r1 == arr.shape[-2] and c1 == arr.shape[-1]:
        return arr, transform
    new_transform, (cropped,) = crop_to_bbox(transform, r0, r1, c0, c1, arr)
    return cropped, new_transform


def build_valid_footprint_polygon(mask, transform):
    """
    Turn a boolean valid-data mask into a single (Multi)Polygon footprint
    in the raster's own CRS, using the affine transform to place it.

    Returns None if the mask has no valid pixels at all.
    """
    mask_u8 = mask.astype(np.uint8)
    geoms = [
        _shp_shape(geom)
        for geom, value in _rio_shapes(mask_u8, mask=mask, transform=transform)
        if value == 1
    ]
    if not geoms:
        return None
    return _shp_unary_union(geoms)


def compute_common_footprint(mask_src, mask_tgt, transform):
    """
    Build the source and target valid-data footprint polygons and return
    their intersection.

    Returns (common_polygon, src_polygon, tgt_polygon). common_polygon is
    None if either input footprint is empty or the two footprints don't
    overlap at all (common_polygon.is_empty would also be True in that
    case, but callers should just check for None/`.is_empty`).
    """
    src_polygon = build_valid_footprint_polygon(mask_src, transform)
    tgt_polygon = build_valid_footprint_polygon(mask_tgt, transform)
    if src_polygon is None or tgt_polygon is None:
        return None, src_polygon, tgt_polygon
    common_polygon = src_polygon.intersection(tgt_polygon)
    if common_polygon.is_empty:
        return None, src_polygon, tgt_polygon
    return common_polygon, src_polygon, tgt_polygon


def polygon_crop(transform, polygon, *arrays):
    """
    Crop `arrays` (each either (bands, rows, cols) or (rows, cols)) down to
    the bounding box of `polygon`, then blank out every pixel that falls
    outside the polygon itself (set to NaN for float arrays, 0 for
    integer/bool arrays) so that only the true common footprint survives --
    no rectangular corners of one-sided nodata left over from a rotated or
    irregular overlap.

    Returns (new_transform, cropped_and_masked_arrays, footprint_mask,
    (r0, r1, c0, c1)) where footprint_mask is the rasterized polygon mask
    at the *cropped* array's resolution (2D, bool) and (r0, r1, c0, c1) are
    the pixel bounds (relative to the input transform) that were cropped to.
    """
    if not arrays:
        raise ValueError("polygon_crop() requires at least one array to crop.")

    height, width = arrays[0].shape[-2], arrays[0].shape[-1]

    minx, miny, maxx, maxy = polygon.bounds
    inv = ~transform
    col_a, row_a = inv * (minx, maxy)
    col_b, row_b = inv * (maxx, miny)
    c0 = max(int(np.floor(min(col_a, col_b))), 0)
    c1 = min(int(np.ceil(max(col_a, col_b))), width)
    r0 = max(int(np.floor(min(row_a, row_b))), 0)
    r1 = min(int(np.ceil(max(row_a, row_b))), height)

    if c1 <= c0 or r1 <= r0:
        return transform, list(arrays), np.zeros((0, 0), dtype=bool), (r0, r0, c0, c0)

    new_transform, cropped = crop_to_bbox(transform, r0, r1, c0, c1, *arrays)

    footprint_mask = _rio_rasterize(
        [(_shp_mapping(polygon), 1)],
        out_shape=(r1 - r0, c1 - c0),
        transform=new_transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    masked = []
    for arr in cropped:
        arr = arr.copy()
        fill_value = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
        if arr.ndim == 3:
            arr[:, ~footprint_mask] = fill_value
        else:
            arr[~footprint_mask] = fill_value
        masked.append(arr)

    return new_transform, masked, footprint_mask, (r0, r1, c0, c1)
