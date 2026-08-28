"""
raster_align.grid

Computing a shared reference grid for src/target rasters, and getting
each raster onto that grid (crop when already aligned, reproject otherwise).
"""

import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import reproject, calculate_default_transform

from .config import DEFAULT_NUM_THREADS
from .crs_utils import crs_equal, bounds_in_crs
from .cropping import crop_to_valid_extent
from .manual_resample import manual_block_average_resample


def _pick_reference_info(info_src, info_tgt, resolution_mode):
    """
    Pick which raster's native pixel grid the common grid should be
    anchored to, so that raster needs only a crop (no resampling) and
    the other one is the only one actually reprojected.

    - "coarser" (default): anchor to whichever raster has the coarser
      (larger) pixel size -- that's the one we resample *to*, so it
      should be the one that doesn't itself need resampling.
    - "finer": anchor to whichever raster has the finer (smaller) pixel
      size, for the same reason.
    - anything else (e.g. "src"): anchor to src, matching prior behavior.
    """
    res_src, res_tgt = info_src["res"][0], info_tgt["res"][0]
    if resolution_mode == "coarser":
        return info_src if res_src >= res_tgt else info_tgt
    elif resolution_mode == "finer":
        return info_src if res_src <= res_tgt else info_tgt
    return info_src


def compute_common_grid(info_src, info_tgt, resolution_mode="coarser", pad_px=32):
    ref_info = _pick_reference_info(info_src, info_tgt, resolution_mode)
    ref_crs = ref_info["crs"]
    ref_transform = ref_info["transform"]

    b_src = bounds_in_crs(info_src["bounds"], info_src["crs"], ref_crs)
    b_tgt = bounds_in_crs(info_tgt["bounds"], info_tgt["crs"], ref_crs)

    left = max(b_src[0], b_tgt[0])
    bottom = max(b_src[1], b_tgt[1])
    right = min(b_src[2], b_tgt[2])
    top = min(b_src[3], b_tgt[3])

    if left >= right or bottom >= top:
        raise ValueError(
            f"[NO OVERLAP] src and target share no overlapping region in world "
            f"coordinates. src bounds (in ref CRS)={b_src}, target bounds (in ref CRS)={b_tgt}"
        )

    target_res = ref_info["res"][0]

    pad_world = pad_px * target_res
    left -= pad_world
    bottom -= pad_world
    right += pad_world
    top += pad_world

    union_left = min(b_src[0], b_tgt[0])
    union_bottom = min(b_src[1], b_tgt[1])
    union_right = max(b_src[2], b_tgt[2])
    union_top = max(b_src[3], b_tgt[3])
    left = max(left, union_left)
    bottom = max(bottom, union_bottom)
    right = min(right, union_right)
    top = min(top, union_top)

    origin_x, pixel_w = ref_transform.c, ref_transform.a
    origin_y, pixel_h = ref_transform.f, ref_transform.e
    left = origin_x + round((left - origin_x) / pixel_w) * pixel_w
    right = origin_x + round((right - origin_x) / pixel_w) * pixel_w
    top = origin_y + round((top - origin_y) / pixel_h) * pixel_h
    bottom = origin_y + round((bottom - origin_y) / pixel_h) * pixel_h

    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    transform = from_bounds(left, bottom, right, top, width, height)

    if width < 4 * pad_px or height < 4 * pad_px:
        print(f"WARNING: Common-grid overlap is small relative to padding "
              f"({width}x{height}px) -- alignment confidence may be low.")

    return {"crs": ref_crs, "transform": transform, "width": width, "height": height, "res": target_res}


def image_already_on_grid(info, grid, tol_frac=0.1):
    if not crs_equal(info["crs"], grid["crs"]):
        return False
    res, transform = info["res"], info["transform"]
    if not np.isclose(res[0], grid["res"], rtol=tol_frac) or not np.isclose(res[1], grid["res"], rtol=tol_frac):
        return False
    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
        return False

    left, bottom, right, top = array_bounds(grid["height"], grid["width"], grid["transform"])

    def aligned(edge_value, origin, pixel_size):
        offset = (edge_value - origin) / pixel_size
        return abs(offset - round(offset)) < tol_frac

    return aligned(left, transform.c, transform.a) and aligned(top, transform.f, transform.e)


def crop_to_grid(arr, src_transform, grid):
    left, bottom, right, top = array_bounds(grid["height"], grid["width"], grid["transform"])
    inv = ~src_transform
    col_off_f, row_off_f = inv * (left, top)
    col_off, row_off = int(round(col_off_f)), int(round(row_off_f))

    n_bands, src_h, src_w = arr.shape
    out = np.full((n_bands, grid["height"], grid["width"]), np.nan, dtype=np.float32)

    row_end, col_end = row_off + grid["height"], col_off + grid["width"]
    s_r0, s_r1 = max(0, row_off), min(src_h, row_end)
    s_c0, s_c1 = max(0, col_off), min(src_w, col_end)

    if s_r0 < s_r1 and s_c0 < s_c1:
        d_r0, d_r1 = s_r0 - row_off, s_r1 - row_off
        d_c0, d_c1 = s_c0 - col_off, s_c1 - col_off
        out[:, d_r0:d_r1, d_c0:d_c1] = arr[:, s_r0:s_r1, s_c0:s_c1]
    return out


def reproject_to_grid(arr, src_transform, src_crs, grid, resampling, num_threads=DEFAULT_NUM_THREADS):
    dest = np.full((arr.shape[0], grid["height"], grid["width"]), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=dest,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=grid["transform"], dst_crs=grid["crs"],
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling,
        num_threads=num_threads,
    )
    return dest

def reproject_to_crs_native_res(arr, src_transform, src_crs, dst_crs,
                                 resampling=Resampling.bilinear, num_threads=DEFAULT_NUM_THREADS):
    """
    Reprojects `arr` from `src_crs` into `dst_crs`, choosing the output
    transform/size via rasterio's calculate_default_transform so the
    output pixel size stays close to the input's own native resolution --
    i.e. "same data, same CRS as the target grid, still (approximately)
    native resolution", as opposed to reproject_to_grid() above, which
    also resamples onto one specific *coarser* destination grid.

    Needed before manual_block_average_resample() (manual_resample.py)
    can be used across a CRS boundary (e.g. adjacent UTM zones): that
    function is pure transform math with no CRS awareness of its own --
    it silently assumes both transforms it's given already share one CRS.
    Passing it a finer-resolution raster's *native* transform (still in
    that raster's own CRS) alongside a *reference* grid transform in a
    different CRS produces boundaries that don't correspond to the same
    ground positions, which surfaces downstream as "no jointly-valid
    pixels" rather than as an obvious error here.

    Returns (reprojected_arr, reprojected_transform) -- both already in
    dst_crs, ready to pass into manual_block_average_resample() alongside
    a reference transform in that same dst_crs.
    """
    n_bands, height, width = arr.shape
    left, bottom, right, top = array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left, bottom, right, top
    )
    dest = np.full((n_bands, dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=dest,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling,
        num_threads=num_threads,
    )
    return dest, dst_transform


def reproject_to_grid_manual(arr, src_transform, src_crs, grid, num_threads=DEFAULT_NUM_THREADS):
    """
    Same job as reproject_to_grid() above -- get `arr` onto `grid` -- but
    via manual_resample.manual_block_average_resample()'s hand-rolled
    exact area-weighted average instead of GDAL's Resampling.average/
    bilinear/etc. Used when the pipeline is run with --resampling manual,
    so the *primary* common-grid arrays (the ones that feed everything
    downstream of the common grid -- ECC alignment, homogeneity,
    composite, the plot) are the manually-averaged ones, not just the
    background QA comparison. Note: by this point `arr` is already
    reflectance, not DN -- compute_common_reflectance() runs once, in
    pipeline.py step "1", BEFORE put_on_common_grid()/this function are
    ever called.

    If src_crs differs from grid["crs"] (e.g. adjacent UTM zones), first
    reprojects `arr` into grid["crs"] at native resolution via
    reproject_to_crs_native_res() -- always with GDAL bilinear regardless
    of the caller's chosen method, since that step only realigns the CRS
    at (approximately) native resolution and isn't itself the
    fine->coarse downsampling step; manual_block_average_resample() then
    does the actual downsampling by hand, with no GDAL/rasterio call
    involved in that part.
    """
    if not crs_equal(src_crs, grid["crs"]):
        arr, src_transform = reproject_to_crs_native_res(
            arr, src_transform, src_crs, grid["crs"], num_threads=num_threads
        )
    return manual_block_average_resample(arr, src_transform, grid["transform"], grid["width"], grid["height"])


def put_on_common_grid(arr_src, info_src, arr_tgt, info_tgt, resolution_mode, pad_px,
                       resampling, num_threads):
    """
    Returns (img_src, img_tgt, ref_transform, ref_crs, ref_width, ref_height,
             resampled_src, resampled_tgt, img_src_detect, img_tgt_detect).

    img_src/img_tgt are resampled with the caller's chosen `resampling`
    method (bilinear/cubic/nearest/average/manual) and are what
    everything downstream (ECC alignment, homogeneity, composite, the
    plot, written outputs) actually uses. Note: arr_src/arr_tgt going IN
    to this function are already TOA reflectance, not DN --
    compute_common_reflectance() runs once, earlier, in pipeline.py step
    "1", before put_on_common_grid() is ever called.

    img_src_detect/img_tgt_detect are a separate always-bilinear pair for
    ECC misalignment detection. When `resampling` is already bilinear (or
    no resampling was needed at all), img_*_detect is just the same array
    as img_src/img_tgt -- no extra work.
    """
    grid_matches = (
        crs_equal(info_src["crs"], info_tgt["crs"])
        and info_src["transform"] == info_tgt["transform"]
        and info_src["width_px"] == info_tgt["width_px"]
        and info_src["height_px"] == info_tgt["height_px"]
    )
    if grid_matches:
        return (arr_src, arr_tgt, info_src["transform"], info_src["crs"],
                info_src["width_px"], info_src["height_px"], False, False,
                arr_src, arr_tgt)

    grid = compute_common_grid(info_src, info_tgt, resolution_mode=resolution_mode, pad_px=pad_px)
    needs_detect_proxy = resampling not in (Resampling.bilinear,)

    if image_already_on_grid(info_src, grid):
        img_src = crop_to_grid(arr_src, info_src["transform"], grid)
        resampled_src = False
        img_src_detect = img_src
    else:
        src_cropped, src_cropped_transform = crop_to_valid_extent(arr_src, info_src["transform"])
        if resampling == "manual":
            img_src = reproject_to_grid_manual(src_cropped, src_cropped_transform, info_src["crs"], grid, num_threads)
        else:
            img_src = reproject_to_grid(src_cropped, src_cropped_transform, info_src["crs"], grid, resampling, num_threads)
        resampled_src = True
        if needs_detect_proxy:
            img_src_detect = reproject_to_grid(
                src_cropped, src_cropped_transform, info_src["crs"], grid, Resampling.bilinear, num_threads
            )
        else:
            img_src_detect = img_src

    if image_already_on_grid(info_tgt, grid):
        img_tgt = crop_to_grid(arr_tgt, info_tgt["transform"], grid)
        resampled_tgt = False
        img_tgt_detect = img_tgt
    else:
        tgt_cropped, tgt_cropped_transform = crop_to_valid_extent(arr_tgt, info_tgt["transform"])
        if resampling == "manual":
            img_tgt = reproject_to_grid_manual(tgt_cropped, tgt_cropped_transform, info_tgt["crs"], grid, num_threads)
        else:
            img_tgt = reproject_to_grid(tgt_cropped, tgt_cropped_transform, info_tgt["crs"], grid, resampling, num_threads)
        resampled_tgt = True
        if needs_detect_proxy:
            img_tgt_detect = reproject_to_grid(
                tgt_cropped, tgt_cropped_transform, info_tgt["crs"], grid, Resampling.bilinear, num_threads
            )
        else:
            img_tgt_detect = img_tgt

    return (img_src, img_tgt, grid["transform"], grid["crs"], grid["width"], grid["height"],
            resampled_src, resampled_tgt, img_src_detect, img_tgt_detect)
