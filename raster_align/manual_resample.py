"""
raster_align.manual_resample

A from-scratch (no GDAL reproject call) area-weighted block-average
resampler. Its output is one of two resampled versions of the same
raster (the other being whatever the pipeline used, e.g. bilinear);
both get scored independently in pipeline.py against the same ground
truth -- the other raster's own actual native data on the common grid
-- via quality.compare_resampled_to_reference, so their PSNR/RMSE/SSIM
numbers are directly comparable.

Works for ANY fine->coarse ratio, integer or fractional (5m->25m is a
clean 5x, but 5m->12m, a 2.4x ratio with no clean pixel alignment, is
handled exactly the same way) and for any sub-pixel offset between the
two rasters' native grids.

How it works: for a north-up (no rotation/shear) raster, resampling is
separable into independent row and column operations. Within each axis,
each fine pixel's value is treated as constant over its own footprint,
which makes the *integral* of the data along that axis a piecewise-linear
function of position. That means we can compute the exact area-weighted
average for ANY output pixel boundary -- including fractional ones -- by
building a cumulative sum once and reading it off with linear
interpolation at the boundary positions, rather than looping over pixels
or relying on GDAL's own Resampling.average implementation.
"""

import numpy as np


def _axis_boundaries(fine_origin, fine_pixel, coarse_origin, coarse_pixel, n_coarse):
    """
    The n_coarse+1 edges of the output (coarse) pixels along one axis,
    expressed as fractional indices into the fine array along that same
    axis. Works regardless of pixel sign convention (rows use a negative
    pixel size in a north-up raster) as long as fine_pixel and
    coarse_pixel use the same convention, which they always do here since
    both come from the same north-up transform family.
    """
    edges_world = coarse_origin + np.arange(n_coarse + 1, dtype=np.float64) * coarse_pixel
    return (edges_world - fine_origin) / fine_pixel


def _cumsum_with_zero(arr, axis):
    cs = np.cumsum(arr, axis=axis, dtype=np.float64)
    pad_shape = list(arr.shape)
    pad_shape[axis] = 1
    zero_pad = np.zeros(pad_shape, dtype=np.float64)
    return np.concatenate([zero_pad, cs], axis=axis)


def _interp_along_axis(cumsum, positions, axis):
    n = cumsum.shape[axis]
    positions = np.clip(positions, 0, n - 1)
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = positions - lo

    lo_vals = np.take(cumsum, lo, axis=axis)
    hi_vals = np.take(cumsum, hi, axis=axis)

    shape = [1] * cumsum.ndim
    shape[axis] = len(positions)
    frac = frac.reshape(shape)

    return lo_vals * (1.0 - frac) + hi_vals * frac


def _collapse_axis(arr, boundaries, axis):
    """
    Exact area-weighted sum of `arr` between each pair of consecutive
    `boundaries` (fractional fine-array indices), along `axis`.
    """
    cumsum = _cumsum_with_zero(arr, axis)
    interp = _interp_along_axis(cumsum, boundaries, axis)

    lo_slice = [slice(None)] * interp.ndim
    hi_slice = [slice(None)] * interp.ndim
    lo_slice[axis] = slice(0, -1)
    hi_slice[axis] = slice(1, None)
    return interp[tuple(hi_slice)] - interp[tuple(lo_slice)]


def _manual_block_average_band(band, native_transform, coarse_transform, coarse_width, coarse_height):
    band = band.astype(np.float64, copy=False)
    valid = np.isfinite(band)
    vals = np.where(valid, band, 0.0)
    weights = valid.astype(np.float64)

    col_boundaries = _axis_boundaries(native_transform.c, native_transform.a,
                                       coarse_transform.c, coarse_transform.a, coarse_width)
    row_boundaries = _axis_boundaries(native_transform.f, native_transform.e,
                                       coarse_transform.f, coarse_transform.e, coarse_height)

    vals_c = _collapse_axis(vals, col_boundaries, axis=1)
    weights_c = _collapse_axis(weights, col_boundaries, axis=1)

    vals_rc = _collapse_axis(vals_c, row_boundaries, axis=0)
    weights_rc = _collapse_axis(weights_c, row_boundaries, axis=0)

    has_data = weights_rc > 1e-9
    out = np.divide(vals_rc, weights_rc, out=np.full_like(vals_rc, np.nan), where=has_data)
    return out.astype(np.float32)


def manual_block_average_resample(arr, native_transform, coarse_transform, coarse_width, coarse_height):
    """
    Resample a (bands, rows, cols) native array onto a coarser common
    grid via exact area-weighted block averaging, computed entirely by
    hand (no rasterio.warp.reproject / GDAL resampling call involved).

    Returns an array of shape (bands, coarse_height, coarse_width) --
    same shape/footprint as what put_on_common_grid() produces for
    img_src/img_tgt, so it can be compared pixel-for-pixel against the
    pipeline's normal (e.g. bilinear) resampled result.
    """
    n_bands = arr.shape[0]
    out = np.empty((n_bands, coarse_height, coarse_width), dtype=np.float32)
    for b in range(n_bands):
        out[b] = _manual_block_average_band(
            arr[b], native_transform, coarse_transform, coarse_width, coarse_height
        )
    return out
