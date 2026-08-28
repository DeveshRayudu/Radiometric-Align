"""
raster_align.quality

Resampling-quality assessment: round-trip reprojection error metrics
(RMSE, MAE, PSNR, SSIM) between a native raster and its resampled version.
"""

import math

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter


def _ssim_uniform(a, b, win_size=7, data_range=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if data_range is None:
        data_range = float(a.max() - a.min())
    if data_range <= 0:
        return None

    win_size = max(3, win_size | 1)
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2

    mu_a = uniform_filter(a, size=win_size)
    mu_b = uniform_filter(b, size=win_size)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = uniform_filter(a * a, size=win_size) - mu_a2
    sigma_b2 = uniform_filter(b * b, size=win_size) - mu_b2
    sigma_ab = uniform_filter(a * b, size=win_size) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    ssim_map = numerator / np.where(denominator == 0, np.nan, denominator)

    pad = win_size // 2
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    return float(np.nanmean(ssim_map))


def compute_error_metrics(original, reconstructed, ssim_win_size=7):
    valid = np.isfinite(original) & np.isfinite(reconstructed)
    n_valid = int(valid.sum())
    total_px = original.size
    coverage = n_valid / total_px if total_px else 0.0

    if n_valid == 0:
        return {
            "rmse": None, "mae": None, "psnr": None, "ssim": None,
            "n_valid_px": 0, "coverage_frac": 0.0, "ssim_gap_filled_frac": None,
        }

    o = original[valid].astype(np.float64)
    r = reconstructed[valid].astype(np.float64)

    diff = o - r
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    data_range = float(np.max(o) - np.min(o))
    if rmse == 0:
        psnr = float("inf")
    elif data_range <= 0:
        psnr = None
    else:
        psnr = 20 * math.log10(data_range) - 10 * math.log10(rmse ** 2)

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    ssim_val = None
    gap_filled_frac = None

    if rows.size and cols.size:
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        o_crop = original[r0:r1, c0:c1].astype(np.float64)
        r_crop = reconstructed[r0:r1, c0:c1].astype(np.float64)
        v_crop = valid[r0:r1, c0:c1]

        if v_crop.all():
            gap_filled_frac = 0.0
        else:
            fill_o = np.nanmean(o_crop[v_crop])
            fill_r = np.nanmean(r_crop[v_crop])
            gap_filled_frac = float((~v_crop).mean())
            o_crop = np.where(v_crop, o_crop, fill_o)
            r_crop = np.where(v_crop, r_crop, fill_r)

        min_side = min(o_crop.shape)
        if min_side >= 3:
            win = min(ssim_win_size, min_side if min_side % 2 == 1 else min_side - 1)
            win = max(3, win)
            data_range_crop = float(o_crop.max() - o_crop.min())
            if data_range_crop > 0:
                ssim_val = _ssim_uniform(o_crop, r_crop, win_size=win, data_range=data_range_crop)

    return {
        "rmse": rmse, "mae": mae, "psnr": psnr, "ssim": ssim_val,
        "n_valid_px": n_valid, "coverage_frac": coverage,
        "ssim_gap_filled_frac": gap_filled_frac,
    }


def compare_resampled_to_reference(resampled_arr, reference_arr, band_index=0, ssim_win_size=7):
    """
    Direct pixel-for-pixel comparison between a raster that was
    resampled onto the common grid and the OTHER raster's own native
    data on that same grid. Both arrays are already co-registered (same
    shape/footprint from put_on_common_grid), so no reprojection is
    needed here -- unlike assess_resampling_quality() (round-trip
    native->resampled->native) or compute_downsampling_error() (actual
    resampling vs the ideal area-average), this answers a more direct
    question: "how much does my resampled image disagree with what the
    other sensor actually measured on this shared grid."
    """
    resampled_band = resampled_arr[band_index] if resampled_arr.ndim == 3 else resampled_arr
    reference_band = reference_arr[band_index] if reference_arr.ndim == 3 else reference_arr
    return compute_error_metrics(resampled_band, reference_band, ssim_win_size=ssim_win_size)


def print_reference_comparison_report(metrics, resampled_label, reference_label):
    print(f"[{resampled_label}] Resampled image vs {reference_label}'s actual native data "
          f"(same common grid):")
    if metrics["rmse"] is None:
        print("  No jointly-valid pixels -- cannot assess.")
        return
    print(f"  valid pixels used: {metrics['n_valid_px']} "
          f"({metrics['coverage_frac'] * 100:.1f}% of frame)")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  MAE  = {metrics['mae']:.4f}")
    psnr = metrics["psnr"]
    print(f"  PSNR = {'inf' if psnr == float('inf') else (f'{psnr:.2f} dB' if psnr is not None else 'N/A (no dynamic range)')}")
    if metrics["ssim"] is not None:
        gap = metrics["ssim_gap_filled_frac"]
        gap_note = f", {gap * 100:.1f}% of SSIM crop was gap-filled" if gap else ""
        print(f"  SSIM = {metrics['ssim']:.4f}{gap_note}")
    else:
        print("  SSIM = N/A (region too small or no dynamic range)")

def print_algorithm_comparison_report(bilinear_metrics, manual_metrics, diff_metrics,
                                       bilinear_label, manual_label):
    """
    PSNR is only meaningful against a true ground truth, so it's reported
    individually for each resampling method vs the reference's actual
    native data (bilinear_metrics/manual_metrics -- both already computed
    against that same reference).

    RMSE/MAE/SSIM are instead computed directly between the two
    resampling methods' own outputs (diff_metrics), independent of the
    reference -- this answers "how much does the choice of algorithm
    change the result", not "how far is each from the ground truth".
    """
    def _fmt_psnr(m):
        if m is None or m.get("psnr") is None:
            return "N/A (no jointly-valid pixels or no dynamic range)"
        return "inf" if m["psnr"] == float("inf") else f"{m['psnr']:.2f} dB"

    print(f"[{bilinear_label}] PSNR "
          f"= {_fmt_psnr(bilinear_metrics)}")
    print(f"[{manual_label}] PSNR "
          f"= {_fmt_psnr(manual_metrics)}")

    print(f"[{manual_label} vs {bilinear_label}] Direct comparison:")
    if diff_metrics["rmse"] is None:
        print("  No jointly-valid pixels -- cannot assess.")
        return
    print(f"  valid pixels used: {diff_metrics['n_valid_px']} "
          f"({diff_metrics['coverage_frac'] * 100:.1f}% of frame)")
    print(f"  RMSE = {diff_metrics['rmse']:.4f}")
    print(f"  MAE  = {diff_metrics['mae']:.4f}")
    if diff_metrics["ssim"] is not None:
        gap = diff_metrics["ssim_gap_filled_frac"]
        gap_note = f", {gap * 100:.1f}% of SSIM crop was gap-filled" if gap else ""
        print(f"  SSIM = {diff_metrics['ssim']:.4f}{gap_note}")
    else:
        print("  SSIM = N/A (region too small or no dynamic range)")


def assess_resampling_quality(arr_native, native_transform, native_crs,
                               resampled_arr, resampled_transform, resampled_crs,
                               band_index=0, resampling=None,
                               num_threads=1, ssim_win_size=7):
    if resampling is None:
        resampling = Resampling.bilinear

    band_native = arr_native[band_index] if arr_native.ndim == 3 else arr_native
    band_resampled = resampled_arr[band_index] if resampled_arr.ndim == 3 else resampled_arr

    h, w = band_native.shape
    back = np.full((1, h, w), np.nan, dtype=np.float32)
    reproject(
        source=band_resampled[np.newaxis, :, :].astype(np.float32),
        destination=back,
        src_transform=resampled_transform, src_crs=resampled_crs,
        dst_transform=native_transform, dst_crs=native_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling,
        num_threads=num_threads,
    )

    metrics = compute_error_metrics(band_native, back[0], ssim_win_size=ssim_win_size)
    metrics["resampling"] = resampling.name
    return metrics


def compute_downsampling_error(arr_native, native_transform, native_crs,
                                coarse_transform, coarse_crs, coarse_width, coarse_height,
                                resampling, band_index=0, num_threads=1, ssim_win_size=7):
    """
    Quantify the error introduced purely by resampling a raster from its
    native (finer) resolution down onto a coarser grid -- as opposed to
    assess_resampling_quality()'s round-trip check, which conflates the
    downsampling error with the error of reprojecting back up to native
    resolution afterwards. This is single-direction: native -> coarse only.

    Returns two independent numbers, answering two different questions:

    1. algorithm_error: how far the ACTUAL resampling method (e.g.
       bilinear/cubic -- whatever `resampling` is) deviates from the
       mathematically "ideal" downsample: the true area-weighted average
       of every native pixel that falls inside each coarse output pixel
       (rasterio's Resampling.average). This is error from the CHOICE of
       interpolation algorithm, and is avoidable -- e.g. by resampling
       with Resampling.average instead.

    2. aggregation_rms_std: the RMS, over all coarse pixels, of the
       standard deviation of the native pixels collapsed into each single
       coarse pixel. This is the IRRECOVERABLE information loss of
       downsampling: real ground variation that no resampling algorithm
       can preserve once several native pixels become one coarse pixel.
       Compare it against your homogeneity threshold -- if it's large
       relative to that threshold, the coarser grid is discarding texture
       your CV-based homogeneity check would otherwise have flagged.
    """
    band = arr_native[band_index] if arr_native.ndim == 3 else arr_native
    band = band.astype(np.float32)

    def _reproject_average(source_band):
        dest = np.full((1, coarse_height, coarse_width), np.nan, dtype=np.float32)
        reproject(
            source=source_band[np.newaxis, :, :],
            destination=dest,
            src_transform=native_transform, src_crs=native_crs,
            dst_transform=coarse_transform, dst_crs=coarse_crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=num_threads,
        )
        return dest[0]

    ideal = _reproject_average(band)

    actual = np.full((1, coarse_height, coarse_width), np.nan, dtype=np.float32)
    reproject(
        source=band[np.newaxis, :, :],
        destination=actual,
        src_transform=native_transform, src_crs=native_crs,
        dst_transform=coarse_transform, dst_crs=coarse_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling,
        num_threads=num_threads,
    )
    actual = actual[0]

    algorithm_error = compute_error_metrics(ideal, actual, ssim_win_size=ssim_win_size)
    algorithm_error["resampling"] = resampling.name

    mean_of_squares = _reproject_average(band ** 2)
    variance = np.clip(mean_of_squares - ideal ** 2, 0, None)
    intra_pixel_std = np.sqrt(variance)
    valid = np.isfinite(intra_pixel_std)
    aggregation_rms_std = (
        float(np.sqrt(np.nanmean(intra_pixel_std[valid] ** 2))) if valid.any() else None
    )

    native_res = abs(native_transform.a)
    coarse_res = abs(coarse_transform.a)

    return {
        "native_res": native_res,
        "coarse_res": coarse_res,
        "downsample_factor": (coarse_res / native_res) if native_res else None,
        "algorithm_error": algorithm_error,
        "aggregation_rms_std": aggregation_rms_std,
    }


def print_downsampling_error_report(result, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Resolution-change error (native {result['native_res']:.3f} -> "
          f"coarse {result['coarse_res']:.3f}, {result['downsample_factor']:.2f}x downsample):")

    alg = result["algorithm_error"]
    if alg["rmse"] is None:
        print(f"{prefix}  No jointly-valid pixels -- cannot assess.")
    else:
        ssim_note = f"  SSIM={alg['ssim']:.4f}" if alg["ssim"] is not None else ""
        print(f"{prefix}  [{alg['resampling']} vs ideal area-average] "
              f"RMSE={alg['rmse']:.4f}  MAE={alg['mae']:.4f}{ssim_note}")

    if result["aggregation_rms_std"] is not None:
        print(f"{prefix}  Intra-pixel RMS std = {result['aggregation_rms_std']:.4f}")
    else:
        print(f"{prefix}  Intra-pixel RMS std = N/A (no valid pixels).")


def print_resampling_quality_report(metrics, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Resampling quality (native <-> reprojected-then-back, [{metrics.get('resampling', '?')}]):")
    if metrics["rmse"] is None:
        print(f"{prefix}  No jointly-valid pixels -- cannot assess.")
        return
    print(f"{prefix}  valid pixels used: {metrics['n_valid_px']} "
          f"({metrics['coverage_frac'] * 100:.1f}% of frame)")
    print(f"{prefix}  RMSE = {metrics['rmse']:.4f}")
    print(f"{prefix}  MAE  = {metrics['mae']:.4f}")
    psnr = metrics["psnr"]
    print(f"{prefix}  PSNR = {'inf' if psnr == float('inf') else (f'{psnr:.2f} dB' if psnr is not None else 'N/A (no dynamic range)')}")
    if metrics["ssim"] is not None:
        gap = metrics["ssim_gap_filled_frac"]
        gap_note = f", {gap * 100:.1f}% of SSIM crop was gap-filled" if gap else ""
        print(f"{prefix}  SSIM = {metrics['ssim']:.4f}{gap_note}")
    else:
        print(f"{prefix}  SSIM = N/A (region too small or no dynamic range)")
