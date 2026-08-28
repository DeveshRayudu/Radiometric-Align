"""
raster_align.alignment

ECC-based shift / rotation / (optional scale) misalignment detection
between src and target, and building the corrected native transform.
"""

import math

import cv2
import numpy as np
from rasterio import Affine


def _to_uint8(arr):
    """Normalize a float array to 0-255 uint8 for OpenCV, masking NaN/Inf to 0."""
    a = arr.astype(np.float32)
    mask = np.isfinite(a)
    out = np.zeros(a.shape, dtype=np.uint8)
    if not mask.any():
        return out, np.zeros(a.shape, dtype=np.uint8)
    mn, mx = a[mask].min(), a[mask].max()
    if mx == mn:
        out[mask] = 128
    else:
        out[mask] = ((a[mask] - mn) / (mx - mn) * 255).astype(np.uint8)
    cv_mask = (mask * 255).astype(np.uint8)
    return out, cv_mask


def estimate_misalignment(band_src, band_tgt, motion_type=cv2.MOTION_EUCLIDEAN,
                           max_iterations=1000, termination_eps=1e-6, gauss_filt_size=5,
                           verbose=True, log_fn=print):
    """
    Estimate geometric misalignment between src and target via ECC
    (Enhanced Correlation Coefficient) image alignment.

    motion_type:
      - cv2.MOTION_EUCLIDEAN: constrains the search to translation + rotation
        only (no scale/shear). Use this when you trust the two rasters'
        pixel sizes and only expect a shift/rotation error.
      - cv2.MOTION_AFFINE: adds scale + shear (6 dof). Use this if you also
        need to correct for slight sensor scaling differences between the
        two rasters.

    verbose/log_fn: this function is called twice per pipeline run -- once
    for the initial detection (wants full console output) and once to
    re-verify the correction on the corrected pair (whose per-band details
    are redundant on console since pipeline.py already prints a concise
    summary of the result). Pass verbose=False with log_fn set to a
    detail-log-only callable for the second call to avoid duplicating the
    full ECC printout on console.
    """
    img_src_u8, mask_src = _to_uint8(band_src)
    img_tgt_u8, mask_tgt = _to_uint8(band_tgt)

    overlap_mask = cv2.bitwise_and(mask_src, mask_tgt)
    n_overlap = int(np.count_nonzero(overlap_mask))
    if n_overlap == 0:
        log_fn("[ECC] No overlapping valid pixels found.")
        return None
    if verbose:
        log_fn(f"[ECC] Overlap pixels: {n_overlap} ({n_overlap / overlap_mask.size * 100:.1f}% of frame)")

    img_src_masked = np.where(overlap_mask > 0, img_src_u8, 0).astype(np.uint8)
    img_tgt_masked = np.where(overlap_mask > 0, img_tgt_u8, 0).astype(np.uint8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            img_tgt_masked, img_src_masked, warp_matrix, motion_type, criteria,
            overlap_mask, gauss_filt_size,
        )
    except cv2.error as exc:
        log_fn(f"[ECC] Did not converge -- {exc}")
        return None

    M = warp_matrix.astype(np.float64)
    dx, dy = float(M[0, 2]), float(M[1, 2])
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    rotation_deg = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))

    if verbose:
        motion_name = {
            cv2.MOTION_EUCLIDEAN: "EUCLIDEAN (translation+rotation only)",
            cv2.MOTION_AFFINE: "AFFINE (translation+rotation+scale/shear)",
        }.get(motion_type, str(motion_type))
        log_fn(f"[ECC] motion model: {motion_name}")
        log_fn(f"[ECC] correlation coefficient: {cc:.4f}")
        log_fn(f"[ECC] Estimated transform: dx={dx:.2f}px, dy={dy:.2f}px, "
               f"rotation={rotation_deg:.3f}deg, scale={scale:.4f}")

    return {"M": M, "dx": dx, "dy": dy, "rotation_deg": rotation_deg, "scale": scale,
            "correlation_coefficient": float(cc)}


def transform_is_negligible(detail, dx_tol=0.5, dy_tol=0.5, rot_tol=0.1, scale_tol=0.01):
    return (abs(detail["dx"]) < dx_tol and abs(detail["dy"]) < dy_tol
            and abs(detail["rotation_deg"]) < rot_tol and abs(detail["scale"] - 1) < scale_tol)


def build_corrected_transform(native_transform, grid_transform, M):
    M_affine = Affine(M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])
    G = grid_transform
    return G * M_affine * (~G) * native_transform
