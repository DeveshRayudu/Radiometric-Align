"""
raster_align.homogeneity

Coefficient-of-variation homogeneity maps in a moving window, NaN-safe,
3-state output (invalid / not-homogeneous / homogeneous).
"""

import numpy as np
from scipy.ndimage import uniform_filter


def homogeneity_cpu(bands_data, window_size=11, threshold=5.0, n_common_bands=None, min_valid_frac=0.5):
    if bands_data.ndim == 2:
        bands_data = bands_data[np.newaxis, :, :]
    if n_common_bands is not None:
        bands_data = bands_data[:n_common_bands]

    num_bands, height, width = bands_data.shape
    total_cv = np.zeros((height, width), dtype=np.float32)
    valid_all = np.ones((height, width), dtype=bool)
    win_area = window_size * window_size

    for b in range(num_bands):
        band = bands_data[b].astype(np.float32, copy=False)
        valid = np.isfinite(band)
        valid_all &= valid

        band_filled = np.where(valid, band, 0.0)
        w = valid.astype(np.float32)

        count = uniform_filter(w, size=window_size, mode="constant", cval=0.0) * win_area
        sum_ = uniform_filter(band_filled, size=window_size, mode="constant", cval=0.0) * win_area
        sumsq = uniform_filter(band_filled ** 2, size=window_size, mode="constant", cval=0.0) * win_area

        count_safe = np.where(count > 0, count, np.nan)
        mean = sum_ / count_safe
        variance = np.clip(sumsq / count_safe - mean ** 2, 0, None)
        std = np.sqrt(variance)
        safe_mean = np.where(np.abs(mean) < 1e-6, np.nan, np.abs(mean))
        cv = (std / safe_mean) * 100.0
        cv = np.where(count >= min_valid_frac * win_area, cv, np.nan)

        total_cv += np.nan_to_num(cv, nan=np.inf, posinf=np.inf, neginf=np.inf)

    homogeneous = (total_cv / num_bands <= threshold) & valid_all

    out = np.full((height, width), 1, dtype=np.uint8)
    out[~valid_all] = 0
    out[homogeneous] = 255
    return out


def compute_homogeneity_maps(img_src, img_tgt, window_size, threshold, n_common_bands):
    map_src = homogeneity_cpu(img_src[:n_common_bands], window_size, threshold)
    map_tgt = homogeneity_cpu(img_tgt[:n_common_bands], window_size, threshold)
    return map_src, map_tgt
