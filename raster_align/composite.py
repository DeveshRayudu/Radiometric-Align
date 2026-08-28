"""
raster_align.composite

Builds the red/green/blue "who is homogeneous" composite from the two
per-image homogeneity maps.
"""

import numpy as np



def build_rgb_composite(binary_map_src, binary_map_tgt, img_src, aligned_img_tgt):
    """
    """
    ref = img_src[0] if img_src.ndim == 3 else img_src
    tgt = aligned_img_tgt[0] if aligned_img_tgt.ndim == 3 else aligned_img_tgt

    if ref.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: src band1={ref.shape}, aligned target band1={tgt.shape}")

    rows, cols = binary_map_src.shape
    valid = np.isfinite(ref) & np.isfinite(tgt)

    src_homog = binary_map_src == 255
    tgt_homog = binary_map_tgt == 255

    both = src_homog & tgt_homog & valid
    only_src = src_homog & ~tgt_homog & valid
    only_tgt = ~src_homog & tgt_homog & valid


    common_map = np.zeros((1, rows, cols), dtype=np.float32)
    common_map[0][both] = 1.0

    rgb = np.zeros((3, rows, cols), dtype=np.uint8)
    rgb[0][only_src] = 255
    rgb[1][both] = 255
    rgb[2][only_tgt] = 255

    return rgb, common_map, valid
