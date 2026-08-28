"""
raster_align.cli

Command-line argument parsing and program entry point.
"""

import argparse
from os.path import join
from os import getcwd
from time import time

import cv2
from rasterio.enums import Resampling

from .config import DEFAULT_NUM_THREADS, DEFAULT_MIN_OVERLAP_PCT
from .pipeline import run_pipeline


RESAMPLING_CHOICES = {
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "nearest": Resampling.nearest,
    "average": Resampling.average,
    "manual": "manual",
}


def _parse_nodata_arg(value):
    if value is None:
        return None, True
    if isinstance(value, str) and value.strip().lower() == "none":
        return None, False
    return float(value), True


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Align a target raster to a source raster and compare their homogeneity."
    )
    parser.add_argument("src", help="Path to the source (reference) raster.")
    parser.add_argument("target", help="Path to the target raster to align.")
    parser.add_argument("-o", "--output-dir", default=join(getcwd(), "outputs"), help="Output directory.")
    parser.add_argument("--window", type=int, default=11, help="Homogeneity moving-window size (pixels).")
    parser.add_argument("--threshold", type=float, default=5.0, help="Homogeneity CV%% threshold.")
    parser.add_argument("--resampling", choices=list(RESAMPLING_CHOICES.keys()), default="bilinear",
                         help="Resampling method used when reprojecting to the common grid (default: "
                              "bilinear). 'manual' uses a from-scratch exact area-weighted block average "
                              "(manual_resample.py) instead of any GDAL resampling call, and skips the "
                              "separate background 'Manual Resample QA' comparison since it would just be "
                              "comparing the chosen method against itself.")
    parser.add_argument("--resolution-mode", choices=["coarser", "finer", "src"], default="coarser",
                         help="Which resolution to resample both images to.")
    parser.add_argument("--min-overlap-pct", type=float, default=DEFAULT_MIN_OVERLAP_PCT,
                         help="Minimum required overlap between src and target footprints, as a %% of "
                              f"the smaller raster's area (default: {DEFAULT_MIN_OVERLAP_PCT:.1f}). Below "
                              "this, the pipeline reports the images as not overlapping and exits before "
                              "any resampling/alignment.")
    parser.add_argument("--pad-px", type=int, default=50,
                         help="Padding (in target-resolution pixels) added around the computed overlap "
                              "to tolerate header/geotransform inaccuracy before alignment.")
    parser.add_argument("--src-nodata", type=str, default=None,
                         help="Explicit nodata value for the src raster, used to OVERRIDE whatever the "
                              "header says (only pass this if the header nodata tag is wrong/missing and "
                              "you know the real value). If omitted (the default), the header's own "
                              "nodata tag is used; if the header has no nodata tag either, falls back to "
                              "0 (LISS dead-zone convention) with a warning. Pass 'none' to disable that "
                              "0-fallback if 0 is legitimate real data.")
    parser.add_argument("--target-nodata", type=str, default=None,
                         help="Explicit nodata value for the target raster. Same override/fallback rules "
                              "as --src-nodata.")
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS,
                         help="Threads GDAL's warp/reproject calls may use internally. Defaults to "
                              f"os.cpu_count() on this machine (currently {DEFAULT_NUM_THREADS}), not "
                              "a fixed cap -- pass a smaller value to leave headroom for other "
                              "processes, or raise it (rarely useful past your physical core count).")
    parser.add_argument("--no-resampling-qa", action="store_true",
                         help="Skip the resampling-quality (reconstruction error) check entirely.")
    parser.add_argument("--motion-type", choices=["euclidean", "affine"], default="euclidean",
                         help="ECC motion model. 'euclidean' restricts detection to translation+rotation "
                              "only. 'affine' also solves for scale/shear -- use this if you also need to "
                              "correct for slight sensor scaling differences between src and target.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    src = args.src
    target = args.target

    motion_type = cv2.MOTION_EUCLIDEAN if args.motion_type == "euclidean" else cv2.MOTION_AFFINE
    resampling = RESAMPLING_CHOICES[args.resampling]

    src_override, src_fallback_on = _parse_nodata_arg(args.src_nodata)
    tgt_override, tgt_fallback_on = _parse_nodata_arg(args.target_nodata)

    t0 = time()
    run_pipeline(src, target, args.output_dir,
                 homogeneity_window=args.window, homogeneity_threshold=args.threshold,
                 resampling=resampling,
                 resolution_mode=args.resolution_mode, pad_px=args.pad_px,
                 min_overlap_pct=args.min_overlap_pct,
                 num_threads=args.num_threads,
                 src_nodata_override=src_override, target_nodata_override=tgt_override,
                 src_nodata_fallback=(0.0 if src_fallback_on else None),
                 target_nodata_fallback=(0.0 if tgt_fallback_on else None),
                 motion_type=motion_type)

    print(f"\nTime Taken = {round(time() - t0)}sec")


if __name__ == "__main__":
    main()
