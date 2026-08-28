"""
raster_align.pipeline

Top-level orchestration: read -> common grid -> ECC alignment -> optical-flow
diagnostics -> resampling QA (background) -> homogeneity -> RGB composite ->
crop -> write outputs -> plots.
"""

import math
import sys
from os import makedirs
from os.path import abspath, join
from time import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from rasterio.enums import Resampling

from .config import DEFAULT_NUM_THREADS, DEFAULT_MIN_OVERLAP_PCT
from .logging_utils import (
    open_run_logs, log_step_start, log_step_end, write_run_header, print_timing_summary,
    print_module_summary,
)
from .io_utils import read_and_prepare, compute_band_stats, build_metadata_row
from .crs_utils import crs_equal, validate_crs, crs_units_compatible, compute_overlap_bounds, \
    compute_overlap_percentage
from .grid import (
    put_on_common_grid, reproject_to_grid, reproject_to_grid_manual,
    reproject_to_crs_native_res,
)
from .alignment import estimate_misalignment, transform_is_negligible, build_corrected_transform
from .quality import (
    compare_resampled_to_reference, print_algorithm_comparison_report,
)
from .manual_resample import manual_block_average_resample
from .homogeneity import compute_homogeneity_maps
from .cropping import bbox_of_mask, crop_to_bbox, compute_common_footprint, polygon_crop
from .composite import build_rgb_composite
from .raster_io import write_raster
from .radiometric_correction import compute_common_reflectance
from .diagnostics import (
    compute_before_displacement, compute_after_displacement, write_shift_histograms, plot_shift_arrow,
    plot_shift_arrow_grid,
)

from .plot_common import plot_common
from .cross_sensor_metrics import compute_cross_sensor_report

MAX_BACKGROUND_WORKERS = 3

TOTAL_PIPELINE_STEPS = 12


def _report_progress(progress_callback, step_index, label):
    if progress_callback is not None:
        progress_callback(step_index, TOTAL_PIPELINE_STEPS, label)


def _resampling_name(resampling):
    """Return a stable label for Rasterio methods and custom sentinels."""
    return resampling if isinstance(resampling, str) else resampling.name


def run_pipeline(src_path, target_path, output_dir,
                  homogeneity_window=11, homogeneity_threshold=5.0,
                  resampling=Resampling.bilinear,
                  resolution_mode="coarser", pad_px=32,
                  num_threads=DEFAULT_NUM_THREADS,
                  src_nodata_override=None, target_nodata_override=None,
                  src_nodata_fallback=0.0, target_nodata_fallback=0.0,
                  min_valid_frac_warn=0.85,
                  min_overlap_pct=DEFAULT_MIN_OVERLAP_PCT,
                  motion_type=cv2.MOTION_EUCLIDEAN,
                  log_filename="run_log.txt",
                  detail_log_filename="run_log_detailed.txt",
                  short_log_callback=None,
                  console_callback=None,
                  progress_callback=None,
                  figure_callback=None):
    """
    Run the full alignment pipeline.

    Unchanged from before: src_path, target_path, and output_dir are plain
    arguments -- the CLI and any other caller (e.g. a GUI) both just pass
    whatever paths/run-specific output directory they've resolved; the
    pipeline itself has no knowledge of where those values came from.

    New, both optional and no-ops if omitted, so existing callers (the CLI)
    are unaffected:
      short_log_callback: callable(line: str), forwarded to open_run_logs.
          Fed exactly what run_log.txt receives -- the run header, one
          result line per pipeline module (plus a few key highlight
          lines, e.g. detected shift), and the final timing totals. This
          is what a GUI console should be wired to, so it mirrors the
          small/precise log rather than the full verbose one.
      console_callback: deprecated alias for short_log_callback, kept for
          backwards compatibility with older callers. Ignored if
          short_log_callback is also given.
      progress_callback: callable(step_index: int, total_steps: int,
          label: str), invoked at the start of each major pipeline stage.
          Lets a GUI drive a progress bar.
      figure_callback: callable(dict), invoked once near the end of the
          run with {filename: live matplotlib Figure} for the
          chart-style outputs (currently: plot_common.png,
          easting_error_histogram.png, northing_error_histogram.png --
          NOT the raster-band diagnostic PNGs, which have no live
          Figure). Left None (the default), this costs nothing extra --
          the underlying plot functions still close their figures
          exactly as before. Given a callback, those figures are kept
          open and handed off instead of closed here; the callback
          becomes responsible for eventually closing them (e.g. when a
          GUI viewer tab for that run is discarded).
    """

    makedirs(output_dir, exist_ok=True)
    plots_dir = join(output_dir, "plots")
    makedirs(plots_dir, exist_ok=True)

    tee_stdout, short_log_file, detail_log_file, detail_log = open_run_logs(
        output_dir, short_filename=log_filename, detail_filename=detail_log_filename,
        console_callback=short_log_callback if short_log_callback is not None else console_callback,
    )
    original_stdout = sys.stdout
    sys.stdout = tee_stdout

    step_times = []
    pipeline_t0 = time()
    bg_pool = ThreadPoolExecutor(max_workers=MAX_BACKGROUND_WORKERS)
    live_figures = {}

    pipeline_result = {"plot_stats": None, "cross_sensor_report": None}

    try:
        t0 = time()
        _report_progress(progress_callback, 1, "Read + prepare rasters")
        log_step_start(detail_log, "1. Read + prepare rasters")
        arr_src, info_src = read_and_prepare(
            src_path, nodata_override=src_nodata_override, fallback_nodata=src_nodata_fallback
        )
        arr_tgt, info_tgt = read_and_prepare(
            target_path, nodata_override=target_nodata_override, fallback_nodata=target_nodata_fallback
        )
        _report_progress(progress_callback, 2, "Reflectance calculation")
        reflectance_src, reflectance_tgt, rad_details = compute_common_reflectance(
            info_src, info_tgt, arr_src, arr_tgt,
        )
        detail_log(
            f"[Radiometric] Converted src DN -> TOA Reflectance at pipeline start "
            f"(sensor={rad_details['sensor_src']}, band={rad_details['band_name_src']}): "
            f"{rad_details['reflectance_src_stats']}"
        )
        detail_log(
            f"[Radiometric] Converted target DN -> TOA Reflectance at pipeline start "
            f"(sensor={rad_details['sensor_tgt']}, band={rad_details['band_name_tgt']}): "
            f"{rad_details['reflectance_tgt_stats']}"
        )
        if rad_details["n_saturated_src"] or rad_details["n_saturated_tgt"]:
            detail_log(
                f"[Saturation] Masked {rad_details['n_saturated_src']:,} saturated src DN pixel(s) "
                f"(max_dn={rad_details['max_dn_src']}) and {rad_details['n_saturated_tgt']:,} saturated "
                f"target DN pixel(s) (max_dn={rad_details['max_dn_tgt']}) to NaN before reflectance "
                f"conversion -- excluded from every downstream stage (alignment, homogeneity, plot, stats)."
            )
        print_module_summary(
            "Radiometric Conversion (DN -> TOA Reflectance)",
            params={
                "src sensor": rad_details["sensor_src"], "src band": rad_details["band_name_src"],
                "target sensor": rad_details["sensor_tgt"], "target band": rad_details["band_name_tgt"],
            },
            checks=[
                f"Resolved calibration metadata -- src: {rad_details['meta_path_src']}, "
                f"target: {rad_details['meta_path_tgt']}",
                f"Checked src DN against max_dn={rad_details['max_dn_src']} for saturation "
                f"({rad_details['n_saturated_src']:,} pixel(s) saturated).",
                f"Checked target DN against max_dn={rad_details['max_dn_tgt']} for saturation "
                f"({rad_details['n_saturated_tgt']:,} pixel(s) saturated).",
            ],
            actions=[
                "Masked saturated DN pixels (if any) to NaN before conversion, for both images.",
                "Converted src DN -> TOA Radiance -> TOA Reflectance.",
                "Converted target DN -> TOA Radiance -> TOA Reflectance.",
                "Done BEFORE common-grid construction/resampling -- every downstream stage "
                "(alignment, resampling, homogeneity, composite) now operates on reflectance.",
            ],
            result=f"OK -- src {rad_details['reflectance_src_stats']}; "
                   f"target {rad_details['reflectance_tgt_stats']}",
        )
        arr_src = reflectance_src
        arr_tgt = reflectance_tgt

        motion_name = {cv2.MOTION_EUCLIDEAN: "MOTION_EUCLIDEAN", cv2.MOTION_AFFINE: "MOTION_AFFINE"}.get(
            motion_type, str(motion_type)
        )
        write_run_header(
            [
                ("src path", abspath(src_path)),
                ("target path", abspath(target_path)),
                ("output dir", abspath(output_dir)),
            ],
            config_params=[
                ("homogeneity window", homogeneity_window),
                ("homogeneity threshold (CV%)", homogeneity_threshold),
                ("resampling method", _resampling_name(resampling)),
                ("resolution mode", resolution_mode),
                ("pad_px", pad_px),
                ("num_threads", num_threads),
                ("src nodata fallback", src_nodata_fallback),
                ("target nodata fallback", target_nodata_fallback),
                ("motion_type", motion_name),
            ],
        )

        img_load_checks = [
            f"src readable via rasterio -- {info_src['width_px']}x{info_src['height_px']}px, "
            f"{info_src['band_count']} band(s), dtype={info_src['dtype']}, driver={info_src['driver']}",
            f"target readable via rasterio -- {info_tgt['width_px']}x{info_tgt['height_px']}px, "
            f"{info_tgt['band_count']} band(s), dtype={info_tgt['dtype']}, driver={info_tgt['driver']}",
            f"src nodata resolved to {info_src['nodata']}",
            f"target nodata resolved to {info_tgt['nodata']}",
        ]
        print_module_summary(
            "Image Loading",
            params={
                "src nodata override": src_nodata_override,
                "target nodata override": target_nodata_override,
                "src nodata fallback": src_nodata_fallback,
                "target nodata fallback": target_nodata_fallback,
            },
            checks=img_load_checks,
            actions=["Loaded both rasters into memory as float32 arrays; nodata pixels (if any) set to NaN."],
            result="OK -- both rasters loaded and nodata-normalized.",
        )


        _report_progress(progress_callback, 3, "CRS validation")
        validate_crs(info_src["crs"], "src")
        validate_crs(info_tgt["crs"], "target")

        src_epsg = info_src["epsg"]
        tgt_epsg = info_tgt["epsg"]
        src_crs_label = f"EPSG:{src_epsg}" if src_epsg is not None else str(info_src["crs"])
        tgt_crs_label = f"EPSG:{tgt_epsg}" if tgt_epsg is not None else str(info_tgt["crs"])

        crs_match = crs_equal(info_src["crs"], info_tgt["crs"])
        units_ok = crs_units_compatible(info_src["crs"], info_tgt["crs"])

        crs_checks = [
            f"src CRS well-formed and defined: {src_crs_label}",
            f"target CRS well-formed and defined: {tgt_crs_label}",
            f"src/target CRS equal: {'yes' if crs_match else 'no'}",
            f"src/target coordinate units compatible (both geographic or both projected): "
            f"{'yes' if units_ok else 'no'}",
        ]
        crs_actions = []
        if crs_match:
            crs_actions.append("No reprojection required for CRS alignment -- src and target already share a CRS.")
        else:
            crs_actions.append(
                f"Reprojection required: target will be reprojected from {tgt_crs_label} "
                f"to src's CRS ({src_crs_label}) during common-grid construction (next module)."
            )
        if not units_ok:
            crs_actions.append(
                "WARNING: src is "
                f"{'geographic (degrees)' if info_src['crs'].is_geographic else 'projected (linear units)'} "
                "and target is "
                f"{'geographic (degrees)' if info_tgt['crs'].is_geographic else 'projected (linear units)'} -- "
                "raw pixel-size comparisons used to pick the reference grid are not meaningful until "
                "reprojection puts both rasters in the same kind of units; this is corrected automatically "
                "downstream."
            )

        stats_src = compute_band_stats(arr_src)
        stats_tgt = compute_band_stats(arr_tgt)
        detail_log(f"[Metadata] src raster: {build_metadata_row(info_src, stats_src)}")
        detail_log(f"[Metadata] target raster: {build_metadata_row(info_tgt, stats_tgt)}")

        print_module_summary(
            "CRS Validation",
            checks=crs_checks,
            actions=crs_actions,
            result="OK -- both CRS valid; " + ("no reprojection needed." if crs_match else "target will be reprojected to src's CRS."),
        )

        dt = time() - t0
        step_times.append(("1. Read + prepare rasters", dt))
        log_step_end(detail_log, "1. Read + prepare rasters", dt)

        t0 = time()
        _report_progress(progress_callback, 4, "Common-area (overlap) check")
        log_step_start(detail_log, "1b. Common-area (overlap) check")
        overlap_bounds = compute_overlap_bounds(
            info_src["bounds"], info_src["crs"], info_tgt["bounds"], info_tgt["crs"]
        )
        overlap_params = {
            "min_overlap_pct required": f"{min_overlap_pct:.1f}%",
        }
        if overlap_bounds is None:
            print_module_summary(
                "Overlap Detection",
                params=overlap_params,
                checks=["Intersected src and target footprints in src's CRS."],
                actions=["No further processing performed -- pipeline halted."],
                result="STOPPED -- no common area (intersection is empty); "
                       "nothing to align/resample.",
            )
            detail_log(f"[Overlap] src bounds={info_src['bounds']}, target bounds={info_tgt['bounds']}, "
                       f"no intersection found.")
            sys.exit(1)

        overlap_pct = compute_overlap_percentage(
            overlap_bounds, info_src["bounds"], info_src["crs"], info_tgt["bounds"], info_tgt["crs"]
        )
        ov_left, ov_bottom, ov_right, ov_top = overlap_bounds

        if overlap_pct < min_overlap_pct:
            print_module_summary(
                "Overlap Detection",
                params=overlap_params,
                checks=[
                    "Intersected src and target footprints in src's CRS.",
                    f"Overlap region: left={ov_left:.6f}, bottom={ov_bottom:.6f}, "
                    f"right={ov_right:.6f}, top={ov_top:.6f}",
                    f"Overlap = {overlap_pct:.2f}% of the smaller raster's footprint "
                    f"(minimum required: {min_overlap_pct:.1f}%)",
                ],
                actions=["No further processing performed -- pipeline halted."],
                result=f"STOPPED -- overlap ({overlap_pct:.2f}%) is below the minimum required "
                       f"({min_overlap_pct:.1f}%); not enough shared ground to align on.",
            )
            detail_log(f"[Overlap] overlap_pct={overlap_pct:.4f}% < min_overlap_pct={min_overlap_pct:.1f}% -- "
                       f"src bounds={info_src['bounds']}, target bounds={info_tgt['bounds']}, "
                       f"overlap_bounds={overlap_bounds}.")
            sys.exit(1)

        n_common = min(info_src["band_count"], info_tgt["band_count"])
        band_action = (
            f"Using first {n_common} common band(s) (src has {info_src['band_count']}, "
            f"target has {info_tgt['band_count']})."
            if info_src["band_count"] != info_tgt["band_count"]
            else f"Using all {n_common} band(s) (both images have the same band count)."
        )

        print_module_summary(
            "Overlap Detection",
            params=overlap_params,
            checks=[
                "Intersected src and target footprints in src's CRS.",
                f"Overlap region: left={ov_left:.6f}, bottom={ov_bottom:.6f}, "
                f"right={ov_right:.6f}, top={ov_top:.6f}",
                f"Overlap = {overlap_pct:.2f}% of the smaller raster's footprint "
                f"(minimum required: {min_overlap_pct:.1f}%)",
                f"Band count -- src: {info_src['band_count']}, target: {info_tgt['band_count']}",
            ],
            actions=[band_action],
            result=f"OK -- sufficient overlap ({overlap_pct:.2f}% >= {min_overlap_pct:.1f}%); continuing.",
        )
        dt = time() - t0
        step_times.append(("1b. Common-area (overlap) check", dt))
        log_step_end(detail_log, "1b. Common-area (overlap) check", dt,
                     overlap_bounds=overlap_bounds, overlap_pct=overlap_pct)

        res_src, res_tgt = info_src["res"][0], info_tgt["res"][0]
        res_match = res_src == res_tgt

        t0 = time()
        _report_progress(progress_callback, 5, "Put on common grid")
        log_step_start(detail_log, "2. Put on common grid", resolution_mode=resolution_mode, pad_px=pad_px)
        img_src, img_tgt, ref_transform, ref_crs, ref_width, ref_height, resampled_src, resampled_tgt, \
            img_src_detect, img_tgt_detect = put_on_common_grid(
            arr_src, info_src, arr_tgt, info_tgt, resolution_mode, pad_px, resampling, num_threads,
        )
        dt = time() - t0
        step_times.append(("2. Put on common grid", dt))
        log_step_end(detail_log, "2. Put on common grid", dt, resampled_src=resampled_src, resampled_tgt=resampled_tgt)

        resample_actions = [
            f"src: {'Resampled to common grid (required -- CRS/transform/size did not already match).' if resampled_src else 'No resampling applied (skipped -- already on the common grid).'}",
            f"target: {'Resampled to common grid (required -- CRS/transform/size did not already match).' if resampled_tgt else 'No resampling applied (skipped -- already on the common grid).'}",
        ]
        print_module_summary(
            "Resolution Matching",
            params={
                "resampling method": _resampling_name(resampling), "resolution_mode": resolution_mode,
                "pad_px": pad_px,
            },
            checks=[
                f"Compared src pixel size ({res_src}) to target pixel size ({res_tgt}).",
                "Checked whether src and target already share CRS/transform/pixel-grid with the target common grid.",
            ],
            actions=(
                ["No resampling required for resolution -- src and target pixel sizes already match."]
                if res_match else
                [f"Resampling required -- src/target pixel sizes differ "
                 f"({res_src} vs {res_tgt}); the finer-resolution image is resampled to the "
                 f"'{resolution_mode}' image's resolution."]
            ) + resample_actions,
            result=f"OK -- common grid established at {ref_width}x{ref_height}px "
                   f"({'resolutions already matched' if res_match else f'resampled to {resolution_mode} resolution'}).",
        )

        manual_resample_future = None
        manual_resample_label = None
        if info_src["res"][0] != info_tgt["res"][0] and resampling != "manual":
            src_is_finer = info_src["res"][0] < info_tgt["res"][0]
            finer_arr = arr_src if src_is_finer else arr_tgt
            finer_transform = info_src["transform"] if src_is_finer else info_tgt["transform"]
            finer_crs = info_src["crs"] if src_is_finer else info_tgt["crs"]
            manual_resample_label = "src" if src_is_finer else "target"
            manual_resample_t0 = time()
            log_step_start(detail_log, "Manual Resample QA (background)", finer=manual_resample_label)
            if not crs_equal(finer_crs, ref_crs):
                finer_arr_for_manual, finer_transform_for_manual = reproject_to_crs_native_res(
                    finer_arr, finer_transform, finer_crs, ref_crs,
                    resampling=Resampling.bilinear, num_threads=num_threads,
                )
            else:
                finer_arr_for_manual, finer_transform_for_manual = finer_arr, finer_transform
            manual_resample_future = bg_pool.submit(
                manual_block_average_resample, finer_arr_for_manual, finer_transform_for_manual,
                ref_transform, ref_width, ref_height
            )

        resampled_vs_reference_metrics = None
        bilinear_resampled_arr = None
        if info_src["res"][0] != info_tgt["res"][0]:
            t0 = time()
            if info_src["res"][0] < info_tgt["res"][0]:
                bilinear_resampled_arr = img_src
                resampled_label = "src"
                metrics = compare_resampled_to_reference(img_src, img_tgt)
            else:
                bilinear_resampled_arr = img_tgt
                resampled_label = "target"
                metrics = compare_resampled_to_reference(img_tgt, img_src)
            resampled_vs_reference_metrics = metrics
            dt = time() - t0
            step_times.append(("2b. Resampled-vs-reference comparison", dt))
            detail_log(f"[Resampled-vs-reference metrics] {metrics}")
            print_module_summary(
            "Resampling QA (resampled vs reference)",
                checks=[f"Compared the {_resampling_name(resampling)}-resampled {resampled_label} directly against the other "
                        f"image's own native data on the shared common grid (full metrics in detail log)."],
                actions=[f"Computed RMSE/MAE/PSNR/SSIM for the resampled {resampled_label}."],
                result=f"OK -- comparison done in {dt:.2f}s.",
            )

        t0 = time()
        motion_name = {cv2.MOTION_EUCLIDEAN: "MOTION_EUCLIDEAN", cv2.MOTION_AFFINE: "MOTION_AFFINE"}.get(
            motion_type, str(motion_type)
        )
        _report_progress(progress_callback, 6, "ECC misalignment detection")
        log_step_start(detail_log, "3. ECC misalignment detection", motion_type=motion_name)
        max_dim = max(img_src.shape[1], img_src.shape[2])
        detect_max_dim = 2000
        step = max(1, math.ceil(max_dim / detect_max_dim))

        if step > 1:
            detail_log(f"[ECC] Decimating by stride {step} for detection ({max_dim}px -> ~{max_dim // step}px).")
            band_src_small = img_src_detect[0][::step, ::step]
            band_tgt_small = img_tgt_detect[0][::step, ::step]
        else:
            band_src_small, band_tgt_small = img_src_detect[0], img_tgt_detect[0]

        detail = estimate_misalignment(band_src_small, band_tgt_small, motion_type=motion_type,
                                        verbose=True, log_fn=detail_log)
        if detail is not None and step > 1:
            detail["dx"] *= step
            detail["dy"] *= step
            detail["M"] = detail["M"].copy()
            detail["M"][0, 2] *= step
            detail["M"][1, 2] *= step
            detail_log(f"[ECC] Scaled back to full resolution: dx={detail['dx']:.2f}px, dy={detail['dy']:.2f}px")

        ecc_correction_applied = detail is not None and not transform_is_negligible(detail)

        ecc_checks = [f"Ran ECC ({motion_name}) misalignment detection between src and target."]
        if detail is None:
            ecc_checks.append("ECC did not converge / found no usable overlap -- no transform estimated.")
        else:
            ecc_checks.append(
                f"Estimated transform: dx={detail['dx']:.2f}px, dy={detail['dy']:.2f}px, "
                f"rotation={detail['rotation_deg']:.3f}deg, scale={detail['scale']:.4f} "
                f"(correlation={detail['correlation_coefficient']:.4f})"
            )
            ecc_checks.append(
                "Checked estimated transform against negligibility tolerances "
                "(dx/dy < 0.5px, rotation < 0.1deg, scale within 1%)."
            )

        if not ecc_correction_applied:
            reason = "no usable overlap for detection" if detail is None else "estimated shift is within tolerance"
            aligned_tgt = img_tgt
            ecc_actions = [f"Skipped correction (not required: {reason})."]
        else:
            grid = {"crs": ref_crs, "transform": ref_transform, "width": ref_width, "height": ref_height}
            corrected_transform = build_corrected_transform(info_tgt["transform"], ref_transform, detail["M"])
            if resampling == "manual":
                aligned_tgt = reproject_to_grid_manual(arr_tgt, corrected_transform, info_tgt["crs"], grid, num_threads)
            else:
                aligned_tgt = reproject_to_grid(arr_tgt, corrected_transform, info_tgt["crs"], grid, resampling, num_threads)

            ecc_actions = [
                f"Applied sub-pixel correction to target (required: detected shift exceeds tolerance): "
                f"dx={detail['dx']:.2f}px, dy={detail['dy']:.2f}px, rotation={detail['rotation_deg']:.3f}deg, "
                f"scale={detail['scale']:.4f}."
            ]

        dt = time() - t0
        step_times.append(("3. ECC misalignment detection + correction", dt))
        log_step_end(detail_log, "3. ECC misalignment detection", dt, detail=detail, correction_applied=ecc_correction_applied)

        if detail is None:
            shift_highlight = "Shift detected: none (ECC did not converge) -- correction NOT applied."
        else:
            shift_highlight = (
                f"Shift detected: dx={detail['dx']:.2f}px, dy={detail['dy']:.2f}px, "
                f"rotation={detail['rotation_deg']:.3f}deg, scale={detail['scale']:.4f} -- "
                f"correction {'APPLIED' if ecc_correction_applied else 'NOT applied (within tolerance)'}."
            )

        print_module_summary(
            "Alignment (ECC)",
            params={"motion_type": motion_name},
            checks=ecc_checks,
            actions=ecc_actions,
            highlights=shift_highlight,
            result=("OK -- correction applied to target." if ecc_correction_applied
                     else "OK -- no correction needed, target used as-is."),
        )

        if ecc_correction_applied:
            t0 = time()
            plot_shift_arrow(img_src[0], detail["dx"], detail["dy"],
                              detail["rotation_deg"], detail["scale"], plots_dir)
            step_times.append(("3b. Shift-direction arrow plot", time() - t0))

        shift_val_t0 = time()
        log_step_start(detail_log, "Shift Validation (background, 2nd ECC pass)")
        aligned_tgt_small = aligned_tgt[0][::step, ::step] if step > 1 else aligned_tgt[0]
        shift_val_future = bg_pool.submit(
            estimate_misalignment, band_src_small, aligned_tgt_small, motion_type,
            1000, 1e-6, 5, False, detail_log,
        )

        flow_t0 = time()
        log_step_start(detail_log, "Optical Flow (background)")
        flow_future = bg_pool.submit(compute_before_displacement, img_src, img_tgt)
        flow_after_future = bg_pool.submit(compute_after_displacement, img_src, aligned_tgt)

        t0 = time()
        _report_progress(progress_callback, 7, "Final Polygon Cropping")
        log_step_start(detail_log, "4. Final Polygon Cropping")

        polygon_crop_checks = [
            "Checked for a jointly-valid (finite in both src and aligned target) pixel region "
            "to determine the initial bounding-box crop extent.",
        ]
        polygon_crop_actions = []
        total_crop_bbox = None
        footprint_mask = None  # set below once the polygon crop is applied

        early_valid_mask = np.isfinite(img_src[0]) & np.isfinite(aligned_tgt[0])
        early_bbox = bbox_of_mask(early_valid_mask)

        if early_bbox is None:
            footprint_mask = np.zeros((ref_height, ref_width), dtype=bool)
            frac_valid_both = 0.0
            polygon_crop_actions.append(
                "Skipped all cropping (not possible: no pixels are valid in both images after alignment)."
            )
            polygon_crop_result = (
                f"WARNING -- no valid common region found; continuing at full padded-grid extent "
                f"({ref_width}x{ref_height}px). All downstream outputs will be all-nodata."
            )
        else:
            er0, er1, ec0, ec1 = early_bbox
            bbox_crop_was_applied = (er0, ec0, er1, ec1) != (0, 0, ref_height, ref_width)

            if bbox_crop_was_applied:
                pre_bbox_shape = (ref_width, ref_height)
                ref_transform, (img_src, img_tgt, aligned_tgt) = crop_to_bbox(
                    ref_transform, er0, er1, ec0, ec1, img_src, img_tgt, aligned_tgt,
                )
                ref_height, ref_width = er1 - er0, ec1 - ec0
                polygon_crop_checks.append(
                    f"Valid-data bbox found: rows[{er0}:{er1}] cols[{ec0}:{ec1}]; "
                    f"bbox is smaller than the full padded grid ({pre_bbox_shape[0]}x{pre_bbox_shape[1]}px)."
                )
                polygon_crop_actions.append(
                    f"Applied bounding-box pre-crop from {pre_bbox_shape[0]}x{pre_bbox_shape[1]}px "
                    f"to {ref_width}x{ref_height}px to eliminate empty padded margins before footprint analysis."
                )
            else:
                polygon_crop_checks.append(
                    f"Valid-data bbox rows[{er0}:{er1}] cols[{ec0}:{ec1}] equals the full grid extent -- "
                    f"no bounding-box pre-crop needed."
                )

            common_footprint, src_footprint, tgt_footprint = compute_common_footprint(
                np.isfinite(img_src[0]), np.isfinite(aligned_tgt[0]), ref_transform
            )
            polygon_crop_checks.append(
                "Built src and aligned-target valid-data footprint polygons and intersected them "
                "to determine the common valid-data footprint."
            )

            src_area = src_footprint.area if src_footprint is not None else 0.0
            tgt_area = tgt_footprint.area if tgt_footprint is not None else 0.0

            if common_footprint is None:
                footprint_mask = np.zeros((ref_height, ref_width), dtype=bool)
                frac_valid_both = 0.0
                if bbox_crop_was_applied:
                    total_crop_bbox = (er0, er1, ec0, ec1)
                polygon_crop_checks.append(
                    f"Source footprint area={src_area:.3f}, target footprint area={tgt_area:.3f}, "
                    f"common footprint area=0.0 (no overlap)."
                )
                polygon_crop_actions.append(
                    "Skipped polygon crop (not possible: src and aligned-target footprints do not intersect) -- "
                    "outputs will be written at the bbox-cropped extent with all-nodata pixels."
                )
                polygon_crop_result = (
                    f"WARNING -- footprints do not overlap; final size {ref_width}x{ref_height}px "
                    f"({ref_width * ref_height:,} px/band, {n_common} band(s)). "
                    f"All downstream outputs will be all-nodata."
                )
            else:
                common_area = common_footprint.area
                polygon_crop_checks.append(
                    f"Source footprint area={src_area:.3f}, target footprint area={tgt_area:.3f}, "
                    f"common footprint area={common_area:.3f}."
                )

                pre_polygon_shape = (ref_width, ref_height)
                (
                    ref_transform,
                    (img_src, img_tgt, aligned_tgt),
                    footprint_mask,
                    (crop_r0, crop_r1, crop_c0, crop_c1),
                ) = polygon_crop(
                    ref_transform, common_footprint,
                    img_src, img_tgt, aligned_tgt,
                )
                ref_height, ref_width = crop_r1 - crop_r0, crop_c1 - crop_c0

                if bbox_crop_was_applied:
                    total_crop_bbox = (
                        er0 + crop_r0, er0 + crop_r1,
                        ec0 + crop_c0, ec0 + crop_c1,
                    )
                else:
                    total_crop_bbox = (crop_r0, crop_r1, crop_c0, crop_c1)

                frac_valid_both = float(footprint_mask.mean()) if footprint_mask.size else 0.0
                n_output_pixels = ref_width * ref_height

                polygon_crop_checks.append(
                    f"Common-footprint bbox (within the pre-cropped grid): "
                    f"rows[{crop_r0}:{crop_r1}] cols[{crop_c0}:{crop_c1}]; "
                    f"output width={ref_width}, height={ref_height}."
                )
                polygon_crop_checks.append(
                    f"{frac_valid_both * 100:.1f}% of the cropped extent is within the common "
                    f"valid-data footprint (minimum expected: {min_valid_frac_warn * 100:.0f}%)."
                )
                polygon_crop_actions.append(
                    f"Polygon-cropped src, target, and aligned target "
                    f"from {pre_polygon_shape[0]}x{pre_polygon_shape[1]}px to {ref_width}x{ref_height}px, "
                    f"blanking any pixel outside the common footprint polygon."
                )

                if frac_valid_both < min_valid_frac_warn:
                    polygon_crop_result = (
                        f"WARNING -- less than {min_valid_frac_warn * 100:.0f}% of the crop is within the "
                        f"common valid-data footprint ({frac_valid_both * 100:.1f}%); "
                        f"final size {ref_width}x{ref_height}px ({n_output_pixels:,} px/band, {n_common} band(s))."
                    )
                else:
                    polygon_crop_result = (
                        f"OK -- final size {ref_width}x{ref_height}px "
                        f"({n_output_pixels:,} px/band, {n_common} band(s))."
                    )

        print_module_summary(
            "Final Polygon Cropping",
            checks=polygon_crop_checks,
            actions=polygon_crop_actions,
            result=polygon_crop_result,
        )

        dt = time() - t0
        step_times.append(("4. Final Polygon Cropping", dt))
        log_step_end(detail_log, "4. Final Polygon Cropping", dt, frac_valid_both=frac_valid_both)

        t0 = time()
        _report_progress(progress_callback, 8, "Homogeneity maps")
        log_step_start(detail_log, "5. Homogeneity maps", window=homogeneity_window, threshold=homogeneity_threshold)
        binary_map_src, binary_map_tgt = compute_homogeneity_maps(
            img_src, aligned_tgt, homogeneity_window, homogeneity_threshold, n_common
        )
        dt = time() - t0
        step_times.append(("5. Homogeneity maps", dt))
        log_step_end(detail_log, "5. Homogeneity maps", dt)

        n_homog_src = int(np.count_nonzero(binary_map_src == 255))
        n_homog_tgt = int(np.count_nonzero(binary_map_tgt == 255))
        n_px_total = binary_map_src.size
        print_module_summary(
            "Homogeneity Analysis",
            params={"window": f"{homogeneity_window}px", "threshold": f"{homogeneity_threshold}% CV"},
            checks=[
                f"Computed moving-window coefficient-of-variation for src -> "
                f"{n_homog_src:,}/{n_px_total:,} px ({n_homog_src / n_px_total * 100:.1f}%) homogeneous.",
                f"Computed moving-window coefficient-of-variation for target -> "
                f"{n_homog_tgt:,}/{n_px_total:,} px ({n_homog_tgt / n_px_total * 100:.1f}%) homogeneous.",
            ],
            actions=["Produced per-image binary homogeneity maps (255=homogeneous, 1=not homogeneous, 0=invalid)."],
            result=f"OK -- homogeneity maps computed in {dt:.2f}s.",
        )

        t0 = time()
        _report_progress(progress_callback, 9, "RGB composite")
        log_step_start(detail_log, "6. RGB composite")
        rgb, common_map, valid_both = build_rgb_composite(
            binary_map_src, binary_map_tgt, img_src[:n_common], aligned_tgt[:n_common]
        )
        dt = time() - t0
        step_times.append(("6. RGB composite", dt))
        log_step_end(detail_log, "6. RGB composite", dt)

        n_both = int(np.count_nonzero(common_map[0] != 0)) if common_map.size else 0
        n_only_src = int(np.count_nonzero((rgb[0] == 255) & (rgb[1] == 0)))
        n_only_tgt = int(np.count_nonzero((rgb[2] == 255) & (rgb[1] == 0)))

        valid_both = valid_both & footprint_mask
        common_map = np.nan_to_num(common_map, nan=0.0)

        _KEEP_PROFILE_KEYS = {"crs", "transform", "width", "height", "count", "dtype", "nodata"}
        base_profile = {k: v for k, v in info_src["profile"].items() if k in _KEEP_PROFILE_KEYS}
        base_profile.update(
            driver="GTiff",
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
        )

        t_rasters0 = time()
        _report_progress(progress_callback, 10, "Write output rasters")
        log_step_start(detail_log, "7. Write output rasters")
        write_raster(output_dir, "homogeneity_sample_src.tif", binary_map_src, base_profile, np.uint8, nodata_value=0)
        write_raster(output_dir, "homogeneity_sample_target.tif", binary_map_tgt, base_profile, np.uint8, nodata_value=0)
        write_raster(output_dir, "common_homogeneous_rgb.tif", rgb, base_profile, np.uint8, nodata_value=None)
        # Save only the common-homogeneous pixels for both src and target.
        # Pixels outside the intersection of both homogeneity maps are set to
        # NaN (nodata) so only common-homogeneous pixels are retained.
        _homo_mask_2d = common_map[0] != 0          # (H, W) bool — True = common + homogeneous
        _homo_tgt = aligned_tgt.astype(np.float32, copy=True)
        _homo_src = img_src.astype(np.float32, copy=True)
        _homo_tgt[:, ~_homo_mask_2d] = np.nan
        _homo_src[:, ~_homo_mask_2d] = np.nan
        write_raster(output_dir, "cropped_aligned_homo_target.tif", _homo_tgt, base_profile,
                     np.float32, nodata_value=np.nan)
        write_raster(output_dir, "cropped_aligned_homo_src.tif", _homo_src, base_profile,
                     np.float32, nodata_value=np.nan)
        dt_rasters = time() - t_rasters0

        written_files = [
            "homogeneity_sample_src.tif", "homogeneity_sample_target.tif",
            "common_homogeneous_rgb.tif",
            "cropped_aligned_homo_target.tif", "cropped_aligned_homo_src.tif",
        ]
        print_module_summary(
            "Output Generation (rasters)",
            params={"output_dir": abspath(output_dir), "final size": f"{ref_width}x{ref_height}px"},
            checks=["Confirmed cropped src/target/homogeneity/composite arrays share the final output shape."],
            actions=[f"Wrote {len(written_files)} GeoTIFF(s): {', '.join(written_files)}."],
            result=f"OK -- {len(written_files)} raster(s) written in {dt_rasters:.2f}s.",
        )

        dx_before, dy_before = flow_future.result()
        dx_after, dy_after = flow_after_future.result()
        flow_dt = time() - flow_t0
        step_times.append(("Optical Flow (ran in background)", flow_dt))
        log_step_end(detail_log, "Optical Flow (background)", flow_dt)

        if total_crop_bbox is not None:
            tr0, tr1, tc0, tc1 = total_crop_bbox
            dx_before = dx_before[tr0:tr1, tc0:tc1]
            dy_before = dy_before[tr0:tr1, tc0:tc1]
            dx_after = dx_after[tr0:tr1, tc0:tc1]
            dy_after = dy_after[tr0:tr1, tc0:tc1]

        flow_actions = []
        if ecc_correction_applied:
            t_arrow_grid0 = time()
            plot_shift_arrow_grid(img_src[0], dx_before, dy_before, plots_dir,
                                   filename="shift_arrow_grid_before.png",
                                   title="Local Displacement Before Alignment")
            step_times.append(("Shift-direction grid arrow plot", time() - t_arrow_grid0))
            flow_actions.append(
                "Wrote shift-direction grid arrow plot (required: a significant misalignment was "
                "detected/corrected, so the pre-alignment flow field is informative)."
            )
        else:
            flow_actions.append(
                "Skipped shift-direction grid arrow plot (not required: no significant misalignment was "
                "detected/corrected, so the raw pre-alignment flow field would just show noise-level arrows)."
            )

        print_module_summary(
            "Optical Flow Diagnostics",
            checks=["Computed dense optical-flow displacement fields between src and target, before and "
                    "after ECC correction (ran in the background, overlapped with steps 4-7)."],
            actions=flow_actions,
            result=f"OK -- optical flow computed in {flow_dt:.2f}s.",
        )

        shift_val_detail = shift_val_future.result()
        shift_val_dt = time() - shift_val_t0
        step_times.append(("Shift Validation (ran in background)", shift_val_dt))
        log_step_end(detail_log, "Shift Validation (background)", shift_val_dt, result=shift_val_detail)
        if shift_val_detail is not None and step > 1:
            shift_val_detail["dx"] *= step
            shift_val_detail["dy"] *= step

        shift_val_checks = ["Ran a 2nd ECC pass on the corrected (src, aligned target) pair, in the "
                             "background, to confirm the residual shift dropped close to zero."]
        if shift_val_detail is None:
            shift_val_checks.append("Re-detection found no usable overlap -- cannot confirm residual shift.")
            shift_val_result = "INCONCLUSIVE -- could not verify residual shift (no usable overlap on re-detection)."
        else:
            shift_val_checks.append(
                f"Residual shift on the corrected pair: dx={shift_val_detail['dx']:.2f}px, "
                f"dy={shift_val_detail['dy']:.2f}px, rotation={shift_val_detail['rotation_deg']:.3f}deg, "
                f"scale={shift_val_detail['scale']:.4f} (correlation={shift_val_detail['correlation_coefficient']:.4f})"
            )
            if ecc_correction_applied and transform_is_negligible(shift_val_detail):
                shift_val_result = "OK -- residual shift is negligible; correction confirmed effective."
            elif ecc_correction_applied:
                shift_val_result = ("WARNING -- residual shift is still non-negligible after correction; "
                                     "the applied transform may not have fully resolved the misalignment.")
            else:
                shift_val_result = "OK -- no correction was applied, so this is a re-confirmation of the original (negligible) estimate."

        print_module_summary(
            "Shift Validation",
            checks=shift_val_checks,
            actions=["Ran second-pass ECC re-detection in the background (overlapped with steps 4-6)."],
            result=shift_val_result,
        )

        if manual_resample_future is not None:
            manual_resampled = manual_resample_future.result()
            manual_resample_dt = time() - manual_resample_t0
            step_times.append(("Manual Resample QA (ran in background)", manual_resample_dt))
            log_step_end(detail_log, "Manual Resample QA (background)", manual_resample_dt)

            reference_arr = img_tgt if manual_resample_label == "src" else img_src
            reference_label = "target" if manual_resample_label == "src" else "src"

            manual_resampled_for_ref = manual_resampled
            if total_crop_bbox is not None:
                mr0, mr1, mc0, mc1 = total_crop_bbox
                manual_resampled_for_ref = manual_resampled[..., mr0:mr1, mc0:mc1]

            manual_metrics = compare_resampled_to_reference(manual_resampled_for_ref, reference_arr)
            detail_log(f"[Manual Resample QA metrics] {manual_metrics}")

            manual_vs_bilinear_metrics = compare_resampled_to_reference(manual_resampled, bilinear_resampled_arr)
            detail_log(f"[Manual vs Bilinear direct comparison metrics] {manual_vs_bilinear_metrics}")

            print_module_summary(
                "Resampling QA (bilinear vs manual block-average)",
                checks=[
                    f"Compared bilinear-resampled {manual_resample_label} against {reference_label}'s native data "
                    f"(PSNR reported vs ground truth).",
                    f"Compared manually block-averaged {manual_resample_label} against {reference_label}'s native "
                    f"data (PSNR reported vs ground truth).",
                    f"Compared bilinear vs manual resampling directly against each other "
                    f"(RMSE/MAE/SSIM, algorithm-choice error only).",
                ],
                actions=["Ran manual block-average resampling in the background (overlapped with steps 3-7) "
                         "as an alternate-algorithm QA check on the bilinear resampling used in the main pipeline."],
                result=f"OK -- resampling QA completed in {manual_resample_dt:.2f}s (full metrics in detail log).",
            )
            print_algorithm_comparison_report(
                bilinear_metrics=resampled_vs_reference_metrics,
                manual_metrics=manual_metrics,
                diff_metrics=manual_vs_bilinear_metrics,
                bilinear_label=f"{manual_resample_label} [{_resampling_name(resampling)}]",
                manual_label=f"{manual_resample_label} [manual]",
            )

        bg_pool.shutdown(wait=True)


        t_hist0 = time()
        _report_progress(progress_callback, 11, "Shift histograms & diagnostics")
        live_figures.update(write_shift_histograms(
                               plots_dir, dx_before, dy_before, dx_after, dy_after,
                               transform=ref_transform, crs=ref_crs,
                               show_after=ecc_correction_applied,
                               keep_figs_open=figure_callback is not None))
        dt_hist = time() - t_hist0

        print_module_summary(
            "Output Generation (shift histograms & diagnostics)",
            params={"plots_dir": abspath(plots_dir)},
            checks=[
                "Optical-flow displacement fields (before/after alignment) computed in the background.",
                f"ECC correction was {'applied' if ecc_correction_applied else 'not applied'}, so the "
                f"'after' ECC reference {'is 0,0 (correction fully applied)' if ecc_correction_applied else 'equals the before value (no correction to show)'}.",
            ],
            actions=[
                "Wrote shift-magnitude/direction histograms (before vs after alignment) to plots_dir.",
                (
                    "Wrote shift-direction grid arrow plot (required: a significant misalignment was detected/corrected)."
                    if ecc_correction_applied else
                    "Skipped shift-direction grid arrow plot (not required: no significant misalignment was detected/corrected)."
                ),
            ],
            result=f"OK -- diagnostics written in {dt_hist:.2f}s.",
        )

        dt_outputs_total = dt_rasters + dt_hist
        step_times.append(("7. Write output rasters & histograms", dt_outputs_total))
        log_step_end(detail_log, "7. Write output rasters & histograms", dt_outputs_total,
                     rasters_s=round(dt_rasters, 3), histograms_s=round(dt_hist, 3))

        t0 = time()
        _report_progress(progress_callback, 12, "Plot")
        common_idx = np.flatnonzero(common_map[0] != 0)
        if common_idx.size == 0:
            print_module_summary(
                "Radiometric Conversion & Density Scatter Plot",
                checks=["Checked for common/homogeneous pixels (common_map) to convert and plot."],
                actions=["Skipped density scatter plot "
                         "(not possible: no common/homogeneous pixels found)."],
                result="SKIPPED -- no common/homogeneous pixels to plot.",
            )
        else:
            reflectance_src_common = img_src[0].ravel()[common_idx]
            reflectance_tgt_common = aligned_tgt[0].ravel()[common_idx]
            detail_log(f"[Common Reflectance Check] N common pixels = {reflectance_src_common.size:,}")
            detail_log(
                f"[Common Reflectance Check] src reflectance: "
                f"min={np.nanmin(reflectance_src_common):.6f}, "
                f"max={np.nanmax(reflectance_src_common):.6f}, "
                f"mean={np.nanmean(reflectance_src_common):.6f}"
            )
            detail_log(
                f"[Common Reflectance Check] tgt reflectance: "
                f"min={np.nanmin(reflectance_tgt_common):.6f}, "
                f"max={np.nanmax(reflectance_tgt_common):.6f}, "
                f"mean={np.nanmean(reflectance_tgt_common):.6f}"
            )
            detail_log(
                f"[Common Reflectance Check] lengths: "
                f"src={reflectance_src_common.size:,}, tgt={reflectance_tgt_common.size:,}"
            )
            _n_sample = min(10, reflectance_src_common.size)
            for _i in range(_n_sample):
                detail_log(
                    f"[Common Reflectance Check]   [{_i}] src={reflectance_src_common[_i]:.6f}  "
                    f"tgt={reflectance_tgt_common[_i]:.6f}"
                )

            plot_y_label = rad_details["plot_label_src"]
            plot_x_label = rad_details["plot_label_tgt"]

            cs_report = compute_cross_sensor_report(
                rho_reference=reflectance_src_common,
                rho_target=reflectance_tgt_common,
                sensor_src_label=rad_details["plot_label_src"],
                sensor_tgt_label=rad_details["plot_label_tgt"],
            )
            pipeline_result["cross_sensor_report"] = cs_report
            detail_log(f"[Cross-sensor stats] {cs_report['stats']}")
            print_module_summary(
                "Cross-Sensor Metrics (RMA)",
                params={},
                checks=cs_report["checks"],
                actions=cs_report["actions"],
                result=cs_report["result"],
            )

            reflectance_src_plot = cs_report["rho_reference_corrected"]
            reflectance_tgt_plot = cs_report["rho_target_corrected"]
            _correction_note = []

            plot_path, plot_stats, plot_fig = plot_common(
                reflectance_src_plot, reflectance_tgt_plot,
                output_dir=plots_dir, value_label="reflectance",
                keep_fig_open=True,
                x_label=plot_x_label, y_label=plot_y_label,
            )
            if plot_fig is not None:
                live_figures["plot_common.png"] = plot_fig
            pipeline_result["plot_stats"] = plot_stats

            fit_checks = []
            if plot_stats is not None:
                r2_str = f"{plot_stats['r_squared']:.4f}" if plot_stats["r_squared"] is not None else "N/A"
                line_error = plot_stats["fit_to_1to1_error_pct"]
                line_error_str = f"{line_error:.2f}%" if line_error is not None else "N/A"
                fit_checks.append(
                    f"Reference vs target fit over common-area reflectance (N={plot_stats['n_common']:,}): "
                    f"slope={plot_stats['slope']:.4f}, intercept={plot_stats['intercept']:.4f}, "
                    f"RMSE={plot_stats['rmse']:.4f}, bias={plot_stats['bias']:.4f}, R\u00b2={r2_str}, "
                    f"fit-vs-1:1 error={line_error_str}"
                )
                if plot_stats.get("slope_robust") is not None:
                    _slope_gap = abs(plot_stats["slope"] - plot_stats["slope_robust"])
                    fit_checks.append(
                        f"Robust regression diagnostic (Theil-Sen): slope={plot_stats['slope_robust']:.4f}, "
                        f"intercept={plot_stats['intercept_robust']:.4f} "
                        f"(vs. OLS/RMA slope={plot_stats['slope']:.4f}, gap={_slope_gap:.4f}"
                        + (" -- notably large; a handful of influential points may be pulling the "
                           "primary fit" if _slope_gap > 0.05 else "")
                        + ")."
                    )

            print_module_summary(
                "Radiometric Conversion & Density Scatter Plot",
                params={"common/homogeneous pixels used": f"{common_idx.size:,}"},
                checks=[
                    "Confirmed common/homogeneous pixels exist (common_map != 0).",
                    f"Located src calibration metadata: sensor={rad_details['sensor_src']}, "
                    f"band={rad_details['band_name_src']}, source={rad_details['meta_path_src']}",
                    f"Located target calibration metadata: sensor={rad_details['sensor_tgt']}, "
                    f"band={rad_details['band_name_tgt']}, source={rad_details['meta_path_tgt']}",
                    rad_details["reflectance_src_stats"],
                    rad_details["reflectance_tgt_stats"],
                ] + fit_checks,
                actions=[
                    "Source: DN -> TOA Radiance -> TOA Reflectance conversion applied at pipeline start "
                    "(before resampling/alignment); selected common-area pixels here.",
                    "Target: DN -> TOA Radiance -> TOA Reflectance conversion applied at pipeline start "
                    "(before resampling/alignment); selected common-area pixels here.",
                    (f"Wrote density scatter plot (source vs target reflectance) "
                     f"to {plot_path}."
                     if plot_path is not None else
                     "Skipped density scatter plot (not possible: no data to plot)."),
                ],
                result="OK -- reflectance computed and comparison plot written." if plot_path is not None
                       else "SKIPPED -- reflectance computed but plot could not be written.",
            )
        dt = time() - t0
        step_times.append(("8. Plot", dt))

        print_timing_summary(step_times, time() - pipeline_t0)
        print(f"\n[Log] Console log saved -> {abspath(join(output_dir, log_filename))}")
        print(f"[Log] Detailed step log saved -> {abspath(join(output_dir, detail_log_filename))}")
        _report_progress(progress_callback, TOTAL_PIPELINE_STEPS, "Done")

        if figure_callback is not None:
            figure_callback(live_figures)

    finally:
        bg_pool.shutdown(wait=False)
        sys.stdout = original_stdout
        short_log_file.close()
        detail_log_file.close()

    return pipeline_result
