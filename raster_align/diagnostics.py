"""
raster_align.diagnostics

Dense Farneback optical-flow displacement measurement (before/after
alignment) and the before-vs-after displacement histograms.
"""

import os

import cv2
import numpy as np


def _to_uint8_masked(arr, mask):
    """
    Normalize a float band to 8-bit (0-255) for calcOpticalFlowFarneback,
    which requires CV_8UC1 input -- NaN/invalid pixels (per `mask`, the
    jointly-finite mask between the two bands being compared) are zeroed
    out before normalizing, and excluded from the min/max stretch itself
    so a big block of NaN-turned-zero pixels can't compress the real data
    range. Same approach as alignment.py's `_to_uint8` (kept separate here
    since this one takes an externally supplied joint mask rather than
    computing its own single-image mask).
    """
    a = np.where(mask, arr, 0.0).astype(np.float32)
    if not mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    mn, mx = a[mask].min(), a[mask].max()
    if mx == mn:
        out = np.full(arr.shape, 128, dtype=np.uint8)
    else:
        out = np.clip((a - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def compute_flow(b1, b2, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                  poly_n=5, poly_sigma=1.2, min_valid_frac=0.5):
    """
    Dense per-pixel displacement field between b1 and b2 via Farneback
    optical flow (cv2.calcOpticalFlowFarneback) -- one (dx, dy) sample
    PER PIXEL, not one per window.

    calcOpticalFlowFarneback requires 8-bit single-channel input, so both
    bands are first normalized to uint8 over their jointly-finite pixels
    (see _to_uint8_masked). Pixels that are NaN in either band (not
    jointly finite) are zeroed going into the flow computation and then
    masked back to NaN in the OUTPUT afterward, so invalid input pixels
    never masquerade as a real (probably near-zero, since both were
    zeroed) flow value.

    If fewer than `min_valid_frac` of the pixels are jointly finite, the
    pair is judged too sparse to trust and both output grids come back
    entirely NaN rather than running Farneback on mostly-fabricated data.

    Returns (dx_grid, dy_grid): 2D arrays the same (H, W) shape as
    b1/b2 -- a full dense field, unlike the coarser one-sample-per-window
    grids a block-based method (e.g. phase correlation) would return.

    These grids are in PIXEL units (fractional pixels of displacement on
    the common grid) -- see compute_world_displacement() to convert a
    grid like this into real-world (meters) displacement using the
    common grid's actual geotransform, rather than a naive per-axis
    pixel-size scalar.
    """
    valid = np.isfinite(b1) & np.isfinite(b2)
    h, w = b1.shape

    if valid.mean() < min_valid_frac:
        return (np.full((h, w), np.nan, dtype=np.float32),
                np.full((h, w), np.nan, dtype=np.float32))

    img1_u8 = _to_uint8_masked(b1, valid)
    img2_u8 = _to_uint8_masked(b2, valid)

    flow = cv2.calcOpticalFlowFarneback(
        img1_u8, img2_u8, None, pyr_scale, levels, winsize, iterations,
        poly_n, poly_sigma, 0,
    )
    dx_grid = flow[..., 0].astype(np.float32)
    dy_grid = flow[..., 1].astype(np.float32)

    dx_grid[~valid] = np.nan
    dy_grid[~valid] = np.nan

    return dx_grid, dy_grid


def compute_before_displacement(img_src, img_tgt, pyr_scale=0.5, levels=3, winsize=15,
                                 iterations=3, poly_n=5, poly_sigma=1.2, min_valid_frac=0.5):
    """
    Dense Farneback displacement field between the un-corrected target
    and src -- one sample per PIXEL (see compute_flow), not per window.

    Argument order matters here: this is passed to compute_flow as
    (target, src) -- NOT (src, target) -- specifically so the resulting
    dx/dy use the same "target -> src" sign convention as ECC's dx/dy
    (see alignment.py's docstring: ECC's warp matrix maps target-grid
    coords -> src-grid coords). calcOpticalFlowFarneback(prev, next)
    reports, for each prev pixel, the displacement that lands it on the
    matching next-image location -- the same "how far does A's content
    move to reach B" convention as ECC's dx/dy. Passing (src, target)
    instead gives the exact opposite sign, which is why the global ECC
    shift line used to look mismatched against this histogram's own
    mean/median.
    """
    return compute_flow(img_tgt[0], img_src[0], pyr_scale=pyr_scale, levels=levels,
                         winsize=winsize, iterations=iterations, poly_n=poly_n,
                         poly_sigma=poly_sigma, min_valid_frac=min_valid_frac)


def compute_after_displacement(img_src, aligned_tgt, pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, min_valid_frac=0.5):
    """
    Same as compute_before_displacement, but against the ECC-corrected
    (aligned) target -- i.e. residual displacement that's still left
    after alignment was applied (or the same uncorrected target again,
    unchanged, if ECC judged the detected shift negligible and skipped
    correction -- see pipeline.py's `ecc_correction_applied` branch).
    Ideally this should cluster tightly around zero if a real correction
    was applied and it worked.

    Same (target, src) argument order as compute_before_displacement, for
    the same sign-convention reason.
    """
    return compute_flow(aligned_tgt[0], img_src[0], pyr_scale=pyr_scale, levels=levels,
                         winsize=winsize, iterations=iterations, poly_n=poly_n,
                         poly_sigma=poly_sigma, min_valid_frac=min_valid_frac)


def mean_residual_magnitude(dx, dy):
    """Convenience summary stat for logging: mean |displacement| in pixels
    across valid flow pixels, ignoring NaNs. Used to report before/after
    dense-correction improvement."""
    return float(np.nanmean(np.hypot(dx, dy)))


def _hist_bins_for(center, half_range=1.0, n_bins=60, clip_min=None):
    """Bin range centered on `center` (the panel's own detected shift,
    typically its median), spanning +/-`half_range` pixels either side.

    Using a fixed, narrow, shift-centered window (instead of a wide
    percentile-clipped one) is what makes the before/after panels precise
    and comparable: the detected shift always lands in the middle of the
    x-axis, and a small span means each bin covers a fraction of a pixel
    instead of a fraction of a many-pixel range.

    `clip_min`, if given, floors the lower edge at that value -- used for
    quantities that can't physically go negative (e.g. a Euclidean
    residual distance), so a tight distribution near zero doesn't get an
    axis window that extends into meaningless negative territory.
    """
    lo = center - half_range
    hi = center + half_range
    if clip_min is not None and lo < clip_min:
        lo = clip_min
        hi = max(hi, lo + 2 * half_range)
    return np.linspace(lo, hi, n_bins + 1)


def _nice_tick_step(span, target_ticks=12):
    """
    Pick a 'nice' tick step that can go below 1.0, so a tight post-alignment
    (or otherwise sub-pixel) distribution still gets multiple readable
    ticks instead of MaxNLocator collapsing to 1-2 ticks because its step
    candidates bottom out at 1.

    Same idea as MaxNLocator's step search, but the candidate set includes
    sub-unit steps (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5 ...) in
    addition to the usual 1/2/5/10 multiples.
    """
    if span <= 0 or not np.isfinite(span):
        return 1.0

    raw_step = span / target_ticks
    nice_multipliers = np.array([1, 2, 2.5, 5, 10])
    exponent = np.floor(np.log10(raw_step))
    base = 10 ** exponent

    candidates = nice_multipliers * base
    diffs = candidates - raw_step
    valid = candidates[diffs >= 0]
    step = valid.min() if valid.size else candidates[np.argmin(np.abs(diffs))]
    return float(step)


def _decimal_places_for_step(step):
    """How many decimal places are needed to distinguish tick labels at
    this step size, e.g. step=0.25 -> 2 decimals, step=5 -> 0 decimals."""
    if step <= 0 or not np.isfinite(step):
        return 2
    max_decimals = 4
    for decimals in range(max_decimals + 1):
        if abs(round(step, decimals) - step) < 1e-9:
            return decimals
    return max_decimals


def _draw_shift_hist_panel(ax, data, subtitle, color,
                            half_range=1.0, pixel_size_m=1.0,
                            x_label="Pixel Shift (pixels)",
                            y_label="Number of Pixels", unit_suffix="px",
                            sample_unit_label="pixels", gsd_m=None,
                            non_negative=False):
    import matplotlib.ticker as ticker

    data_m = data * pixel_size_m
    half_range_m = half_range * pixel_size_m

    valid = np.isfinite(data_m)
    vals = data_m[valid]

    if vals.size == 0:
        ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", transform=ax.transAxes)
        if subtitle:
            ax.set_title(subtitle, fontsize=12)
        ax.axis("off")
        return

    mean_v, median_v, std_v = float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))

    center = median_v
    bins = _hist_bins_for(center, half_range=half_range_m, clip_min=0.0 if non_negative else None)

    def _px_suffix(value_m):
        if gsd_m is None or gsd_m <= 0:
            return ""
        return f" (~{value_m / gsd_m:.2f}px)"

    clipped_vals = np.clip(vals, bins[0], bins[-1])

    counts, edges, patches = ax.hist(
        clipped_vals, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4
    )
    peak = counts.max() if counts.max() > 0 else 1
    for c, p in zip(counts, patches):
        p.set_alpha(0.35 + 0.55 * (c / peak))

    peak_count = float(counts.max()) if counts.size else 0.0
    ax.set_ylim(0, peak_count * 1.22 if peak_count > 0 else 1.0)

    ax.axvline(0, color="black", linewidth=1.4, linestyle="--", alpha=0.7, label="Zero shift")
    ax.axvline(mean_v, color="crimson", linewidth=1.4, linestyle="-", alpha=0.8,
               label=f"Mean = {mean_v:.3f}{unit_suffix}{_px_suffix(mean_v)}")
    ax.axvline(median_v, color="darkorange", linewidth=1.2, linestyle=":", alpha=0.8,
               label=f"Median = {median_v:.3f}{unit_suffix}{_px_suffix(median_v)}")

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    gsd_note = f", GSD = {gsd_m:.3f}m/px" if gsd_m else ""
    stats_line = (f"std = {std_v:.3f}{unit_suffix}{_px_suffix(std_v)}, "
                  f"n = {vals.size:,} {sample_unit_label}")
    ax.set_title(f"{subtitle}\n({stats_line})" if subtitle else stats_line, fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    span = bins[-1] - bins[0]
    tick_step = _nice_tick_step(span, target_ticks=16)
    tick_start = np.ceil(bins[0] / tick_step) * tick_step
    major_ticks = np.arange(tick_start, bins[-1] + tick_step * 0.5, tick_step)
    decimals = _decimal_places_for_step(tick_step)

    ax.set_xticks(major_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(9)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def plot_shift_hist(data_before, data_after, title, color, filename, hist_dir,
                     label_before="Before Alignment", label_after="After Alignment",
                     half_range=1.0,
                     pixel_size_m=1.0, x_label="Pixel Shift (pixels)",
                     y_label="Number of Pixels", unit_suffix="px",
                     sample_unit_label="pixels", gsd_m=None, non_negative=False,
                     show_after=True, keep_fig_open=False):
    """
    Two-subplot figure: left panel = `data_before`, right panel =
    `data_after`. Both panels are drawn to the same fixed +/-`half_range`
    pixel window, centered on that panel's own detected shift (its
    median), so the shift always sits in the middle of the axis and the
    two panels are directly, precisely comparable.

    `data_before`/`data_after` are expected to be the dense
    one-sample-per-pixel Farneback flow grids returned by
    diagnostics.compute_flow -- each entry is an independent per-pixel
    shift estimate, so the histogram's "n" and bin heights reflect the
    real number of valid pixels that went into it. Every valid pixel is
    shown as measured -- near-zero values are NOT filtered out.

    `pixel_size_m` converts `data_before`/`data_after` (and `half_range`,
    still expressed in pixels) from pixels to meters before plotting --
    leave at 1.0 to keep everything in pixels, or if the caller (e.g.
    write_shift_histograms) has already converted the data to meters
    itself. `x_label`/`y_label`/`unit_suffix` should be updated to match
    whichever unit is actually being plotted. `sample_unit_label` names
    what each sample actually is (e.g. "pixels") for the n=... annotation.

    `show_after`: when False, draws a SINGLE panel (data_before only)
    instead of the before/after pair. Use this when no real correction
    was ever applied (e.g. ECC's detected shift was negligible) --
    `data_after` would just be an identical copy of `data_before` in
    that case, so a side-by-side "before vs after" comparison would be
    misleading (implying a correction happened when none did). In this
    mode the panel is NOT labeled "Before Alignment" -- there is no
    "after" for "before" to be contrasted against, so that label (and
    any "no correction applied" caveat) would just be confusing noise.
    `title` is shown as-is with no such caveat appended.

    Defaults to the ECC before/after-alignment labels, but `label_before`/
    `label_after` can be overridden to reuse this for other displacement
    comparisons -- e.g. ECC-only residual vs ECC+dense-correction residual.
    These are only used when `show_after=True`; the single-panel case
    ignores them.

    By default the figure is closed before returning (returns None) so
    plain batch/CLI callers don't accumulate open figures. Pass
    `keep_fig_open=True` to get the live Figure back instead -- caller
    then owns it (same convention as plot_common.plot_common).
    """
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if show_after:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(18, 6.5), sharey=False)
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(9, 6.5))

    _draw_shift_hist_panel(ax_before, data_before, label_before if show_after else None, color,
                            half_range=half_range, pixel_size_m=pixel_size_m,
                            x_label=x_label, y_label=y_label, unit_suffix=unit_suffix,
                            sample_unit_label=sample_unit_label, gsd_m=gsd_m, non_negative=non_negative)
    if show_after:
        _draw_shift_hist_panel(ax_after, data_after, label_after, color,
                                half_range=half_range, pixel_size_m=pixel_size_m,
                                x_label=x_label, y_label=y_label, unit_suffix=unit_suffix,
                                sample_unit_label=sample_unit_label, gsd_m=gsd_m, non_negative=non_negative)
    fig.suptitle(title, fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(hist_dir, filename), dpi=300, bbox_inches="tight")
    if keep_fig_open:
        return fig
    plt.close(fig)
    return None


def compute_world_displacement(dx_px, dy_px, transform, crs=None):
    """
    Convert a dense PIXEL-space displacement field (as returned by
    compute_before_displacement / compute_after_displacement /
    compute_flow) into a dense WORLD-space displacement field, in meters,
    by looking up each pixel's own real-world coordinate on the common
    grid -- its lat/lon (geographic CRS) or easting/northing (projected
    CRS) -- via `transform`, and the real-world coordinate of the matched
    location Farneback found for it, then differencing the two.

    This replaces the previous approach of multiplying the pixel offset
    by a constant per-axis pixel-size scalar (`pixel_size_x_m`,
    `pixel_size_y_m` = abs(transform.a), abs(transform.e)), which silently
    assumed the common grid is north-up and unrotated (transform.b == 0
    and transform.d == 0) and that its units are already meters. Using
    the full affine transform instead correctly handles a rotated/sheared
    grid, and detecting a geographic CRS (crs.is_geographic) means a
    degree of longitude is no longer mistakenly treated as if it were the
    same fixed distance as a degree of latitude.

    Given transform = | a  b  c |
                       | d  e  f |
    a pixel (col, row)'s world coordinate is:
        x = a*col + b*row + c   (easting, or longitude)
        y = d*col + e*row + f   (northing, or latitude)
    For each src pixel at grid index (row, col), the location Farneback
    says its content actually matches is (row + dy_px, col + dx_px) --
    that pair is run through the same transform to get its world
    coordinate, and the (dx_world, dy_world) reported here is the
    difference between the two real-world coordinates -- i.e. actually
    comparing "lat/lon of this src pixel" against "lat/lon of where its
    content was found in the other image", not a synthetic pixel-count
    scaling.

    For a geographic CRS, the resulting lon/lat difference is converted
    to meters via a local equirectangular approximation (meters-per-
    degree evaluated at each pixel's own latitude). This is accurate to
    well under a meter at the sub-few-pixel displacement scales this
    diagnostic deals with, so a full geodesic solve isn't needed.

    Returns (dx_m, dy_m): float32 grids the same (H, W) shape as
    dx_px/dy_px, in meters, along the transform's x-axis (easting) and
    y-axis (northing) respectively. NaN pixels in the input stay NaN.
    """
    h, w = dx_px.shape
    row_idx, col_idx = np.indices((h, w), dtype=np.float64)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

    x1 = a * col_idx + b * row_idx + c
    y1 = d * col_idx + e * row_idx + f

    col2 = col_idx + dx_px.astype(np.float64)
    row2 = row_idx + dy_px.astype(np.float64)
    x2 = a * col2 + b * row2 + c
    y2 = d * col2 + e * row2 + f

    is_geographic = bool(getattr(crs, "is_geographic", False))
    if is_geographic:
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(y1))
        dx_m = (x2 - x1) * meters_per_deg_lon
        dy_m = (y2 - y1) * meters_per_deg_lat
    else:
        dx_m = x2 - x1
        dy_m = y2 - y1

    return dx_m.astype(np.float32), dy_m.astype(np.float32)


def compute_residual_distance_m(dx_m, dy_m):
    """
    Euclidean residual distance per pixel, in meters: sqrt(dx_m**2 +
    dy_m**2), given the world-space (dx_m, dy_m) displacement grids from
    compute_world_displacement -- i.e. the straight-line real-world
    distance between "where this src pixel is" and "where its content was
    found", combining the easting and northing components into one
    magnitude rather than reporting them as two separate values.
    """
    return np.hypot(dx_m, dy_m).astype(np.float32)


def _nominal_pixel_size_m(transform, crs, grid_shape):
    """
    Nominal (approximate) ground footprint of one pixel, in meters, along
    the transform's x-axis and y-axis -- used only to pick a sensible
    default `half_range` window width for the histograms, not for the
    actual per-pixel displacement values (see compute_world_displacement,
    which is exact per-pixel rather than this single nominal figure).

    Uses hypot(a, d) / hypot(b, e) rather than plain abs(a)/abs(e), so a
    rotated/sheared transform still gets a correct pixel footprint instead
    of just its axis-aligned component.
    """
    import math

    px_x = math.hypot(transform.a, transform.d)
    px_y = math.hypot(transform.b, transform.e)
    if crs is not None and bool(getattr(crs, "is_geographic", False)):
        h, w = grid_shape
        center_lat = transform.f + transform.e * (h / 2.0) + transform.b * (w / 2.0)
        px_x *= 111320.0 * math.cos(math.radians(center_lat))
        px_y *= 111320.0
    return px_x, px_y


def write_shift_histograms(output_dir, dx_before, dy_before, dx_after, dy_after,
                            transform=None, crs=None, show_after=True, keep_figs_open=False):
    """
    Returns a dict {filename: fig} of the live matplotlib Figures behind
    the two saved PNGs (easting_error_histogram.png,
    northing_error_histogram.png) when `keep_figs_open=True` -- the
    caller then owns those figures (same convention as
    plot_shift_hist/plot_common). Default is an empty dict and both
    figures are closed as usual, so plain batch/CLI callers see no
    change in behavior.
    """

    common_valid = (
        np.isfinite(dx_before) & np.isfinite(dy_before)
        & np.isfinite(dx_after) & np.isfinite(dy_after)
    )

    dx_before = np.where(common_valid, dx_before, np.nan)
    dy_before = np.where(common_valid, dy_before, np.nan)
    dx_after = np.where(common_valid, dx_after, np.nan)
    dy_after = np.where(common_valid, dy_after, np.nan)

    if transform is not None:
        dx_before_m, dy_before_m = compute_world_displacement(dx_before, dy_before, transform, crs)
        dx_after_m, dy_after_m = compute_world_displacement(dx_after, dy_after, transform, crs)
        px_x_m, px_y_m = _nominal_pixel_size_m(transform, crs, dx_before.shape)
        half_range_x_m = max(px_x_m, 1e-6)
        half_range_y_m = max(px_y_m, 1e-6)
        gsd_x_m, gsd_y_m = px_x_m, px_y_m
        unit_suffix, sample_unit_label = "m", "pixels"
        axis_unit = "meters"
    else:
        dx_before_m, dy_before_m = dx_before, dy_before
        dx_after_m, dy_after_m = dx_after, dy_after
        half_range_x_m = half_range_y_m = 1.0
        gsd_x_m = gsd_y_m = None
        unit_suffix, sample_unit_label = "px", "pixels"
        axis_unit = "pixels"

    hist_dir = output_dir
    os.makedirs(hist_dir, exist_ok=True)

    hist_title_suffix = " (Before vs After Alignment)" if show_after else ""

    figs = {}

    figs["easting_error_histogram.png"] = plot_shift_hist(
                     dx_before_m, dx_after_m, f"Easting Error{hist_title_suffix}",
                     "steelblue", "easting_error_histogram.png", hist_dir,
                     half_range=half_range_x_m, pixel_size_m=1.0,
                     x_label=f"Easting error, in {axis_unit}",
                     y_label="Frequency of observed easting error",
                     unit_suffix=unit_suffix, sample_unit_label=sample_unit_label,
                     gsd_m=gsd_x_m, non_negative=False, show_after=show_after,
                     keep_fig_open=keep_figs_open)

    figs["northing_error_histogram.png"] = plot_shift_hist(
                     dy_before_m, dy_after_m, f"Northing Error{hist_title_suffix}",
                     "indianred", "northing_error_histogram.png", hist_dir,
                     half_range=half_range_y_m, pixel_size_m=1.0,
                     x_label=f"Northing error, in {axis_unit}",
                     y_label="Frequency of observed northing error",
                     unit_suffix=unit_suffix, sample_unit_label=sample_unit_label,
                     gsd_m=gsd_y_m, non_negative=False, show_after=show_after,
                     keep_fig_open=keep_figs_open)

    if not keep_figs_open:
        return {}
    return figs


def plot_shift_arrow(background, dx, dy, rotation_deg, scale, output_dir,
                      filename="shift_arrow.png", title="Detected target shift (ECC)"):
    """
    Draws a single large arrow over `background` (typically img_src band 1,
    pre-crop, in image/pixel space) showing the direction and magnitude of
    the misalignment ECC detected in the TARGET image, i.e. "target needs
    to move this way to line up with src".

    Sign convention: `dx`/`dy` here are exactly detail["dx"]/detail["dy"]
    from estimate_misalignment() -- M maps target-grid coords -> src-grid
    coords (see alignment.py's docstring), so (dx, dy) already IS the
    src-space displacement that explains where target sits relative to
    src. The arrow is drawn from the image center pointing along (dx, dy),
    scaled up for visibility since true sub/low-pixel shifts are otherwise
    invisible at image scale.

    Saved as its own standalone PNG, separate from plot_common.png and the
    histograms.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    band = background[0] if background.ndim == 3 else background
    h, w = band.shape
    finite = np.isfinite(band)
    if finite.any():
        vmin, vmax = np.nanpercentile(band[finite], [2, 98])
    else:
        vmin, vmax = 0, 1

    mag = float(np.hypot(dx, dy))
    min_len = 0.06 * min(h, w)
    draw_len = max(min_len, min(0.35 * min(h, w), mag * 15))
    if mag > 1e-9:
        ux, uy = dx / mag, dy / mag
    else:
        ux, uy = 0.0, 0.0

    cx, cy = w / 2.0, h / 2.0
    ex, ey = cx + ux * draw_len, cy + uy * draw_len

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(band, cmap="gray", vmin=vmin, vmax=vmax)
    ax.annotate(
        "", xy=(ex, ey), xytext=(cx, cy),
        arrowprops=dict(facecolor="red", edgecolor="red", width=4, headwidth=16, headlength=18),
    )
    ax.plot(cx, cy, "o", color="red", markersize=5)
    ax.set_title(
        f"{title}\ndx={dx:.2f}px  dy={dy:.2f}px  |shift|={mag:.2f}px  "
        f"rotation={rotation_deg:.3f}deg  scale={scale:.4f}\n"
        f"(arrow direction/rotation to true scale, length exaggerated for visibility)",
        fontsize=10,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Shift Arrow] Saved -> {out_path}")
    return out_path


def plot_shift_arrow_grid(background, dx, dy, output_dir, n_rows=None, n_cols=None,
                           cell_px=None, min_valid_frac=0.5, arrow_scale=25.0,
                           filename="shift_arrow_grid.png",
                           title="Local Displacement (Easting/Northing Error)"):
    """
    Grid version of plot_shift_arrow: instead of one big arrow for the
    whole-image ECC shift, this bins the DENSE per-pixel (dx, dy) field
    (e.g. from compute_before_displacement / compute_after_displacement)
    into a rows x cols grid and draws one small arrow per cell showing
    that cell's own local displacement -- the same style as e.g. USGS
    geometric-accuracy figures: a yellow grid over the image with a red
    easting/northing error arrow at the center of each cell.

    `dx`/`dy` must be dense, same (H, W) shape as `background` -- pass
    the per-pixel Farneback flow field, NOT the single scalar ECC
    dx/dy (that's what plot_shift_arrow is for).

    Grid size: pass either (n_rows, n_cols) directly, or `cell_px` (cell
    side length in pixels, one number used for both axes) to derive the
    count from a target cell size. If none of those are given, defaults
    to a fixed 20x20 grid regardless of image size/aspect ratio -- cells
    are simply width/20 by height/20, so they're square-ish only when
    the image itself is roughly square; a wide or tall image just gets
    wide or tall cells, same as it would under any other sizing scheme.

    Each cell's arrow direction/length comes from the MEDIAN of that
    cell's finite dx/dy pixels (robust to a few bad flow samples inside
    the cell). A cell is skipped entirely (no arrow) if fewer than
    `min_valid_frac` of its pixels are finite, rather than plotting a
    fabricated direction from mostly-invalid data.

    `arrow_scale` exaggerates true (usually sub/low-pixel) displacement
    for visibility, same idea as plot_shift_arrow's `min_len`/`mag * 15`
    -- here each arrow's drawn length is additionally capped to roughly
    the cell's own size so dense grids stay readable instead of
    neighboring arrows overlapping.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow

    os.makedirs(output_dir, exist_ok=True)

    band = background[0] if background.ndim == 3 else background
    h, w = band.shape

    if dx.shape != band.shape or dy.shape != band.shape:
        raise ValueError(
            f"[Shift Arrow Grid] dx/dy shape {dx.shape} must match background shape "
            f"{band.shape} -- pass the dense flow field, not a scalar ECC dx/dy."
        )

    if n_rows is None or n_cols is None:
        if cell_px is None:
            FIXED_GRID_CELLS = 20
            n_cols = FIXED_GRID_CELLS
            n_rows = FIXED_GRID_CELLS
        else:
            n_cols = max(1, round(w / cell_px))
            n_rows = max(1, round(h / cell_px))

    finite = np.isfinite(band)
    if finite.any():
        vmin, vmax = np.nanpercentile(band[finite], [2, 98])
    else:
        vmin, vmax = 0, 1

    row_edges = np.linspace(0, h, n_rows + 1)
    col_edges = np.linspace(0, w, n_cols + 1)

    cell_h = h / n_rows
    cell_w = w / n_cols
    cell_min = min(cell_h, cell_w)
    head_len = 0.14 * cell_min
    head_w = 0.14 * cell_min
    shaft_w = 0.035 * cell_min
    max_len = 0.42 * cell_min
    min_len = 0.30 * cell_min

    fig, ax = plt.subplots(figsize=(9, 9 * h / w if w > 0 else 9))
    ax.imshow(band, cmap="gray", vmin=vmin, vmax=vmax)

    n_drawn, n_skipped = 0, 0
    for i in range(n_rows):
        r0, r1 = int(round(row_edges[i])), int(round(row_edges[i + 1]))
        for j in range(n_cols):
            c0, c1 = int(round(col_edges[j])), int(round(col_edges[j + 1]))
            if r1 <= r0 or c1 <= c0:
                continue

            cell_dx = dx[r0:r1, c0:c1]
            cell_dy = dy[r0:r1, c0:c1]
            cell_valid = np.isfinite(cell_dx) & np.isfinite(cell_dy)

            if cell_valid.mean() < min_valid_frac:
                n_skipped += 1
                continue

            med_dx = float(np.nanmedian(cell_dx[cell_valid]))
            med_dy = float(np.nanmedian(cell_dy[cell_valid]))
            mag = float(np.hypot(med_dx, med_dy))

            if mag > 1e-9:
                ux, uy = med_dx / mag, med_dy / mag
            else:
                ux, uy = 0.0, 0.0

            draw_len = max(min_len, min(max_len, mag * arrow_scale))
            ccx, ccy = (c0 + c1) / 2.0, (r0 + r1) / 2.0
            hx, hy = ux * draw_len / 2.0, uy * draw_len / 2.0

            ax.add_patch(FancyArrow(
                ccx - hx, ccy - hy, 2 * hx, 2 * hy,
                width=shaft_w,
                head_width=head_w,
                head_length=head_len,
                length_includes_head=True,
                facecolor="red", edgecolor="red", zorder=3,
            ))
            n_drawn += 1

    for r in row_edges:
        ax.axhline(r, color="yellow", linewidth=0.6, zorder=2)
    for c in col_edges:
        ax.axvline(c, color="yellow", linewidth=0.6, zorder=2)

    legend_elems = [
        Line2D([0], [0], color="red", marker=">", markersize=8, linestyle="-",
               linewidth=2, label="Easting and northing error"),
        Line2D([0], [0], color="yellow", linewidth=1.2, label="Grid"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, frameon=True, title="EXPLANATION", title_fontsize=10)

    ax.set_title(
        f"{title}\n{n_rows}x{n_cols} grid, {n_drawn} arrows drawn, {n_skipped} cells skipped "
        f"(insufficient valid data)\n(arrow direction/relative magnitude true, length exaggerated "
        f"{arrow_scale:.0f}x for visibility)",
        fontsize=10,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Shift Arrow Grid] Saved -> {out_path} ({n_drawn} arrows, {n_skipped} cells skipped)")
    return out_path
