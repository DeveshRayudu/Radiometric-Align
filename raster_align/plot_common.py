"""
raster_align.plot_common

2D density scatter plot of image1 vs image2 DN values at pixels marked
common/homogeneous, with a 1:1 reference line and a least-squares fit
overlaid, plus RMSE and R² (coefficient of determination) computed
between the two images over that same pixel set.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from scipy.stats import theilslopes
from pylr2 import regress2


def _point_density(x, y, bins=150, smooth_sigma=1.5):
    """
    Per-point density estimate that keeps every point as its own marker
    (unlike a 2D histogram/hexbin, which merges points into a single
    lumped cell). Points are only ever an INPUT to this -- never
    aggregated into an output plotted shape:

      1. Bin the points onto a `bins` x `bins` grid (fast, O(n)) just to
         get a rough occupancy count per cell.
      2. Gaussian-smooth that grid (`smooth_sigma`, in grid-cell units)
         so it becomes a continuous density surface instead of blocky
         counts -- this also washes out the quantization banding that a
         raw histogram would show, without touching the real x/y values.
      3. Interpolate that smoothed surface back at each point's exact,
         original (x, y) location, via `interpn`. This is the key step:
         the output is one density value per point, not a grid of
         cells, so the caller can color individual scatter markers by
         it and every original point stays visible and separate.

    Returns an array the same length as x/y: each point's estimated
    local density (arbitrary units -- only relative magnitude across
    points is meaningful, for colormap purposes).
    """
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    counts = gaussian_filter(counts, sigma=smooth_sigma)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    z = interpn(
        (x_centers, y_centers), counts, np.column_stack([x, y]),
        method="linear", bounds_error=False, fill_value=None,
    )
    return np.nan_to_num(z, nan=0.0)


def _rma_linear_fit(x, y):

    results = regress2(x, y, _method_type_2 = "reduced major axis")
    return results['slope'], results['intercept']


def plot_common(img1, img2, c_map=None, output_dir=None, density_bins=150, value_label="Radiance",
                 keep_fig_open=False, x_label=None, y_label=None):
    """
    2D density plot of image1 vs image2 values, with two reference lines
    overlaid:
      - the 1:1 line (x == y, i.e. zero difference between the images)
      - a least-squares linear fit line (y = m*x + c) that follows the
        actual point cloud, so wherever it departs from the 1:1 line,
        that gap IS the systematic difference between the two images.

    `c_map` is optional:
      - If given (a 2D array, same shape as img1/img2), only pixels
        where c_map != 0 (common/homogeneous) are plotted -- img1/img2
        are indexed down to those pixels here.
      - If omitted (None), img1/img2 are used as-is: this is the
        expected path for callers who already extracted just the
        common-area values themselves (e.g. computing radiance only for
        the DN values at common-area pixels, rather than converting a
        whole image) and don't need c_map re-applied -- it's only kept
        around as the reference used to pick those pixels in the first
        place, not something that needs converting itself.

    Any pixel that's NaN/Inf in *either* image (e.g. leftover nodata) is
    dropped before anything else -- density histogram, fit line, RMSE,
    and R² are all computed only over pixels valid in both.

    `value_label` controls the axis/units label only (e.g. "Radiance"
    or "DN") -- it does not affect what's plotted, just how it's
    described. Defaults to "Radiance" since callers now typically pass
    TOA-radiance-converted values rather than raw DN.

    `x_label` / `y_label` optionally override the full axis label (e.g.
    "Resourcesat-2A LISS4 B3 (Target)" / "Sentinel-2 B04 (Reference)")
    instead of the generic "Target ({value_label})" / "Reference
    ({value_label})" text. If omitted, the generic labels are used as
    before.

    Rendered as a plain scatter plot (semi-transparent points so
    overlapping regions still read as denser). Saved to disk (never
    displayed interactively). `density_bins` no longer sets a 2D
    histogram bin count (this hasn't been a histogram plot for a
    while) -- it now sets the scale of a small on-screen-only jitter
    applied to the scatter coordinates, to blur the diagonal stripes
    that quantized source values (e.g. DN * a fixed gain/offset)
    otherwise produce. The jitter never touches arr_orig/arr_shift
    themselves, so the fit line, RMSE, R², and bias are unaffected.

    RMSE (root-mean-square error, direct pixel-to-pixel: reference minus
    target, no fit involved) and R² (coefficient of determination of the
    reference-vs-target linear regression -- i.e. how well the fit line
    explains the point cloud, not just squared Pearson correlation) are
    both computed and drawn on the figure as a text box.

    In addition to direct pixel RMSE, `stats["fit_to_1to1_error_pct"]`
    quantifies how far the fitted line is from the ideal line.  It is the
    RMS vertical separation between `y = m*x + c` and `y = x`, evaluated at
    the plotted target values, normalized by the full plotted value range:

        100 * RMS((m - 1) * x + c) / (max(x, y) - min(x, y))

    Thus 0% means the fitted line is exactly the 1:1 line.  It is a
    line-comparison metric, not a replacement for pixel RMSE.

    This function never prints -- it returns (output_path, stats, fig), or
    (None, None, None) if the plot was skipped for lack of data, so the
    caller can report that itself via logging_utils.print_module_summary()
    instead of a duplicate standalone message here.

    `fig` is the live matplotlib Figure used to render the PNG. By
    default it's closed before returning (`fig` is then None) so plain
    batch/CLI callers don't accumulate open figures across a run.
    Pass `keep_fig_open=True` to get the Figure back instead -- a GUI
    caller can hand it to a FigureCanvasQTAgg for a live, vector-crisp
    interactive view instead of re-scaling the saved PNG's pixels. The
    caller then owns it and is responsible for eventually closing it.
    """
    if c_map is not None:
        idx = np.flatnonzero(c_map != 0)
        if idx.size == 0:
            return None, None, None
        arr_orig = img1.ravel()[idx]
        arr_shift = img2.ravel()[idx]
    else:
        arr_orig = np.asarray(img1).ravel()
        arr_shift = np.asarray(img2).ravel()
        if arr_orig.size == 0:
            return None, None, None

    finite_mask = np.isfinite(arr_orig) & np.isfinite(arr_shift)
    arr_orig = arr_orig[finite_mask]
    arr_shift = arr_shift[finite_mask]
    if arr_orig.size == 0:
        return None, None, None

    # Normalise to the physical TOA-reflectance range [0, 1].
    # Values outside this band are sensor artefacts or calibration edge
    # cases; clipping keeps the plot axes clean and the statistics honest.
    arr_orig  = np.clip(arr_orig,  0.0, 1.0)
    arr_shift = np.clip(arr_shift, 0.0, 1.0)

    rmse = float(np.sqrt(np.mean((arr_orig - arr_shift) ** 2)))
    bias = float(np.mean(arr_orig - arr_shift))
    n_common = int(arr_orig.size)

    r_squared = None
    if np.std(arr_shift) > 0 and np.std(arr_orig) > 0:
        fit_m, fit_c = _rma_linear_fit(arr_shift, arr_orig)
        residual_sum_squares = float(np.sum((arr_orig - (fit_m * arr_shift + fit_c)) ** 2))
        total_sum_squares = float(np.sum((arr_orig - np.mean(arr_orig)) ** 2))
        r_squared = 1.0 - residual_sum_squares / total_sum_squares
    else:
        fit_m, fit_c = 0.0, float(np.mean(arr_orig))

    lo = min(np.min(arr_orig), np.min(arr_shift))
    hi = max(np.max(arr_orig), np.max(arr_shift))
    # Span the full [0, 1] axis so the 1:1 line and fit line reach both corners.
    linear = np.linspace(0.0, 1.0, 200)
    fit_line = fit_m * linear + fit_c

    value_range = float(hi - lo)
    fit_identity_rmse = float(np.sqrt(np.mean(((fit_m - 1.0) * arr_shift + fit_c) ** 2)))
    fit_to_1to1_error_pct = (
        100.0 * fit_identity_rmse / value_range if value_range > 0 else None
    )

    rng = np.random.default_rng(0)
    jitter = (hi - lo) / density_bins if density_bins else 0.0
    plot_x = arr_shift + rng.uniform(-jitter / 2, jitter / 2, size=arr_shift.shape)
    plot_y = arr_orig + rng.uniform(-jitter / 2, jitter / 2, size=arr_orig.shape)

    density = _point_density(arr_shift, arr_orig, bins=density_bins)
    order = np.argsort(density)
    plot_x, plot_y, density = plot_x[order], plot_y[order], density[order]

    _THEIL_SEN_MAX_N = 5000
    slope_robust = intercept_robust = None
    if arr_shift.size >= 2 and np.std(arr_shift) > 0:
        if arr_shift.size > _THEIL_SEN_MAX_N:
            rng = np.random.default_rng(0)
            _sub_idx = rng.choice(arr_shift.size, size=_THEIL_SEN_MAX_N, replace=False)
            _ts_x, _ts_y = arr_shift[_sub_idx], arr_orig[_sub_idx]
        else:
            _ts_x, _ts_y = arr_shift, arr_orig
        try:
            slope_robust, intercept_robust, _lo_slope, _hi_slope = theilslopes(_ts_y, _ts_x)
            slope_robust = float(slope_robust)
            intercept_robust = float(intercept_robust)
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(8, 6.5))
    r2_text = f"{r_squared:.4f}" if r_squared is not None else "N/A"
    ax.set_title(
        f"Reference vs Target [N={n_common:,}, RMSE: {rmse:.4f}, "
        f"R\u00b2: {r2_text}]"
    )
    # Always prefix axes with TOA_Reflectance_ so units are unambiguous.
    _x_lbl = f"TOA_Reflectance_{x_label}" if x_label is not None else f"TOA_Reflectance_Target"
    _y_lbl = f"TOA_Reflectance_{y_label}" if y_label is not None else f"TOA_Reflectance_Reference"
    ax.set_xlabel(_x_lbl)
    ax.set_ylabel(_y_lbl)

    sc = ax.scatter(plot_x, plot_y, s=4, c=density, cmap="viridis", alpha=0.7,
                     edgecolors="none", zorder=1, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Point density (local estimate)")

    ax.plot(linear, linear, color="#222222", linestyle="--", linewidth=1.4, alpha=0.85,
            label="1:1 line (ideal)", zorder=2)

    fit_label = f"RMA fit: y = {fit_m:.3f}x "

    if fit_c < 0:
        fit_label += f"- {abs(fit_c):.3f}"
    elif fit_c > 0:
        fit_label += f"+ {fit_c:.3f}"

    ax.plot(linear, fit_line, color="orangered", linestyle="-.", linewidth=1.8, label=fit_label, zorder=3)

    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")


    # Fix both axes to the full TOA-reflectance range so the 1:1 line
    # always spans the entire physical domain and comparisons across runs
    # are visually consistent.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    if output_dir is None:
        from os import getcwd
        output_dir = join(getcwd(), "outputs", "data_plots")
    makedirs(output_dir, exist_ok=True)

    out_path = join(output_dir, "plot_common.png")
    fig.savefig(out_path, dpi=150)
    stats = {
        "rmse": rmse, "r_squared": r_squared, "n_common": n_common,
        "slope": fit_m, "intercept": fit_c, "bias": bias,
        "fit_to_1to1_error_pct": fit_to_1to1_error_pct,
        "slope_robust": slope_robust, "intercept_robust": intercept_robust,
    }
    if keep_fig_open:
        return out_path, stats, fig
    plt.close(fig)
    return out_path, stats, None
