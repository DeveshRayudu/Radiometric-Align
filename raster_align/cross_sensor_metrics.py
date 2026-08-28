"""Cross-sensor regression and error statistics for matched reflectance pixels."""

import math
import numpy as np


def _odr_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Analytical Orthogonal Distance Regression (ODR / major-axis regression)
    for the symmetric error-in-variables case.

    The slope is:

        m = (S_yy - S_xx + sqrt((S_yy - S_xx)^2 + 4 * S_xy^2)) / (2 * S_xy)

    where S_xx, S_yy, S_xy are sums of squares/cross-products relative to
    the means.  This is the standard solution for bivariate error-in-variables
    regression (equal weights); see Isobe et al. (1990, ApJ).

    Returns (slope, intercept).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_bar, y_bar = float(np.mean(x)), float(np.mean(y))
    dx = x - x_bar
    dy = y - y_bar

    S_xx = float(np.sum(dx * dx))
    S_yy = float(np.sum(dy * dy))
    S_xy = float(np.sum(dx * dy))

    if S_xy == 0.0:
        if S_xx == 0.0:
            return 0.0, y_bar
        return float(S_xy / S_xx), y_bar - (S_xy / S_xx) * x_bar

    discriminant = (S_yy - S_xx) ** 2 + 4.0 * S_xy ** 2
    slope = ((S_yy - S_xx) + math.sqrt(discriminant)) / (2.0 * S_xy)
    intercept = y_bar - slope * x_bar
    return slope, intercept


def compute_cross_sensor_statistics(
    rho_reference: np.ndarray,
    rho_target: np.ndarray,
) -> dict:
    """
    Compute the full suite of cross-sensor comparison statistics per the
    methodology document Section 10, over matched reflectance pixel arrays.

    Parameters
    ----------
    rho_reference : np.ndarray, shape (N,)
        Reference-sensor surface reflectance values (the "ground truth" side,
        typically the better-calibrated sensor: Landsat-8/9 or Sentinel-2).
    rho_target : np.ndarray, shape (N,)
        Target-sensor surface reflectance values (the sensor being assessed).
        Must be the same length as rho_reference.

    Returns
    -------
    dict with keys:
        n_pixels          int    -- number of valid matched pixels used.
        mbe               float  -- Mean Bias Error (reference - target).
        relative_bias_pct float  -- MBE / mean(reference) * 100  [%].
        mae               float  -- Mean Absolute Error.
        rmse              float  -- Root Mean Square Error.
        relative_rmse_pct float  -- RMSE / mean(reference) * 100  [%].
        pearson_r         float  -- Pearson correlation coefficient.
        odr_slope         float  -- ODR regression slope.
        odr_intercept     float  -- ODR regression intercept.
        ci95_mbe          float  -- 95% CI half-width on MBE (Â± this value).
        ci95_rmse         float  -- 95% CI half-width on RMSE (approx.).
        std_residuals     float  -- Std deviation of (reference - target).
        r_squared         float  -- Coefficient of determination (ODR fit).

    Notes
    -----
    NaN/Inf values in either array are silently dropped before all
    computations (identical to plot_common behaviour).
    """
    ref = np.asarray(rho_reference, dtype=np.float64).ravel()
    tgt = np.asarray(rho_target, dtype=np.float64).ravel()

    valid = np.isfinite(ref) & np.isfinite(tgt)
    ref = ref[valid]
    tgt = tgt[valid]
    n = int(ref.size)

    if n == 0:
        return {k: None for k in (
            "n_pixels", "mbe", "relative_bias_pct", "mae", "rmse",
            "relative_rmse_pct", "pearson_r", "odr_slope", "odr_intercept",
            "ci95_mbe", "ci95_rmse", "std_residuals", "r_squared",
        )} | {"n_pixels": 0}

    residuals = ref - tgt
    mbe = float(np.mean(residuals))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    std_res = float(np.std(residuals))

    mean_ref = float(np.mean(ref))
    relative_bias_pct = (mbe / mean_ref * 100.0) if mean_ref != 0.0 else None
    relative_rmse_pct = (rmse / mean_ref * 100.0) if mean_ref != 0.0 else None

    if np.std(ref) > 0 and np.std(tgt) > 0:
        pearson_r = float(np.corrcoef(ref, tgt)[0, 1])
    else:
        pearson_r = None

    odr_slope, odr_intercept = _odr_slope_intercept(tgt, ref)

    fitted = odr_slope * tgt + odr_intercept
    ss_res = float(np.sum((ref - fitted) ** 2))
    ss_tot = float(np.sum((ref - mean_ref) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    sem = std_res / math.sqrt(n) if n > 0 else None
    ci95_mbe = float(1.96 * sem) if sem is not None else None

    ci95_rmse = float(1.96 * std_res / math.sqrt(2 * n)) if n > 30 else None

    return {
        "n_pixels": n,
        "mbe": mbe,
        "relative_bias_pct": relative_bias_pct,
        "mae": mae,
        "rmse": rmse,
        "relative_rmse_pct": relative_rmse_pct,
        "pearson_r": pearson_r,
        "odr_slope": odr_slope,
        "odr_intercept": odr_intercept,
        "ci95_mbe": ci95_mbe,
        "ci95_rmse": ci95_rmse,
        "std_residuals": std_res,
        "r_squared": r_squared,
    }



def format_stats_table(stats: dict, sensor_src_label: str, sensor_tgt_label: str) -> list[str]:
    """
    Format the cross-sensor statistics dict into a list of log-friendly
    strings that can be passed to logging_utils.print_module_summary().

    Parameters
    ----------
    stats : dict
        Output of compute_cross_sensor_statistics().
    sensor_src_label : str
        Human-readable label for the reference sensor (e.g. "Sentinel-2_MSI_B04").
    sensor_tgt_label : str
        Human-readable label for the target sensor (e.g. "Resourcesat-2A_LISS4_B3").

    Returns
    -------
    list[str]
        One string per metric, ready to pass as ``checks`` to print_module_summary.
    """
    if stats.get("n_pixels") == 0 or stats.get("n_pixels") is None:
        return ["No valid matched pixels found -- statistics not computed."]

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    n = stats["n_pixels"]
    lines = [
        f"Reference sensor: {sensor_src_label}",
        f"Target sensor: {sensor_tgt_label}",
        f"N matched pixels: {n:,}",
        f"MBE (ref - tgt): {_fmt(stats['mbe'])}  "
        f"(relative bias: {_fmt(stats['relative_bias_pct'], '.2f')} %,  "
        f"95% CI: Â±{_fmt(stats['ci95_mbe'])})",
        f"MAE: {_fmt(stats['mae'])}",
        f"RMSE: {_fmt(stats['rmse'])}  "
        f"(relative RMSE: {_fmt(stats['relative_rmse_pct'], '.2f')} %,  "
        f"95% CI: Â±{_fmt(stats['ci95_rmse'])})",
        f"Std of residuals: {_fmt(stats['std_residuals'])}",
        f"Pearson r: {_fmt(stats['pearson_r'])}",
        f"RMA slope: {_fmt(stats['odr_slope'])}  |  "
        f"RMA intercept: {_fmt(stats['odr_intercept'])}",
        f"RÂ² (RMA fit): {_fmt(stats['r_squared'])}",
    ]
    return lines




def compute_cross_sensor_report(
    rho_reference: np.ndarray,
    rho_target: np.ndarray,
    sensor_src_label: str,
    sensor_tgt_label: str,
) -> dict:
    """Compute statistics directly on the matched reflectance arrays."""
    rho_ref = np.asarray(rho_reference, dtype=np.float32)
    rho_tgt = np.asarray(rho_target, dtype=np.float32)
    stats = compute_cross_sensor_statistics(rho_ref, rho_tgt)
    checks = format_stats_table(stats, sensor_src_label, sensor_tgt_label)
    if stats.get("n_pixels", 0) == 0:
        result = "SKIPPED -- no valid matched pixels for cross-sensor statistics."
        actions = ["Skipped statistical comparison because no valid matched pixels were available."]
    else:
        result = (
            f"OK -- N={stats['n_pixels']:,}, RMSE={stats['rmse']:.6f}, "
            f"RMA slope={stats['odr_slope']:.6f}, RMA intercept={stats['odr_intercept']:.6f}."
        )
        actions = ["Computed cross-sensor statistics directly from the matched reflectance arrays."]
    return {
        "stats": stats,
        "checks": checks,
        "actions": actions,
        "result": result,
        "rho_reference_corrected": rho_ref,
        "rho_target_corrected": rho_tgt,
    }
