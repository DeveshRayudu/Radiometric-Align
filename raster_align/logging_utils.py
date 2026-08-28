"""
raster_align.logging_utils

Two log files are produced per run:
  - run_log.txt          : a small, precise summary -- run header, one
    result line per pipeline module (plus a few key highlight lines, e.g.
    detected shift), and the final timing totals. Meant to be skimmed --
    and this is also what a GUI console (console_callback) is fed, so the
    GUI's CLI view shows this same small/precise content.
  - run_log_detailed.txt : an exact mirror of everything printed to the
    real terminal via print(...) -- every module's full parameter/checks/
    actions block, step timing, warnings, everything.

Use `print(...)` as normal for anything that should show up on the real
terminal and in run_log_detailed.txt. The short run_log.txt (and the GUI
console) is populated automatically and only from: write_run_header(),
print_module_summary()'s `result`/`highlights`, and
print_timing_summary()'s totals -- nothing else needs to touch it.
"""

import sys
from datetime import datetime
from os.path import join



_active_short_log = None


def _short(msg):
    if _active_short_log is not None:
        _active_short_log(msg)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def open_run_logs(output_dir, short_filename="run_log.txt", detail_filename="run_log_detailed.txt",
                   console_callback=None):
    """
    Opens the two log files and returns everything the caller needs:

      tee_stdout   -- install this as sys.stdout. Mirrors every print() to
                      the real terminal console AND the detail log file
                      (the short log file is populated separately, see
                      below). This stream is NOT forwarded to
                      console_callback -- a GUI console should show the
                      small run_log.txt content, not the full detail feed.
      short_log_file, detail_log_file -- file handles, for closing later.
      detail_log   -- callable(msg): no-op, kept only for backwards
                      compatibility with existing call sites.

    console_callback, if given, is called with each line as it's written
    to run_log.txt (header, one result line per module, key highlights,
    timing totals) -- i.e. a GUI console mirrors the short/precise log,
    not the full print() stream.
    """
    global _active_short_log

    short_log_file = open(join(output_dir, short_filename), "a", encoding="utf-8")
    detail_log_file = open(join(output_dir, detail_filename), "a", encoding="utf-8")

    run_marker = f"\n\n########## RUN START {datetime.now().isoformat(timespec='seconds')} ##########\n"
    short_log_file.write(run_marker)
    detail_log_file.write(run_marker)
    short_log_file.flush()
    detail_log_file.flush()

    tee_stdout = _Tee(sys.stdout, detail_log_file)

    def _short_log_line(msg):
        short_log_file.write(msg + "\n")
        short_log_file.flush()
        if console_callback is not None:
            console_callback(msg)

    _active_short_log = _short_log_line

    def detail_log(_msg):
        pass

    return tee_stdout, short_log_file, detail_log_file, detail_log


def log_step_start(detail_log, label, **params):
    ts = datetime.now().isoformat(timespec="milliseconds")
    suffix = f" | params: {params}" if params else ""
    detail_log(f"[STEP START] {label} @ {ts}{suffix}")


def log_step_end(detail_log, label, elapsed, **extra):
    parts = [f"elapsed={elapsed:.3f}s"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    detail_log(f"[STEP END]   {label} | " + " | ".join(parts))


def write_run_header(path_params, config_params=None):
    """
    Print the run-start header block (console + detail log), and also
    write a compact version of it to run_log.txt.

    path_params    : iterable of (label, value) for the path rows
                     (src path, target path, output dir).
    config_params  : iterable of (label, value) for the run's key config
                     values (homogeneity window/threshold, resampling
                     method, resolution mode, pad_px, num_threads, nodata
                     fallbacks, motion_type, ...), printed as a second
                     block right after the paths.
    """
    lines = ["=" * 78, f"Run started : {datetime.now().isoformat(timespec='seconds')}", ""]
    for label, value in path_params:
        lines.append(f"{label:<28s}: {value}")
    if config_params:
        lines.append("")
        for label, value in config_params:
            lines.append(f"{label:<28s}: {value}")
    lines.append("=" * 78)

    for line in lines:
        print(line)
    for line in lines:
        _short(line)


def print_module_summary(module_name, params=None,
                          checks=None, actions=None, result=None, highlights=None):
    """
    Print a single, consistently-formatted summary block for one
    preprocessing module/stage, so every stage of the pipeline reports
    itself the same way on console (and both logs, since this just uses
    print()).

    Every argument except module_name is optional -- callers pass only
    what's relevant to that stage (e.g. a check-only stage like the
    overlap check has no "actions taken"; a stage with no separate
    reference/target image, like homogeneity, can omit those).

    module_name : str
        Name of the module/stage, e.g. "Image Loading", "Resolution Check".
    reference, target : str or None
        Accepted for backwards compatibility with existing call sites, but
        no longer printed here -- the reference/target image names are
        printed once, at the start of the run (see print_run_images()),
        rather than repeated in every module's summary block.
    params : dict or None
        Key input parameters used by this stage (e.g. {"window": 11,
        "threshold": 5.0}).
    checks : list[str] or None
        Checks performed, each as a short human-readable line already
        stating what was checked and, ideally, what was found.
    actions : list[str] or None
        Actions taken as a result of those checks. Each entry should be
        explicit about whether the action was required or skipped, which
        image (if any) it applied to, and why -- e.g. "Resampled target
        to source resolution (required: target pixel size 5.0m != source
        2.5m)" or "Skipped resampling target (not required: resolutions
        already match)".
    result : str or None
        One-line final status/outcome of this module, e.g. "OK -- src and
        target now share a common grid" or "Stopped -- no overlap found".
    highlights : str, list[str], or None
        A small number of key facts worth surfacing in run_log.txt even
        though it otherwise only gets the result line -- e.g. for the
        alignment module, the detected shift and whether it was applied
        ("Shift detected: dx=1.32px, dy=-0.87px, rotation=0.021deg --
        correction APPLIED"). Kept short and console/detail-log-visible
        too (printed as an "Info:" line in the full block), NOT a
        substitute for `checks`/`actions`, which stay detail-log-only in
        terms of density.
    """
    if isinstance(highlights, str):
        highlights = [highlights]

    header = f" Module: {module_name} "
    print("\n" + header.center(78, "="))
    if params:
        print("  Parameters:")
        for k, v in params.items():
            print(f"    - {k}: {v}")
    if checks:
        print("  Checks performed:")
        for c in checks:
            print(f"    - {c}")
    if actions:
        print("  Actions taken:")
        for a in actions:
            print(f"    - {a}")
    if highlights:
        print("  Info:")
        for h in highlights:
            print(f"    - {h}")
    if result is not None:
        print(f"  Result: {result}")
    print("=" * 78)

    if result is not None:
        _short(f"{module_name:<45s}: {result}")
    else:
        _short(f"{module_name}")
    if highlights:
        for h in highlights:
            _short(f"    - {h}")


def print_run_images(reference, target):
    """
    Print the reference/target image names once, at the start of the run
    (right after they're read), instead of repeating them in every
    module's summary block.
    """
    print(f"  Reference image : {reference}")
    print(f"  Target image    : {target}")


def print_timing_summary(step_times, total_time):
    lines = ["\n" + "=" * 78, "Step timing summary", "-" * 78]
    for label, dt in step_times:
        lines.append(f"  {label:<45s} {dt:8.2f}s")
    lines.append("-" * 78)
    lines.append(f"  {'TOTAL (sum of timed steps)':<45s} {sum(dt for _, dt in step_times):8.2f}s")
    lines.append(f"  {'TOTAL (wall clock, full pipeline)':<45s} {total_time:8.2f}s")
    lines.append("=" * 78)

    for line in lines:
        print(line)

    _short("-" * 78)
    _short(f"  {'TOTAL (wall clock, full pipeline)':<45s} {total_time:8.2f}s")
    _short("=" * 78)
