"""
gui.worker

Runs the processing pipeline (raster_align.run_pipeline) in a background
thread.

Responsibilities (and only these):
  - Execute pipeline.py's run_pipeline(...) with the paths/output dir the
    controller gives it.
  - Forward log messages to the GUI, via a Qt signal (thread-safe) fed by
    run_pipeline's short_log_callback hook. This is the compact
    run_log.txt content (run header, one result line per module, key
    highlights, final timing totals) -- not the full run_log_detailed.txt
    stream -- so the GUI console stays readable during a run.
  - Emit a completion/error signal when the run finishes either way.
  - Do all of the above off the GUI thread, so the UI never freezes
    while a run is in progress.

No processing logic lives here either -- this is purely a QThread shim
around raster_align.run_pipeline. It doesn't validate paths, build the
output directory, or decide what happens on success/failure (dialogs,
re-enabling controls, etc.) -- that's gui.controller's job; this module
only runs the pipeline and reports what happened.
"""

import threading

from PyQt5.QtCore import QThread, pyqtSignal

from raster_align.pipeline import run_pipeline


class RunCancelled(Exception):
    """Raised (from inside the progress_callback hook) to unwind out of
    run_pipeline when the user has pressed Stop. Caught in run() below,
    same treatment as the SystemExit case."""


class PipelineWorker(QThread):
    # One line of the compact run_log.txt output, forwarded live.
    console_line = pyqtSignal(str)
    # (step_index, total_steps, label), forwarded from progress_callback.
    progress = pyqtSignal(int, int, str)
    # {filename: matplotlib Figure} for the chart-style outputs (plot_common,
    # shift histograms) -- emitted once, just before `finished`, only on a
    # successful run. The Figures are fully rendered (Agg backend) but never
    # touched a Qt widget, so handing them to the GUI thread here is safe;
    # only the receiving slot should wrap them in a FigureCanvasQTAgg.
    figures_ready = pyqtSignal(dict)
    # (success, message) -- emitted exactly once, when the run ends.
    finished = pyqtSignal(bool, str)

    def __init__(self, src_path, target_path, output_dir, pipeline_kwargs=None, parent=None):
        super().__init__(parent)
        self.src_path = src_path
        self.target_path = target_path
        self.output_dir = output_dir
        # Extra run_pipeline kwargs from the GUI's Parameters panel (e.g.
        # homogeneity_threshold, resampling, resolution_mode...) -- see
        # gui.main_window.ParametersPanel.get_pipeline_kwargs(). Empty dict
        # means "use run_pipeline's own defaults", same as the old behavior.
        self.pipeline_kwargs = pipeline_kwargs or {}
        # Set by request_stop() (called from the GUI thread); checked from
        # _emit_progress() (called from this worker thread) at the start of
        # every major pipeline stage. run_pipeline has no cancellation hook
        # of its own, so this is the only place a Stop request can actually
        # interrupt an in-progress run -- it takes effect at the next stage
        # boundary, not instantly mid-stage.
        self._stop_requested = threading.Event()

    def request_stop(self):
        """Ask the run to cancel at the next stage boundary. Thread-safe --
        called from the GUI thread while this worker is running."""
        self._stop_requested.set()

    def run(self):
        try:
            run_pipeline(
                self.src_path,
                self.target_path,
                self.output_dir,
                short_log_callback=self.console_line.emit,
                progress_callback=self._emit_progress,
                figure_callback=self.figures_ready.emit,
                **self.pipeline_kwargs,
            )
        except RunCancelled:
            message = "Run stopped by user."
            self.console_line.emit(f"[GUI] {message}")
            self.finished.emit(False, message)
        except SystemExit:
            # run_pipeline calls sys.exit(1) on fatal-but-expected
            # conditions (e.g. src/target don't overlap enough to align)
            # after already printing a clear reason to the console --
            # that message already reached the GUI via short_log_callback
            # above, so just report the run as failed, don't crash the
            # worker thread.
            message = "Pipeline stopped early -- see the console output above for the reason."
            self.console_line.emit(f"[GUI] {message}")
            self.finished.emit(False, message)
        except Exception as exc:
            self.console_line.emit(f"[GUI] ERROR: {exc}")
            self.finished.emit(False, str(exc))
        else:
            self.finished.emit(True, f"Outputs written to: {self.output_dir}")

    def _emit_progress(self, step_index, total_steps, label):
        if self._stop_requested.is_set():
            raise RunCancelled()
        self.progress.emit(step_index, total_steps, label)
