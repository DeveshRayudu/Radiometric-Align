"""
gui.controller

Controls the complete workflow between the GUI and the processing
backend (raster_align.run_pipeline). This is the module that knows about
output-directory layout, logging setup, and the background worker thread
-- main_window.py stays a pure view and never touches any of this
directly; it only calls start_run(...) and exposes a couple of callback
targets (append_console_line, set_progress) for this controller to push
updates into.
"""

import os
import time

from PyQt5.QtWidgets import QMessageBox

from raster_align.config import DEFAULT_OUTPUT_DIR, RUN_DIR_PREFIX
from gui.worker import PipelineWorker


class Controller:
    def __init__(self, main_window):
        self.window = main_window
        self.worker = None
        self.output_dir = None
        self.is_running = False

    # -- Entry point, called by MainWindow._on_start_clicked -------------
    def start_run(self, src_path, target_path, pipeline_kwargs=None):
        error = self._validate_paths(src_path, target_path)
        if error:
            self._show_error(error)
            return

        output_dir = self._create_run_output_dir()
        self.output_dir = output_dir

        self.window.clear_console()
        self.window.append_console_line(f"[GUI] Output folder for this run: {output_dir}")
        self.window.set_progress(0, 1, "Starting...")
        self.is_running = True
        self.window.set_running_state(True)
        self._set_controls_enabled(False)

        self.worker = PipelineWorker(src_path, target_path, output_dir, pipeline_kwargs=pipeline_kwargs)
        self.worker.console_line.connect(self.window.append_console_line)
        self.worker.progress.connect(self.window.set_progress)
        self.worker.figures_ready.connect(self.window.image_viewer.set_live_figures)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    # -- Entry point, called by MainWindow when Stop is clicked mid-run --
    def stop_run(self):
        if self.worker is None or not self.is_running:
            return
        self.window.append_console_line("[GUI] Stop requested -- finishing current stage, then cancelling...")
        self.window.set_stopping_state()
        self.worker.request_stop()

    # -- Validation --------------------------------------------------------
    def _validate_paths(self, src_path, target_path):
        if not src_path or not target_path:
            return "Please select both a source and a target image."
        if not os.path.isfile(src_path):
            return f"Source image not found:\n{src_path}"
        if not os.path.isfile(target_path):
            return f"Target image not found:\n{target_path}"
        return None

    # -- Output directory: base "outputs" dir + epoch-named subfolder ------
    def _create_run_output_dir(self):
        base_output_dir = os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)
        os.makedirs(base_output_dir, exist_ok=True)
        epoch = int(time.time())
        run_name = f"{RUN_DIR_PREFIX}_{epoch}"
        run_output_dir = os.path.join(base_output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        return run_output_dir

    # -- Worker completion ---------------------------------------------------
    def _on_worker_finished(self, success, message):
        was_cancelled = (not success) and message == "Run stopped by user."
        self.is_running = False
        self._set_controls_enabled(True)
        self.window.set_running_state(False)
        if success:
            self.window.append_console_line("[GUI] Run finished successfully.")
            # plot_common.png is now shown inside the viewer — no external opener needed
            self.window.show_run_results(self.output_dir)
        elif was_cancelled:
            self.window.append_console_line(f"[GUI] {message}")
            self.window.set_stopped_state()
        else:
            self.window.append_console_line(f"[GUI] Run failed: {message}")
            self.window.set_failed_state(message)
            QMessageBox.critical(self.window, "Run failed", message or "The pipeline run failed.")

    # -- GUI control enable/disable during a run ------------------------------
    def _set_controls_enabled(self, enabled):
        self.window.src_browse_btn.setEnabled(enabled)
        self.window.target_browse_btn.setEnabled(enabled)
        self.window.src_path_edit.setEnabled(enabled)
        self.window.target_path_edit.setEnabled(enabled)
        self.window.params_panel.setEnabled(enabled)

    def _show_error(self, message):
        QMessageBox.warning(self.window, "Invalid input", message)
