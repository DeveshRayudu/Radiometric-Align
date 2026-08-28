"""
gui.signals

Centralized communication between the worker (background QThread) and
the GUI. All signals a worker needs to talk back to the GUI thread live
here, in one place, instead of being scattered/redefined across worker
classes.

Responsibilities (and only these):
  - Log message signal.
  - Error signal.
  - Finished signal.
  - (Optional) Progress signal, for future use.

This module only defines signals -- it doesn't emit them, connect them,
or decide what happens when they fire. Emitting is gui.worker's job;
connecting/handling is gui.controller's job.
"""

from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    # One line of runtime/log output from the pipeline.
    log_message = pyqtSignal(str)

    # An error occurred during the run. Carries a human-readable message.
    error = pyqtSignal(str)

    # The run ended, one way or the other. success is True/False; message
    # is a human-readable summary (e.g. the output dir on success, the
    # error text on failure).
    finished = pyqtSignal(bool, str)

    # Optional, for future use: (step_index, total_steps, label).
    progress = pyqtSignal(int, int, str)
