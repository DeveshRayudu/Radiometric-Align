"""
gui.main_window

Main GUI window — improved UI with:
  - Custom-painted animated pipeline step indicator
  - Prominent current-stage display with pulsing dot
  - Gradient progress bar with step counter
  - plot_common shown inside the viewer (not opened externally)
"""

import os
import math

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QDragEnterEvent, QDropEvent, QKeySequence,
    QPainter, QPen, QBrush, QColor, QFont, QPixmap,
)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout,
    QSizePolicy, QToolButton, QStyle, QShortcut, QScrollArea,
    QStackedWidget, QComboBox, QSpinBox, QDoubleSpinBox,
)

from gui.controller import Controller
from gui.console import ConsoleWidget
from gui.image_viewer import ImageViewerPanel
from raster_align.cli import RESAMPLING_CHOICES, _parse_nodata_arg
from raster_align.config import DEFAULT_NUM_THREADS, DEFAULT_MIN_OVERLAP_PCT
import cv2


RASTER_FILTER = "Raster files (*.tif *.tiff *.jp2 *.img);;All files (*.*)"

RASTER_OUTPUT_EXTENSIONS = {".tif", ".tiff", ".jp2", ".img"}
PLOT_OUTPUT_EXTENSIONS   = {".png", ".jpg", ".jpeg"}
LOG_OUTPUT_EXTENSIONS    = {".txt", ".log"}

PIPELINE_STAGES = [
    "Read & prepare",
    "Reflectance",
    "CRS check",
    "Overlap check",
    "Common grid",
    "ECC align",
    "Crop",
    "Homogeneity",
    "RGB composite",
    "Write rasters",
    "Histograms",
    "Plot",
]

PAGE_HOME   = 0
PAGE_VIEWER = 1

RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources"
)
ISRO_LOGO_PATH = os.path.join(RESOURCES_DIR, "isro_logo.png")


# ---------------------------------------------------------------------------
# Custom-painted pipeline step indicator
# ---------------------------------------------------------------------------

class PipelineStepIndicator(QWidget):
    """
    Animated horizontal step indicator.
    Circles: gray = pending, black pulsing = active, green check = done.
    A progress line connects all stages.
    """

    _CR = 9          # circle radius
    _CY = 20         # circle centre Y
    _LBL_GAP = 5     # gap between circle bottom and label top
    _LBL_H  = 30     # label area height
    _LBL_W  = 72     # label width (also drives the margin so labels stay centred)
    _H_MARGIN = 36   # = _LBL_W // 2  — guarantees no clamping needed on first/last stage

    _C_LINE_DONE   = QColor("#1E7A34")
    _C_LINE_PEND   = QColor("#E8E8E8")
    _C_DOT_DONE_BG = QColor("#1E7A34")
    _C_DOT_DONE_FG = QColor("#FFFFFF")
    _C_DOT_ACT_BG  = QColor("#111111")
    _C_DOT_ACT_FG  = QColor("#FFFFFF")
    _C_DOT_PEND_BG = QColor("#F5F5F5")
    _C_DOT_PEND_BD = QColor("#D0D0D0")
    _C_DOT_PEND_FG = QColor("#BBBBBB")
    _C_LBL_DONE    = QColor("#1E7A34")
    _C_LBL_ACT     = QColor("#111111")
    _C_LBL_PEND    = QColor("#CCCCCC")

    def __init__(self, stages, parent=None):
        super().__init__(parent)
        self._stages = list(stages)
        self._step = 0        # 1-based; 0 = idle
        self._pulse = 0.0     # 0.0 → 1.0 → 0.0 breathing
        self._pulse_dir = 1.0

        # Continuous line-fill position, in stage-index space (0-based,
        # fractional). Creeps toward `_line_target` every tick instead of
        # jumping straight to the newly-completed stage, so the line visibly
        # advances at the pace of whichever module is currently running.
        self._line_progress = 0.0
        self._line_target = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(35)
        self._timer.timeout.connect(self._tick)

        total_h = self._CY + self._CR + self._LBL_GAP + self._LBL_H + 6
        self.setFixedHeight(total_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Pipeline stages — gray: pending  ●: active  ✓: completed")

    # -- Public API ---------------------------------------------------------

    def reset(self):
        self._step = 0
        self._pulse = 0.0
        self._line_progress = 0.0
        self._line_target = 0.0
        self._timer.stop()
        self.update()

    def set_step(self, step_index, total_steps):
        """step_index is 1-based; 0 = idle."""
        self._step = step_index
        n = len(self._stages)
        if step_index <= 0:
            self._line_progress = 0.0
            self._line_target = 0.0
        elif step_index > n:
            # All done — let the line finish creeping the rest of the way
            # instead of snapping, then it settles at the far end.
            self._line_target = float(n - 1)
        else:
            active_idx = step_index - 1
            gap = 0.15 if active_idx > 0 else 0.0
            self._line_target = max(0.0, active_idx - gap)
        if 0 < step_index <= n or step_index > n:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
        self.update()

    # -- Animation ----------------------------------------------------------

    def _tick(self):
        self._pulse += 0.045 * self._pulse_dir
        if self._pulse >= 1.0:
            self._pulse = 1.0
            self._pulse_dir = -1.0
        elif self._pulse <= 0.0:
            self._pulse = 0.0
            self._pulse_dir = 1.0

        diff = self._line_target - self._line_progress
        if abs(diff) > 0.002:
            self._line_progress += diff * 0.06
        else:
            self._line_progress = self._line_target
            if self._step > len(self._stages):
                # Reached the end after the run finished — nothing left to
                # animate, stop the timer.
                self._timer.stop()

        self.update()

    # -- Painting -----------------------------------------------------------

    def _stage_xs(self):
        n = len(self._stages)
        if n <= 1:
            return [self.width() // 2]
        avail = self.width() - 2 * self._H_MARGIN
        step_w = avail / (n - 1)
        return [int(self._H_MARGIN + i * step_w) for i in range(n)]

    def _line_progress_x(self, xs):
        """Convert the continuous _line_progress (stage-index space) into a
        pixel x-position by interpolating between the two nearest stage
        marker positions."""
        n = len(xs)
        if n <= 1:
            return xs[0]
        p = max(0.0, min(self._line_progress, n - 1))
        idx = int(p)
        if idx >= n - 1:
            return xs[-1]
        frac = p - idx
        return int(xs[idx] + (xs[idx + 1] - xs[idx]) * frac)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        n = len(self._stages)
        if n == 0:
            return

        xs = self._stage_xs()
        step = self._step
        all_done = step > n
        CY, CR = self._CY, self._CR
        LY = CY + CR + self._LBL_GAP
        LH = self._LBL_H

        # -- Connecting line -------------------------------------------------
        painter.setPen(QPen(self._C_LINE_PEND, 1.5))
        painter.drawLine(xs[0], CY, xs[-1], CY)
        if step > 1 or all_done:
            done_x = self._line_progress_x(xs)
            painter.setPen(QPen(self._C_LINE_DONE, 1.5))
            painter.drawLine(xs[0], CY, done_x, CY)

        # -- Circles + labels -----------------------------------------------
        f_base = QFont(painter.font())
        f_base.setPointSize(7)
        f_base.setBold(False)
        f_bold = QFont(f_base)
        f_bold.setBold(True)
        f_tiny = QFont(f_base)
        f_tiny.setPointSize(6)
        f_tiny_bold = QFont(f_tiny)
        f_tiny_bold.setBold(True)

        for idx in range(n):
            x = xs[idx]
            snum = idx + 1  # 1-based

            is_done   = all_done or snum < step
            is_active = (not all_done) and snum == step
            is_pend   = (not all_done) and snum > step

            if is_done:
                painter.setBrush(QBrush(self._C_DOT_DONE_BG))
                painter.setPen(QPen(self._C_DOT_DONE_BG, 1))
                painter.drawEllipse(x - CR, CY - CR, CR * 2, CR * 2)
                painter.setFont(f_bold)
                painter.setPen(QPen(self._C_DOT_DONE_FG, 1))
                painter.drawText(x - CR, CY - CR, CR * 2, CR * 2, Qt.AlignCenter, "✓")
                painter.setFont(f_tiny)
                painter.setPen(QPen(self._C_LBL_DONE, 1))

            elif is_active:
                # Outer pulse ring
                ring_r = CR + int(self._pulse * 5)
                ring_alpha = int(55 * (1.0 - self._pulse))
                ring_c = QColor(17, 17, 17, ring_alpha)
                painter.setPen(QPen(ring_c, 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(x - ring_r, CY - ring_r, ring_r * 2, ring_r * 2)
                # Main filled circle
                painter.setBrush(QBrush(self._C_DOT_ACT_BG))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(x - CR, CY - CR, CR * 2, CR * 2)
                painter.setFont(f_bold)
                painter.setPen(QPen(self._C_DOT_ACT_FG, 1))
                painter.drawText(x - CR, CY - CR, CR * 2, CR * 2, Qt.AlignCenter, str(snum))
                painter.setFont(f_tiny_bold)
                painter.setPen(QPen(self._C_LBL_ACT, 1))

            else:  # pending
                painter.setBrush(QBrush(self._C_DOT_PEND_BG))
                painter.setPen(QPen(self._C_DOT_PEND_BD, 1))
                painter.drawEllipse(x - CR, CY - CR, CR * 2, CR * 2)
                painter.setFont(f_tiny)
                painter.setPen(QPen(self._C_DOT_PEND_FG, 1))
                painter.drawText(x - CR, CY - CR, CR * 2, CR * 2, Qt.AlignCenter, str(snum))
                painter.setFont(f_tiny)
                painter.setPen(QPen(self._C_LBL_PEND, 1))

            # Label below circle — centred exactly on the dot (no clamping needed
            # because _H_MARGIN == _LBL_W // 2, so first/last labels stay in bounds).
            lw = self._LBL_W
            lx = x - lw // 2
            painter.drawText(
                lx, LY, lw, LH,
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                self._stages[idx],
            )

        painter.end()


# ---------------------------------------------------------------------------
# Raster input card
# ---------------------------------------------------------------------------

class RasterInputCard(QFrame):
    pathChanged = pyqtSignal(str)

    def __init__(self, title, description, placeholder, parent=None):
        super().__init__(parent)
        self.setObjectName("rasterCard")
        self.setAcceptDrops(True)
        self._title = title
        self._description = description

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.icon_label = QLabel("▣", self)
        self.icon_label.setObjectName("rasterCardIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(32, 32)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("rasterCardTitle")
        self.description_label = QLabel(description, self)
        self.description_label.setObjectName("rasterCardDescription")
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.description_label)

        top_row.addWidget(self.icon_label)
        top_row.addLayout(title_col, 1)

        self.state_label = QLabel("●  NOT SELECTED", self)
        self.state_label.setObjectName("rasterCardState")
        self.state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_row.addWidget(self.state_label, 0, Qt.AlignTop)
        layout.addLayout(top_row)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setClearButtonEnabled(True)
        self.path_edit.setObjectName("rasterPathEdit")
        self.path_edit.setToolTip("Full raster path. You can also drag a raster file here.")
        self.path_edit.textChanged.connect(self._on_path_changed)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setMinimumWidth(92)
        self.browse_button.setToolTip(f"Browse for the {title.lower()}.")
        self.browse_button.clicked.connect(self._browse)

        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)
        layout.addLayout(path_row)

        self.file_info = QLabel("Drop a GeoTIFF/JP2/IMG file here, or use Browse.", self)
        self.file_info.setObjectName("rasterFileInfo")
        self.file_info.setWordWrap(False)
        layout.addWidget(self.file_info)

    @property
    def path(self):
        return self.path_edit.text().strip()

    def set_path(self, path):
        self.path_edit.setText(path or "")

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {self._title.lower()}", "", RASTER_FILTER)
        if path:
            self.set_path(path)

    def _on_path_changed(self, path):
        path = path.strip()
        if not path:
            self.state_label.setText("●  NOT SELECTED")
            self.state_label.setProperty("state", "empty")
            self.file_info.setText("Drop a GeoTIFF/JP2/IMG file here, or use Browse.")
        elif os.path.isfile(path):
            filename = os.path.basename(path)
            size = os.path.getsize(path)
            self.state_label.setText("✓  READY")
            self.state_label.setProperty("state", "ready")
            self.file_info.setText(f"{filename}  •  {self._format_size(size)}")
            self.file_info.setToolTip(path)
        else:
            self.state_label.setText("⚠  FILE NOT FOUND")
            self.state_label.setProperty("state", "invalid")
            self.file_info.setText("The selected path does not exist.")
            self.file_info.setToolTip(path)

        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.pathChanged.emit(path)

    @staticmethod
    def _format_size(size):
        if size < 1024:         return f"{size} B"
        if size < 1024**2:      return f"{size/1024:.1f} KB"
        if size < 1024**3:      return f"{size/1024**2:.1f} MB"
        return f"{size/1024**3:.1f} GB"

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._has_raster_urls(event.mimeData()):
            self.setProperty("dragActive", True)
            self.style().unpolish(self); self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self)
        event.accept()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if self._is_raster(path):
                    self.set_path(path)
                    event.acceptProposedAction()
                    self.setProperty("dragActive", False)
                    self.style().unpolish(self); self.style().polish(self)
                    return
        event.ignore()

    @staticmethod
    def _is_raster(path):
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in {".tif", ".tiff", ".jp2", ".img"}

    def _has_raster_urls(self, mime):
        return any(u.isLocalFile() and self._is_raster(u.toLocalFile()) for u in mime.urls())


# ---------------------------------------------------------------------------
# Parameters panel (unchanged except object name kept identical)
# ---------------------------------------------------------------------------

_MOTION_TYPE_CHOICES = {
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine":    cv2.MOTION_AFFINE,
}

_MAX_THREADS = max(1, os.cpu_count() or DEFAULT_NUM_THREADS)


class ParametersPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sectionFrame")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("Parameters", self)
        title.setObjectName("sectionTitle")
        hint = QLabel("Advanced — defaults match the CLI", self)
        hint.setObjectName("sectionHint")
        self.toggle_btn = QToolButton(self)
        self.toggle_btn.setObjectName("paramsToggleButton")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setText("Show")
        self.toggle_btn.toggled.connect(self._on_toggle)
        self.reset_btn = QToolButton(self)
        self.reset_btn.setObjectName("paramsResetButton")
        self.reset_btn.setText("Reset to defaults")
        self.reset_btn.setToolTip("Restore every parameter below to its default value.")
        self.reset_btn.clicked.connect(self.reset_defaults)
        header_row.addWidget(title)
        header_row.addWidget(hint)
        header_row.addStretch(1)
        header_row.addWidget(self.reset_btn)
        header_row.addWidget(self.toggle_btn)
        outer.addLayout(header_row)

        self.body = QWidget(self)
        grid = QGridLayout(self.body)
        grid.setContentsMargins(4, 8, 4, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        outer.addWidget(self.body)
        self.body.setVisible(False)

        col1 = QVBoxLayout(); col2 = QVBoxLayout(); col3 = QVBoxLayout()
        col1.setSpacing(8); col2.setSpacing(8); col3.setSpacing(8)

        self.homogeneity_window_spin = QSpinBox(self.body)
        self.homogeneity_window_spin.setRange(3, 999)
        self.homogeneity_window_spin.setSingleStep(2)
        self.homogeneity_window_spin.setValue(11)
        self.homogeneity_window_spin.setToolTip("Homogeneity moving-window size, in pixels (odd values recommended). Default: 11.")
        col1.addWidget(self._labeled("Homogeneity window (px)", self.homogeneity_window_spin))

        self.homogeneity_threshold_spin = QDoubleSpinBox(self.body)
        self.homogeneity_threshold_spin.setRange(0.0, 1000.0)
        self.homogeneity_threshold_spin.setDecimals(2)
        self.homogeneity_threshold_spin.setValue(5.0)
        self.homogeneity_threshold_spin.setSuffix(" %")
        self.homogeneity_threshold_spin.setToolTip("Homogeneity CV threshold (%). Default: 5.0%.")
        col1.addWidget(self._labeled("Homogeneity threshold", self.homogeneity_threshold_spin))

        self.resolution_mode_combo = QComboBox(self.body)
        self.resolution_mode_combo.addItems(["coarser", "finer", "src"])
        self.resolution_mode_combo.setToolTip("Resolution both images are resampled to. Default: coarser.")
        col1.addWidget(self._labeled("Resolution mode", self.resolution_mode_combo))

        self.pad_px_spin = QSpinBox(self.body)
        self.pad_px_spin.setRange(0, 100000)
        self.pad_px_spin.setValue(50)
        self.pad_px_spin.setToolTip("Padding (px) around the overlap before alignment. Default: 50.")
        col1.addWidget(self._labeled("Padding (px)", self.pad_px_spin))

        self.resampling_combo = QComboBox(self.body)
        self.resampling_combo.addItems(list(RESAMPLING_CHOICES.keys()))
        self.resampling_combo.setCurrentText("bilinear")
        self.resampling_combo.setToolTip("Resampling method for the common grid. Default: bilinear.")
        col2.addWidget(self._labeled("Resampling method", self.resampling_combo))

        self.motion_type_combo = QComboBox(self.body)
        self.motion_type_combo.addItems(list(_MOTION_TYPE_CHOICES.keys()))
        self.motion_type_combo.setToolTip("ECC motion model. Default: euclidean.")
        col2.addWidget(self._labeled("ECC motion model", self.motion_type_combo))

        self.min_overlap_spin = QDoubleSpinBox(self.body)
        self.min_overlap_spin.setRange(0.0, 100.0)
        self.min_overlap_spin.setDecimals(2)
        self.min_overlap_spin.setValue(DEFAULT_MIN_OVERLAP_PCT)
        self.min_overlap_spin.setSuffix(" %")
        self.min_overlap_spin.setToolTip(f"Minimum required overlap (%). Default: {DEFAULT_MIN_OVERLAP_PCT:.1f}%.")
        col2.addWidget(self._labeled("Min. overlap", self.min_overlap_spin))

        self.src_nodata_edit = QLineEdit(self.body)
        self.src_nodata_edit.setPlaceholderText("(use header nodata)")
        self.src_nodata_edit.setToolTip("Explicit nodata override for source raster. Leave blank to use header.")
        col3.addWidget(self._labeled("Source nodata", self.src_nodata_edit))

        self.target_nodata_edit = QLineEdit(self.body)
        self.target_nodata_edit.setPlaceholderText("(use header nodata)")
        self.target_nodata_edit.setToolTip("Explicit nodata override for target raster. Leave blank to use header.")
        col3.addWidget(self._labeled("Target nodata", self.target_nodata_edit))

        self.num_threads_spin = QSpinBox(self.body)
        self.num_threads_spin.setRange(1, _MAX_THREADS)
        self.num_threads_spin.setValue(DEFAULT_NUM_THREADS)
        self.num_threads_spin.setToolTip(f"GDAL warp threads. Capped at {_MAX_THREADS} (this machine's CPU count).")
        col3.addWidget(self._labeled(f"Threads (max {_MAX_THREADS})", self.num_threads_spin))

        col1.addStretch(1); col2.addStretch(1); col3.addStretch(1)
        grid.addLayout(col1, 0, 0)
        grid.addLayout(col2, 0, 1)
        grid.addLayout(col3, 0, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)

    @staticmethod
    def _labeled(text, widget):
        col = QVBoxLayout()
        col.setSpacing(3)
        label = QLabel(text)
        label.setObjectName("paramFieldLabel")
        col.addWidget(label)
        col.addWidget(widget)
        wrapper = QWidget()
        wrapper.setLayout(col)
        return wrapper

    def _on_toggle(self, checked):
        self.body.setVisible(checked)
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggle_btn.setText("Hide" if checked else "Show")

    def reset_defaults(self):
        self.homogeneity_window_spin.setValue(11)
        self.homogeneity_threshold_spin.setValue(5.0)
        self.resolution_mode_combo.setCurrentText("coarser")
        self.pad_px_spin.setValue(50)
        self.resampling_combo.setCurrentText("bilinear")
        self.motion_type_combo.setCurrentText("euclidean")
        self.min_overlap_spin.setValue(DEFAULT_MIN_OVERLAP_PCT)
        self.src_nodata_edit.setText("")
        self.target_nodata_edit.setText("")
        self.num_threads_spin.setValue(DEFAULT_NUM_THREADS)

    def get_pipeline_kwargs(self):
        resampling_key = self.resampling_combo.currentText()
        motion_key     = self.motion_type_combo.currentText()
        src_text = self.src_nodata_edit.text().strip()
        tgt_text = self.target_nodata_edit.text().strip()
        src_override, src_fallback_on = _parse_nodata_arg(src_text if src_text else None)
        tgt_override, tgt_fallback_on = _parse_nodata_arg(tgt_text if tgt_text else None)
        return dict(
            homogeneity_window=self.homogeneity_window_spin.value(),
            homogeneity_threshold=self.homogeneity_threshold_spin.value(),
            resampling=RESAMPLING_CHOICES[resampling_key],
            resolution_mode=self.resolution_mode_combo.currentText(),
            pad_px=self.pad_px_spin.value(),
            num_threads=self.num_threads_spin.value(),
            src_nodata_override=src_override,
            target_nodata_override=tgt_override,
            src_nodata_fallback=(0.0 if src_fallback_on else None),
            target_nodata_fallback=(0.0 if tgt_fallback_on else None),
            min_overlap_pct=self.min_overlap_spin.value(),
            motion_type=_MOTION_TYPE_CHOICES[motion_key],
        )


# ---------------------------------------------------------------------------
# Run output panel
# ---------------------------------------------------------------------------

def _count_output_files(output_dir):
    counts = {"raster": 0, "plot": 0, "log": 0}
    if not output_dir or not os.path.isdir(output_dir):
        return counts
    for _r, _d, files in os.walk(output_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in RASTER_OUTPUT_EXTENSIONS:  counts["raster"] += 1
            elif ext in PLOT_OUTPUT_EXTENSIONS:   counts["plot"]   += 1
            elif ext in LOG_OUTPUT_EXTENSIONS:    counts["log"]    += 1
    return counts


class RunOutputPanel(QFrame):
    resetRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sectionFrame")
        self._output_dir = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("Run output", self)
        title.setObjectName("sectionTitle")
        self.status_label = QLabel("✓  Completed", self)
        self.status_label.setObjectName("outputStatusLabel")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.status_label)
        layout.addLayout(title_row)

        self.dir_label = QLabel("", self)
        self.dir_label.setObjectName("outputDirLabel")
        self.dir_label.setWordWrap(True)
        self.dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.dir_label)

        counts_row = QHBoxLayout()
        counts_row.setSpacing(10)
        self.raster_count_label = QLabel("", self)
        self.raster_count_label.setObjectName("outputCountLabel")
        self.plot_count_label = QLabel("", self)
        self.plot_count_label.setObjectName("outputCountLabel")
        self.log_count_label = QLabel("", self)
        self.log_count_label.setObjectName("outputCountLabel")
        counts_row.addWidget(self.raster_count_label)
        counts_row.addWidget(self.plot_count_label)
        counts_row.addWidget(self.log_count_label)
        counts_row.addStretch(1)
        layout.addLayout(counts_row)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)
        self.reset_btn = QPushButton("Reset", self)
        self.reset_btn.setObjectName("secondaryActionButton")
        self.reset_btn.setToolTip("Clear and start a new run. (Ctrl+R)")
        self.reset_btn.clicked.connect(self.resetRequested.emit)
        actions_row.addWidget(self.reset_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.hide()

    def show_results(self, output_dir):
        self._output_dir = output_dir
        counts = _count_output_files(output_dir)
        self.dir_label.setText(output_dir)
        self.dir_label.setToolTip(output_dir)
        self.raster_count_label.setText(f"▣  {counts['raster']} raster outputs")
        self.plot_count_label.setText(f"▤  {counts['plot']} plot/diagnostic outputs")
        self.log_count_label.setText(f"▥  {counts['log']} log files")
        self.show()

    @property
    def output_dir(self):
        return self._output_dir


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------

class NavButton(QToolButton):
    def __init__(self, glyph, label, parent=None):
        super().__init__(parent)
        self.setObjectName("navButton")
        self.setCheckable(True)
        self.setFixedSize(60, 60)
        self.setToolTip(label)
        self._glyph = glyph
        self._label = label
        self._locked = False
        self._update_text()

    def set_locked(self, locked: bool):
        self._locked = locked
        self.setEnabled(not locked)
        if locked:
            self.setToolTip(f"{self._label} — available after a successful run")
        else:
            self.setToolTip(self._label)
        self._update_text()

    def _update_text(self):
        self.setText(f"{self._glyph}\n{self._label}")


class Sidebar(QWidget):
    pageRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)

        self._buttons: list = []

        self.home_btn   = self._add_btn("⌂",  "Home",  PAGE_HOME)
        self.viewer_btn = self._add_btn("▤", "Plots", PAGE_VIEWER)

        layout.addStretch(1)

        self.viewer_btn.set_locked(True)
        self.home_btn.setChecked(True)

    def _add_btn(self, glyph, label, page_index):
        btn = NavButton(glyph, label, self)
        btn.clicked.connect(lambda _checked, idx=page_index: self._on_clicked(idx))
        self.layout().addWidget(btn)
        self._buttons.append(btn)
        return btn

    def _on_clicked(self, page_index):
        self._select(page_index)
        self.pageRequested.emit(page_index)

    def select_page(self, page_index):
        self._select(page_index)

    def _select(self, page_index):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == page_index)

    def unlock_viewer(self):
        self.viewer_btn.set_locked(False)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Radiometric Alignment")
        self.resize(1120, 720)
        self.setMinimumSize(900, 650)

        self.controller = Controller(self)

        # Pulsing dot timer for the "running" state indicator
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(600)
        self._dot_timer.timeout.connect(self._pulse_dot)
        self._dot_state = True

        self._build_ui()
        self._connect_signals()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("rootWidget")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar = Sidebar(root)
        self.sidebar.pageRequested.connect(self._switch_page)
        root_layout.addWidget(self.sidebar)

        sep = QFrame(root)
        sep.setObjectName("sidebarSeparator")
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        root_layout.addWidget(sep)

        self._page_stack = QStackedWidget(root)
        root_layout.addWidget(self._page_stack, 1)

        home_page = self._build_home_page()
        self._page_stack.addWidget(home_page)           # PAGE_HOME   = 0

        self.image_viewer = ImageViewerPanel(self._page_stack)
        self._page_stack.addWidget(self.image_viewer)   # PAGE_VIEWER = 1

        self._page_stack.setCurrentIndex(PAGE_HOME)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self._build_shortcuts()

    def _build_home_page(self):
        central = QWidget()
        central.setObjectName("pageContent")
        central.setAttribute(Qt.WA_StyledBackground, True)

        scroll = QScrollArea()
        scroll.setObjectName("pageScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(central)

        main = QVBoxLayout(central)
        main.setContentsMargins(24, 20, 24, 12)
        main.setSpacing(16)

        # -- Header --
        header = QFrame(central)
        header.setObjectName("headerFrame")
        hdr_layout = QHBoxLayout(header)
        hdr_layout.setContentsMargins(2, 0, 2, 2)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("RADIOMETRIC ALIGNMENT", header)
        title.setObjectName("appTitle")
        subtitle = QLabel("Geospatial image registration and remote-sensing analysis", header)
        subtitle.setObjectName("appSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        # ISRO/NRSC logo, shown beside the title. The PNG already has a
        # transparent background, so it composites cleanly over the
        # header without any white box around it.
        logo_label = QLabel(header)
        logo_label.setObjectName("appLogo")
        if os.path.exists(ISRO_LOGO_PATH):
            logo_pix = QPixmap(ISRO_LOGO_PATH)
            if not logo_pix.isNull():
                logo_label.setPixmap(
                    logo_pix.scaledToHeight(56, Qt.SmoothTransformation)
                )
        hdr_layout.addWidget(logo_label)
        hdr_layout.addSpacing(12)
        hdr_layout.addLayout(title_col)
        hdr_layout.addStretch(1)
        main.addWidget(header)

        # -- Input section --
        input_frame = QFrame(central)
        input_frame.setObjectName("sectionFrame")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(18, 16, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Input data", input_frame)
        input_title.setObjectName("sectionTitle")
        input_hint = QLabel(
            "Choose the reference/source raster and target raster. "
            "You can browse or drag files onto either card.", input_frame)
        input_hint.setObjectName("sectionHint")
        input_layout.addWidget(input_title)
        input_layout.addWidget(input_hint)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.src_card = RasterInputCard(
            "SOURCE IMAGE", "Reference raster", "Source raster path...", input_frame)
        self.target_card = RasterInputCard(
            "TARGET IMAGE", "Raster to align", "Target raster path...", input_frame)

        self.src_path_edit     = self.src_card.path_edit
        self.src_browse_btn    = self.src_card.browse_button
        self.target_path_edit  = self.target_card.path_edit
        self.target_browse_btn = self.target_card.browse_button

        cards.addWidget(self.src_card, 1)
        cards.addWidget(self.target_card, 1)
        input_layout.addLayout(cards)
        main.addWidget(input_frame)

        # -- Parameters section --
        self.params_panel = ParametersPanel(central)
        main.addWidget(self.params_panel)

        # -- Processing section --
        proc_frame = QFrame(central)
        proc_frame.setObjectName("sectionFrame")
        proc_layout = QVBoxLayout(proc_frame)
        proc_layout.setContentsMargins(18, 14, 18, 16)
        proc_layout.setSpacing(10)

        # Current stage display — prominent pulsing dot + stage name + step counter
        stage_row = QHBoxLayout()
        stage_row.setSpacing(8)
        self._run_dot = QLabel("●", proc_frame)
        self._run_dot.setObjectName("runDot")
        self._run_dot.setProperty("dotState", "idle")
        self._run_dot.setFixedWidth(14)

        self.progress_label = QLabel("Ready to start", proc_frame)
        self.progress_label.setObjectName("stageNameLabel")

        self._step_counter = QLabel("", proc_frame)
        self._step_counter.setObjectName("stepCounterLabel")
        self._step_counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        stage_row.addWidget(self._run_dot)
        stage_row.addWidget(self.progress_label, 1)
        stage_row.addWidget(self._step_counter)
        proc_layout.addLayout(stage_row)

        # Start / stop button
        self.start_btn = QPushButton("Start processing", proc_frame)
        self.start_btn.setObjectName("startButton")
        self.start_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.setToolTip("Start processing the selected rasters. (F5)")
        self.start_btn.setShortcut(QKeySequence("F5"))
        start_row = QHBoxLayout()
        start_row.addStretch(1)
        start_row.addWidget(self.start_btn)
        start_row.addStretch(1)
        proc_layout.addLayout(start_row)

        # Custom-painted step indicator (replaces scrollable card strip)
        self.stage_indicator = PipelineStepIndicator(PIPELINE_STAGES, proc_frame)
        proc_layout.addWidget(self.stage_indicator)

        main.addWidget(proc_frame)

        # -- Console section --
        con_frame = QFrame(central)
        con_frame.setObjectName("sectionFrame")
        con_layout = QVBoxLayout(con_frame)
        con_layout.setContentsMargins(18, 14, 18, 18)
        con_layout.setSpacing(8)

        con_title_row = QHBoxLayout()
        con_title = QLabel("Process log", con_frame)
        con_title.setObjectName("sectionTitle")
        con_hint = QLabel("Live pipeline output", con_frame)
        con_hint.setObjectName("sectionHint")
        self.clear_console_btn = QToolButton(con_frame)
        self.clear_console_btn.setObjectName("clearConsoleButton")
        self.clear_console_btn.setText("Clear")
        self.clear_console_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.clear_console_btn.setToolTip("Clear the process log. (Ctrl+L)")
        self.clear_console_btn.clicked.connect(self.clear_console)
        con_title_row.addWidget(con_title)
        con_title_row.addStretch(1)
        con_title_row.addWidget(con_hint)
        con_title_row.addWidget(self.clear_console_btn)
        con_layout.addLayout(con_title_row)

        self.console_widget = ConsoleWidget(con_frame)
        self.console_widget.setMinimumHeight(180)
        con_layout.addWidget(self.console_widget, 1)
        main.addWidget(con_frame, 1)

        # -- Run output summary --
        self.output_panel = RunOutputPanel(central)
        main.addWidget(self.output_panel)

        return scroll

    def _build_shortcuts(self):
        self.clear_log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_log_shortcut.activated.connect(self.clear_console)
        self.reset_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.reset_shortcut.activated.connect(self.reset_ui)
        self.browse_src_shortcut = QShortcut(QKeySequence("Ctrl+Shift+O"), self)
        self.browse_src_shortcut.activated.connect(self.src_card._browse)
        self.browse_target_shortcut = QShortcut(QKeySequence("Ctrl+Alt+O"), self)
        self.browse_target_shortcut.activated.connect(self.target_card._browse)

    def _connect_signals(self):
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.src_card.pathChanged.connect(self._update_input_summary)
        self.target_card.pathChanged.connect(self._update_input_summary)
        self.output_panel.resetRequested.connect(self.reset_ui)

    # -----------------------------------------------------------------------
    # Page switching
    # -----------------------------------------------------------------------

    def _switch_page(self, page_index):
        if page_index == PAGE_VIEWER and self.sidebar.viewer_btn._locked:
            return
        self._page_stack.setCurrentIndex(page_index)
        self.sidebar.select_page(page_index)

    # -----------------------------------------------------------------------
    # Called by Controller on successful run completion
    # -----------------------------------------------------------------------

    def show_run_results(self, output_dir):
        # Advance the step indicator past the last stage so every circle turns green.
        self.stage_indicator.set_step(len(PIPELINE_STAGES) + 1, len(PIPELINE_STAGES))
        self.progress_label.setText("Complete")
        self._step_counter.setText(f"{len(PIPELINE_STAGES)} / {len(PIPELINE_STAGES)}")
        self.output_panel.show_results(output_dir)
        self.sidebar.unlock_viewer()
        self.image_viewer.set_output_dir(output_dir)
        self._switch_page(PAGE_VIEWER)

    # -----------------------------------------------------------------------
    # Status / progress helpers
    # -----------------------------------------------------------------------

    def _on_start_clicked(self):
        if self.controller.is_running:
            self.controller.stop_run()
            return
        self.controller.start_run(
            src_path=self.src_path_edit.text().strip(),
            target_path=self.target_path_edit.text().strip(),
            pipeline_kwargs=self.params_panel.get_pipeline_kwargs(),
        )

    def _update_input_summary(self, _path):
        selected = bool(self.src_card.path and self.target_card.path)
        valid = (selected
                 and os.path.isfile(self.src_card.path)
                 and os.path.isfile(self.target_card.path))
        if valid:
            self.status_bar.showMessage("Both input rasters are ready.")
        elif selected:
            self.status_bar.showMessage("Check the selected raster paths.")
        else:
            self.status_bar.showMessage("Select a source and target raster to begin.")

    def append_console_line(self, line):
        self.console_widget.append_message(line)

    def clear_console(self):
        self.console_widget.clear_console()

    def set_progress(self, step_index, total_steps, label):
        self.progress_label.setText(label)
        total = total_steps or 1
        self._step_counter.setText(f"{step_index} / {total}")
        self.status_bar.showMessage(f"[{step_index}/{total}] {label}")
        self.stage_indicator.set_step(step_index, total_steps)

    def set_running_state(self, running):
        if running:
            self._set_status_state("processing", "●  PROCESSING")
            self.progress_label.setText("Initialising…")
            self._step_counter.setText("")
            self.stage_indicator.reset()
            self.output_panel.hide()
            self._set_start_btn_mode("stop")
            # Start dot pulse animation
            self._dot_state = True
            self._set_dot_state("active")
            self._dot_timer.start()
        else:
            self._dot_timer.stop()
            if getattr(self, "_status_state", "ready") not in ("failed", "stopped"):
                self._set_status_state("ready", "✓  READY")
            self._set_dot_state("idle")
            self._set_start_btn_mode("start")

    def _pulse_dot(self):
        self._dot_state = not self._dot_state
        self._set_dot_state("active" if self._dot_state else "dim")

    def _set_dot_state(self, state):
        self._run_dot.setProperty("dotState", state)
        self._run_dot.style().unpolish(self._run_dot)
        self._run_dot.style().polish(self._run_dot)

    def set_stopping_state(self):
        self.progress_label.setText("Stopping…")
        self.status_bar.showMessage("Stopping after the current stage finishes…")
        self.start_btn.setEnabled(False)

    def set_stopped_state(self):
        self._set_status_state("stopped", "■  STOPPED")
        self.progress_label.setText("Stopped")
        self._step_counter.setText("")
        self.status_bar.showMessage("Run stopped by user.")
        self._set_dot_state("idle")

    def set_failed_state(self, message=None):
        self._set_status_state("failed", "✕  FAILED")
        self.progress_label.setText("Failed")
        self._step_counter.setText("")
        self._set_dot_state("idle")
        if message:
            self.status_bar.showMessage(f"Run failed: {message}")

    def _set_start_btn_mode(self, mode):
        if mode == "stop":
            self.start_btn.setText("Stop processing")
            self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.start_btn.setToolTip("Stop the current run. (F5)")
            self.start_btn.setProperty("mode", "stop")
        else:
            self.start_btn.setText("Start processing")
            self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.start_btn.setToolTip("Start processing the selected rasters. (F5)")
            self.start_btn.setProperty("mode", "start")
        self.start_btn.setEnabled(True)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)

    def _set_status_state(self, state, text):
        # Status tag removed from the header — no widget to update, but we
        # still track the state so callers that check it (e.g. run-finished
        # logic) keep working correctly.
        self._status_state = state

    def reset_ui(self):
        self.src_card.set_path("")
        self.target_card.set_path("")
        self.params_panel.reset_defaults()
        self.clear_console()
        self.progress_label.setText("Ready to start")
        self._step_counter.setText("")
        self.stage_indicator.reset()
        self.output_panel.hide()
        self._set_status_state("ready", "✓  READY")
        self._set_start_btn_mode("start")
        self._set_dot_state("idle")
        self._dot_timer.stop()
        self.status_bar.showMessage("Select a source and target raster to begin.")
        self._page_stack.setCurrentIndex(PAGE_HOME)
        self.sidebar.select_page(PAGE_HOME)
