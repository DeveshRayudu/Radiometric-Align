"""
gui.image_viewer

Interactive plot/image viewer with Desmos-style infinite canvas.
Sequential loading: northing → easting → plot_common → others,
each with its own loading indicator that operates independently.
"""

import os

from PyQt5.QtCore import (
    Qt, QPoint, QPointF, QRectF, QSize, QTimer, pyqtSignal
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QWheelEvent, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QToolButton, QFileDialog,
    QStackedWidget, QFrame,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np

PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# All plots now shown inside the viewer (plot_common included)
GUI_EXCLUDED_PLOTS: set = set()

# Sequential load priority: lower number = loaded first
LOAD_PRIORITY = {
    "northing_error_histogram.png": 0,
    "easting_error_histogram.png":  1,
    "plot_common.png":              2,
}

# Human-readable names & subtitles for each plot in the loading widget
PLOT_DISPLAY_INFO = {
    "northing_error_histogram.png": ("Northing Error",  "Loading northing shift histogram…"),
    "easting_error_histogram.png":  ("Easting Error",   "Loading easting shift histogram…"),
    "plot_common.png":              ("Pixel Scatter",   "Loading density scatter plot…"),
}


# ---------------------------------------------------------------------------
# Loading overlay shown on the canvas while each image is being revealed
# ---------------------------------------------------------------------------

class LoadingWidget(QWidget):
    """Per-plot loading pane: shows the plot name + animated dot sequence."""

    _DOT_PATTERNS = ["●  ○  ○", "○  ●  ○", "○  ○  ●"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loadingWidget")

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setSpacing(14)

        self._title = QLabel("", self)
        self._title.setObjectName("loadingTitle")
        self._title.setAlignment(Qt.AlignCenter)

        self._subtitle = QLabel("", self)
        self._subtitle.setObjectName("loadingSubtitle")
        self._subtitle.setAlignment(Qt.AlignCenter)

        self._dots = QLabel(self._DOT_PATTERNS[0], self)
        self._dots.setObjectName("loadingDots")
        self._dots.setAlignment(Qt.AlignCenter)

        outer.addWidget(self._title)
        outer.addWidget(self._subtitle)
        outer.addWidget(self._dots)

        self._idx = 0
        self._timer = QTimer(self)
        self._timer.setInterval(380)
        self._timer.timeout.connect(self._cycle)

    def set_plot(self, filename: str):
        """Configure for a specific output file and start animation."""
        base = os.path.basename(filename).lower()
        title, subtitle = PLOT_DISPLAY_INFO.get(base, (os.path.basename(filename), "Loading…"))
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self._idx = 0
        self._dots.setText(self._DOT_PATTERNS[0])
        self._timer.start()

    def stop_animation(self):
        self._timer.stop()

    def _cycle(self):
        self._idx = (self._idx + 1) % len(self._DOT_PATTERNS)
        self._dots.setText(self._DOT_PATTERNS[self._idx])


# ---------------------------------------------------------------------------
# Zoomable canvas
# ---------------------------------------------------------------------------

class ZoomableCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.setCursor(Qt.OpenHandCursor)

        self._pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._drag_active = False
        self._drag_last = QPoint()

        self._zoom_target = 1.0
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setInterval(16)
        self._zoom_timer.timeout.connect(self._smooth_zoom_step)
        self._zoom_anchor = QPointF(0, 0)

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.fit_in_view()

    def fit_in_view(self):
        if not self._pixmap:
            return
        w, h = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw == 0 or ih == 0:
            return
        scale = min(w / iw, h / ih) * 0.92
        self._scale = scale
        self._zoom_target = scale
        self._offset = QPointF((w - iw * scale) / 2, (h - ih * scale) / 2)
        self.update()

    def zoom_in(self):
        self._start_zoom(self._scale * 1.25, QPointF(self.width() / 2, self.height() / 2))

    def zoom_out(self):
        self._start_zoom(self._scale / 1.25, QPointF(self.width() / 2, self.height() / 2))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_checkerboard(painter)
        if self._pixmap:
            iw = self._pixmap.width() * self._scale
            ih = self._pixmap.height() * self._scale
            dest = QRectF(self._offset.x(), self._offset.y(), iw, ih)
            painter.drawPixmap(dest, self._pixmap, QRectF(self._pixmap.rect()))
        painter.end()

    def _draw_checkerboard(self, painter):
        cell = 16
        c1 = QColor("#F0F0F0")
        c2 = QColor("#E0E0E0")
        cols = (self.width() // cell) + 2
        rows = (self.height() // cell) + 2
        for row in range(rows):
            for col in range(cols):
                color = c1 if (row + col) % 2 == 0 else c2
                painter.fillRect(col * cell, row * cell, cell, cell, color)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else (1 / 1.12)
        new_scale = max(0.02, min(self._scale * factor, 64.0))
        self._start_zoom(new_scale, QPointF(event.pos()))
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_active:
            delta = event.pos() - self._drag_last
            self._offset += QPointF(delta)
            self._drag_last = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.fit_in_view()

    def resizeEvent(self, event):
        if self._pixmap:
            self.fit_in_view()
        super().resizeEvent(event)

    def _start_zoom(self, target_scale, anchor: QPointF):
        self._zoom_anchor = anchor
        self._zoom_target = max(0.02, min(target_scale, 64.0))
        if not self._zoom_timer.isActive():
            self._zoom_timer.start()

    def _smooth_zoom_step(self):
        diff = self._zoom_target - self._scale
        if abs(diff) < 0.0005:
            self._scale = self._zoom_target
            self._zoom_timer.stop()
        else:
            self._scale += diff * 0.28
        anchor = self._zoom_anchor
        old_scale = self._scale - (self._zoom_target - self._scale) * 0.28
        ratio = self._scale / max(old_scale, 1e-9)
        self._offset = QPointF(
            anchor.x() - ratio * (anchor.x() - self._offset.x()),
            anchor.y() - ratio * (anchor.y() - self._offset.y()),
        )
        self.update()


# ---------------------------------------------------------------------------
# Interactive matplotlib canvas
# ---------------------------------------------------------------------------

class InteractiveFigureCanvas(FigureCanvasQTAgg):
    def __init__(self, fig, parent=None):
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setCursor(Qt.OpenHandCursor)

        self._home_limits = {}
        try:
            self._home_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in fig.get_axes()}
        except Exception:
            pass

        self._pan_ax = None
        self._pan_xpress = self._pan_ypress = None
        self._pan_xlim = self._pan_ylim = None

        try:
            for ax in fig.get_axes():
                ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
                ax.callbacks.connect("ylim_changed", self._on_ylim_changed)
                self._update_x_grid(ax)
                self._update_y_grid(ax)
        except Exception:
            pass

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def fit_in_view(self):
        try:
            for ax, (xlim, ylim) in self._home_limits.items():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.draw_idle()
        except Exception:
            pass

    def zoom_in(self):
        self._zoom_all(1.25)

    def zoom_out(self):
        self._zoom_all(1 / 1.25)

    def close_figure(self):
        try:
            plt.close(self.figure)
        except Exception:
            pass

    @staticmethod
    def _nice_step(span, target=7):
        if span <= 0:
            return 1.0
        raw = span / target
        exp = np.floor(np.log10(raw))
        base = 10 ** exp
        for mult in (1, 2, 2.5, 5, 10):
            if base * mult >= raw:
                return base * mult
        return base * 10

    @staticmethod
    def _decimals_for(step):
        if step >= 1:
            return 0
        return max(0, int(np.ceil(-np.log10(step))))

    def _update_x_grid(self, ax):
        try:
            import matplotlib.ticker as mticker
            xlo, xhi = ax.get_xlim()
            span = xhi - xlo
            step = self._nice_step(span)
            minor_step = step / 5
            start = np.ceil(xlo / step) * step
            majors = np.arange(start, xhi + step * 0.01, step)
            ax.set_xticks(majors)
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(minor_step))
            dec = self._decimals_for(step)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{dec}f"))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")
                lbl.set_fontsize(9)
        except Exception:
            pass

    def _update_y_grid(self, ax):
        try:
            import matplotlib.ticker as mticker
            ylo, yhi = ax.get_ylim()
            span = yhi - ylo
            step = self._nice_step(span)
            minor_step = step / 5
            start = np.ceil(ylo / step) * step
            majors = np.arange(start, yhi + step * 0.01, step)
            ax.set_yticks(majors)
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(minor_step))
            dec = self._decimals_for(step)
            if dec == 0:
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
                )
            else:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{dec}f"))
        except Exception:
            pass

    def _on_xlim_changed(self, ax):
        self._update_x_grid(ax)

    def _on_ylim_changed(self, ax):
        self._update_y_grid(ax)

    def _zoom_all(self, factor):
        try:
            for ax in self.figure.get_axes():
                self._zoom_axes(ax, factor, center=None)
            self.draw_idle()
        except Exception:
            pass

    def _zoom_axes(self, ax, factor, center):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if center is None or center[0] is None:
            cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        else:
            cx, cy = center
        new_w = (xlim[1] - xlim[0]) / factor
        new_h = (ylim[1] - ylim[0]) / factor
        left_frac = (cx - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
        bot_frac  = (cy - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.5
        ax.set_xlim(cx - new_w * left_frac, cx + new_w * (1 - left_frac))
        ax.set_ylim(cy - new_h * bot_frac,  cy + new_h * (1 - bot_frac))

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        factor = 1.15 if event.step > 0 else (1 / 1.15)
        self._zoom_axes(event.inaxes, factor, center=(event.xdata, event.ydata))
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes is None:
            return
        if event.dblclick:
            self.fit_in_view()
            return
        if event.button != 1:
            return
        self._pan_ax = event.inaxes
        self._pan_xpress, self._pan_ypress = event.xdata, event.ydata
        self._pan_xlim, self._pan_ylim = event.inaxes.get_xlim(), event.inaxes.get_ylim()
        self.setCursor(Qt.ClosedHandCursor)

    def _on_motion(self, event):
        if self._pan_ax is None or event.inaxes != self._pan_ax or event.xdata is None:
            return
        dx = event.xdata - self._pan_xpress
        dy = event.ydata - self._pan_ypress
        self._pan_ax.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
        self._pan_ax.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
        self.draw_idle()

    def _on_release(self, event):
        self._pan_ax = None
        self._pan_xpress = self._pan_ypress = None
        self._pan_xlim = self._pan_ylim = None
        self.setCursor(Qt.OpenHandCursor)


# ---------------------------------------------------------------------------
# Thumbnail button
# ---------------------------------------------------------------------------

class ThumbnailButton(QToolButton):
    def __init__(self, path, parent=None, loading=False):
        super().__init__(parent)
        self.path = path
        self.setFixedSize(72, 72)
        self.setCheckable(True)
        self.setObjectName("thumbnailButton")
        self.setToolTip(os.path.basename(path))
        self._loaded = False
        if loading:
            self._show_loading_state()
        else:
            self._load_pixmap()

    def _show_loading_state(self):
        """Show a compact placeholder until the image is revealed."""
        self.setIcon(QIcon())
        self.setIconSize(QSize(64, 64))
        # Short name as placeholder text
        name = os.path.splitext(os.path.basename(self.path))[0]
        short = name[:10] + "…" if len(name) > 10 else name
        self.setText(short)
        self.setProperty("thumbnailLoading", "true")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_loaded(self):
        """Transition from loading state to showing the actual image."""
        if self._loaded:
            return
        self._load_pixmap()
        self.setProperty("thumbnailLoading", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def _load_pixmap(self):
        self._loaded = True
        try:
            pix = QPixmap(self.path)
            if not pix.isNull():
                pix = pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setIcon(QIcon(pix))
                self.setIconSize(QSize(64, 64))
                self.setText("")
                return
        except Exception:
            pass
        # Fallback: text label
        name = os.path.splitext(os.path.basename(self.path))[0]
        self.setText(name[:10])


# ---------------------------------------------------------------------------
# Thumbnail strip
# ---------------------------------------------------------------------------

class ThumbnailStrip(QScrollArea):
    imageSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("thumbnailStrip")
        self.setFixedWidth(90)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 8, 6, 8)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._buttons: list = []
        self._paths: list = []

    # ---- original instant load (used for manual folder open) ----
    def set_images(self, paths):
        self._clear_buttons()
        self._paths = list(paths)
        for path in self._paths:
            btn = ThumbnailButton(path, self._container, loading=False)
            btn.clicked.connect(lambda checked, p=path: self._on_clicked(p))
            self._layout.insertWidget(self._layout.count() - 1, btn)
            self._buttons.append(btn)
        if self._buttons:
            self._buttons[0].setChecked(True)

    # ---- staged load: all thumbnails start in "loading" state ----
    def begin_loading(self, paths):
        self._clear_buttons()
        self._paths = list(paths)
        for path in self._paths:
            btn = ThumbnailButton(path, self._container, loading=True)
            btn.clicked.connect(lambda checked, p=path: self._on_clicked(p))
            self._layout.insertWidget(self._layout.count() - 1, btn)
            self._buttons.append(btn)

    def reveal_thumbnail(self, path):
        """Transition a specific thumbnail from loading to loaded state."""
        for btn in self._buttons:
            if btn.path == path:
                btn.set_loaded()
                self.select_path(path)
                break

    def select_path(self, path):
        for btn in self._buttons:
            btn.setChecked(btn.path == path)

    def _on_clicked(self, path):
        self.select_path(path)
        self.imageSelected.emit(path)

    def _clear_buttons(self):
        for btn in self._buttons:
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()


# ---------------------------------------------------------------------------
# Viewer toolbar
# ---------------------------------------------------------------------------

class ViewerToolbar(QWidget):
    fitClicked = pyqtSignal()
    zoomInClicked = pyqtSignal()
    zoomOutClicked = pyqtSignal()
    openFolderClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewerToolbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.fit_btn      = self._make_btn("Fit",      "Reset zoom to fit image (double-click canvas)", self.fitClicked)
        self.zoom_in_btn  = self._make_btn("＋",       "Zoom in",  self.zoomInClicked)
        self.zoom_out_btn = self._make_btn("－",       "Zoom out", self.zoomOutClicked)

        layout.addWidget(self.fit_btn)
        layout.addWidget(self.zoom_in_btn)
        layout.addWidget(self.zoom_out_btn)
        layout.addStretch(1)

        self.open_folder_btn = self._make_btn("Open folder…", "Browse for a folder containing plots", self.openFolderClicked)
        self.open_folder_btn.setObjectName("secondaryActionButton")
        layout.addWidget(self.open_folder_btn)

        self.filename_label = QLabel("", self)
        self.filename_label.setObjectName("viewerFilenameLabel")
        layout.addWidget(self.filename_label)

    def set_filename(self, name):
        self.filename_label.setText(name)

    @staticmethod
    def _make_btn(text, tooltip, signal):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setObjectName("viewerToolbarButton")
        btn.setFixedHeight(28)
        btn.clicked.connect(signal)
        return btn


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class EmptyState(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("▤", self)
        icon.setObjectName("emptyStateIcon")
        icon.setAlignment(Qt.AlignCenter)

        msg = QLabel("No plots yet.\nRun the pipeline or open a folder to view outputs.", self)
        msg.setObjectName("emptyStateMessage")
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)

        layout.addWidget(icon)
        layout.addSpacing(10)
        layout.addWidget(msg)


# ---------------------------------------------------------------------------
# Top-level viewer panel
# ---------------------------------------------------------------------------

class ImageViewerPanel(QWidget):
    """
    Full viewer panel with sequential loading:
    northing → easting → plot_common → others.
    Each image gets its own brief loading overlay before appearing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewerPanel")
        self._current_dir = None
        self._image_paths = []
        self._live_figures = {}
        self._figure_canvases = {}
        self._active_zoomable = None
        self._current_path = None

        # Sequential load queue
        self._reveal_queue: list = []

        self.setFocusPolicy(Qt.StrongFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setSpacing(0)
        body.setContentsMargins(0, 0, 0, 0)

        self.strip = ThumbnailStrip(self)
        self.strip.imageSelected.connect(self._load_image)
        body.addWidget(self.strip)

        right = QVBoxLayout()
        right.setSpacing(0)
        right.setContentsMargins(0, 0, 0, 0)

        self.toolbar = ViewerToolbar(self)
        self.toolbar.fitClicked.connect(self._on_fit)
        self.toolbar.zoomInClicked.connect(self._on_zoom_in)
        self.toolbar.zoomOutClicked.connect(self._on_zoom_out)
        self.toolbar.openFolderClicked.connect(self._on_open_folder)
        right.addWidget(self.toolbar)

        self._stack = QStackedWidget(self)
        self._empty   = EmptyState(self._stack)       # idx 0
        self._canvas  = ZoomableCanvas(self._stack)   # idx 1
        self._loading = LoadingWidget(self._stack)    # idx 2
        self._stack.addWidget(self._empty)
        self._stack.addWidget(self._canvas)
        self._stack.addWidget(self._loading)
        self._stack.setCurrentIndex(0)
        right.addWidget(self._stack, 1)

        body.addLayout(right, 1)
        outer.addLayout(body, 1)

    # -- Public API ---------------------------------------------------------

    def set_output_dir(self, path):
        self._current_dir = path
        self._refresh_images()

    def set_live_figures(self, figures):
        self._release_live_figures()
        self._live_figures = dict(figures or {})

    def clear(self):
        self._current_dir = None
        self._image_paths = []
        self._reveal_queue.clear()
        self.strip.set_images([])
        self._stack.setCurrentIndex(0)
        self.toolbar.set_filename("")
        self._active_zoomable = None
        self._current_path = None
        self._release_live_figures()

    # -- Internal -----------------------------------------------------------

    def _release_live_figures(self):
        for canvas in self._figure_canvases.values():
            try:
                self._stack.removeWidget(canvas)
                canvas.close_figure()
                canvas.deleteLater()
            except Exception:
                pass
        self._figure_canvases.clear()
        self._live_figures.clear()

    @staticmethod
    def _sort_by_priority(paths):
        def key(p):
            base = os.path.basename(p).lower()
            return LOAD_PRIORITY.get(base, 100), base
        return sorted(paths, key=key)

    def _refresh_images(self):
        if not self._current_dir or not os.path.isdir(self._current_dir):
            return
        paths = []
        try:
            for root, _, files in os.walk(self._current_dir):
                for f in sorted(files):
                    base = f.lower()
                    if (os.path.splitext(f)[1].lower() in PLOT_EXTENSIONS
                            and base not in GUI_EXCLUDED_PLOTS):
                        paths.append(os.path.join(root, f))
        except OSError:
            return

        self._image_paths = self._sort_by_priority(paths)
        self._reveal_queue = list(self._image_paths)

        if not self._image_paths:
            self._stack.setCurrentIndex(0)
            return

        # Set up strip in loading state
        self.strip.begin_loading(self._image_paths)

        # Kick off sequential reveal
        self._do_next_reveal()

    def _do_next_reveal(self):
        """Reveal one image from the queue: show loading state → then image."""
        if not self._reveal_queue:
            self._loading.stop_animation()
            return

        path = self._reveal_queue.pop(0)

        # 1. Show the per-plot loading widget on canvas
        self._loading.set_plot(path)
        self._stack.setCurrentWidget(self._loading)

        # 2. After a brief moment, load + show the actual image
        QTimer.singleShot(280, lambda p=path: self._reveal_image(p))

    def _reveal_image(self, path):
        """Actually load the image and transition strip thumbnail to loaded."""
        self._loading.stop_animation()
        try:
            self.strip.reveal_thumbnail(path)
            self._load_image(path)
        except Exception:
            pass

        # Schedule next reveal with a gap so each feels independent
        if self._reveal_queue:
            QTimer.singleShot(500, self._do_next_reveal)

    def _load_image(self, path):
        if not os.path.isfile(path):
            return
        try:
            fig = self._live_figures.get(os.path.basename(path))
            if fig is not None:
                self._show_figure(path, fig)
            else:
                self._show_pixmap(path)
        except Exception:
            pass

    def _show_pixmap(self, path):
        try:
            pix = QPixmap(path)
            if pix.isNull():
                return
            self._canvas.set_pixmap(pix)
            self._stack.setCurrentWidget(self._canvas)
            self._active_zoomable = self._canvas
            self._current_path = path
            self.strip.select_path(path)
            self.toolbar.set_filename(os.path.basename(path))
        except Exception:
            pass

    def _show_figure(self, path, fig):
        try:
            name = os.path.basename(path)
            canvas = self._figure_canvases.get(name)
            if canvas is None:
                canvas = InteractiveFigureCanvas(fig, self._stack)
                self._stack.addWidget(canvas)
                self._figure_canvases[name] = canvas
            self._stack.setCurrentWidget(canvas)
            self._active_zoomable = canvas
            self._current_path = path
            self.strip.select_path(path)
            self.toolbar.set_filename(name)
        except Exception:
            # Fallback to pixmap if live figure fails
            self._show_pixmap(path)

    def _on_fit(self):
        if self._active_zoomable is not None:
            try:
                self._active_zoomable.fit_in_view()
            except Exception:
                pass

    def _on_zoom_in(self):
        if self._active_zoomable is not None:
            try:
                self._active_zoomable.zoom_in()
            except Exception:
                pass

    def _on_zoom_out(self):
        if self._active_zoomable is not None:
            try:
                self._active_zoomable.zoom_out()
            except Exception:
                pass

    def _on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.set_output_dir(path)

    # -- Keyboard navigation -------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self._step_plot(1)
            event.accept()
        elif event.key() == Qt.Key_Left:
            self._step_plot(-1)
            event.accept()
        else:
            super().keyPressEvent(event)

    def _step_plot(self, direction):
        """Move to the next/previous loaded plot (direction = +1 or -1)."""
        if not self._image_paths:
            return
        if self._current_path in self._image_paths:
            idx = self._image_paths.index(self._current_path)
        else:
            idx = -1 if direction > 0 else 0
        new_idx = (idx + direction) % len(self._image_paths)
        self._load_image(self._image_paths[new_idx])
