"""
gui.styles

Application-wide visual theme — Stark Light / Frameless Minimalism.
Improved in this version:
  - Gradient progress bar (7 px tall, no text on bar)
  - Step counter label and stage name label styles
  - Pulsing run-dot styles
  - Loading widget styles for image viewer
  - Thumbnail loading-state style
"""

FONT_FAMILY = "Segoe UI"
FONT_SIZE_PT = 10

FONT_FAMILY_STACK = ["Segoe UI", "Segoe UI Symbol", "Segoe UI Emoji", "Arial"]

CONSOLE_FONT_FAMILY = "Consolas"
CONSOLE_FONT_SIZE_PT = 10

# -- Palette (Stark Light) ---------------------------------------------------
COLOR_BACKGROUND    = "#FFFFFF"
COLOR_SURFACE       = "#FFFFFF"
COLOR_SURFACE_ALT   = "#F5F5F5"
COLOR_INPUT         = "#FFFFFF"
COLOR_BORDER        = "#E0E0E0"
COLOR_BORDER_HOVER  = "#BFBFBF"

COLOR_TEXT_PRIMARY   = "#222222"
COLOR_TEXT_SECONDARY = "#666666"
COLOR_TEXT_MUTED     = "#999999"

COLOR_ACCENT         = "#111111"
COLOR_ACCENT_HOVER   = "#333333"
COLOR_ACCENT_PRESSED = "#000000"

COLOR_SUCCESS = "#1E7A34"
COLOR_WARNING = "#946200"
COLOR_ERROR   = "#B3261E"

COLOR_CONSOLE_BACKGROUND = "#FAFAFA"
COLOR_CONSOLE_TEXT       = "#222222"

# -- Main stylesheet -------------------------------------------------------
WINDOW_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLOR_BACKGROUND};
}}

QWidget {{
    color: {COLOR_TEXT_PRIMARY};
    font-family: {", ".join(f'"{n}"' for n in FONT_FAMILY_STACK)};
    font-size: {FONT_SIZE_PT}pt;
}}

QLabel {{
    color: {COLOR_TEXT_PRIMARY};
    background: transparent;
}}

QLabel#appTitle {{
    color: #000000;
    font-size: 22pt;
    font-weight: 700;
}}

QLabel#appSubtitle {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 10pt;
}}

QLabel#sectionTitle {{
    color: #000000;
    font-size: 10.5pt;
    font-weight: 700;
    letter-spacing: 1px;
}}

QLabel#sectionHint {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 9pt;
}}

QLabel#paramFieldLabel {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
    font-weight: 600;
}}

/* ── Current-stage display ────────────────────────────────────────── */
QLabel#stageNameLabel {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: 9.5pt;
    font-weight: 600;
}}

QLabel#stepCounterLabel {{
    color: {COLOR_TEXT_MUTED};
    font-size: 8.5pt;
    font-weight: 600;
    padding-right: 2px;
}}

/* Pulsing run-dot */
QLabel#runDot {{
    font-size: 9pt;
    color: {COLOR_TEXT_MUTED};
    background: transparent;
}}

QLabel#runDot[dotState="active"] {{
    color: {COLOR_ACCENT};
}}

QLabel#runDot[dotState="dim"] {{
    color: {COLOR_BORDER_HOVER};
}}

QLabel#runDot[dotState="idle"] {{
    color: {COLOR_TEXT_MUTED};
}}

/* ── Parameters panel ─────────────────────────────────────────────── */
QToolButton#paramsToggleButton {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 3px 8px;
    background-color: transparent;
    font-size: 8.5pt;
}}

QToolButton#paramsResetButton {{
    border: none;
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
    text-decoration: underline;
    padding: 3px 6px;
    background-color: transparent;
}}

QToolButton#paramsResetButton:hover {{
    color: {COLOR_ACCENT};
}}

/* ── Status indicator (top-right badge) ───────────────────────────── */
QLabel#statusIndicator {{
    color: {COLOR_SUCCESS};
    font-size: 9pt;
    font-weight: 650;
    padding: 5px 10px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    background-color: transparent;
}}

QLabel#statusIndicator[statusState="ready"] {{
    color: {COLOR_SUCCESS};
    border: 1px solid {COLOR_BORDER};
    background-color: transparent;
}}

QLabel#statusIndicator[statusState="processing"] {{
    color: {COLOR_ACCENT};
    border: 1px solid #BFBFBF;
    background-color: transparent;
}}

QLabel#statusIndicator[statusState="failed"] {{
    color: {COLOR_ERROR};
    border: 1px solid rgba(179, 38, 30, 0.35);
    background-color: transparent;
}}

QLabel#statusIndicator[statusState="stopped"] {{
    color: {COLOR_TEXT_MUTED};
    border: 1px solid {COLOR_BORDER};
    background-color: transparent;
}}

/* ── Start / Stop button ──────────────────────────────────────────── */
QPushButton#startButton[mode="stop"] {{
    background-color: {COLOR_ERROR};
}}

QPushButton#startButton[mode="stop"]:hover {{
    background-color: #942019;
}}

QPushButton#startButton[mode="stop"]:pressed {{
    background-color: #7A1A15;
}}

/* ── Section frames ───────────────────────────────────────────────── */
QFrame#headerFrame {{
    background-color: transparent;
    border: none;
}}

QFrame#sectionFrame {{
    background-color: {COLOR_SURFACE};
    border: none;
    border-top: 1px solid {COLOR_BORDER};
    border-radius: 0px;
}}

/* ── Line edits ───────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {COLOR_INPUT};
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 9px 2px;
    min-height: 18px;
    color: {COLOR_TEXT_PRIMARY};
    selection-background-color: {COLOR_ACCENT};
    selection-color: #FFFFFF;
}}

QLineEdit:hover {{
    border-bottom: 1px solid {COLOR_BORDER_HOVER};
}}

QLineEdit:focus {{
    border-bottom: 2px solid #000000;
}}

QLineEdit:disabled {{
    color: {COLOR_TEXT_MUTED};
    background-color: {COLOR_SURFACE_ALT};
}}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {COLOR_ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 0px;
    padding: 9px 18px;
    min-height: 18px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {COLOR_ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {COLOR_ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: {COLOR_SURFACE_ALT};
    color: {COLOR_TEXT_MUTED};
}}

QPushButton#browseButton {{
    background-color: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_PRIMARY};
}}

QPushButton#browseButton:hover {{
    background-color: {COLOR_SURFACE_ALT};
    border-color: {COLOR_BORDER_HOVER};
}}

QPushButton#startButton {{
    min-height: 24px;
    padding: 11px 28px;
    font-size: 10pt;
}}

/* ── Raster cards ─────────────────────────────────────────────────── */
QFrame#rasterCard {{
    background-color: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
}}

QFrame#rasterCard:hover {{
    border-color: {COLOR_BORDER_HOVER};
}}

QFrame#rasterCard[dragActive="true"] {{
    border: 1px solid #000000;
    background-color: {COLOR_SURFACE_ALT};
}}

QLabel#rasterCardIcon {{
    color: #000000;
    background-color: transparent;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    font-size: 14pt;
    font-weight: 700;
}}

QLabel#rasterCardTitle {{
    color: #000000;
    font-size: 10pt;
    font-weight: 700;
}}

QLabel#rasterCardDescription {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
}}

QLabel#rasterCardState {{
    color: {COLOR_TEXT_MUTED};
    font-size: 8pt;
    font-weight: 650;
    padding: 4px 7px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    background-color: transparent;
}}

QLabel#rasterCardState[state="ready"] {{
    color: {COLOR_SUCCESS};
    border-color: rgba(30, 122, 52, 0.35);
}}

QLabel#rasterCardState[state="invalid"] {{
    color: {COLOR_WARNING};
    border-color: rgba(148, 98, 0, 0.40);
}}

QLabel#rasterFileInfo {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
    padding-left: 2px;
}}

QLineEdit#rasterPathEdit {{
    min-height: 20px;
}}

/* ── Console ──────────────────────────────────────────────────────── */
QTextEdit {{
    background-color: {COLOR_CONSOLE_BACKGROUND};
    color: {COLOR_CONSOLE_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 9px;
    selection-background-color: {COLOR_ACCENT};
    selection-color: #FFFFFF;
}}

/* ── Status bar ───────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_SECONDARY};
    border-top: 1px solid {COLOR_BORDER};
    padding: 2px 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* ── Scroll areas ─────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: {COLOR_BACKGROUND};
}}

QScrollBar:vertical {{
    background: {COLOR_BACKGROUND};
    width: 8px;
    margin: 2px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    min-height: 28px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLOR_BORDER_HOVER};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollArea#pageScrollArea, QScrollArea#pageScrollArea > QWidget, QWidget#pageContent {{
    background-color: {COLOR_BACKGROUND};
    border: none;
}}

/* ── Console toolbar button ───────────────────────────────────────── */
QToolButton#clearConsoleButton {{
    background-color: transparent;
    color: {COLOR_TEXT_SECONDARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 3px 10px;
    font-size: 8pt;
    font-weight: 650;
}}

QToolButton#clearConsoleButton:hover {{
    background-color: {COLOR_SURFACE_ALT};
    border-color: {COLOR_BORDER_HOVER};
    color: {COLOR_TEXT_PRIMARY};
}}

QToolButton#clearConsoleButton:pressed {{
    background-color: {COLOR_BORDER};
}}

/* ── Run output panel ─────────────────────────────────────────────── */
QLabel#outputStatusLabel {{
    color: {COLOR_SUCCESS};
    font-size: 9pt;
    font-weight: 650;
    padding: 4px 10px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    background-color: transparent;
}}

QLabel#outputDirLabel {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
    font-family: "{CONSOLE_FONT_FAMILY}";
    background-color: {COLOR_SURFACE_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 7px 10px;
}}

QLabel#outputCountLabel {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: 8.5pt;
    font-weight: 600;
    padding: 4px 9px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    background-color: transparent;
}}

QPushButton#secondaryActionButton {{
    background-color: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_PRIMARY};
    font-weight: 600;
}}

QPushButton#secondaryActionButton:hover {{
    background-color: {COLOR_SURFACE_ALT};
    border-color: {COLOR_BORDER_HOVER};
}}

QPushButton#secondaryActionButton:pressed {{
    background-color: {COLOR_BORDER};
}}

/* ── Tooltips ─────────────────────────────────────────────────────── */
QToolTip {{
    background-color: #FFFFFF;
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_HOVER};
    border-radius: 0px;
    padding: 5px 8px;
    font-size: 8.5pt;
}}

QPushButton#browseButton,
QPushButton#startButton,
QPushButton#secondaryActionButton {{
    text-align: center;
}}

/* ── Loading widget (image viewer) ────────────────────────────────── */
QWidget#loadingWidget {{
    background-color: {COLOR_BACKGROUND};
}}

QLabel#loadingTitle {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: 13pt;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#loadingSubtitle {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 9pt;
}}

QLabel#loadingDots {{
    color: {COLOR_TEXT_MUTED};
    font-size: 11pt;
    letter-spacing: 4px;
}}
"""

BUTTON_STYLESHEET = ""

DIALOG_STYLESHEET = f"""
QMessageBox {{
    background-color: {COLOR_SURFACE};
}}
QMessageBox QLabel {{
    color: {COLOR_TEXT_PRIMARY};
    background: transparent;
}}
QMessageBox QPushButton {{
    min-width: 72px;
}}
"""

CONSOLE_STYLESHEET = f"""
QTextEdit {{
    background-color: {COLOR_CONSOLE_BACKGROUND};
    color: {COLOR_CONSOLE_TEXT};
    font-family: "{CONSOLE_FONT_FAMILY}";
    font-size: {CONSOLE_FONT_SIZE_PT}pt;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 9px;
}}
"""

SIDEBAR_STYLESHEET = f"""
QWidget#sidebar {{
    background-color: {COLOR_SURFACE_ALT};
    border: none;
}}

QFrame#sidebarSeparator {{
    color: {COLOR_BORDER};
    background-color: {COLOR_BORDER};
    border: none;
}}

QWidget#rootWidget {{
    background-color: {COLOR_BACKGROUND};
}}

QToolButton#navButton {{
    background-color: transparent;
    color: {COLOR_TEXT_MUTED};
    border: none;
    border-radius: 0px;
    font-size: 7.5pt;
    font-weight: 600;
    padding: 2px 0px;
    text-align: center;
}}

QToolButton#navButton:hover {{
    background-color: {COLOR_BORDER};
    color: {COLOR_TEXT_PRIMARY};
}}

QToolButton#navButton:checked {{
    background-color: transparent;
    color: #000000;
    border-left: 2px solid #000000;
    font-weight: 700;
}}

QToolButton#navButton:disabled {{
    background-color: transparent;
    color: #CCCCCC;
    border: none;
    font-weight: 600;
}}
"""

VIEWER_STYLESHEET = f"""
QWidget#imageViewerPanel {{
    background-color: {COLOR_BACKGROUND};
}}

QScrollArea#thumbnailStrip {{
    background-color: {COLOR_SURFACE_ALT};
    border: none;
    border-right: 1px solid {COLOR_BORDER};
}}

QScrollArea#thumbnailStrip > QWidget {{
    background-color: {COLOR_SURFACE_ALT};
}}

QToolButton#thumbnailButton {{
    background-color: transparent;
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 2px;
    color: {COLOR_TEXT_MUTED};
    font-size: 7pt;
}}

QToolButton#thumbnailButton:hover {{
    border-color: {COLOR_BORDER_HOVER};
    background-color: {COLOR_BORDER};
}}

QToolButton#thumbnailButton:checked {{
    border: 2px solid #000000;
    background-color: {COLOR_SURFACE_ALT};
}}

/* Loading-state thumbnail: hatched background, muted text */
QToolButton#thumbnailButton[thumbnailLoading="true"] {{
    background-color: {COLOR_SURFACE_ALT};
    color: {COLOR_TEXT_MUTED};
    border: 1px dashed {COLOR_BORDER};
    font-size: 7pt;
    font-weight: 600;
}}

QWidget#viewerToolbar {{
    background-color: {COLOR_SURFACE};
    border-bottom: 1px solid {COLOR_BORDER};
}}

QPushButton#viewerToolbarButton {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 0px;
    padding: 3px 12px;
    font-size: 9pt;
    font-weight: 600;
    min-width: 32px;
}}

QPushButton#viewerToolbarButton:hover {{
    background-color: {COLOR_SURFACE_ALT};
    border-color: {COLOR_BORDER_HOVER};
}}

QPushButton#viewerToolbarButton:pressed {{
    background-color: {COLOR_BORDER};
}}

QLabel#viewerFilenameLabel {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 8.5pt;
    font-family: "{CONSOLE_FONT_FAMILY}";
    padding-right: 4px;
}}

QLabel#emptyStateIcon {{
    font-size: 40pt;
    color: {COLOR_BORDER_HOVER};
}}

QLabel#emptyStateMessage {{
    color: {COLOR_TEXT_MUTED};
    font-size: 10pt;
    max-width: 300px;
}}
"""


def full_application_stylesheet():
    return WINDOW_STYLESHEET + DIALOG_STYLESHEET + SIDEBAR_STYLESHEET + VIEWER_STYLESHEET


def apply_stark_light(app):
    app.setStyleSheet(full_application_stylesheet())
