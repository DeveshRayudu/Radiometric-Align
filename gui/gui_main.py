"""
gui.gui_main

GUI application entry point.
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory

from gui.main_window import MainWindow
from gui.styles import apply_stark_light


def main():
    # Explicitly ensure native dialogs are used (this is Qt's default,
    # but stated here so the Browse buttons' QFileDialog.getOpenFileName
    # calls -- which no longer pass DontUseNativeDialog -- reliably get
    # the OS's own Explorer-style picker rather than a Qt-drawn one).
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, False)

    app = QApplication(sys.argv)
    app.setApplicationName("Radiometric Alignment")
    app.setApplicationDisplayName("Radiometric Alignment")

    # Fusion still styles the app's own widgets (buttons, line edits,
    # frames, scroll bars, etc.) consistently across platforms; it has
    # no effect on native dialogs, which the OS draws itself.
    app.setStyle(QStyleFactory.create("Fusion"))
    apply_stark_light(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
