"""
gui.console

Embedded terminal/console window.

Responsibilities (and only these):
  - Display all runtime messages (one line at a time, as they arrive).
  - Automatically scroll to the latest message.
  - Prevent editing.

This is a display-only widget. It doesn't know where messages come from
(worker threads, the controller, etc.) or what they mean -- callers just
call append_message(line); everything else (deciding what to show,
wiring it up to PipelineWorker.console_line, etc.) is main_window.py's
and controller.py's job.
"""

from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor


class ConsoleWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Prevent editing -- this widget only ever displays messages, it
        # never accepts input.
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.setUndoRedoEnabled(False)

        font = self.font()
        font.setFamily("Courier New")
        self.setFont(font)

    def append_message(self, message):
        """Append one runtime message and scroll to show it."""
        self.append(message)
        self._scroll_to_bottom()

    def clear_console(self):
        """Clear all displayed messages (e.g. when a new run starts)."""
        self.clear()

    def _scroll_to_bottom(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
