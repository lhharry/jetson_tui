"""Console log widget with a loguru sink."""

from __future__ import annotations

from datetime import datetime

from textual.widgets import RichLog


class ConsoleLog(RichLog):
    DEFAULT_CSS = """
    ConsoleLog {
        border: round $secondary;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="console", markup=False, wrap=True, highlight=False)
        self.border_title = "Console"

    def write_line(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.write(f"{ts} {message}")

    def loguru_sink(self, message) -> None:
        # `message` is loguru's Message object; calling str() yields the formatted line.
        text = str(message).rstrip()
        if not text:
            return
        self.write(text)
