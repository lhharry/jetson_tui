"""Main screen: status, two readouts, console, key bar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header

from jetson_imu_tui.widgets.console_log import ConsoleLog
from jetson_imu_tui.widgets.imu_readout import IMUReadout
from jetson_imu_tui.widgets.status_panel import StatusPanel


class MainScreen(Screen):
    DEFAULT_CSS = """
    MainScreen #readouts { height: 1fr; }
    MainScreen IMUReadout { width: 1fr; }
    """

    BINDINGS = [
        ("c", "app.connect", "Connect"),
        ("s", "app.toggle_stream", "Stream"),
        ("r", "app.toggle_record", "Record"),
        ("f", "app.set_frequency", "Freq"),
        ("l", "app.set_log_dir", "Log dir"),
        ("p", "app.plot", "Plot"),
        ("q", "app.quit_app", "Quit"),
    ]

    def __init__(self, labels: list[str]) -> None:
        super().__init__()
        self._labels = labels

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusPanel()
        readouts = [IMUReadout(label, subtitle=f"{label} thigh") for label in self._labels]
        yield Horizontal(*readouts, id="readouts")
        yield Vertical(ConsoleLog(), id="console-wrap")
        yield Footer()

    @property
    def status(self) -> StatusPanel:
        return self.query_one(StatusPanel)

    @property
    def console(self) -> ConsoleLog:
        return self.query_one(ConsoleLog)

    def readout(self, label: str) -> IMUReadout | None:
        try:
            return self.query_one(f"#readout-{label.lower()}", IMUReadout)
        except Exception:
            return None
