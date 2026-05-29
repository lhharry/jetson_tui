"""Top status panel."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static


class StatusPanel(Static):
    DEFAULT_CSS = """
    StatusPanel {
        height: 4;
        padding: 0 1;
        background: $boost;
        color: $text;
        border: round $primary;
    }
    StatusPanel Horizontal { height: 1; }
    StatusPanel .field { padding-right: 2; }
    """

    def __init__(self) -> None:
        super().__init__(id="status-panel")
        self.border_title = "Status"
        self._imus: dict[str, str] = {}
        self._streaming = False
        self._recording = False
        self._record_hz = 100
        self._log_dir = Path("./logs")

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("Status:", classes="field"),
            Static("(disconnected)", id="imu-states", classes="field"),
            Static("Streaming: OFF", id="streaming-state", classes="field"),
            Static("Rec: OFF", id="recording-state", classes="field"),
            Static("Data: -- Hz", id="data-hz", classes="field"),
        )
        yield Horizontal(
            Static("Freq: 100 Hz", id="freq-state", classes="field"),
            Static("Logs: ./logs", id="log-state", classes="field"),
        )

    def set_imus(self, mapping: dict[str, str]) -> None:
        self._imus = mapping
        self._refresh_imus()

    def set_streaming(self, on: bool) -> None:
        self._streaming = on
        self.query_one("#streaming-state", Static).update(f"Streaming: {'ON' if on else 'OFF'}")

    def set_recording(self, on: bool) -> None:
        self._recording = on
        self.query_one("#recording-state", Static).update(f"Rec: {'ON' if on else 'OFF'}")

    def set_record_hz(self, hz: int) -> None:
        self._record_hz = hz
        self.query_one("#freq-state", Static).update(f"Freq: {hz} Hz")

    def set_log_dir(self, path: Path) -> None:
        self._log_dir = path
        self.query_one("#log-state", Static).update(f"Logs: {path}")

    def set_data_hz(self, hz: int) -> None:
        text = "Data: -- Hz" if hz <= 0 else f"Data: {hz} Hz"
        self.query_one("#data-hz", Static).update(text)

    def _refresh_imus(self) -> None:
        if not self._imus:
            text = "(disconnected)"
        else:
            text = "  ".join(f"{label}: {state}" for label, state in self._imus.items())
        self.query_one("#imu-states", Static).update(text)
