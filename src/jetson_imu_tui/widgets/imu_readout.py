"""Per-IMU readout widget."""

from __future__ import annotations

from imu_python.base_classes import IMUData
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jetson_imu_tui.ring_buffer import RAD_TO_DEG


class IMUReadout(Static):
    DEFAULT_CSS = """
    IMUReadout {
        border: round $accent;
        padding: 0 1;
        height: 1fr;
    }
    IMUReadout .row { height: 1; }
    """

    def __init__(self, label: str, subtitle: str = "") -> None:
        super().__init__(id=f"readout-{label.lower()}")
        self._label = label
        self._subtitle = subtitle
        self.border_title = label

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._subtitle, id="subtitle", classes="row"),
            Static("Eul x --", id="euler-x", classes="row"),
            Static("Eul y --", id="euler-y", classes="row"),
            Static("Eul z --", id="euler-z", classes="row"),
            Static("Accel --", id="accel", classes="row"),
            Static("Gyro  --", id="gyro", classes="row"),
            Static("Quat  --", id="quat", classes="row"),
        )

    def set_subtitle(self, text: str) -> None:
        self._subtitle = text
        self.query_one("#subtitle", Static).update(text)

    def update_from(self, data: IMUData | None) -> None:
        if data is None:
            self.query_one("#euler-x", Static).update("Eul x --")
            self.query_one("#euler-y", Static).update("Eul y --")
            self.query_one("#euler-z", Static).update("Eul z --")
            self.query_one("#accel", Static).update("Accel --")
            self.query_one("#gyro", Static).update("Gyro  --")
            self.query_one("#quat", Static).update("Quat  --")
            return
        e = data.quat.to_euler("ZYX")
        self.query_one("#euler-x", Static).update(f"Eul x {e.x * RAD_TO_DEG:+8.2f}°")
        self.query_one("#euler-y", Static).update(f"Eul y {e.y * RAD_TO_DEG:+8.2f}°")
        self.query_one("#euler-z", Static).update(f"Eul z {e.z * RAD_TO_DEG:+8.2f}°")
        a = data.device_data.accel
        self.query_one("#accel", Static).update(
            f"Accel x {a.x:+7.3f}  y {a.y:+7.3f}  z {a.z:+7.3f}"
        )
        g = data.device_data.gyro
        self.query_one("#gyro", Static).update(
            f"Gyro  x {g.x:+7.3f}  y {g.y:+7.3f}  z {g.z:+7.3f}"
        )
        q = data.quat
        self.query_one("#quat", Static).update(
            f"Quat  w {q.w:+6.3f}  x {q.x:+6.3f}  y {q.y:+6.3f}  z {q.z:+6.3f}"
        )
