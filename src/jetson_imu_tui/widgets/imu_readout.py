"""Per-IMU readout widget."""

from __future__ import annotations

from collections import deque

from imu_python.base_classes import IMUData
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Sparkline, Static

from jetson_imu_tui.ring_buffer import RAD_TO_DEG


SPARK_LEN = 60
SPARK_UPDATE_EVERY = 3  # at ui_refresh_hz=30 this is ~10 Hz


class IMUReadout(Static):
    DEFAULT_CSS = """
    IMUReadout {
        border: round $accent;
        padding: 0 1;
        height: 1fr;
    }
    IMUReadout .row { height: 1; }
    IMUReadout Sparkline { height: 1; margin: 0; }
    """

    def __init__(self, label: str, subtitle: str = "") -> None:
        super().__init__(id=f"readout-{label.lower()}")
        self._label = label
        self._subtitle = subtitle
        self.border_title = label
        self._spark_x: deque[float] = deque(maxlen=SPARK_LEN)
        self._spark_y: deque[float] = deque(maxlen=SPARK_LEN)
        self._spark_z: deque[float] = deque(maxlen=SPARK_LEN)
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._subtitle, id="subtitle", classes="row"),
            Static("Eul x --", id="euler-x", classes="row"),
            Sparkline([], id="spark-x"),
            Static("Eul y --", id="euler-y", classes="row"),
            Sparkline([], id="spark-y"),
            Static("Eul z --", id="euler-z", classes="row"),
            Sparkline([], id="spark-z"),
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
            self._spark_x.clear()
            self._spark_y.clear()
            self._spark_z.clear()
            self.query_one("#spark-x", Sparkline).data = []
            self.query_one("#spark-y", Sparkline).data = []
            self.query_one("#spark-z", Sparkline).data = []
            return
        e = data.quat.to_euler("ZYX")
        x = e.x * RAD_TO_DEG
        y = e.y * RAD_TO_DEG
        z = e.z * RAD_TO_DEG
        self.query_one("#euler-x", Static).update(f"Eul x {x:+8.2f}°")
        self.query_one("#euler-y", Static).update(f"Eul y {y:+8.2f}°")
        self.query_one("#euler-z", Static).update(f"Eul z {z:+8.2f}°")
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
        self._spark_x.append(x)
        self._spark_y.append(y)
        self._spark_z.append(z)
        self._frame += 1
        if self._frame % SPARK_UPDATE_EVERY == 0:
            self.query_one("#spark-x", Sparkline).data = list(self._spark_x)
            self.query_one("#spark-y", Sparkline).data = list(self._spark_y)
            self.query_one("#spark-z", Sparkline).data = list(self._spark_z)
