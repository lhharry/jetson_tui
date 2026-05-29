"""Per-IMU readout widget."""

from __future__ import annotations

from imu_python.base_classes import IMUData
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jetson_imu_tui.ring_buffer import RAD_TO_DEG

# Axis colors shared with the plot (x/y/z = red/green/blue).
_AX = ("red", "green", "blue")


class IMUReadout(Static):
    DEFAULT_CSS = """
    IMUReadout {
        border: round $accent;
        padding: 0 1;
        height: 1fr;
    }
    IMUReadout .row { height: 1; }
    IMUReadout #subtitle { color: $text-muted; }
    """

    def __init__(self, label: str, subtitle: str = "") -> None:
        super().__init__(id=f"readout-{label.lower()}")
        self._label = label
        self._subtitle = subtitle
        self.border_title = label
        # Build child widgets up front and keep references, so updates never need
        # a query_one() DOM lookup on the hot path.
        self._subtitle_w = Static(subtitle, id="subtitle", classes="row")
        self._euler_x = Static("Eul x   --", classes="row")
        self._euler_y = Static("Eul y   --", classes="row")
        self._euler_z = Static("Eul z   --", classes="row")
        self._accel = Static("Accel --", classes="row")
        self._gyro = Static("Gyro  --", classes="row")
        self._quat = Static("Quat  --", classes="row")
        self._last: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(
            self._subtitle_w,
            self._euler_x,
            self._euler_y,
            self._euler_z,
            self._accel,
            self._gyro,
            self._quat,
        )

    @staticmethod
    def _set(widget: Static, text: str, cache: dict[int, str]) -> None:
        key = id(widget)
        if cache.get(key) == text:
            return
        cache[key] = text
        widget.update(text)

    def set_subtitle(self, text: str) -> None:
        self._subtitle = text
        self._subtitle_w.update(text)

    def update_from(self, data: IMUData | None, euler=None) -> None:
        cache = self._last
        if data is None:
            self._set(self._euler_x, "Eul x   --", cache)
            self._set(self._euler_y, "Eul y   --", cache)
            self._set(self._euler_z, "Eul z   --", cache)
            self._set(self._accel, "Accel --", cache)
            self._set(self._gyro, "Gyro  --", cache)
            self._set(self._quat, "Quat  --", cache)
            return
        if euler is None:
            euler = data.quat.to_euler("ZYX")
        x = euler.x * RAD_TO_DEG
        y = euler.y * RAD_TO_DEG
        z = euler.z * RAD_TO_DEG
        self._set(self._euler_x, f"Eul [{_AX[0]}]x[/] {x:+8.2f}°", cache)
        self._set(self._euler_y, f"Eul [{_AX[1]}]y[/] {y:+8.2f}°", cache)
        self._set(self._euler_z, f"Eul [{_AX[2]}]z[/] {z:+8.2f}°", cache)
        a = data.device_data.accel
        self._set(
            self._accel,
            f"Accel [{_AX[0]}]x{a.x:+7.3f}[/] [{_AX[1]}]y{a.y:+7.3f}[/] [{_AX[2]}]z{a.z:+7.3f}[/]",
            cache,
        )
        g = data.device_data.gyro
        self._set(
            self._gyro,
            f"Gyro  [{_AX[0]}]x{g.x:+7.3f}[/] [{_AX[1]}]y{g.y:+7.3f}[/] [{_AX[2]}]z{g.z:+7.3f}[/]",
            cache,
        )
        q = data.quat
        self._set(
            self._quat,
            f"Quat  w{q.w:+6.3f} [{_AX[0]}]x{q.x:+6.3f}[/] "
            f"[{_AX[1]}]y{q.y:+6.3f}[/] [{_AX[2]}]z{q.z:+6.3f}[/]",
            cache,
        )
