"""Live plot screen using a Unicode-Braille canvas (low CPU)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from jetson_imu_tui.ring_buffer import RingBuffers, SIGNAL_AXES
from jetson_imu_tui.widgets.braille_canvas import BrailleCanvas

LABEL_COLORS = {"Left": "magenta", "Right": "cyan"}
DEFAULT_COLORS = ("magenta", "cyan", "yellow", "green")

SIGNAL_TITLES = {
    "euler": "Euler (deg, ZYX)",
    "accel": "Accel (m/s^2)",
    "gyro": "Gyro (rad/s)",
    "quat": "Quaternion",
}


def _smooth(seq: list[float], win: int) -> list[float]:
    """Centered moving average (O(n)). ``win <= 1`` returns the input unchanged."""
    n = len(seq)
    if win <= 1 or n == 0:
        return seq
    half = win // 2
    csum = [0.0]
    for v in seq:
        csum.append(csum[-1] + v)
    out: list[float] = []
    for i in range(n):
        lo = i - half if i - half > 0 else 0
        hi = i + half + 1 if i + half + 1 < n else n
        out.append((csum[hi] - csum[lo]) / (hi - lo))
    return out


class PlotScreen(Screen):
    DEFAULT_CSS = """
    PlotScreen #plot-status { height: 1; padding: 0 1; background: $boost; }
    PlotScreen #plot-area { height: 1fr; }
    """

    BINDINGS = [
        ("1", "set_signal('euler')", "Euler"),
        ("2", "set_signal('accel')", "Accel"),
        ("3", "set_signal('gyro')", "Gyro"),
        ("4", "set_signal('quat')", "Quat"),
        ("space", "toggle_pause", "Pause"),
        ("q", "app.pop_screen", "Back"),
    ]

    def __init__(
        self,
        buffers: RingBuffers,
        fps: int = 15,
        window_s: float = 10.0,
        smoothing: int = 1,
    ) -> None:
        super().__init__()
        self._buffers = buffers
        self._signal = "euler"
        self._paused = False
        self._fps = max(1, int(fps))
        self._window_s = float(window_s)
        self._smoothing = int(smoothing)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(self._status_text(), id="plot-status")
        yield Vertical(id="plot-area")
        yield Footer()

    async def on_mount(self) -> None:
        await self._rebuild_canvases()
        self.set_interval(1.0 / self._fps, self._redraw)
        self._redraw()

    async def action_set_signal(self, signal: str) -> None:
        if signal not in SIGNAL_AXES or signal == self._signal:
            return
        self._signal = signal
        await self._rebuild_canvases()
        self._refresh_status()
        self._redraw()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        self._refresh_status()

    def _status_text(self) -> str:
        suffix = " [PAUSED]" if self._paused else ""
        return (
            f"Signal: {self._signal}{suffix}   window {self._window_s:g}s @ {self._fps} fps   "
            "(1=Eul 2=Acc 3=Gyr 4=Quat  space=pause  q=back)"
        )

    def _refresh_status(self) -> None:
        self.query_one("#plot-status", Static).update(self._status_text())

    async def _rebuild_canvases(self) -> None:
        area = self.query_one("#plot-area", Vertical)
        await area.remove_children()
        if self._signal == "quat":
            await area.mount(BrailleCanvas(id="bc-quat"))
        else:
            canvases = [BrailleCanvas(id=f"bc-{axis}") for axis in SIGNAL_AXES[self._signal]]
            await area.mount(*canvases)

    def _now(self, labels: list[str]) -> float:
        t_now = 0.0
        for label in labels:
            tl = self._buffers.time.get(label)
            if tl:
                last = tl[-1]
                if last > t_now:
                    t_now = last
        return t_now

    def _series_xy(self, label: str, axis: str, t_now: float):
        t_full = self._buffers.time.get(label)
        y_full = self._buffers.data.get((label, self._signal, axis))
        if not t_full or not y_full or len(t_full) != len(y_full):
            return None
        lo = -self._window_s
        xs: list[float] = []
        ys: list[float] = []
        for t, y in zip(t_full, y_full):
            dx = t - t_now
            if dx >= lo:
                xs.append(dx)
                ys.append(y)
        if not xs:
            return None
        if self._smoothing > 1:
            ys = _smooth(ys, self._smoothing)
        return xs, ys

    def _redraw(self) -> None:
        if self._paused:
            return
        axes = SIGNAL_AXES[self._signal]
        labels = self._buffers.labels
        t_now = self._now(labels)
        if t_now <= 0.0:
            return
        if self._signal == "quat":
            self._redraw_overlay(labels, axes, t_now)
        else:
            self._redraw_per_axis(labels, axes, t_now)

    def _redraw_overlay(self, labels: list[str], axes: tuple[str, ...], t_now: float) -> None:
        try:
            canvas = self.query_one("#bc-quat", BrailleCanvas)
        except Exception:
            return
        series: list[tuple[str, str, list[float], list[float]]] = []
        ylo = yhi = None
        for li, label in enumerate(labels):
            for ai, axis in enumerate(axes):
                xy = self._series_xy(label, axis, t_now)
                if xy is None:
                    continue
                xs, ys = xy
                color = DEFAULT_COLORS[(li * len(axes) + ai) % len(DEFAULT_COLORS)]
                series.append((f"{label}_{axis}", color, xs, ys))
                lo, hi = min(ys), max(ys)
                ylo = lo if ylo is None else min(ylo, lo)
                yhi = hi if yhi is None else max(yhi, hi)
        if not series:
            return
        canvas.set_plot(series, (-self._window_s, 0.0), (ylo, yhi), title=SIGNAL_TITLES["quat"])

    def _redraw_per_axis(self, labels: list[str], axes: tuple[str, ...], t_now: float) -> None:
        for axis in axes:
            try:
                canvas = self.query_one(f"#bc-{axis}", BrailleCanvas)
            except Exception:
                continue
            series: list[tuple[str, str, list[float], list[float]]] = []
            ylo = yhi = None
            for label in labels:
                xy = self._series_xy(label, axis, t_now)
                if xy is None:
                    continue
                xs, ys = xy
                color = LABEL_COLORS.get(label, DEFAULT_COLORS[0])
                series.append((label, color, xs, ys))
                lo, hi = min(ys), max(ys)
                ylo = lo if ylo is None else min(ylo, lo)
                yhi = hi if yhi is None else max(yhi, hi)
            if not series:
                continue
            title = f"{SIGNAL_TITLES[self._signal]} — {axis}"
            canvas.set_plot(series, (-self._window_s, 0.0), (ylo, yhi), title=title)
