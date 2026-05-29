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

PLOT_REDRAW_INTERVAL = 0.1
PLOT_MAX_POINTS = 150


def _downsample(seq: list[float], cap: int) -> list[float]:
    n = len(seq)
    if n <= cap:
        return seq
    step = n // cap
    return seq[::step][-cap:]


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

    def __init__(self, buffers: RingBuffers) -> None:
        super().__init__()
        self._buffers = buffers
        self._signal = "euler"
        self._paused = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(self._status_text(), id="plot-status")
        yield Vertical(id="plot-area")
        yield Footer()

    async def on_mount(self) -> None:
        await self._rebuild_canvases()
        self.set_interval(PLOT_REDRAW_INTERVAL, self._redraw)
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
            f"Signal: {self._signal}{suffix}   "
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

    def _redraw(self) -> None:
        if self._paused:
            return
        axes = SIGNAL_AXES[self._signal]
        labels = self._buffers.labels
        if self._signal == "quat":
            self._redraw_overlay(labels, axes)
        else:
            self._redraw_per_axis(labels, axes)

    def _redraw_overlay(self, labels: list[str], axes: tuple[str, ...]) -> None:
        try:
            canvas = self.query_one("#bc-quat", BrailleCanvas)
        except Exception:
            return
        series: list[tuple[str, str, list[float], list[float]]] = []
        all_t: list[float] = []
        all_y: list[float] = []
        for li, label in enumerate(labels):
            t_full = list(self._buffers.time[label])
            if not t_full:
                continue
            t0 = t_full[0]
            ts = _downsample([x - t0 for x in t_full], PLOT_MAX_POINTS)
            for ai, axis in enumerate(axes):
                y_full = list(self._buffers.data[(label, "quat", axis)])
                if len(y_full) != len(t_full):
                    continue
                y = _downsample(y_full, PLOT_MAX_POINTS)
                color = DEFAULT_COLORS[(li * len(axes) + ai) % len(DEFAULT_COLORS)]
                series.append((f"{label}_{axis}", color, ts, y))
                all_t.extend(ts)
                all_y.extend(y)
        if not series:
            return
        x_range = (min(all_t), max(all_t))
        y_range = _pad_range(min(all_y), max(all_y))
        canvas.set_plot(series, x_range, y_range, title=SIGNAL_TITLES["quat"])

    def _redraw_per_axis(self, labels: list[str], axes: tuple[str, ...]) -> None:
        for axis in axes:
            try:
                canvas = self.query_one(f"#bc-{axis}", BrailleCanvas)
            except Exception:
                continue
            series: list[tuple[str, str, list[float], list[float]]] = []
            all_t: list[float] = []
            all_y: list[float] = []
            for label in labels:
                t_full = list(self._buffers.time[label])
                y_full = list(self._buffers.data[(label, self._signal, axis)])
                if not t_full or len(t_full) != len(y_full):
                    continue
                t0 = t_full[0]
                ts = _downsample([x - t0 for x in t_full], PLOT_MAX_POINTS)
                y = _downsample(y_full, PLOT_MAX_POINTS)
                color = LABEL_COLORS.get(label, DEFAULT_COLORS[0])
                series.append((label, color, ts, y))
                all_t.extend(ts)
                all_y.extend(y)
            if not series:
                continue
            x_range = (min(all_t), max(all_t))
            y_range = _pad_range(min(all_y), max(all_y))
            title = f"{SIGNAL_TITLES[self._signal]} — {axis}"
            canvas.set_plot(series, x_range, y_range, title=title)


def _pad_range(lo: float, hi: float) -> tuple[float, float]:
    if hi - lo < 1e-6:
        return (lo - 0.5, hi + 0.5)
    return (lo, hi)
