"""Live plot screen using textual-plotext."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual_plotext import PlotextPlot

from jetson_imu_tui.ring_buffer import RingBuffers, SIGNAL_AXES

LABEL_COLORS = {"Left": "magenta+", "Right": "cyan+"}
DEFAULT_COLORS = ("magenta+", "cyan+", "yellow+", "green+")

SIGNAL_TITLES = {
    "euler": "Euler angles (deg)",
    "accel": "Acceleration (m/s^2)",
    "gyro": "Gyroscope (rad/s)",
    "quat": "Quaternion",
}


class PlotScreen(Screen):
    DEFAULT_CSS = """
    PlotScreen PlotextPlot { height: 1fr; }
    PlotScreen #plot-status { height: 1; padding: 0 1; background: $boost; }
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
        yield Static("Signal: euler   (1=Eul 2=Acc 3=Gyr 4=Quat  space=pause  q=back)", id="plot-status")
        yield Vertical(PlotextPlot(id="plot"))
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._redraw)
        self._redraw()

    def action_set_signal(self, signal: str) -> None:
        if signal in SIGNAL_AXES:
            self._signal = signal
            self._refresh_status()
            self._redraw()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        self._refresh_status()

    def _refresh_status(self) -> None:
        suffix = " [PAUSED]" if self._paused else ""
        self.query_one("#plot-status", Static).update(
            f"Signal: {self._signal}{suffix}   (1=Eul 2=Acc 3=Gyr 4=Quat  space=pause  q=back)"
        )

    def _redraw(self) -> None:
        if self._paused:
            return
        widget = self.query_one("#plot", PlotextPlot)
        plt = widget.plt
        plt.clear_figure()
        axes = SIGNAL_AXES[self._signal]
        labels = self._buffers.labels
        if self._signal == "quat":
            for li, label in enumerate(labels):
                t = list(self._buffers.time[label])
                if not t:
                    continue
                t0 = t[0]
                ts = [x - t0 for x in t]
                for ai, axis in enumerate(axes):
                    y = list(self._buffers.data[(label, "quat", axis)])
                    if len(y) != len(ts):
                        continue
                    color = DEFAULT_COLORS[(li * len(axes) + ai) % len(DEFAULT_COLORS)]
                    plt.plot(ts, y, label=f"{label}_{axis}", color=color)
            plt.title(SIGNAL_TITLES[self._signal])
            plt.xlabel("t (s)")
        else:
            plt.subplots(len(axes), 1)
            for ai, axis in enumerate(axes, start=1):
                sub = plt.subplot(ai, 1)
                for label in labels:
                    t = list(self._buffers.time[label])
                    y = list(self._buffers.data[(label, self._signal, axis)])
                    if not t or len(t) != len(y):
                        continue
                    t0 = t[0]
                    ts = [x - t0 for x in t]
                    color = LABEL_COLORS.get(label, DEFAULT_COLORS[0])
                    sub.plot(ts, y, label=label, color=color)
                sub.title(f"{SIGNAL_TITLES[self._signal]} — {axis}")
                if ai == len(axes):
                    sub.xlabel("t (s)")
        widget.refresh()
