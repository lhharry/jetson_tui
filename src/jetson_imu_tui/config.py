"""Configuration loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass
class AppConfig:
    bus_labels: dict[int, str] = field(default_factory=lambda: {1: "Left", 7: "Right"})
    log_dir: Path = Path("./logs")
    data_hz: int = 30
    ui_refresh_hz: int = 15
    plot_fps: int = 15
    plot_window_seconds: float = 10.0
    plot_smoothing: int = 1
    record_hz: int = 100
    plot_window_samples: int = 600

    @property
    def labels(self) -> list[str]:
        return [self.bus_labels[k] for k in sorted(self.bus_labels)]


def load_config(path: Path | None = None) -> AppConfig:
    src = path or DEFAULT_CONFIG
    with open(src, "rb") as fh:
        raw = tomllib.load(fh)
    buses_raw = raw.get("buses", {})
    bus_labels = {int(k): str(v) for k, v in buses_raw.items()}
    defaults = raw.get("defaults", {})
    return AppConfig(
        bus_labels=bus_labels or {1: "Left", 7: "Right"},
        log_dir=Path(defaults.get("log_dir", "./logs")).expanduser(),
        data_hz=int(defaults.get("data_hz", 30)),
        ui_refresh_hz=int(defaults.get("ui_refresh_hz", 15)),
        plot_fps=int(defaults.get("plot_fps", 15)),
        plot_window_seconds=float(defaults.get("plot_window_seconds", 10.0)),
        plot_smoothing=int(defaults.get("plot_smoothing", 1)),
        record_hz=int(defaults.get("record_hz", 100)),
        plot_window_samples=int(defaults.get("plot_window_samples", 600)),
    )
