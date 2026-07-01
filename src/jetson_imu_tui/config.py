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
    plot_fps: int = 15
    plot_window_seconds: float = 10.0
    record_hz: int = 100
    web_host: str = "::"
    web_port: int = 8000
    # Real-time activity classification (CLS page). Disabled unless a checkpoint is present.
    cls_enabled: bool = True
    cls_model_path: str = ""
    cls_sensor: str = "Left"
    cls_target_hz: float = 10.0
    cls_window: int = 20
    cls_stride: int = 10

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
    cls = raw.get("cls", {})
    return AppConfig(
        bus_labels=bus_labels or {1: "Left", 7: "Right"},
        log_dir=Path(defaults.get("log_dir", "./logs")).expanduser(),
        plot_fps=int(defaults.get("plot_fps", 15)),
        plot_window_seconds=float(defaults.get("plot_window_seconds", 10.0)),
        record_hz=int(defaults.get("record_hz", 100)),
        web_host=str(defaults.get("web_host", "::")),
        web_port=int(defaults.get("web_port", 8000)),
        cls_enabled=bool(cls.get("enabled", True)),
        cls_model_path=str(cls.get("model_path", "")),
        cls_sensor=str(cls.get("sensor", "Left")),
        cls_target_hz=float(cls.get("target_hz", 10.0)),
        cls_window=int(cls.get("window", 20)),
        cls_stride=int(cls.get("stride", 10)),
    )
