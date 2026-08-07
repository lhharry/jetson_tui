"""Configuration loader."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib is stdlib only since 3.11
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass
class AppConfig:
    bus_labels: dict[int, str] = field(default_factory=lambda: {1: "Left", 7: "Right"})
    log_dir: Path = Path("./logs")
    sample_hz: int = 100
    plot_fps: int = 15
    plot_window_seconds: float = 10.0
    record_hz: int = 100
    web_host: str = "::"
    web_port: int = 8000
    # Sample source: the two I2C BNO055s, or one IMU streaming binary frames over serial.
    # ``sample_hz`` is the wire rate in both cases (it drives the CLS decimation).
    source_kind: str = "i2c"
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    serial_label: str = "Left"
    serial_magic: str = ""
    serial_gyro_units: str = "deg"
    # Real-time activity classification (CLS page). Disabled unless a checkpoint is present.
    cls_enabled: bool = True
    cls_model_path: str = ""
    cls_sensor: str = "Left"
    cls_target_hz: float = 10.0
    cls_window: int = 20
    cls_stride: int = 1

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
    source = raw.get("source", {})
    return AppConfig(
        bus_labels=bus_labels or {1: "Left", 7: "Right"},
        log_dir=Path(defaults.get("log_dir", "./logs")).expanduser(),
        sample_hz=int(defaults.get("sample_hz", 100)),
        plot_fps=int(defaults.get("plot_fps", 15)),
        plot_window_seconds=float(defaults.get("plot_window_seconds", 10.0)),
        record_hz=int(defaults.get("record_hz", 100)),
        web_host=str(defaults.get("web_host", "::")),
        web_port=int(defaults.get("web_port", 8000)),
        source_kind=str(source.get("kind", "i2c")).lower(),
        serial_port=str(source.get("port", "/dev/ttyACM0")),
        serial_baud=int(source.get("baud", 115200)),
        serial_label=str(source.get("label", "Left")),
        serial_magic=str(source.get("magic", "")),
        serial_gyro_units=str(source.get("gyro_units", "deg")).lower(),
        cls_enabled=bool(cls.get("enabled", True)),
        cls_model_path=str(cls.get("model_path", "")),
        cls_sensor=str(cls.get("sensor", "Left")),
        cls_target_hz=float(cls.get("target_hz", 10.0)),
        cls_window=int(cls.get("window", 20)),
        cls_stride=int(cls.get("stride", 1)),
    )
