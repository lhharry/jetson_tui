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
    # CSV rows per second; 0 = every frame the source produced. Not a polling cadence —
    # see recorder.Recorder, whose DRAIN_HZ constant owns that.
    record_hz: int = 0
    web_host: str = "::"
    web_port: int = 8000
    # Sample source: the two I2C BNO055s, or one IMU streaming binary frames over serial.
    # ``source_kind`` is only the *boot* default — the web UI switches at runtime.
    source_kind: str = "i2c"
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    serial_label: str = "Left"
    serial_magic: str = ""
    # Frame layout: float count, whether a source timestamp rides along, and channel order.
    # See read_serial.LAYOUTS — the order is undetectable from the bytes, so it is declared.
    serial_layout: str = "accel_gyro_t"
    serial_gyro_units: str = "deg"
    # Wire rate of the serial device. Defaults to ``sample_hz`` (the I2C rate); it exists
    # separately because the two sources can be swapped at runtime and the rate drives the
    # CLS decimation, so one global value cannot be right for both if they differ.
    serial_sample_hz: int = 100
    # Per-telemetry-group absolute limit: values outside +/- it are clamped before display and
    # recording (see SerialImuService._clean). A group absent from the table passes through
    # untouched — which is why "state" must never appear in it, its t_src being a device clock
    # that climbs into the thousands.
    telemetry_clip: dict[str, float] = field(default_factory=dict)
    # Host-side axis remap, applied to both sources (see imu_common.AxisState). Op names from
    # ``imu_common.OPS``, applied in list order. This is only the *default*: a mapping set from
    # the web UI is written to ``<log_dir>/axis_remap.json`` and takes precedence at startup.
    axis_ops: list[str] = field(default_factory=list)
    # Real-time activity classification (CLS page). Disabled unless a checkpoint is present.
    cls_enabled: bool = True
    cls_model_path: str = ""
    cls_sensor: str = "Left"
    cls_target_hz: float = 10.0
    cls_window: int = 20
    cls_stride: int = 1
    # Temporal aggregation of frame predictions into one stable decision (see cls/vote.py).
    vote_enabled: bool = True
    vote_window: int = 5
    vote_emit_every: int = 5
    vote_hysteresis: int = 0
    # Absolute path of the TOML this was loaded from. Recorded into every session folder so
    # a recording can be traced back to the settings that produced it.
    config_path: str = ""

    @property
    def labels(self) -> list[str]:
        return [self.bus_labels[k] for k in sorted(self.bus_labels)]

    def sample_hz_for(self, kind: str) -> int:
        """Wire rate of one source kind — what ``start_sampling`` and CLS decimation need."""
        return self.serial_sample_hz if kind == "serial" else self.sample_hz


def load_config(path: Path | None = None) -> AppConfig:
    src = path or DEFAULT_CONFIG
    with open(src, "rb") as fh:
        raw = tomllib.load(fh)
    buses_raw = raw.get("buses", {})
    bus_labels = {int(k): str(v) for k, v in buses_raw.items()}
    defaults = raw.get("defaults", {})
    cls = raw.get("cls", {})
    vote = cls.get("vote", {})
    source = raw.get("source", {})
    axis = raw.get("axis", {})
    clip = raw.get("telemetry", {}).get("clip", {})
    sample_hz = int(defaults.get("sample_hz", 100))
    return AppConfig(
        config_path=str(Path(src).resolve()),
        bus_labels=bus_labels or {1: "Left", 7: "Right"},
        log_dir=Path(defaults.get("log_dir", "./logs")).expanduser(),
        sample_hz=sample_hz,
        plot_fps=int(defaults.get("plot_fps", 15)),
        plot_window_seconds=float(defaults.get("plot_window_seconds", 10.0)),
        record_hz=max(0, int(defaults.get("record_hz", 0))),
        web_host=str(defaults.get("web_host", "::")),
        web_port=int(defaults.get("web_port", 8000)),
        source_kind=str(source.get("kind", "i2c")).lower(),
        serial_port=str(source.get("port", "/dev/ttyACM0")),
        serial_baud=int(source.get("baud", 115200)),
        serial_label=str(source.get("label", "Left")),
        serial_magic=str(source.get("magic", "")),
        serial_layout=str(source.get("layout", "accel_gyro_t")).lower(),
        serial_gyro_units=str(source.get("gyro_units", "deg")).lower(),
        serial_sample_hz=int(source.get("sample_hz", sample_hz)),
        telemetry_clip={str(k): abs(float(v)) for k, v in clip.items()},
        axis_ops=[str(op) for op in axis.get("ops", [])],
        cls_enabled=bool(cls.get("enabled", True)),
        cls_model_path=str(cls.get("model_path", "")),
        cls_sensor=str(cls.get("sensor", "Left")),
        cls_target_hz=float(cls.get("target_hz", 10.0)),
        cls_window=int(cls.get("window", 20)),
        cls_stride=int(cls.get("stride", 1)),
        vote_enabled=bool(vote.get("enabled", True)),
        vote_window=int(vote.get("window", 5)),
        vote_emit_every=int(vote.get("emit_every", 5)),
        vote_hysteresis=int(vote.get("hysteresis", 0)),
    )
