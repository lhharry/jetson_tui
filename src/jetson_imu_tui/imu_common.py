"""Sensor-source pieces shared by the I2C and serial services — no hardware imports.

``imu_service`` pulls in ``adafruit_bno055`` / Blinka at module scope, so a machine running the
serial source (an Arduino streaming BNO055 frames) cannot import anything from it. Everything
both services need lives here instead: the axis-remap tables and decoding (still meaningful over
serial as *reported* state, even though only I2C can write the registers), the tare helper, and
the one-method protocol CLS consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Default mapping = P1 (identity: X->X, Y->Y, Z->Z, all positive).
DEFAULT_CONFIG = 0x24
DEFAULT_SIGN = 0x00

# Datasheet §3.4 (p.27) mounting placements: name -> (config_byte, sign_byte).
PLACEMENTS: dict[str, tuple[int, int]] = {
    "P0": (0x21, 0x04),
    "P1": (0x24, 0x00),
    "P2": (0x24, 0x06),
    "P3": (0x21, 0x02),
    "P4": (0x24, 0x03),
    "P5": (0x21, 0x01),
    "P6": (0x21, 0x07),
    "P7": (0x24, 0x05),
}

_AXIS_LETTERS = {0: "X", 1: "Y", 2: "Z", 3: "INVALID"}

# The four signal keys every source reports per sample. A source that cannot produce one
# (the serial stream has no fusion output) reports None for it rather than dropping the key —
# the browser derives its label list from ``euler``'s keys.
SIGNAL_KEYS = ("euler", "accel", "gyro", "quat")


@dataclass
class ImuInfo:
    label: str
    bus_id: int
    sensor_name: str


@runtime_checkable
class SensorSource(Protocol):
    """What ``ClsService`` requires of a sensor source — one method, nothing else.

    Both ``ImuService`` and ``SerialImuService`` satisfy it; CLS never touches the rest of
    either class, which is why the serial source can feed the model unchanged."""

    def raw_samples_since(self, label: str, t: float, limit: int | None = None) -> list[dict]:
        ...


def is_valid_config(config: int) -> bool:
    """Each output axis must map to a *distinct* source axis (no duplicates / no 0b11)."""
    fields = [config & 0b11, (config >> 2) & 0b11, (config >> 4) & 0b11]
    return sorted(fields) == [0, 1, 2]


def decode_axis_remap(config: int, sign: int) -> dict:
    """Human-readable mapping for the three outputs: which source axis + sign each takes."""
    srcs = {"x": config & 0b11, "y": (config >> 2) & 0b11, "z": (config >> 4) & 0b11}
    signs = {"x": (sign >> 2) & 0b1, "y": (sign >> 1) & 0b1, "z": sign & 0b1}
    return {
        out: {"axis": _AXIS_LETTERS.get(srcs[out], "INVALID"), "sign": "-" if signs[out] else "+"}
        for out in ("x", "y", "z")
    }


def placement_for(config: int, sign: int) -> str | None:
    for name, (cc, ss) in PLACEMENTS.items():
        if cc == config and ss == sign:
            return name
    return None


def apply_offset(sig: dict | None, o: dict[str, list[float]] | None) -> dict | None:
    """Copy of ``sig``'s four signal keys with the tare offset applied. Copies always —
    buffer samples are shared with other consumers and must never be mutated.

    A None signal value (serial has no euler/quat) stays None and is never offset, as is
    quaternion, which is passed through by design."""
    if sig is None:
        return None
    quat = sig.get("quat")
    out: dict[str, list[float] | None] = {"quat": list(quat) if quat is not None else None}
    for key in ("euler", "accel", "gyro"):
        vals = sig.get(key)
        if vals is None:
            out[key] = None
            continue
        # ``o.get``: a serial tare only captures accel/gyro, so the euler entry is absent.
        ref = o.get(key) if o else None
        out[key] = [v - ov for v, ov in zip(vals, ref)] if ref else list(vals)
    return out
