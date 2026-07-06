"""IMU lifecycle, signals, calibration and axis-remap — backed by the official Adafruit
``adafruit_bno055`` driver using the BNO055's **onboard** sensor fusion.

The chip runs in **IMUPLUS** mode (relative 6-DOF orientation from accelerometer + gyroscope,
magnetometer OFF), so euler / quaternion / acceleration / gyroscope are read directly from the
chip's fused output — no software Madgwick filter. Each configured I2C bus (from
``config/default.toml`` ``[buses]``) is opened with ``adafruit_extended_bus.ExtendedI2C`` and a
``BNO055_I2C`` driver at address 0x28.

Downstream consumers (``web_server._payload`` and ``recorder``) only use ``signals()`` →
``{label: {"euler","accel","gyro","quat"}}``, so this module owns all sensor specifics.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C

from jetson_imu_tui.ring_buffer import RingBuffer

# --- BNO055 specifics ------------------------------------------------------
BNO055_ADDRESS = 0x28  # both buses use the default address
REG_AXIS_MAP_CONFIG = 0x41
REG_AXIS_MAP_SIGN = 0x42

# Onboard fusion mode this project runs (accel+gyro, magnetometer off → no figure-8,
# no magnetic-distortion heading errors; orientation is relative).
FUSION_MODE = adafruit_bno055.IMUPLUS_MODE
CONFIG_MODE = adafruit_bno055.CONFIG_MODE

# Gyro output normalization to rad/s (the UI label and CSV expect rad/s). The Adafruit
# driver's units depend on the library version / UNIT_SEL — VERIFY ON DEVICE: rotate at a
# known rate; if values are ~57x too large the lib is returning deg/s, so set this to
# math.pi / 180. Leave at 1.0 if the lib already returns rad/s.
_GYRO_TO_RADS = 1.0

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


def _is_valid_config(config: int) -> bool:
    """Each output axis must map to a *distinct* source axis (no duplicates / no 0b11)."""
    fields = [config & 0b11, (config >> 2) & 0b11, (config >> 4) & 0b11]
    return sorted(fields) == [0, 1, 2]


def _decode_axis_remap(config: int, sign: int) -> dict:
    """Human-readable mapping for the three outputs: which source axis + sign each takes."""
    srcs = {"x": config & 0b11, "y": (config >> 2) & 0b11, "z": (config >> 4) & 0b11}
    signs = {"x": (sign >> 2) & 0b1, "y": (sign >> 1) & 0b1, "z": sign & 0b1}
    return {
        out: {"axis": _AXIS_LETTERS.get(srcs[out], "INVALID"), "sign": "-" if signs[out] else "+"}
        for out in ("x", "y", "z")
    }


def _placement_for(config: int, sign: int) -> str | None:
    for name, (cc, ss) in PLACEMENTS.items():
        if cc == config and ss == sign:
            return name
    return None


@dataclass
class ImuInfo:
    label: str
    bus_id: int
    sensor_name: str


class ImuService:
    def __init__(self, bus_labels: dict[int, str], state_path: Path | str | None = None) -> None:
        self._bus_labels = dict(bus_labels)
        self.sensors: dict[str, adafruit_bno055.BNO055_I2C] = {}
        self._buses: dict[str, ExtendedI2C] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._state_path = Path(state_path) if state_path else None
        self._axis_lock = threading.Lock()
        self._axis_config = DEFAULT_CONFIG
        self._axis_sign = DEFAULT_SIGN
        self._load_state()
        # Per-label zero offset for euler/accel/gyro (tare). None = no offset.
        self._offset: dict[str, dict[str, list[float]]] | None = None
        self._offset_lock = threading.Lock()
        # Background sampling: one thread per sensor fills a ring buffer; consumers
        # (web, recorder, CLS) read the buffers so I2C is polled once per period total.
        self._buffers: dict[str, RingBuffer] = {}
        self._sample_threads: dict[str, threading.Thread] = {}
        self._sampling_stop = threading.Event()

    @property
    def labels(self) -> list[str]:
        return [self._bus_labels[k] for k in sorted(self._bus_labels)]

    def connect(self) -> list[ImuInfo]:
        if self.sensors:
            return self.info()
        for bus_id in sorted(self._bus_labels):
            label = self._bus_labels[bus_id]
            try:
                i2c = ExtendedI2C(bus_id)
                sensor = adafruit_bno055.BNO055_I2C(i2c, address=BNO055_ADDRESS)
                sensor.mode = FUSION_MODE
                self._buses[label] = i2c
                self.sensors[label] = sensor
                self._locks[label] = threading.Lock()
                self._buffers[label] = RingBuffer()
            except Exception as err:  # pragma: no cover - hardware dependent
                logger.warning(f"{label} (bus {bus_id}): no BNO055 ({err})")
        # Axis remap is volatile (lost on power-cycle): re-apply a persisted non-default map.
        if self.sensors and (self._axis_config, self._axis_sign) != (DEFAULT_CONFIG, DEFAULT_SIGN):
            try:
                self.set_axis_remap(self._axis_config, self._axis_sign, persist=False)
            except Exception as err:  # pragma: no cover - hardware dependent
                logger.warning(f"axis-remap re-apply on connect failed: {err}")
        return self.info()

    def disconnect(self) -> None:
        self.stop_sampling()
        for i2c in self._buses.values():
            try:
                i2c.deinit()
            except Exception:
                pass
        self.sensors.clear()
        self._buses.clear()
        self._locks.clear()
        self._buffers.clear()

    def info(self) -> list[ImuInfo]:
        out: list[ImuInfo] = []
        for label, _sensor in self.sensors.items():
            bus_id = next((b for b, lab in self._bus_labels.items() if lab == label), -1)
            out.append(ImuInfo(label=label, bus_id=bus_id, sensor_name="BNO055"))
        return out

    def is_connected(self) -> bool:
        return bool(self.sensors)

    # --- background sampling -------------------------------------------------
    @property
    def sampling(self) -> bool:
        return bool(self._sample_threads)

    def start_sampling(self, hz: float = 100.0) -> None:
        """Start one sampler thread per connected sensor, filling its ring buffer at ``hz``.

        The two I2C buses run in parallel (one thread each) instead of the old serial
        read in ``signals()``. Once running, ``signals``/``read_raw``/``samples_since``
        are memory reads — the sensors see exactly one poll per period regardless of how
        many consumers (web tabs, recorder, CLS) are attached."""
        if self._sample_threads or not self.sensors:
            return
        self._sampling_stop.clear()
        for label in self.sensors:
            th = threading.Thread(
                target=self._sample_loop,
                args=(label, float(hz)),
                daemon=True,
                name=f"imu-sampler-{label}",
            )
            self._sample_threads[label] = th
            th.start()

    def stop_sampling(self) -> None:
        self._sampling_stop.set()
        for th in self._sample_threads.values():
            th.join(timeout=2.0)
        self._sample_threads.clear()

    def _sample_loop(self, label: str, hz: float) -> None:
        period = 1.0 / hz
        buf = self._buffers[label]
        sensor = self.sensors[label]
        next_tick = time.monotonic()
        stat_start = next_tick
        samples = overruns = bad_reads = 0
        durations: list[float] = []
        while not self._sampling_stop.is_set():
            t0 = time.perf_counter()
            sig = self._read(label, sensor)
            durations.append(time.perf_counter() - t0)
            if sig is not None:
                sig["t"] = time.monotonic()
                buf.append(sig)
                samples += 1
            else:
                bad_reads += 1
            now = time.monotonic()
            if now - stat_start >= 5.0:
                durations.sort()
                p50 = durations[len(durations) // 2] * 1e3
                p95 = durations[int(len(durations) * 0.95)] * 1e3
                # logger.info(
                #     f"sampler[{label}]: {samples / (now - stat_start):.1f} Hz (target {hz:.0f}) · "
                #     f"read p50={p50:.1f}ms p95={p95:.1f}ms · overruns={overruns} bad_reads={bad_reads}"
                # )
                stat_start = now
                samples = overruns = bad_reads = 0
                durations.clear()
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._sampling_stop.wait(sleep_for):
                    break
            else:
                overruns += 1
                next_tick = time.monotonic()  # fell behind — resync rather than spin

    # --- data --------------------------------------------------------------
    def _read(self, label: str, sensor: adafruit_bno055.BNO055_I2C) -> dict[str, list[float]] | None:
        """Read the chip's fused outputs under the per-sensor lock. None on a bad/partial read."""
        lock = self._locks.get(label)
        try:
            if lock is not None:
                with lock:
                    eul = sensor.euler
                    quat = sensor.quaternion
                    acc = sensor.acceleration
                    gyr = sensor.gyro
            else:  # pragma: no cover - locks always present for connected sensors
                eul, quat, acc, gyr = sensor.euler, sensor.quaternion, sensor.acceleration, sensor.gyro
        except Exception:
            return None
        vals = (eul, quat, acc, gyr)
        if any(v is None for v in vals) or any(c is None for v in vals for c in v):
            return None
        # Adafruit euler is (heading, roll, pitch) in degrees → keep the existing
        # contract x=roll, y=pitch, z=heading/yaw (degrees). quaternion is (w, x, y, z).
        return {
            "euler": [float(eul[1]), float(eul[2]), float(eul[0])],
            "accel": [float(acc[0]), float(acc[1]), float(acc[2])],
            "gyro": [float(gyr[0]) * _GYRO_TO_RADS, float(gyr[1]) * _GYRO_TO_RADS, float(gyr[2]) * _GYRO_TO_RADS],
            "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        }

    def _latest_raw(self, label: str) -> dict | None:
        """Newest raw sample for ``label`` — from the ring buffer while the sampler runs,
        falling back to a direct I2C read otherwise (library use without start_sampling)."""
        if self._sample_threads:
            buf = self._buffers.get(label)
            return buf.latest() if buf is not None else None
        sensor = self.sensors.get(label)
        return self._read(label, sensor) if sensor is not None else None

    @staticmethod
    def _with_offset(sig: dict | None, o: dict[str, list[float]] | None) -> dict | None:
        """Copy of ``sig``'s four signal keys with the tare offset applied. Copies always —
        buffer samples are shared with other consumers and must never be mutated."""
        if sig is None:
            return None
        out: dict[str, list[float]] = {"quat": list(sig["quat"])}
        for key in ("euler", "accel", "gyro"):
            vals = sig[key]
            out[key] = [v - ov for v, ov in zip(vals, o[key])] if o else list(vals)
        return out

    def signals(self) -> dict[str, dict[str, list[float]] | None]:
        """Latest derived signals per label, with the zero offset applied when active."""
        off = self._offset
        out: dict[str, dict[str, list[float]] | None] = {}
        for label in self.sensors:
            sig = self._latest_raw(label)
            out[label] = self._with_offset(sig, off.get(label) if off else None)
        return out

    def read_raw(self, label: str) -> dict[str, list[float]] | None:
        """Raw fused outputs for one sensor with the zero/tare offset NOT applied.

        The CLS classifier needs gravity-inclusive accel, so it must bypass the tare that
        ``signals()`` applies. Returns None if the label is unknown or no data is available."""
        if label not in self.sensors:
            return None
        sig = self._latest_raw(label)
        if sig is None:
            return None
        return {key: list(sig[key]) for key in ("euler", "accel", "gyro", "quat")}

    def raw_samples_since(self, label: str, t: float, limit: int | None = None) -> list[dict]:
        """Raw (tare-NOT-applied) ring-buffer samples ``{"t","euler","accel","gyro","quat"}``
        for one sensor with ``sample["t"] > t``, oldest first. Copies so callers never mutate
        buffer state shared with other consumers.

        CLS block-averages the full 100 Hz batch since its last tick to match training's
        anti-aliasing downsample, so it needs every raw sample — not just ``read_raw``'s
        latest one. Requires the sampler to be running (returns [] otherwise, since without
        the buffer there is nothing to average over)."""
        if label not in self.sensors:
            return []
        buf = self._buffers.get(label)
        if buf is None:
            return []
        return [
            {"t": s["t"], **{key: list(s[key]) for key in ("euler", "accel", "gyro", "quat")}}
            for s in buf.since(t, limit=limit)
        ]

    def samples_since(self, t: float, limit: int = 300) -> list[dict]:
        """Payload-shaped samples newer than monotonic time ``t``, oldest first.

        The first label is the time master; for each of its samples the other labels
        contribute their nearest-in-time sample (the sub-period misalignment between the
        independent sampler threads is invisible at plot scale). Tare is applied to
        euler/accel/gyro per sample; quaternions are never offset."""
        labels = [lab for lab in self.labels if lab in self._buffers]
        if not labels or not self._sample_threads:
            return []
        master = labels[0]
        master_samples = self._buffers[master].since(t, limit=limit)
        if not master_samples:
            return []
        others: dict[str, list[dict]] = {
            lab: self._buffers[lab].since(master_samples[0]["t"] - 0.05) for lab in labels[1:]
        }
        off = self._offset
        idx = {lab: 0 for lab in others}
        out: list[dict] = []
        for sm in master_samples:
            row: dict = {"t": sm["t"], "euler": {}, "accel": {}, "gyro": {}, "quat": {}}
            per_label: dict[str, dict | None] = {master: sm}
            for lab, arr in others.items():
                i = idx[lab]
                while i + 1 < len(arr) and abs(arr[i + 1]["t"] - sm["t"]) <= abs(arr[i]["t"] - sm["t"]):
                    i += 1
                idx[lab] = i
                per_label[lab] = arr[i] if arr else None
            for lab in labels:
                sig = self._with_offset(per_label.get(lab), off.get(lab) if off else None)
                for key in ("euler", "accel", "gyro", "quat"):
                    row[key][lab] = sig[key] if sig is not None else None
            out.append(row)
        return out

    # --- zero / tare -------------------------------------------------------
    @property
    def is_zeroed(self) -> bool:
        return self._offset is not None

    def zero_toggle(self) -> bool:
        """Capture the current euler/accel/gyro as the zero reference, or clear it. Returns
        True if now zeroed. Quaternion is never offset."""
        with self._offset_lock:
            if self._offset is None:
                captured: dict[str, dict[str, list[float]]] = {}
                for label in self.sensors:
                    sig = self._latest_raw(label)
                    if sig is not None:
                        captured[label] = {
                            "euler": list(sig["euler"]),
                            "accel": list(sig["accel"]),
                            "gyro": list(sig["gyro"]),
                        }
                self._offset = captured
            else:
                self._offset = None
            return self._offset is not None

    # --- calibration (status only) ----------------------------------------
    def calibration_status(self) -> dict[str, dict | None]:
        """Per-label BNO055 calibration levels (0-3). In IMUPLUS the magnetometer is unused,
        so readiness is based on gyro + accel."""
        out: dict[str, dict | None] = {}
        for label, sensor in self.sensors.items():
            lock = self._locks.get(label)
            try:
                if lock is not None:
                    with lock:
                        sys_c, gyro_c, accel_c, mag_c = sensor.calibration_status
                else:  # pragma: no cover
                    sys_c, gyro_c, accel_c, mag_c = sensor.calibration_status
                out[label] = {
                    "sys": int(sys_c),
                    "gyro": int(gyro_c),
                    "accel": int(accel_c),
                    "mag": int(mag_c),
                    "ready": int(gyro_c) >= 3 and int(accel_c) >= 3,
                }
            except Exception:
                out[label] = None
        return out

    # --- axis remap --------------------------------------------------------
    def get_axis_remap(self) -> dict:
        c, s = self._axis_config, self._axis_sign
        return {
            "config": c,
            "sign": s,
            "config_hex": f"0x{c:02X}",
            "sign_hex": f"0x{s:02X}",
            "mapping": _decode_axis_remap(c, s),
            "placement": _placement_for(c, s),
            "valid": _is_valid_config(c),
        }

    def set_axis_remap(self, config_byte: int, sign_byte: int, *, persist: bool = True) -> dict:
        config_byte &= 0xFF
        sign_byte &= 0xFF
        with self._axis_lock:
            valid = _is_valid_config(config_byte)
            result: dict = {
                "config": config_byte,
                "sign": sign_byte,
                "config_hex": f"0x{config_byte:02X}",
                "sign_hex": f"0x{sign_byte:02X}",
                "mapping": _decode_axis_remap(config_byte, sign_byte),
                "placement": _placement_for(config_byte, sign_byte),
                "valid": valid,
                "ok": False,
                "hardware": False,
                "applied": {},
                "message": "",
            }
            if not valid:
                result["message"] = (
                    "Invalid mapping: each output axis must map to a distinct source axis."
                )
                return result

            any_hw = False
            all_ok = True
            for label, sensor in self.sensors.items():
                entry = self._apply_axis(label, sensor, config_byte, sign_byte)
                result["applied"][label] = entry
                any_hw = any_hw or entry["hardware"]
                all_ok = all_ok and entry["ok"]

            self._axis_config = config_byte
            self._axis_sign = sign_byte
            result["hardware"] = any_hw
            result["ok"] = all_ok
            result["message"] = (
                "Applied to hardware." if any_hw else "Stored (no sensors connected)."
            )
            if persist:
                self._save_state()
            return result

    def _apply_axis(self, label: str, sensor: adafruit_bno055.BNO055_I2C, config_byte: int, sign_byte: int) -> dict:
        """Write AXIS_MAP_CONFIG/SIGN on one sensor (CONFIG mode → write → restore fusion),
        then read back. The Adafruit ``mode`` setter handles the 19/7 ms switch delays."""
        entry: dict = {
            "ok": False,
            "hardware": False,
            "readback_config": None,
            "readback_sign": None,
            "error": None,
        }
        lock = self._locks.get(label)
        try:
            with lock:  # type: ignore[arg-type]
                sensor.mode = CONFIG_MODE
                sensor._write_register(REG_AXIS_MAP_CONFIG, config_byte)
                sensor._write_register(REG_AXIS_MAP_SIGN, sign_byte)
                sensor.mode = FUSION_MODE
                rc = sensor._read_register(REG_AXIS_MAP_CONFIG)
                rs = sensor._read_register(REG_AXIS_MAP_SIGN)
            entry["readback_config"] = rc
            entry["readback_sign"] = rs
            entry["hardware"] = True
            entry["ok"] = rc == config_byte and rs == sign_byte
            if not entry["ok"]:
                entry["error"] = "readback mismatch (mapping may have been rejected by the chip)"
        except Exception as err:  # pragma: no cover - hardware dependent
            entry["error"] = f"{type(err).__name__}: {err}"
        return entry

    # --- persistence -------------------------------------------------------
    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
            self._axis_config = int(data["config"]) & 0xFF
            self._axis_sign = int(data["sign"]) & 0xFF
        except Exception as err:
            logger.warning(f"axis-remap: failed to load {self._state_path}: {err}")

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"config": self._axis_config, "sign": self._axis_sign})
            )
        except Exception as err:
            logger.warning(f"axis-remap: failed to save {self._state_path}: {err}")
