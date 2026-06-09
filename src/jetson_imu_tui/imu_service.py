"""IMU lifecycle and snapshot helpers wrapping imu-python.

Also implements BNO055 **axis remap** (datasheet §3.4): write AXIS_MAP_CONFIG (0x41)
and AXIS_MAP_SIGN (0x42) while the chip is in CONFIG_MODE so the reported axes match the
physical mounting. A single shared mapping is applied to every connected sensor.

The register I/O path through `imu-python` is not part of its public API, so the low-level
read/write handle is *resolved defensively* at runtime (`_resolve_regio`): we look for an
adafruit_bno055-style driver (``_write_register``/``_read_register``) first, then a
busio-style I2C bus (``writeto``/``writeto_then_readfrom``). On mock IMUs (dev host) or when
no handle is found, writes become safe no-ops and the chosen bytes are still stored/persisted
so the UI and persistence flow keep working.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from imu_python.base_classes import IMUData
from imu_python.definitions import MOCK_NAME
from imu_python.factory import IMUFactory
from imu_python.sensor_manager import IMUManager

# --- BNO055 registers / modes (Page 0) -------------------------------------
REG_PAGE_ID = 0x07
REG_OPR_MODE = 0x3D
REG_AXIS_MAP_CONFIG = 0x41
REG_AXIS_MAP_SIGN = 0x42
MODE_CONFIG = 0x00
MODE_AMG = 0x07  # the operation mode this project runs (raw accel/mag/gyro)
BNO055_ADDRESS = 0x28  # both buses use the default address (see README)

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


# --- defensive register-I/O resolution -------------------------------------
class _RegIO:
    """Tiny adapter exposing ``write(reg, val)`` / ``read(reg) -> int`` over whatever
    low-level handle we managed to find on a manager."""

    def __init__(self, write, read) -> None:
        self.write = write
        self.read = read


def _candidate_objects(root, max_depth: int = 3) -> list:
    """Shallow BFS over instance attributes to collect candidate handle objects."""
    seen: set[int] = set()
    out: list = []
    frontier = [(root, 0)]
    while frontier:
        obj, depth = frontier.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        out.append(obj)
        if depth >= max_depth:
            continue
        try:
            attrs = vars(obj)
        except TypeError:
            continue
        for value in attrs.values():
            if value is None or isinstance(
                value, (int, float, str, bytes, bool, list, dict, tuple, set, frozenset)
            ):
                continue
            frontier.append((value, depth + 1))
    return out


def _acquire(try_lock) -> bool:
    if not callable(try_lock):
        return True
    for _ in range(2000):
        if try_lock():
            return True
    return False


def _resolve_regio(manager: IMUManager) -> _RegIO | None:
    """Find a register read/write path on a manager. Prefer an adafruit_bno055-style driver
    (it manages its own I2C locking); fall back to a raw busio.I2C bus at 0x28."""
    objs = _candidate_objects(manager)

    # 1) adafruit_bno055-style driver: _write_register(reg, val) / _read_register(reg)
    for obj in objs:
        w = getattr(obj, "_write_register", None)
        r = getattr(obj, "_read_register", None)
        if callable(w) and callable(r):
            return _RegIO(
                lambda reg, val, w=w: w(reg & 0xFF, val & 0xFF),
                lambda reg, r=r: int(r(reg & 0xFF)) & 0xFF,
            )

    # 2) busio.I2C-style bus: writeto / writeto_then_readfrom (lock around transactions)
    for obj in objs:
        writeto = getattr(obj, "writeto", None)
        wtrf = getattr(obj, "writeto_then_readfrom", None)
        if callable(writeto) and callable(wtrf):
            try_lock = getattr(obj, "try_lock", None)
            unlock = getattr(obj, "unlock", None)
            addr = BNO055_ADDRESS

            def _w(reg, val, bus=obj, addr=addr, try_lock=try_lock, unlock=unlock):
                locked = _acquire(try_lock)
                try:
                    bus.writeto(addr, bytes([reg & 0xFF, val & 0xFF]))
                finally:
                    if locked and callable(unlock):
                        unlock()

            def _r(reg, bus=obj, addr=addr, wtrf=wtrf, try_lock=try_lock, unlock=unlock):
                buf = bytearray(1)
                locked = _acquire(try_lock)
                try:
                    bus.writeto_then_readfrom(addr, bytes([reg & 0xFF]), buf)
                finally:
                    if locked and callable(unlock):
                        unlock()
                return buf[0]

            return _RegIO(_w, _r)

    return None


@dataclass
class ImuInfo:
    label: str
    bus_id: int
    sensor_name: str
    is_mock: bool


class ImuService:
    def __init__(self, bus_labels: dict[int, str], state_path: Path | str | None = None) -> None:
        self._bus_labels = dict(bus_labels)
        self.managers: dict[str, IMUManager] = {}
        self._state_path = Path(state_path) if state_path else None
        self._axis_lock = threading.Lock()
        self._axis_config = DEFAULT_CONFIG
        self._axis_sign = DEFAULT_SIGN
        self._load_state()

    @property
    def labels(self) -> list[str]:
        return [self._bus_labels[k] for k in sorted(self._bus_labels)]

    def connect(self) -> list[ImuInfo]:
        if self.managers:
            return self.info()
        # free_threading is auto-gated inside imu-python; pass False to keep things
        # predictable on stock CPython.
        managers = IMUFactory.detect_and_create(free_threading=False, log_data=False)
        labeled: dict[str, IMUManager] = {}
        for m in managers:
            bus_id = int(m.i2c_id) if m.i2c_id is not None else -1
            label = self._bus_labels.get(bus_id, f"bus_{bus_id}")
            labeled[label] = m
        for m in labeled.values():
            m.start()
        self.managers = labeled
        # Axis remap is volatile (lost on power-cycle): re-apply a persisted non-default map.
        if (self._axis_config, self._axis_sign) != (DEFAULT_CONFIG, DEFAULT_SIGN):
            try:
                self.set_axis_remap(self._axis_config, self._axis_sign, persist=False)
            except Exception as err:  # pragma: no cover - hardware dependent
                logger.warning(f"axis-remap re-apply on connect failed: {err}")
        return self.info()

    def disconnect(self) -> None:
        for m in self.managers.values():
            m.stop()
        self.managers.clear()

    def info(self) -> list[ImuInfo]:
        out: list[ImuInfo] = []
        for label, m in self.managers.items():
            bus_id = int(m.i2c_id) if m.i2c_id is not None else -1
            name = m.imu_descriptor.name
            out.append(
                ImuInfo(label=label, bus_id=bus_id, sensor_name=name, is_mock=name == MOCK_NAME)
            )
        return out

    def is_connected(self) -> bool:
        return bool(self.managers)

    def snapshot(self) -> dict[str, IMUData | None]:
        out: dict[str, IMUData | None] = {}
        for label, m in self.managers.items():
            try:
                out[label] = m.get_data()
            except Exception:
                # A manager may be momentarily stopped during an axis-remap apply.
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
            for label, m in self.managers.items():
                entry = self._apply_to_manager(m, config_byte, sign_byte)
                result["applied"][label] = entry
                any_hw = any_hw or entry["hardware"]
                all_ok = all_ok and entry["ok"]

            # Store + persist even if there were no managers (e.g. set before connect).
            self._axis_config = config_byte
            self._axis_sign = sign_byte
            result["hardware"] = any_hw
            result["ok"] = all_ok
            result["message"] = (
                "Applied to hardware."
                if any_hw
                else "Stored (no hardware write — mock/dev host or no I2C handle resolved)."
            )
            if persist:
                self._save_state()
            return result

    def _apply_to_manager(self, m: IMUManager, config_byte: int, sign_byte: int) -> dict:
        """Stop the manager, run the CONFIG_MODE write sequence, read back, restart."""
        entry: dict = {
            "ok": False,
            "hardware": False,
            "readback_config": None,
            "readback_sign": None,
            "error": None,
        }
        name = getattr(getattr(m, "imu_descriptor", None), "name", "")
        if name == MOCK_NAME:
            entry["ok"] = True
            entry["error"] = "mock (simulated)"
            return entry

        io = _resolve_regio(m)
        if io is None:
            # Not a hard failure: the UI/persistence flow continues, but flag no hardware.
            entry["ok"] = True
            entry["error"] = "no register I/O handle resolved (see plan Step 1 / fallback)"
            logger.warning(
                "axis-remap: could not resolve a register I/O handle on the manager; "
                "mapping stored but NOT written to the chip."
            )
            return entry

        try:
            try:
                m.stop()
            except Exception:
                pass
            # Preserve and restore the current operation mode (default AMG).
            try:
                prev_mode = io.read(REG_OPR_MODE) & 0x0F
            except Exception:
                prev_mode = None
            op_mode = prev_mode if prev_mode not in (None, MODE_CONFIG) else MODE_AMG

            io.write(REG_PAGE_ID, 0x00)
            io.write(REG_OPR_MODE, MODE_CONFIG)
            time.sleep(0.02)  # any -> CONFIG takes 19 ms
            io.write(REG_AXIS_MAP_CONFIG, config_byte)
            io.write(REG_AXIS_MAP_SIGN, sign_byte)
            io.write(REG_OPR_MODE, op_mode)
            time.sleep(0.01)  # CONFIG -> any takes 7 ms

            rc = io.read(REG_AXIS_MAP_CONFIG)
            rs = io.read(REG_AXIS_MAP_SIGN)
            entry["readback_config"] = rc
            entry["readback_sign"] = rs
            entry["hardware"] = True
            entry["ok"] = rc == config_byte and rs == sign_byte
            if not entry["ok"]:
                entry["error"] = "readback mismatch (mapping may have been rejected by the chip)"
        except Exception as err:  # pragma: no cover - hardware dependent
            entry["error"] = f"{type(err).__name__}: {err}"
        finally:
            try:
                m.start()
            except Exception as err:  # pragma: no cover - hardware dependent
                if entry["error"] is None:
                    entry["error"] = f"manager restart failed: {err}"
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
