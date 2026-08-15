"""Sensor-source pieces shared by the I2C and serial services — no hardware imports.

``imu_service`` pulls in ``adafruit_bno055`` / Blinka at module scope, so a machine running the
serial source (an Arduino streaming BNO055 frames) cannot import anything from it. Everything
both services need lives here instead: the axis remap (tables, ops, and the transform itself),
the tare helper, and the one-method protocol CLS consumes.

**Axis remap is applied here, in software, for both sources.** It used to be written to the
BNO055's ``AXIS_MAP_CONFIG``/``AXIS_MAP_SIGN`` registers, which meant the serial source could
not use it at all (the mapping lived in the transmitting device's firmware) and the registers
had to be re-sent on every connect because they are volatile. Now the chip is left at its P1
identity default and the permutation happens on the host, so one code path serves both sources.

The transform is always a *signed permutation*: every composition of 90-degree rotations and
per-axis negations is one. That is the same shape as the old register encoding, so the byte
form is kept as a compact interchange format (``to_bytes``/``from_bytes``) and the datasheet
P0-P7 presets still work as shortcuts.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

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


# --- axis transform: matrices ----------------------------------------------

Matrix3 = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

IDENTITY_M: Matrix3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

# Active right-hand-rule rotations of the *readings* — the "direct" convention the UI exposes:
# picking X / +90 rotates the measured vector by +90 deg about X, so (x, y, z) -> (x, -z, y).
# The inverse convention ("the sensor is mounted 90 deg off, cancel it") is exactly these
# transposed; do not mix the two.
OPS: dict[str, Matrix3] = {
    "rot_x_90": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "rot_x_180": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "rot_x_270": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "rot_y_90": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "rot_y_180": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "rot_y_270": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "rot_z_90": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "rot_z_180": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "rot_z_270": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    # A single negation is a mirror (det = -1), not a mounting orientation. It is allowed
    # because the old register encoding allowed it, but quaternion/euler cannot follow it.
    "flip_x": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "flip_y": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "flip_z": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
}


def _matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    """Matrix product ``a @ b``. Input/output: 3x3 tuples of int. Exact — no float error."""
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _det3(m: Matrix3) -> int:
    """Determinant of a 3x3 int matrix. +1 = rotation, -1 = mirror. Output: int."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def compose_ops(ops) -> Matrix3:
    """Compose op names into one matrix, **applied in list order** (ops[0] first).

    Input: iterable of str, each a key of ``OPS``. Output: 3x3 tuple of int.
    Raises ValueError on an unknown name so a typo in the config is loud, not silent."""
    m = IDENTITY_M
    for name in ops:
        step = OPS.get(str(name))
        if step is None:
            raise ValueError(f"unknown axis op '{name}' (expected one of: {', '.join(OPS)})")
        m = _matmul(step, m)  # later ops apply on top of earlier ones
    return m


def _perm_signs(m: Matrix3) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Signed permutation form of ``m``: ``out[i] = signs[i] * v[perm[i]]``.

    Input: 3x3 tuple of int. Output: (perm, signs), both 3-tuples of int; signs are +/-1.
    Raises ValueError if ``m`` is not a signed permutation (cannot happen from ``OPS``)."""
    perm: list[int] = []
    signs: list[int] = []
    for row in m:
        nz = [(j, v) for j, v in enumerate(row) if v != 0]
        if len(nz) != 1 or abs(nz[0][1]) != 1:
            raise ValueError(f"not a signed permutation matrix: {m}")
        perm.append(nz[0][0])
        signs.append(nz[0][1])
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"not a permutation: {m}")
    return (perm[0], perm[1], perm[2]), (signs[0], signs[1], signs[2])


# --- axis transform: quaternion helpers ------------------------------------

def quat_mul(a, b) -> tuple[float, float, float, float]:
    """Hamilton product ``a (x) b``. Input/output: (w, x, y, z) sequences of float."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q) -> tuple[float, float, float, float]:
    """Conjugate (== inverse for a unit quaternion). Input/output: (w, x, y, z) float."""
    return (q[0], -q[1], -q[2], -q[3])


def mat_to_quat(m: Matrix3) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion. Input: 3x3 int, det must be +1.

    Output: (w, x, y, z) float. Shepperd's method — the branch matters here because a 90 deg
    signed permutation often has trace <= 0, where the naive trace formula divides by ~0."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def quat_to_euler(q) -> list[float]:
    """Unit quaternion -> this project's euler contract.

    Input: (w, x, y, z) float. Output: [roll, pitch, yaw] in **degrees**, ZYX intrinsic,
    with yaw normalized to [0, 360) to match what the BNO055 reports for heading."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw) % 360.0]


# --- axis transform: the transform object ----------------------------------

@dataclass(frozen=True)
class AxisTransform:
    """An immutable signed-permutation remap of the sensor axes, applied on the host.

    Built via ``from_ops`` / ``from_bytes``; the fields are derived and consistent by
    construction. Immutable so a reader thread can grab it without a lock."""

    ops: tuple[str, ...] = ()
    matrix: Matrix3 = IDENTITY_M
    perm: tuple[int, int, int] = (0, 1, 2)
    signs: tuple[int, int, int] = (1, 1, 1)
    det: int = 1
    # Rotation quaternion of ``matrix``, or None when det == -1 (a mirror has none).
    rot_quat: tuple[float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def _build(cls, ops, m: Matrix3) -> "AxisTransform":
        perm, signs = _perm_signs(m)
        det = _det3(m)
        return cls(tuple(str(o) for o in ops), m, perm, signs, det, mat_to_quat(m) if det == 1 else None)

    @classmethod
    def from_ops(cls, ops) -> "AxisTransform":
        """Input: iterable of op names (see ``OPS``). Output: AxisTransform.
        Raises ValueError on an unknown name."""
        ops = list(ops or [])
        return cls._build(ops, compose_ops(ops))

    @classmethod
    def from_bytes(cls, config: int, sign: int) -> "AxisTransform":
        """Input: the two BNO055-style bytes (int). Output: AxisTransform with empty ``ops``
        (the op chain that produced them is not recoverable, only the resulting mapping).
        Raises ValueError if ``config`` is not a valid permutation."""
        config &= 0xFF
        sign &= 0xFF
        if not is_valid_config(config):
            raise ValueError(f"invalid axis config 0x{config:02X}: outputs must map to distinct axes")
        perm = (config & 0b11, (config >> 2) & 0b11, (config >> 4) & 0b11)
        signs = (-1 if (sign >> 2) & 1 else 1, -1 if (sign >> 1) & 1 else 1, -1 if sign & 1 else 1)
        m = tuple(  # type: ignore[assignment]
            tuple(signs[i] if j == perm[i] else 0 for j in range(3)) for i in range(3)
        )
        return cls._build((), m)  # type: ignore[arg-type]

    def to_bytes(self) -> tuple[int, int]:
        """Output: (config_byte, sign_byte) int, the compact BNO055-style encoding."""
        config = self.perm[0] | (self.perm[1] << 2) | (self.perm[2] << 4)
        sign = ((1 if self.signs[0] < 0 else 0) << 2) | ((1 if self.signs[1] < 0 else 0) << 1) | (
            1 if self.signs[2] < 0 else 0
        )
        return config, sign

    @property
    def is_identity(self) -> bool:
        return self.perm == (0, 1, 2) and self.signs == (1, 1, 1)

    @property
    def is_mirror(self) -> bool:
        """True when an odd number of negations makes this a reflection, not a rotation.
        Vectors still transform correctly; quaternion/euler cannot follow."""
        return self.det == -1

    def apply_vec3(self, v) -> list[float]:
        """Input: 3-sequence of float. Output: new list[float] of length 3.

        ``+ 0.0`` normalizes the ``-1 * 0.0 = -0.0`` that would otherwise reach the CSVs."""
        p, s = self.perm, self.signs
        return [s[0] * v[p[0]] + 0.0, s[1] * v[p[1]] + 0.0, s[2] * v[p[2]] + 0.0]

    def apply_quat(self, q) -> list[float]:
        """Re-express an orientation quaternion in the remapped body frame.

        Input: (w, x, y, z) float. Output: new list[float] of length 4.

        ``q`` maps body coords to world coords. The remap redefines body coords as
        ``v' = R v``, so ``v = R^T v'`` and therefore ``q' = q (x) conj(q_R)``.
        A mirror has no ``q_R``; the input is passed through unchanged."""
        if self.rot_quat is None:
            return list(q)
        return list(quat_mul(q, quat_conj(self.rot_quat)))

    def describe(self) -> dict:
        """Output: JSON-ready dict for ``GET /axis-remap`` and the UI."""
        c, s = self.to_bytes()
        return {
            "ops": list(self.ops),
            "config": c,
            "sign": s,
            "config_hex": f"0x{c:02X}",
            "sign_hex": f"0x{s:02X}",
            "mapping": decode_axis_remap(c, s),
            "placement": placement_for(c, s),
            "valid": True,
            "identity": self.is_identity,
            "mirror": self.is_mirror,
        }


IDENTITY_TRANSFORM = AxisTransform()


def apply_axis_transform(sig: dict | None, tf: AxisTransform | None) -> dict | None:
    """Copy of ``sig``'s four signal keys with the axis remap applied.

    Input: ``sig`` = ``{"euler","accel","gyro","quat"}`` with list[float] or None values
    (the serial source has no euler/quat), ``tf`` = AxisTransform or None.
    Output: a **new** dict with new lists — buffer samples are shared with other consumers
    and must never be mutated. Callers can therefore drop their own copy step.

    ``accel``/``gyro`` are vectors and always follow the transform. ``quat`` is re-expressed
    and ``euler`` is recomputed from it, but only when the transform is a real rotation *and*
    non-identity: at identity the chip's own euler is passed through untouched, so enabling
    this feature cannot perturb the default case with quaternion-to-euler rounding."""
    if sig is None:
        return None
    ident = tf is None or tf.is_identity
    out: dict[str, list[float] | None] = {}
    for key in ("accel", "gyro"):
        vals = sig.get(key)
        if vals is None:
            out[key] = None
        else:
            out[key] = list(vals) if ident else tf.apply_vec3(vals)  # type: ignore[union-attr]
    quat = sig.get("quat")
    euler = sig.get("euler")
    if ident or quat is None or tf.is_mirror:  # type: ignore[union-attr]
        out["quat"] = list(quat) if quat is not None else None
        out["euler"] = list(euler) if euler is not None else None
    else:
        new_q = tf.apply_quat(quat)  # type: ignore[union-attr]
        out["quat"] = new_q
        out["euler"] = quat_to_euler(new_q)
    return out


# --- axis transform: shared state + persistence ----------------------------

class AxisState:
    """Current axis remap plus its persistence — owned by both ``ImuService`` and
    ``SerialImuService`` so the two sources behave identically.

    Startup precedence: ``<log_dir>/axis_remap.json`` (last UI change) overrides the
    ``[axis] ops`` default from the TOML. Deleting the JSON reverts to the TOML."""

    def __init__(self, state_path: Path | str | None = None, default_ops=()) -> None:
        self._path = Path(state_path) if state_path else None
        self._lock = threading.Lock()
        try:
            self._tf = AxisTransform.from_ops(default_ops)
        except ValueError as err:
            logger.warning(f"axis: bad [axis] ops in config ({err}) — using identity")
            self._tf = IDENTITY_TRANSFORM
        self._load()

    @property
    def transform(self) -> AxisTransform:
        """Output: the current AxisTransform. No lock needed — the object is immutable and
        the attribute rebind is atomic, so a reader sees either the old or the new one."""
        return self._tf

    def describe(self) -> dict:
        return self._tf.describe()

    def set(self, *, ops=None, config: int | None = None, sign: int | None = None,
            persist: bool = True) -> dict:
        """Replace the transform, from either an op list or the two bytes.

        Input: ``ops`` = iterable of op names, or ``config``/``sign`` = int.
        Output: ``describe()`` plus ``ok``/``message``. Never raises — an invalid request
        comes back as ``ok: False`` so the HTTP layer can render it."""
        with self._lock:
            try:
                if ops is not None:
                    tf = AxisTransform.from_ops(ops)
                elif config is not None and sign is not None:
                    tf = AxisTransform.from_bytes(int(config), int(sign))
                else:
                    return {**self._tf.describe(), "ok": False,
                            "message": "provide 'ops', or 'placement', or numeric 'config' and 'sign'"}
            except (ValueError, TypeError) as err:
                return {**self._tf.describe(), "ok": False, "message": str(err)}
            self._tf = tf
            if persist:
                self._save()
            msg = "Applied." if not tf.is_mirror else (
                "Applied — mirrored mapping: accel/gyro follow it, quaternion/euler cannot."
            )
            return {**tf.describe(), "ok": True, "message": msg}

    def _load(self) -> None:
        """Read the persisted mapping, if any. Accepts both the new ``ops`` form and the old
        ``{"config","sign"}`` files written by the register-based version."""
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if data.get("ops"):
                self._tf = AxisTransform.from_ops(data["ops"])
            elif "config" in data and "sign" in data:
                self._tf = AxisTransform.from_bytes(int(data["config"]), int(data["sign"]))
        except Exception as err:
            logger.warning(f"axis: failed to load {self._path}: {err}")

    def _save(self) -> None:
        if not self._path:
            return
        try:
            c, s = self._tf.to_bytes()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"ops": list(self._tf.ops), "config": c, "sign": s}))
        except Exception as err:
            logger.warning(f"axis: failed to save {self._path}: {err}")
