"""Decoder for the binary frames an Arduino / Simulink model streams over serial.

A frame is an optional header followed by little-endian float32. What those floats mean is
declared by ``layout``, which is an ordered list of **channel blocks**. Blocks rather than one
enum per combination: the frame grew from "one IMU" to "one IMU + both knees + motor feedback +
controller trace" and will likely grow again, and a block list expresses that as one more table
row instead of a new decoder branch.

==========  ======  ==================================================================
block       floats  contents
==========  ======  ==================================================================
``enable``       1  the device's SWITCH line; non-zero means the controller may drive
``gyro``         3  gx gy gz, in the device's units (``serial_service`` converts)
``accel``        3  ax ay az, m/s^2, gravity included
``knee4``        4  ang_r vel_r ang_l vel_l -- knee angle + angular velocity, both legs
``fb6``          6  pos_r speed_r torque_r pos_l speed_l torque_l -- motor feedback
``trace5``       5  finalClass LU_AVEL_F L_KVEL L_KWRAP MotorCom_L -- controller trace
``t_src``        1  the device's own clock in seconds, as **data**
``t``            1  the device's own clock in seconds, as a **sync anchor**; last only
==========  ======  ==================================================================

``t`` and ``t_src`` carry the same quantity and differ only in what the decoder does with it, so
a layout may hold at most one of them:

* ``t`` is pinned last because alignment reads it as ``v[-1]``, and its monotonicity gates
  sync — a jump forces a re-align.
* ``t_src`` is an ordinary channel. Nothing gates on it; it exists so the host can measure
  dropped frames and jitter against the device's own clock.

Prefer ``t_src`` on a long frame. float32 quantises: once its ulp exceeds the frame period,
adjacent timestamps round to the same value, ``t - last_t`` reads 0 < ``DT_MIN`` and a healthy
stream is declared out of sync. That happens at t >= 2**17 s (~36 h) at 100 Hz and ~18 h at
200 Hz. A long frame does not need the help anyway: a 2-byte header matching at 9 consecutive
frame boundaries is a ~(1/65536)**9 false lock.

Byte alignment has two mechanisms and needs at least one. With ``t``, it is recovered from the
last float increasing monotonically — a misaligned read is virtually never 8 consecutive
plausible step sizes — so the header is optional. Without ``t`` the header is the only thing to
lock onto and is therefore **required**; alignment means finding a phase where it appears at 8
consecutive frame boundaries, and a header that later goes missing forces a re-sync exactly as a
timestamp jump does. Neither mechanism looks at what the blocks mean, so adding a block costs
nothing here beyond a longer frame.

Channel order is not detectable from the bytes and getting it wrong is silent, so it is declared
rather than guessed: gravity landing in the gyro channels degrades the classifier without any
error. Whatever the layout, this module always yields each block under its own key, and a block
the layout does not carry is reported as None rather than omitted — so a consumer can tell "this
link has no knee channels" from "this frame happened to be empty".

This module is only the decoder: ``read_frames`` is a silent generator over the frame dict — no
files, no printing, no threads. ``serial_service.SerialImuService`` is what turns it into a
sensor source the web server, recorder and CLS can consume.

Two entry points, differing only in who owns the port. ``read_frames`` opens and closes it, which
is all a read-only consumer needs. ``decode_frames`` takes an already-open port and leaves it
open, so a caller that also has to *write* to the device — sending inference results back to the
Arduino — can hold the single handle and share it between its reader thread and its writer.
"""

from __future__ import annotations

import struct
import threading
from collections.abc import Iterator
from typing import NamedTuple

import serial

FLOAT_SIZE = 4
SYNC_FRAMES = 8                 # consecutive good frames required to call a phase aligned
DT_MIN, DT_MAX = 0.001, 0.5     # plausible interval between adjacent timestamps
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200

# Channel block -> how many float32 it occupies. A layout is an ordered tuple of these names,
# so a frame's meaning is fully described by its block list and nothing else.
BLOCK_SIZES: dict[str, int] = {
    "enable": 1,
    "accel": 3,
    "gyro": 3,
    "knee4": 4,
    "fb6": 6,
    "trace5": 5,
    "t_src": 1,
    "t": 1,
}

# Same quantity, different role (see the module docstring). A layout may carry at most one.
_CLOCK_BLOCKS = ("t", "t_src")

# Canonical channel order inside the composite blocks, for callers that want to name a column.
KNEE4_CHANNELS = ("ang_r", "vel_r", "ang_l", "vel_l")
FB6_CHANNELS = ("pos_r", "speed_r", "torque_r", "pos_l", "speed_l", "torque_l")
TRACE5_CHANNELS = ("finalClass", "LU_AVEL_F", "L_KVEL", "L_KWRAP", "MotorCom_L")


class Layout(NamedTuple):
    """One frame format. Built by ``_layout`` from a block list — never written out by hand.

    Fields:
        n_floats: int, total float32 in the frame, header excluded. Deliberately field 0:
                  ``others/tools/serial_monitor.py`` reads it positionally as ``[0]``.
        has_ts:   bool, True when the last block is ``t`` — i.e. when sync may key on it.
        blocks:   tuple[str, ...], the declared wire order.
        offsets:  dict[str, int], block name -> index of its first float within the frame.
                  Precomputed so ``decode_frames`` never re-derives it per frame.
    """

    n_floats: int
    has_ts: bool
    blocks: tuple[str, ...]
    offsets: dict[str, int]


def _layout(*blocks: str) -> Layout:
    """Build a ``Layout`` from its block names, deriving every other field.

    Args:
        *blocks: str, block names in wire order; each should be a key of ``BLOCK_SIZES``
                 (an unknown name contributes 0 floats here and is rejected by
                 ``_check_layouts``, which runs once the whole table exists).

    Returns:
        Layout. Because n_floats/has_ts/offsets are all computed here, a table entry cannot
        disagree with itself — the class of bug where a hand-written float count drifts away
        from the channel list is unrepresentable rather than merely tested for.
    """
    offsets: dict[str, int] = {}
    pos = 0
    for b in blocks:
        offsets[b] = pos
        pos += BLOCK_SIZES.get(b, 0)
    return Layout(
        n_floats=pos,
        has_ts=bool(blocks) and blocks[-1] == "t",
        blocks=tuple(blocks),
        offsets=offsets,
    )


# layout name -> frame format. The four original names keep their exact previous meaning.
LAYOUTS: dict[str, Layout] = {
    # --- IMU only: a plain BNO055 forwarder --------------------------------------------
    "accel_gyro_t": _layout("accel", "gyro", "t"),
    "gyro_accel_t": _layout("gyro", "accel", "t"),
    "accel_gyro": _layout("accel", "gyro"),
    "gyro_accel": _layout("gyro", "accel"),
    # --- the exoskeleton rig's full uplink: IMU + both knees + motor feedback + trace ---
    # 23 floats = 94 bytes with the 2-byte header. Header-only sync by design; the clock
    # rides along as data (``t_src``) so nothing gates on float32 quantisation.
    "exo_v1": _layout("enable", "gyro", "accel", "knee4", "fb6", "trace5", "t_src"),
}
DEFAULT_LAYOUT = "accel_gyro_t"

PAYLOAD = 7 * FLOAT_SIZE        # frame payload of the default layout, kept for reference


def _check_layouts() -> None:
    """Validate every ``LAYOUTS`` entry; called once at import.

    Args:    none — reads the module-level ``LAYOUTS``.
    Returns: None. Raises ValueError naming the first bad entry.

    A wrong block list shifts every channel after the mistake, and on the wire that failure is
    completely silent: the classifier merely degrades and the plots merely show the wrong
    column. Refusing to import is the point — this module's standing rule is that frame meaning
    is declared rather than guessed, so a declaration that cannot be true must not load.
    """
    for name, lay in LAYOUTS.items():
        unknown = [b for b in lay.blocks if b not in BLOCK_SIZES]
        if unknown:
            raise ValueError(f"layout {name!r}: unknown block(s) {unknown}")
        if len(set(lay.blocks)) != len(lay.blocks):
            raise ValueError(f"layout {name!r}: repeated block in {lay.blocks}")
        if "t" in lay.blocks and lay.blocks[-1] != "t":
            raise ValueError(
                f"layout {name!r}: 't' must be the last block — alignment reads it as v[-1]"
            )
        if sum(1 for b in lay.blocks if b in _CLOCK_BLOCKS) > 1:
            raise ValueError(
                f"layout {name!r}: at most one of {_CLOCK_BLOCKS} may appear — they are the "
                f"same quantity, differing only in whether sync keys on it"
            )
        if lay.n_floats != sum(BLOCK_SIZES[b] for b in lay.blocks):  # pragma: no cover
            raise ValueError(f"layout {name!r}: n_floats disagrees with its blocks")


_check_layouts()


def _frame_at(b, i: int, magic: bytes, frame_len: int, fmt: str):
    """The floats of the frame at ``b[i:i+frame_len]``; None if a header is set and misses."""
    hdr = len(magic)
    if magic and b[i:i + hdr] != magic:
        return None
    return struct.unpack(fmt, bytes(b[i + hdr:i + frame_len]))


def _find_offset(b, magic: bytes, frame_len: int, fmt: str, has_ts: bool) -> int | None:
    """Index of the first whole frame in ``b``, or None if no phase aligns.

    A phase qualifies when SYNC_FRAMES+1 consecutive frames decode: with a header that means it
    is present at every frame boundary, and with a timestamp that the steps all fall inside
    (DT_MIN, DT_MAX). Formats carrying both must satisfy both, which locks on fastest."""
    sync_bytes = frame_len * (SYNC_FRAMES + 1)
    for off in range(frame_len):
        if len(b) - off < sync_bytes:
            break
        vals = []
        for k in range(SYNC_FRAMES + 1):
            v = _frame_at(b, off + k * frame_len, magic, frame_len, fmt)
            if v is None:
                break
            vals.append(v)
        if len(vals) != SYNC_FRAMES + 1:
            continue
        if not has_ts:
            return off          # the header matched at every boundary — nothing else to check
        ts = [v[-1] for v in vals]
        if all(DT_MIN < y - x < DT_MAX for x, y in zip(ts, ts[1:])):
            return off
    return None


def decode_frames(
    ser: "serial.Serial",
    *,
    magic: str = "",
    layout: str = DEFAULT_LAYOUT,
    stop: threading.Event | None = None,
) -> Iterator[dict]:
    """Yield one frame dict per decoded frame off an **already open** port.

    Args:
        ser:    serial.Serial, already open. Never closed here — see below.
        magic:  str, frame header as hex (e.g. "aa55"); "" means no header, which only a
                layout carrying ``t`` can align without.
        layout: str, a key of ``LAYOUTS``.
        stop:   threading.Event | None. Set it to end the generator; None runs forever.

    Yields:
        dict carrying one key per block the layout *could* hold, so the shape is stable
        across layouts:
          "t":        float | None            -- sync clock, seconds (the ``t`` block)
          "t_src":    float | None            -- data clock, seconds (the ``t_src`` block)
          "accel":    list[float] len 3 | None -- ax ay az, m/s^2, gravity included
          "gyro":     list[float] len 3 | None -- gx gy gz, device units, unconverted
          "joints":   list[float] len 4 | None -- ang_r vel_r ang_l vel_l, device units
          "feedback": list[float] len 6 | None -- pos/speed/torque per side, device units
          "trace":    list[float] len 5 | None -- finalClass LU_AVEL_F L_KVEL L_KWRAP MotorCom_L
          "enable":   float | None             -- SWITCH line; non-zero = may drive
        A None always means "this layout carries no such block", never "the value was
        missing" — a frame that decodes at all has every channel its layout declares.

    The port belongs to the caller: it is never closed here, so the caller can keep writing to it
    (or reopen on its own schedule). Loss of sync (header mismatch or a timestamp jump) silently
    re-runs alignment and keeps going, so the generator only ends when ``stop`` is set or the
    caller closes it.
    """
    try:
        lay = LAYOUTS[layout]
    except KeyError:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {', '.join(sorted(LAYOUTS))}"
        ) from None
    n_floats, has_ts = lay.n_floats, lay.has_ts
    magic_b = bytes.fromhex(magic) if magic else b""
    if not has_ts and not magic_b:
        raise ValueError(
            f"layout {layout!r} has no sync timestamp, so a header is the only way to find the "
            f"frame boundaries — set [source] magic (e.g. \"aa55\")"
        )
    fmt = f"<{n_floats}f"
    frame_len = len(magic_b) + n_floats * FLOAT_SIZE
    sync_bytes = frame_len * (SYNC_FRAMES + 1)
    # Resolve block positions once per call rather than per frame. An offset of 0 is a valid
    # position, so every test below is against None, never falsiness.
    off_accel = lay.offsets.get("accel")
    off_gyro = lay.offsets.get("gyro")
    off_knee = lay.offsets.get("knee4")
    off_fb = lay.offsets.get("fb6")
    off_trace = lay.offsets.get("trace5")
    off_enable = lay.offsets.get("enable")
    off_tsrc = lay.offsets.get("t_src")

    ser.reset_input_buffer()
    buf = bytearray()
    synced = False
    last_t = None
    while stop is None or not stop.is_set():
        chunk = ser.read(max(1, ser.in_waiting))
        if chunk:
            buf += chunk

        if not synced:
            if len(buf) < sync_bytes:
                continue
            off = _find_offset(buf, magic_b, frame_len, fmt, has_ts)
            if off is None:
                del buf[:len(buf) - sync_bytes + 1]   # every phase in here is ruled out
                continue
            del buf[:off]
            synced = True
            last_t = None

        while len(buf) >= frame_len:
            v = _frame_at(buf, 0, magic_b, frame_len, fmt)
            if v is None:                     # only reachable with a header configured
                synced = False
                break
            t = v[-1] if has_ts else None
            # Timestamp sanity. Don't consume the frame — let _find_offset decide whether
            # bytes were lost or the source's clock just restarted.
            if last_t is not None and not (DT_MIN < t - last_t < DT_MAX):
                synced = False
                break
            del buf[:frame_len]
            last_t = t
            yield {
                "t": t,
                "t_src": float(v[off_tsrc]) if off_tsrc is not None else None,
                "accel": list(v[off_accel:off_accel + 3]) if off_accel is not None else None,
                "gyro": list(v[off_gyro:off_gyro + 3]) if off_gyro is not None else None,
                "joints": list(v[off_knee:off_knee + 4]) if off_knee is not None else None,
                "feedback": list(v[off_fb:off_fb + 6]) if off_fb is not None else None,
                "trace": list(v[off_trace:off_trace + 5]) if off_trace is not None else None,
                "enable": float(v[off_enable]) if off_enable is not None else None,
            }


def read_frames(
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    *,
    magic: str = "",
    layout: str = DEFAULT_LAYOUT,
    stop: threading.Event | None = None,
) -> Iterator[dict]:
    """``decode_frames`` on a port this function opens and closes.

    Args:
        port:   str, device path or COM name (e.g. "/dev/ttyACM0", "COM17").
        baud:   int, bits per second; must match the device.
        magic:  str, frame header as hex, as in ``decode_frames``.
        layout: str, a key of ``LAYOUTS``.
        stop:   threading.Event | None. Set it to end the generator.

    Yields:
        dict, exactly the shape ``decode_frames`` yields.

    Opening is lazy (generator semantics): the port opens on the first item and is closed when the
    generator ends — whether that is ``stop`` being set or the caller closing it. Use this for a
    read-only consumer; a caller that must also *write* to the device wants ``decode_frames`` so
    it can own the single handle.

        for f in read_frames("COM17", 115200, magic="aa55", layout="gyro_accel"):
            print(f["accel"], f["gyro"])
    """
    ser = serial.Serial(port, baud, timeout=1)
    try:
        yield from decode_frames(ser, magic=magic, layout=layout, stop=stop)
    finally:
        ser.close()
