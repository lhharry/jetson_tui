"""Decoder for the BNO055 binary frames an Arduino / Simulink model streams over serial.

A frame is an optional header followed by 6 or 7 little-endian float32, described by ``layout``:

===============  =====================  ==========================================
layout           floats                 typical source
===============  =====================  ==========================================
accel_gyro_t     ax ay az gx gy gz t    Simulink Serial Transmit, header 5A A5
gyro_accel_t     gx gy gz ax ay az t    as above, channels swapped
accel_gyro       ax ay az gx gy gz      Arduino sketch, no clock
gyro_accel       gx gy gz ax ay az      Arduino sketch, no clock, header AA 55
===============  =====================  ==========================================

Byte alignment has two mechanisms and needs at least one. With a timestamp, it is recovered from
the last float increasing monotonically — a misaligned read is virtually never 8 consecutive
plausible step sizes — so the header is optional. Without a timestamp the header is the only
thing to lock onto and is therefore **required**; alignment means finding a phase where it
appears at 8 consecutive frame boundaries, and a header that later goes missing forces a re-sync
exactly as a timestamp jump does.

Channel order is not detectable from the bytes and getting it wrong is silent, so it is declared
rather than guessed: gravity landing in the gyro channels degrades the classifier without any
error. Whatever the layout, this module always yields accel and gyro under their own keys.

This module is only the decoder: ``read_frames`` is a silent generator over
``{"t", "accel", "gyro"}`` — no files, no printing, no threads. ``serial_service.SerialImuService``
is what turns it into a sensor source the web server, recorder and CLS can consume.

Two entry points, differing only in who owns the port. ``read_frames`` opens and closes it, which
is all a read-only consumer needs. ``decode_frames`` takes an already-open port and leaves it
open, so a caller that also has to *write* to the device — sending inference results back to the
Arduino — can hold the single handle and share it between its reader thread and its writer.
"""

from __future__ import annotations

import struct
import threading
from collections.abc import Iterator

import serial

FLOAT_SIZE = 4
SYNC_FRAMES = 8                 # consecutive good frames required to call a phase aligned
DT_MIN, DT_MAX = 0.001, 0.5     # plausible interval between adjacent timestamps
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200

# layout name -> (float count, carries a source timestamp, gyro comes before accel)
LAYOUTS: dict[str, tuple[int, bool, bool]] = {
    "accel_gyro_t": (7, True, False),
    "gyro_accel_t": (7, True, True),
    "accel_gyro": (6, False, False),
    "gyro_accel": (6, False, True),
}
DEFAULT_LAYOUT = "accel_gyro_t"

PAYLOAD = 7 * FLOAT_SIZE        # frame payload of the default layout, kept for reference


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
    """Yield one ``{"t", "accel": [ax,ay,az], "gyro": [gx,gy,gz]}`` per frame off an **already
    open** port. ``t`` is seconds on the source's own clock, or None for a layout without one.

    The port belongs to the caller: it is never closed here, so the caller can keep writing to it
    (or reopen on its own schedule). ``magic`` is the header as a hex string (e.g. "aa55"); empty
    means no header, which only a layout carrying a timestamp can align without. Loss of sync
    (header mismatch or a timestamp jump) silently re-runs alignment and keeps going, so the
    generator only ends when ``stop`` is set or the caller closes it.
    """
    try:
        n_floats, has_ts, gyro_first = LAYOUTS[layout]
    except KeyError:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {', '.join(sorted(LAYOUTS))}"
        ) from None
    magic_b = bytes.fromhex(magic) if magic else b""
    if not has_ts and not magic_b:
        raise ValueError(
            f"layout {layout!r} carries no timestamp, so a header is the only way to find the "
            f"frame boundaries — set [source] magic (e.g. \"aa55\")"
        )
    fmt = f"<{n_floats}f"
    frame_len = len(magic_b) + n_floats * FLOAT_SIZE
    sync_bytes = frame_len * (SYNC_FRAMES + 1)

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
            first, second = list(v[0:3]), list(v[3:6])
            accel, gyro = (second, first) if gyro_first else (first, second)
            yield {"t": t, "accel": accel, "gyro": gyro}


def read_frames(
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    *,
    magic: str = "",
    layout: str = DEFAULT_LAYOUT,
    stop: threading.Event | None = None,
) -> Iterator[dict]:
    """``decode_frames`` on a port this function opens and closes.

    Opening is lazy (generator semantics): the port opens on the first item and is closed when the
    generator ends — whether that is ``stop`` being set or the caller closing it.

        for f in read_frames("COM17", 115200, magic="aa55", layout="gyro_accel"):
            print(f["accel"], f["gyro"])
    """
    ser = serial.Serial(port, baud, timeout=1)
    try:
        yield from decode_frames(ser, magic=magic, layout=layout, stop=stop)
    finally:
        ser.close()
