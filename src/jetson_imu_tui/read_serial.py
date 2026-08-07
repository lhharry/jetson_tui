"""Decoder for the BNO055 binary frames an Arduino / Simulink model streams over serial.

Wire format is 7 float32 per frame (ax ay az gx gy gz t). The header is optional: a Simulink
Serial Transmit block with Header [90 165] means 5A A5, empty means no header. Without one,
byte alignment is found from the timestamp (last float) increasing monotonically — a misaligned
read is virtually never 8 consecutive plausible step sizes.

This module is only the decoder: ``read_frames`` is a silent generator over
``{"t", "accel", "gyro"}`` — no files, no printing, no threads. ``serial_service.SerialImuService``
is what turns it into a sensor source the web server, recorder and CLS can consume.
"""

from __future__ import annotations

import struct
import threading
from collections.abc import Iterator

import serial

PAYLOAD = 28                    # 7 float32
FMT = "<7f"
SYNC_FRAMES = 8                 # consecutive valid timestamp steps required to call it aligned
DT_MIN, DT_MAX = 0.001, 0.5     # plausible interval between adjacent timestamps
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200


def _frame_at(b, i: int, magic: bytes, frame_len: int):
    """The 7 floats of the frame at ``b[i:i+frame_len]``; None if a header is set and misses."""
    hdr = len(magic)
    if magic and b[i:i + hdr] != magic:
        return None
    return struct.unpack(FMT, bytes(b[i + hdr:i + frame_len]))


def _find_offset(b, magic: bytes, frame_len: int) -> int | None:
    """Index of the first whole frame in ``b``, or None if no phase aligns.

    Alignment means SYNC_FRAMES consecutive timestamp steps inside (DT_MIN, DT_MAX); with a
    header ``_frame_at`` also checks it, which locks on faster."""
    sync_bytes = frame_len * (SYNC_FRAMES + 1)
    for off in range(frame_len):
        if len(b) - off < sync_bytes:
            break
        ts = []
        for k in range(SYNC_FRAMES + 1):
            v = _frame_at(b, off + k * frame_len, magic, frame_len)
            if v is None:
                break
            ts.append(v[6])
        if len(ts) == SYNC_FRAMES + 1 and all(DT_MIN < y - x < DT_MAX for x, y in zip(ts, ts[1:])):
            return off
    return None


def read_frames(
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    *,
    magic: str = "",
    stop: threading.Event | None = None,
) -> Iterator[dict]:
    """Yield one ``{"t", "accel": [ax,ay,az], "gyro": [gx,gy,gz]}`` per frame, ``t`` in seconds
    on the source's own clock.

    ``magic`` is the header as a hex string (e.g. "5aa5"); empty means no header. Loss of sync
    (header mismatch or a timestamp jump) silently re-runs alignment and keeps going, so the
    generator only ends when ``stop`` is set or the caller closes it — the port is closed either
    way. Opening is lazy (generator semantics): the port opens on the first item.

        for f in read_frames("COM5", 115200):
            print(f["t"], f["accel"], f["gyro"])
    """
    magic_b = bytes.fromhex(magic) if magic else b""
    frame_len = len(magic_b) + PAYLOAD
    sync_bytes = frame_len * (SYNC_FRAMES + 1)

    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()
    buf = bytearray()
    synced = False
    last_t = None
    try:
        while stop is None or not stop.is_set():
            chunk = ser.read(max(1, ser.in_waiting))
            if chunk:
                buf += chunk

            if not synced:
                if len(buf) < sync_bytes:
                    continue
                off = _find_offset(buf, magic_b, frame_len)
                if off is None:
                    del buf[:len(buf) - sync_bytes + 1]   # every phase in here is ruled out
                    continue
                del buf[:off]
                synced = True
                last_t = None

            while len(buf) >= frame_len:
                v = _frame_at(buf, 0, magic_b, frame_len)
                if v is None:                     # only reachable with a header configured
                    synced = False
                    break
                ax, ay, az, gx, gy, gz, t = v
                # Timestamp sanity. Don't consume the frame — let _find_offset decide whether
                # bytes were lost or the source's clock just restarted.
                if last_t is not None and not (DT_MIN < t - last_t < DT_MAX):
                    synced = False
                    break
                del buf[:frame_len]
                last_t = t
                yield {"t": t, "accel": [ax, ay, az], "gyro": [gx, gy, gz]}
    finally:
        ser.close()
