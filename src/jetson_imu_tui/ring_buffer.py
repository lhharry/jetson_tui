"""Thread-safe ring buffer of timestamped IMU samples + shared unit constant.

One ``RingBuffer`` per sensor is filled by the sampler thread in ``imu_service`` at the
configured ``sample_hz``; the recorder, the web server and the CLS service all consume
from these buffers instead of issuing their own I2C transactions. Samples are stored
raw (no zero/tare offset) — consumers apply the offset where needed.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right
from collections import deque

RAD_TO_DEG = 180.0 / math.pi

# 2048 samples ≈ 20 s at 100 Hz — comfortably more than the browser's poll interval
# and the plot window's needs after a dropped poll or two.
DEFAULT_MAXLEN = 2048


class RingBuffer:
    """Fixed-capacity FIFO of samples ``{"t": monotonic, "euler", "accel", "gyro", "quat"}``.

    Appends come from a single sampler thread; reads come from the recorder, web and CLS
    threads, so every access is guarded by a lock. ``since()``/``nearest()`` rely on ``t``
    being monotonically increasing, which ``time.monotonic()`` guarantees per process.
    """

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: dict) -> None:
        with self._lock:
            self._buf.append(sample)

    def latest(self) -> dict | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def since(self, t: float, limit: int | None = None) -> list[dict]:
        """Samples with ``sample["t"] > t``, oldest first; at most the newest ``limit``."""
        with self._lock:
            ts = [s["t"] for s in self._buf]
            idx = bisect_right(ts, t)
            out = list(self._buf)[idx:]
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        return out

    def nearest(self, t: float) -> dict | None:
        """The sample whose timestamp is closest to ``t`` (None if empty)."""
        with self._lock:
            if not self._buf:
                return None
            ts = [s["t"] for s in self._buf]
            idx = bisect_right(ts, t)
            candidates = []
            if idx > 0:
                candidates.append(self._buf[idx - 1])
            if idx < len(self._buf):
                candidates.append(self._buf[idx])
            return min(candidates, key=lambda s: abs(s["t"] - t))

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
