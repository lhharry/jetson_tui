"""ClsService — background 10 Hz block-averaging sampler + sliding-window inference.

Every ~1/``target_hz`` s it pulls *all* raw (gravity-inclusive, tare-bypassed) 100 Hz
samples that arrived since the last tick via ``ImuService.raw_samples_since`` and
block-averages them into one 10 Hz vector. This mirrors training's anti-aliasing
downsample (``dataset/jetson_leg.down_sample``): plain decimation (one instantaneous
sample per tick) would alias >5 Hz energy and feed the model out-of-distribution input,
hurting the dynamic classes (jog / stairs) most. It keeps a rolling window of the last
``window`` averaged vectors and, every ``stride`` new vectors, runs the vendored BERT
classifier, appending a timestamped result to a capped log the web layer polls via
``GET /cls``. A time gap between consecutive averaged vectors (sensor stall / reconnect)
clears the window so inference never runs across a discontinuity.

The web UI can ``pause()``/``resume()`` the service at runtime (``POST /cls/toggle``):
while paused the loop idles without pulling samples or running the model, so inference
stops competing with the sampler threads; the checkpoint stays loaded for instant resume.

Fails safe: if ``torch`` or the checkpoint is missing, the service stays ``enabled=False``
and never touches the sensor, so the rest of the TUI is unaffected.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.ring_buffer import RingBuffer

if TYPE_CHECKING:  # ImuService pulls in the Linux-only hardware stack; only needed for hints.
    from jetson_imu_tui.imu_service import ImuService


class ClsService:
    def __init__(
        self,
        service: "ImuService",
        model_path: Path | str,
        *,
        sensor: str = "Left",
        target_hz: float = 10.0,
        window: int = 20,
        stride: int = 10,
        log_size: int = 500,
    ) -> None:
        self._service = service
        self._model_path = Path(model_path)
        self._sensor = sensor
        self._period = 1.0 / float(target_hz)
        self._window = int(window)
        self._stride = int(stride)
        # Clear the rolling window if consecutive averaged vectors are more than this far
        # apart (a stalled/reconnecting sensor) so a window never spans a discontinuity.
        self._gap_reset = 2.5 * self._period

        self._clf = None
        self._enabled = False
        self._reason = "not started"

        # Runtime switch (web UI): while paused the loop idles — no raw-sample pulls, no
        # inference — so CLS stops competing with the sampler threads for CPU.
        self._paused = False
        self._cursor_reset = threading.Event()  # tells the loop to skip the paused backlog

        self._buf: deque[list[float]] = deque(maxlen=self._window)
        # Every 6-channel vector fed to the model, timestamped (monotonic). The recorder
        # drains this into model_input.csv so a recording captures the exact model input.
        self._input_buf = RingBuffer()
        self._since_pred = 0
        self._log: deque[dict] = deque(maxlen=int(log_size))
        self._current: dict | None = None
        self._next_id = 1
        self._log_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Load the model and start the sampler thread. Self-disables on any failure."""
        if not self._model_path.exists():
            self._reason = f"checkpoint not found: {self._model_path}"
            logger.warning(f"CLS disabled — {self._reason}")
            return
        try:
            from jetson_imu_tui.cls.classifier import ActivityClassifier

            self._clf = ActivityClassifier(self._model_path)
        except Exception as err:  # torch missing / bad checkpoint / etc.
            self._reason = f"model load failed: {err}"
            logger.warning(f"CLS disabled — {self._reason}")
            return
        self._enabled = True
        self._reason = "ok"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"CLS enabled on '{self._sensor}' (device={self._clf.device})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # --- sampling + inference ---------------------------------------------
    def _loop(self) -> None:
        next_tick = time.monotonic()
        cursor = time.monotonic()  # only consume raw samples newer than this
        last_t: float | None = None  # monotonic time of the last accepted averaged vector
        while not self._stop.is_set():
            if self._cursor_reset.is_set():
                self._cursor_reset.clear()
                cursor = time.monotonic()
                last_t = None
                self._buf.clear()  # a tick racing pause() may have appended one vector
                self._since_pred = 0
            if self._paused:
                batch = []
            else:
                batch = self._service.raw_samples_since(self._sensor, cursor)
            if batch:
                cursor = batch[-1]["t"]
                # Block-average the batch → one 10 Hz vector (matches down_sample).
                acc = np.mean([s["accel"] for s in batch], axis=0)
                gyr = np.mean([s["gyro"] for s in batch], axis=0)
                sample_t = batch[-1]["t"]
                if last_t is not None and (sample_t - last_t) > self._gap_reset:
                    self._buf.clear()  # discontinuity — never window across a stall
                last_t = sample_t
                self._buf.append([acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2]])
                self._input_buf.append(
                    {"t": sample_t, "acc": [float(v) for v in acc], "gyr": [float(v) for v in gyr]}
                )
                self._since_pred += 1
                if self._since_pred >= self._stride and len(self._buf) >= self._window:
                    self._since_pred = 0
                    self._infer()
            next_tick += self._period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                next_tick = time.monotonic()  # fell behind — resync

    def _infer(self) -> None:
        window = np.asarray(self._buf, dtype=np.float32)  # (window, 6)
        try:
            cls_name, conf, probs = self._clf.predict(window)
        except Exception as err:  # pragma: no cover - runtime safety
            logger.warning(f"CLS inference error: {err}")
            return
        entry = {
            "id": self._next_id,
            "t": time.time(),
            "clock": datetime.now().strftime("%H:%M:%S"),
            "cls": cls_name,
            "conf": conf,
            "probs": [float(p) for p in probs],
        }
        with self._log_lock:
            self._next_id += 1
            self._log.append(entry)
            self._current = entry

    # --- runtime switch ------------------------------------------------------
    def pause(self) -> None:
        """Suspend sampling + inference (model stays loaded). Idempotent."""
        with self._log_lock:
            self._paused = True
            self._current = None  # don't show / record a stale prediction
        self._buf.clear()
        self._since_pred = 0

    def resume(self) -> None:
        """Resume sampling + inference, skipping everything buffered while paused."""
        self._cursor_reset.set()
        with self._log_lock:
            self._paused = False

    def toggle_running(self) -> bool:
        """Flip paused/running; returns True if now running."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return not self._paused

    @property
    def running(self) -> bool:
        return self._enabled and not self._paused

    # --- accessors ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def classes(self) -> list[str]:
        """Label order the model emits probabilities in (matches ``predict`` / CLASSES)."""
        return list(CLASSES)

    def current(self) -> dict | None:
        """Thread-safe copy of the latest prediction (``cls``/``conf``/``probs``), or None.

        The recorder polls this per drain to persist the held prediction at 100 Hz."""
        with self._log_lock:
            return dict(self._current) if self._current else None

    def inputs_since(self, t: float, limit: int | None = None) -> list[dict]:
        """Model-input samples ``{"t","acc","gyr"}`` newer than monotonic ``t``, oldest first.

        These are the exact 6-channel vectors fed to the model (raw accel+gyro of the CLS
        sensor at the model's ``target_hz``); the recorder drains them into model_input.csv."""
        return self._input_buf.since(t, limit=limit)

    # --- web accessor ------------------------------------------------------
    def snapshot(self, since: int = 0) -> dict:
        """Payload for GET /cls: enabled flag, current prediction, entries after ``since``."""
        if not self._enabled:
            return {"enabled": False, "reason": self._reason, "current": None, "entries": []}
        with self._log_lock:
            entries = [e for e in self._log if e["id"] > since]
            current = dict(self._current) if self._current else None
            running = not self._paused
        return {
            "enabled": True,
            "running": running,
            "sensor": self._sensor,
            "current": current,
            "entries": entries,
        }
