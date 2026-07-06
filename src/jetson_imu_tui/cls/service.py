"""ClsService — background 10 Hz sampler + sliding-window activity inference.

Samples one IMU (the ``sensor`` label) at a fixed 10 Hz via ``ImuService.read_raw`` (raw,
gravity-inclusive), keeps a rolling window of the last ``window`` samples, and every
``stride`` new samples runs the vendored BERT classifier, appending a timestamped result
to a capped log the web layer polls through ``GET /cls``.

Fails safe: if ``torch`` or the checkpoint is missing, the service stays ``enabled=False``
and never touches the sensor, so the rest of the TUI is unaffected.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.imu_service import ImuService


class ClsService:
    def __init__(
        self,
        service: ImuService,
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

        self._clf = None
        self._enabled = False
        self._reason = "not started"

        self._buf: deque[list[float]] = deque(maxlen=self._window)
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
        while not self._stop.is_set():
            raw = self._service.read_raw(self._sensor)
            if raw is not None:
                acc, gyr = raw["accel"], raw["gyro"]
                self._buf.append([acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2]])
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

    # --- web accessor ------------------------------------------------------
    def snapshot(self, since: int = 0) -> dict:
        """Payload for GET /cls: enabled flag, current prediction, entries after ``since``."""
        if not self._enabled:
            return {"enabled": False, "reason": self._reason, "current": None, "entries": []}
        with self._log_lock:
            entries = [e for e in self._log if e["id"] > since]
            current = dict(self._current) if self._current else None
        return {"enabled": True, "sensor": self._sensor, "current": current, "entries": entries}
