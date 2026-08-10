"""ClsService — background block-averaging sampler + sliding-window inference.

Raw (gravity-inclusive, tare-bypassed) samples are pulled from ``raw_samples_since`` and
block-averaged ``decim = sample_hz / target_hz`` at a time into one model-rate vector. That one
method is the entire contract with the sensor source (``imu_common.SensorSource``), so either
source works unchanged: ``ImuService`` over I2C or ``SerialImuService`` over a serial link.
This mirrors training's anti-aliasing downsample (``dataset/jetson_leg.down_sample``): plain
decimation (one instantaneous sample per tick) would alias >5 Hz energy and feed the model
out-of-distribution input, hurting the dynamic classes (jog / stairs) most.

Grouping is driven by *raw sample count*, not by tick timing: the CLS tick and the sampler
thread drift independently, so a tick can deliver 9, 11 or (after a stall) ~20 samples. Only
a full ``decim`` samples ever form a vector, so a late tick yields two correct vectors rather
than one over-wide one. A raw-resolution gap check drops the whole window at a discontinuity
so inference never runs across a stall.

Per-frame predictions are not the service's output. They are pushed through an injected
``aggregator`` (``cls.vote.SoftVoter``) which averages several frames into one stable
**decision** — frame-level predictions are too noisy for a downstream controller to act on
directly. Only decisions reach ``on_result``, the sink that writes the class index back to the
device. The aggregator is only ever asked to ``push`` and ``reset``, so this module knows nothing
about how it aggregates, and nothing about where the result goes.

The web UI can ``pause()``/``resume()`` the service at runtime (``POST /cls/toggle``):
while paused the loop idles without pulling samples or running the model, so inference
stops competing with the sampler threads; the checkpoint stays loaded for instant resume.
``set_source`` likewise re-points the service at a different sensor source (the web UI switching
between the I2C IMUs and a serial one) without reloading the checkpoint.
All buffer mutation happens on the loop thread (``pause``/``resume``/``set_source`` only signal
via ``_cursor_reset``), so the window can never be cleared mid-inference by a web request.

Fails safe: if ``torch`` or the checkpoint is missing, or ``sample_hz`` is not an integer
multiple of ``target_hz``, the service stays ``enabled=False`` and never touches the
sensor, so the rest of the TUI is unaffected.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.cls.vote import SoftVoter
from jetson_imu_tui.ring_buffer import RingBuffer

if TYPE_CHECKING:  # hint only — importing a concrete source here would drag in its deps.
    from jetson_imu_tui.imu_common import SensorSource

# Clear the rolling window if consecutive *raw* samples are further apart than this.
# Count-based grouping already keeps every group at exactly ``decim`` samples, so this
# guard only needs to catch true discontinuities (sensor stall / reconnect / resume —
# hundreds of ms and up), not I2C jitter (tens of ms), which merely stretches a group
# by a sample or two. 100 ms sits between the two regimes.
MAX_RAW_GAP_S = 0.1


class ClsService:
    def __init__(
        self,
        service: "SensorSource",
        model_path: Path | str,
        *,
        sensor: str = "Left",
        sample_hz: float = 100.0,
        target_hz: float = 10.0,
        window: int = 20,
        stride: int = 1,
        log_size: int = 3000,
        aggregator: SoftVoter | None = None,
        on_result: Callable[[int], None] | None = None,
    ) -> None:
        self._service = service
        self._model_path = Path(model_path)
        self._sensor = sensor
        self._sample_hz = float(sample_hz)
        self._target_hz = float(target_hz)
        self._period = 1.0 / self._target_hz
        self._window = int(window)
        self._stride = int(stride)
        # Raw samples per model-rate vector (100 Hz / 10 Hz = 10). ``start()`` refuses to run
        # unless the ratio is an exact integer — the reshape in ``_infer`` relies on it.
        self._decim = max(1, int(round(self._sample_hz / self._target_hz)))

        # Frame predictions -> stable decisions. Injected so the scheme is swappable; the
        # default (window=1) is an exact passthrough, i.e. one decision per inference.
        self._agg = aggregator if aggregator is not None else SoftVoter(window=1, emit_every=1)
        self._on_result = on_result

        self._clf = None
        self._enabled = False
        self._reason = "not started"

        # Runtime switch (web UI): while paused the loop idles — no raw-sample pulls, no
        # inference — so CLS stops competing with the sampler threads for CPU.
        self._paused = False
        self._cursor_reset = threading.Event()  # tells the loop to drop cursor + window
        # Sensor source swap requested by a web thread, applied by the loop thread (see
        # ``set_source``). Guarded by ``_log_lock``.
        self._pending_source: tuple["SensorSource", float] | None = None

        # Rolling *raw* window: exactly window*decim samples, i.e. the span one inference
        # needs. Kept un-averaged so every window is rebuilt on a clean grid and can never
        # inherit a malformed group. Only ever touched by the loop thread.
        self._raw: deque[list[float]] = deque(maxlen=self._window * self._decim)
        self._group: list[list[float]] = []  # raw samples accumulating into the next vector
        self._last_raw_t: float | None = None
        self._groups_since_pred = 0
        # Every 6-channel vector fed to the model, timestamped (monotonic). The recorder
        # drains this into model_input.csv so a recording captures the exact model input.
        self._input_buf = RingBuffer()
        # Every aggregated decision, timestamped (monotonic) — the recorder drains this into
        # cls_vote.csv, giving a recording both the frame-level and the post-vote stream.
        self._decision_buf = RingBuffer()
        self._log: deque[dict] = deque(maxlen=int(log_size))
        self._current: dict | None = None
        self._current_decision: dict | None = None
        self._next_id = 1
        self._log_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Load the model and start the sampler thread. Self-disables on any failure."""
        if abs(self._sample_hz / self._target_hz - self._decim) > 1e-9:
            self._reason = (
                f"sample_hz ({self._sample_hz:g}) must be an integer multiple of "
                f"target_hz ({self._target_hz:g})"
            )
            logger.warning(f"CLS disabled — {self._reason}")
            return
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
        while not self._stop.is_set():
            if self._pending_source is not None:
                self._apply_pending_source()
            if self._cursor_reset.is_set():
                self._cursor_reset.clear()
                cursor = time.monotonic()
                self._reset_window()  # runs even while paused, so pause() never races us
            if not self._paused:
                for sample in self._service.raw_samples_since(self._sensor, cursor):
                    cursor = sample["t"]
                    self._push_raw(sample)
            next_tick += self._period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                next_tick = time.monotonic()  # fell behind — resync

    def _reset_window(self) -> None:
        """Drop all buffered raw data so the next window starts on a clean grid.

        The aggregator is reset with it: a partial vote window spanning a discontinuity would
        average predictions from either side of a stall, a pause or a source switch.

        Loop-thread only: web-thread callers signal via ``_cursor_reset`` instead."""
        self._raw.clear()
        self._group.clear()
        self._last_raw_t = None
        self._groups_since_pred = 0
        self._agg.reset()

    def _apply_pending_source(self) -> None:
        """Land a ``set_source`` request. Loop-thread only — ``_raw`` is rebuilt here.

        Re-pointing is all that is needed for the model itself: the window is rebuilt from raw
        samples on every inference, so nothing carries over from the old source."""
        with self._log_lock:
            pending, self._pending_source = self._pending_source, None
        if pending is None:
            return
        service, sample_hz = pending
        self._service = service
        self._sample_hz = float(sample_hz)
        self._decim = max(1, int(round(self._sample_hz / self._target_hz)))
        # maxlen is fixed at construction, so a changed decim needs a fresh deque.
        self._raw = deque(maxlen=self._window * self._decim)
        self._reset_window()
        logger.info(
            f"CLS source swapped (sample_hz={self._sample_hz:g}, decim={self._decim})"
        )

    def _push_raw(self, sample: dict) -> None:
        """Feed one raw sample: emits a model-rate vector every ``decim`` samples and runs
        inference every ``stride`` vectors. Group size is fixed by sample count, so batching
        by the caller — 9, 11 or 20 samples in one tick — cannot change what the model sees."""
        t = sample["t"]
        if self._last_raw_t is not None and (t - self._last_raw_t) > MAX_RAW_GAP_S:
            self._reset_window()  # discontinuity — never average or window across a stall
        self._last_raw_t = t
        vec = [*sample["accel"], *sample["gyro"]]
        self._raw.append(vec)
        self._group.append(vec)
        if len(self._group) < self._decim:
            return
        # One full group -> one model-rate vector (== down_sample's integer branch).
        avg = np.mean(self._group, axis=0)
        self._group.clear()
        self._input_buf.append(
            {"t": t, "acc": [float(v) for v in avg[:3]], "gyr": [float(v) for v in avg[3:]]}
        )
        self._groups_since_pred += 1
        if self._groups_since_pred >= self._stride and len(self._raw) == self._raw.maxlen:
            self._groups_since_pred = 0
            self._infer(t)

    def _infer(self, t: float) -> None:
        try:
            # Rebuild the window from raw: exactly ``decim`` samples per row, grid-aligned
            # because we only get here on a group boundary. Bit-identical to
            # ``down_sample(raw, sample_hz, target_hz)`` — see others/tests/test_cls_downsample.py.
            window = (
                np.asarray(self._raw, dtype=np.float32)
                .reshape(self._window, self._decim, 6)
                .mean(axis=1)
            )
            cls_name, conf, probs = self._clf.predict(window)
        except Exception as err:  # pragma: no cover - runtime safety, never kill the thread
            logger.warning(f"CLS inference error: {err}")
            return
        idx = int(np.argmax(probs))
        # Aggregate before building the entry: the entry dict is handed to HTTP threads via
        # ``_log`` and must never be mutated afterwards, so "did this frame produce a decision"
        # has to be known up front.
        decision = self._agg.push(probs)
        entry = {
            "id": self._next_id,
            "t": time.time(),
            "clock": datetime.now().strftime("%H:%M:%S"),
            "cls": cls_name,
            "conf": conf,
            "idx": idx,
            "probs": [float(p) for p in probs],
            "decision": decision.index if decision is not None else None,
        }
        decided: dict | None = None
        if decision is not None:
            decided = {
                "t": t,
                "clock": entry["clock"],
                "idx": decision.index,
                "cls": self._class_name(decision.index),
                "conf": decision.confidence,
                "probs": decision.probs,
                "n": decision.n_frames,
                "held": decision.held,
            }
        with self._log_lock:
            self._next_id += 1
            self._log.append(entry)
            self._current = entry
            if decided is not None:
                self._current_decision = decided
        if decided is not None:
            self._decision_buf.append(decided)
            # Outside the lock: the sink writes to a serial port, and a decision must never be
            # able to stall an HTTP thread or kill this one.
            sink = self._on_result
            if sink is not None:
                try:
                    sink(decided["idx"])
                except Exception as err:  # pragma: no cover - runtime safety
                    logger.warning(f"CLS result sink failed: {err}")

    @staticmethod
    def _class_name(idx: int) -> str:
        return CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx)

    # --- runtime switch ------------------------------------------------------
    def pause(self) -> None:
        """Suspend sampling + inference (model stays loaded). Idempotent.

        Buffers are cleared by the loop thread on its next tick (via ``_cursor_reset``);
        touching them here would race an in-flight ``_infer``."""
        with self._log_lock:
            self._paused = True
            # Don't show / record a stale prediction or decision.
            self._current = None
            self._current_decision = None
        self._cursor_reset.set()

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

    def set_source(self, service: "SensorSource", *, sample_hz: float | None = None) -> str | None:
        """Re-point at a different sensor source. Returns an error string, or None on success.

        The checkpoint is *not* reloaded — the model is agnostic to where its samples came from,
        so switching the web UI between the I2C IMUs and a serial one costs nothing but a window
        refill. Like ``pause``/``resume``, this only signals: the loop thread performs the swap
        and rebuilds the buffers on its next tick (≤ one CLS period), so it can never race an
        in-flight inference.

        A non-integral ``sample_hz / target_hz`` is refused rather than applied: the reshape in
        ``_infer`` depends on that ratio, so the service pauses with a reason instead of feeding
        the model mis-sized windows."""
        hz = self._sample_hz if sample_hz is None else float(sample_hz)
        if not self._enabled:
            # No loop thread, so swap directly — and leave ``_reason`` alone: it holds why CLS
            # is disabled (missing torch, bad checkpoint), which the page still needs to show.
            self._service = service
            self._sample_hz = hz
            return None
        decim = max(1, int(round(hz / self._target_hz)))
        if abs(hz / self._target_hz - decim) > 1e-9:
            reason = (
                f"sample_hz ({hz:g}) must be an integer multiple of "
                f"target_hz ({self._target_hz:g})"
            )
            logger.warning(f"CLS paused — {reason}")
            self.pause()
            with self._log_lock:
                self._reason = reason
            return reason
        with self._log_lock:
            self._pending_source = (service, hz)
            if self._reason != "ok":
                self._reason = "ok"
        self._cursor_reset.set()
        return None

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

    @property
    def vote_config(self) -> dict[str, int]:
        """The aggregator's knobs, for ``/cls`` and the UI."""
        return self._agg.config

    def current(self) -> dict | None:
        """Thread-safe copy of the latest prediction (``cls``/``conf``/``probs``), or None.

        The recorder polls this per drain to persist the held prediction at 100 Hz."""
        with self._log_lock:
            return dict(self._current) if self._current else None

    def current_decision(self) -> dict | None:
        """Thread-safe copy of the latest aggregated decision — the service's actual output."""
        with self._log_lock:
            return dict(self._current_decision) if self._current_decision else None

    def inputs_since(self, t: float, limit: int | None = None) -> list[dict]:
        """Model-input samples ``{"t","acc","gyr"}`` newer than monotonic ``t``, oldest first.

        These are the exact 6-channel vectors fed to the model (raw accel+gyro of the CLS
        sensor at the model's ``target_hz``); the recorder drains them into model_input.csv."""
        return self._input_buf.since(t, limit=limit)

    def decisions_since(self, t: float, limit: int | None = None) -> list[dict]:
        """Aggregated decisions newer than monotonic ``t``, oldest first.

        One entry per decision at its own rate (``target_hz / (stride * emit_every)``), not
        step-held — the recorder writes these to cls_vote.csv so a session holds the frame-level
        stream and the post-vote stream side by side for offline comparison."""
        return self._decision_buf.since(t, limit=limit)

    # --- web accessor ------------------------------------------------------
    def snapshot(self, since: int = 0) -> dict:
        """Payload for GET /cls: enabled flag, latest decision, frame entries after ``since``."""
        if not self._enabled:
            return {
                "enabled": False,
                "reason": self._reason,
                "current": None,
                "decision": None,
                "entries": [],
            }
        with self._log_lock:
            entries = [e for e in self._log if e["id"] > since]
            current = dict(self._current) if self._current else None
            decision = dict(self._current_decision) if self._current_decision else None
            running = not self._paused
            reason = self._reason
        return {
            "enabled": True,
            "running": running,
            "sensor": self._sensor,
            "current": current,
            "decision": decision,
            "vote": self._agg.config,
            # Surfaced while enabled too: a refused ``set_source`` pauses the service, and
            # without this the page would just go quiet with no explanation.
            "reason": reason,
            "entries": entries,
        }
