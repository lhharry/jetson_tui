"""ClsService — background block-averaging sampler + sliding-window inference.

Raw (gravity-inclusive, tare-bypassed) samples are pulled from ``raw_samples_since`` and
block-averaged ``sample_hz / target_hz`` at a time into one model-rate vector. That one method
is the entire contract with the sensor source (``imu_common.SensorSource``), so either source
works unchanged: ``ImuService`` over I2C or ``SerialImuService`` over a serial link. This mirrors
training's anti-aliasing downsample (``dataset/jetson_leg.down_sample``) in BOTH of its
branches: an integer ratio means fixed groups of ``ratio`` samples; a non-integer ratio (33.3 Hz
into 10 Hz is 3.33) alternates groups of ``floor`` and ``floor + 1`` samples on a running
remainder, exactly as the training code does, so the stream is resampled at the trained rate
whatever the device sends. Plain decimation (one instantaneous sample per tick) would alias
>5 Hz energy and feed the model out-of-distribution input, hurting the dynamic classes (jog /
stairs) most.

Grouping is driven by *raw sample count*, not by tick timing: the CLS tick and the sampler
thread drift independently, so a tick can deliver 9, 11 or (after a stall) ~20 samples. Only
a full group ever forms a vector, so a late tick yields two correct vectors rather than one
over-wide one. A raw-resolution gap check drops the whole window at a discontinuity so
inference never runs across a stall. That check reads the **device clock** (``t_src``) when the
source carries one: host arrival time reads a host scheduling stall -- frames waiting in the OS
serial buffer, then arriving in a burst -- as a hole in the data, and each false reset costs a
full window (2 s) of silence. Host time is the fallback for sources without a device clock.

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

Fails safe: if ``torch`` or the checkpoint is missing, or ``sample_hz`` is below
``target_hz`` (nothing to average), the service stays ``enabled=False`` and never touches the
sensor, so the rest of the TUI is unaffected.
"""

from __future__ import annotations

import math
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

# Clear the rolling window if consecutive *raw* samples are further apart than this, in seconds
# of the DEVICE clock (``t_src``) when the source has one, else of host arrival time.
# Count-based grouping already keeps every group the right size, so this guard only needs to
# catch true discontinuities (sensor stall / reconnect / resume — hundreds of ms and up), not
# jitter (tens of ms), which merely stretches a group by a sample or two. 100 ms sits between
# the two regimes.
#
# Why the device clock: on a serial link the host stamps ``t`` when the reader thread decodes a
# frame, so a 300 ms scheduling stall on the host (recorder, Flask, GIL) shows up as a 300 ms
# hole in ``t`` while the frames sat intact in the OS buffer and their device clocks continued
# 30 ms apart. Resetting there threw away a full window -- 2 s with no decision -- for data that
# was never missing. Recorded sessions showed every one of those holes lining up with a 2 s gap
# in cls_vote.csv. A device restart (clock going backwards) or a real device-side gap still
# resets, which is what the check is for.
MAX_RAW_GAP_S = 0.1

# One log line per this many seconds while the window keeps being reset, however many resets.
# A stall storm should be one warning, not one per stall; the counter in ``snapshot`` carries
# the number.
RESET_LOG_EVERY_S = 10.0


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
        self._target_hz = float(target_hz)
        self._period = 1.0 / self._target_hz
        self._window = int(window)
        self._stride = int(stride)
        # Raw samples per model-rate vector, as a ratio (100 Hz / 10 Hz = 10; 33.3 / 10 = 3.33).
        # ``_base`` / ``_frac`` are its integer and fractional parts, which drive the group
        # sizes below the same way training's ``down_sample`` drives them.
        self._sample_hz = 0.0
        self._ratio = 1.0
        self._base = 1
        self._frac = 0.0
        self._configure_rate(sample_hz)

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

        # Rolling window of model-rate vectors: exactly ``window`` block means, i.e. what one
        # inference consumes. Each vector is the mean of one complete group, so the window is
        # rebuilt on a clean grid after any reset and can never inherit a malformed group.
        # Only ever touched by the loop thread.
        self._vecs: deque[np.ndarray] = deque(maxlen=self._window)
        self._group: list[list[float]] = []  # raw samples accumulating into the next vector
        self._group_target = 0               # raw samples the current group needs
        self._remainder = 0.0                # training's running remainder for non-integer ratios
        self._last_raw_t: float | None = None    # host arrival time of the previous raw sample
        self._last_src_t: float | None = None    # its device clock, when the source has one
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

        # Health, for /cls: how often the window was thrown away and why, how long the model
        # takes, and how far behind sample arrival the inference runs. Written by the loop
        # thread under ``_log_lock``, read by HTTP threads under the same lock.
        self._resets = 0
        self._last_reset: dict | None = None
        self._infer_ms: float | None = None
        self._lag_s: float | None = None
        self._reset_logged_at = 0.0              # loop-thread only

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- rate ---------------------------------------------------------------
    def _configure_rate(self, sample_hz: float) -> None:
        """Derive the grouping from a wire rate. Loop-thread (or pre-start) only.

        Args:    sample_hz: float, raw samples per second the source delivers.
        Returns: None. Sets ``_sample_hz``, ``_ratio``, ``_base``, ``_frac``.

        ``_base`` is floored at 1 so a ratio below 1 (refused by ``start``/``set_source`` before
        it can matter) still produces well-formed groups rather than an infinite loop.
        """
        self._sample_hz = float(sample_hz)
        self._ratio = self._sample_hz / self._target_hz
        self._base = max(1, int(self._ratio))
        self._frac = max(0.0, self._ratio - self._base)

    def _rate_error(self, sample_hz: float) -> str | None:
        """Why ``sample_hz`` cannot feed the model, or None if it can.

        Args:    sample_hz: float.
        Returns: str | None.

        The only hard limit is training's: ``down_sample`` raises for a raw rate below the
        target, since a group would need less than one sample. Any ratio >= 1, integer or not,
        resamples the same way training did.
        """
        if not sample_hz or sample_hz < self._target_hz:
            return (
                f"sample_hz ({sample_hz:g}) must be >= target_hz ({self._target_hz:g})"
            )
        return None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Load the model and start the sampler thread. Self-disables on any failure."""
        err = self._rate_error(self._sample_hz)
        if err is not None:
            self._reason = err
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
        logger.info(
            f"CLS enabled on '{self._sensor}' (device={self._clf.device}, "
            f"sample_hz={self._sample_hz:g}, ratio={self._ratio:.3g})"
        )

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
                    # Re-checked per sample, not just per tick: a batch can be large after a
                    # stall, and pressing Stop must stop inference — and the results being
                    # transmitted downstream — promptly rather than at the end of the batch.
                    if self._paused:
                        break
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
        """Drop all buffered data so the next window starts on a clean grid.

        The aggregator is reset with it: a partial vote window spanning a discontinuity would
        average predictions from either side of a stall, a pause or a source switch. The
        grouping remainder restarts at zero too, which is where training's ``down_sample``
        starts every file.

        Loop-thread only: web-thread callers signal via ``_cursor_reset`` instead."""
        self._vecs.clear()
        self._group.clear()
        self._group_target = 0
        self._remainder = 0.0
        self._last_raw_t = None
        self._last_src_t = None
        self._groups_since_pred = 0
        self._agg.reset()

    def _note_reset(self, reason: str, gap_s: float | None) -> None:
        """Reset the window because the raw stream broke, and make that visible.

        Args:
            reason: str, what broke ("device clock gap", "host clock went backwards", ...).
            gap_s:  float | None, the measured step in seconds; None when no step applies.

        Returns: None.

        Only the resets the data forced are counted here. ``pause``/``resume``/``set_source``/
        ``reset_window`` also clear the window, but those are operator actions with their own
        UI, and folding them in would make the counter read "faults" when it means "clicks".
        The log is rate-limited by time (``RESET_LOG_EVERY_S``): a stall storm is one warning
        with the count in ``snapshot``, not a line per stall. Loop-thread only.
        """
        self._reset_window()
        info = {
            "reason": reason,
            "gap_s": None if gap_s is None else float(gap_s),
            "clock": datetime.now().strftime("%H:%M:%S"),
        }
        with self._log_lock:
            self._resets += 1
            self._last_reset = info   # replaced whole: published dicts are never mutated
            n = self._resets
        now = time.monotonic()
        if now - self._reset_logged_at >= RESET_LOG_EVERY_S:
            self._reset_logged_at = now
            step = "" if gap_s is None else f", step {gap_s:+.3f}s"
            logger.warning(
                f"CLS window reset ({reason}{step}) — {n} since start; the next decision needs "
                f"{self._window} fresh vectors (~{self._window / self._target_hz:.0f} s)"
            )

    def _apply_pending_source(self) -> None:
        """Land a ``set_source`` request. Loop-thread only — the grouping is rebuilt here.

        Re-pointing is all that is needed for the model itself: the window is rebuilt from
        fresh groups after the reset, so nothing carries over from the old source."""
        with self._log_lock:
            pending, self._pending_source = self._pending_source, None
        if pending is None:
            return
        service, sample_hz = pending
        self._service = service
        self._configure_rate(sample_hz)
        self._reset_window()
        logger.info(
            f"CLS source swapped (sample_hz={self._sample_hz:g}, ratio={self._ratio:.3g})"
        )

    def _push_raw(self, sample: dict) -> None:
        """Feed one raw sample: emits a model-rate vector when a group completes and runs
        inference every ``stride`` vectors.

        Args:    sample: dict with ``t`` (host monotonic), ``accel``/``gyro`` (list[float] or
                 None), and optionally ``t_src`` (device clock, seconds, float or None).
        Returns: None.

        Group size is fixed by sample count, so batching by the caller — 9, 11 or 20 samples in
        one tick — cannot change what the model sees. Sizes follow training's ``down_sample``:
        every group takes ``_base`` samples, plus one whenever the running remainder of the
        fractional part crosses 1, so 3.33 comes out as 3, 3, 4, 3, 3, 4, ... and an integer
        ratio degenerates to the fixed groups it always had.
        """
        t = sample["t"]
        t_src = sample.get("t_src")
        if t_src is not None and not math.isfinite(t_src):
            t_src = None   # a NaN clock is no clock (the serial filter already refuses these)
        if self._last_raw_t is not None:
            # Device clock when both sides have one, host arrival time otherwise. A repeated
            # device timestamp (dt == 0) is normal — float32 quantisation, and historically
            # a device resending each sample — and is not a break.
            if t_src is not None and self._last_src_t is not None:
                gap, clock = t_src - self._last_src_t, "device clock"
            else:
                gap, clock = t - self._last_raw_t, "host clock"
            if gap < 0.0:
                self._note_reset(f"{clock} went backwards", gap)     # device restart
            elif gap > MAX_RAW_GAP_S:
                self._note_reset(f"{clock} gap", gap)                # a real hole in the data
        self._last_raw_t, self._last_src_t = t, t_src
        acc, gyr = sample["accel"], sample["gyro"]
        # A source reports an unusable component as None (serial does this for a value that
        # arrived non-finite). That is a hole in the signal, not a zero: substituting one would
        # feed the model a reading the sensor never took, and passing None through reaches
        # ``np.mean`` and kills this thread — ``_push_raw`` has no exception handler, and the
        # loop would stop without CLS ever reporting itself as stopped.
        if acc is None or gyr is None or any(v is None for v in (*acc, *gyr)):
            self._note_reset("invalid sample", None)
            return
        if not self._group:
            # A new group: decide its size the way training does, remainder first.
            self._remainder += self._frac
            if self._remainder >= 1.0:
                self._remainder -= 1.0
                self._group_target = self._base + 1
            else:
                self._group_target = self._base
        self._group.append([*acc, *gyr])
        if len(self._group) < self._group_target:
            return
        # One full group -> one model-rate vector (== down_sample's block mean).
        avg = np.mean(self._group, axis=0)
        self._group.clear()
        self._vecs.append(avg)
        self._input_buf.append(
            {"t": t, "acc": [float(v) for v in avg[:3]], "gyr": [float(v) for v in avg[3:]]}
        )
        self._groups_since_pred += 1
        if self._groups_since_pred >= self._stride and len(self._vecs) == self._window:
            self._groups_since_pred = 0
            self._infer(t)

    def _infer(self, t: float) -> None:
        # How far behind arrival this thread runs, measured before the model adds its own time.
        lag = time.monotonic() - t
        try:
            # The window is the last ``window`` group means, grid-aligned because a vector only
            # exists once its whole group is in. Bit-identical to ``down_sample(raw, sample_hz,
            # target_hz)`` up to the float32 cast — see others/tests/test_cls_downsample.py.
            window = np.asarray(self._vecs, dtype=np.float32)
            t0 = time.perf_counter()
            cls_name, conf, probs = self._clf.predict(window)
            ms = (time.perf_counter() - t0) * 1e3
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
            # Timing is recorded even for an inference that ends up unpublished: it measures
            # the thread, not the result.
            self._lag_s = lag
            self._infer_ms = ms if self._infer_ms is None else 0.8 * self._infer_ms + 0.2 * ms
            # Stopped while this inference was running. ``pause`` sets the flag under this same
            # lock, so checking it here makes the two mutually exclusive: once Stop returns,
            # nothing further is published and — crucially — no further byte is transmitted.
            if self._paused:
                return
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

    def reset_window(self) -> None:
        """Drop the partial window and vote buffer — for a discontinuity the gap check cannot
        see, such as the axis remap changing under us (same timestamps, new coordinate frame).

        Signals only: the loop thread does the clearing on its next tick, so this is safe to
        call from a web thread mid-inference. No-op beyond a flag if CLS is disabled."""
        self._cursor_reset.set()

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

        A ``sample_hz`` below ``target_hz`` is refused rather than applied: there would be less
        than one raw sample per model vector, so the service pauses with a reason instead of
        feeding the model nothing."""
        hz = self._sample_hz if sample_hz is None else float(sample_hz)
        if not self._enabled:
            # No loop thread, so swap directly — and leave ``_reason`` alone: it holds why CLS
            # is disabled (missing torch, bad checkpoint), which the page still needs to show.
            self._service = service
            self._configure_rate(hz)
            return None
        reason = self._rate_error(hz)
        if reason is not None:
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
    def decim(self) -> float:
        """Raw samples block-averaged into one model-rate vector, on average.

        Output: float >= 1 — ``sample_hz / target_hz``. An integer (10.0) means every group is
        that size; 3.33 means groups of 3 and 4 in training's remainder pattern."""
        return self._ratio

    def timing(self, observed_hz: float | None = None) -> dict:
        """What the model window actually spans, in seconds of real time.

        Args:
            observed_hz: float | None, the wire rate measured by the source. None (or a
                         non-positive value) falls back to the configured ``sample_hz``.

        Returns:
            dict: {"decim": float, "window": int, "sample_hz": float, "target_hz": float,
                   "window_s": float, "trained_window_s": float, "nominal": bool}
            ``nominal`` is True when no measurement was available, i.e. ``window_s`` is
            derived from the declared rate rather than the observed one.

        The denominator has to be the OBSERVED rate. Using ``sample_hz`` makes this
        ``window * decim / sample_hz``, which reduces to ``window / target_hz`` -- the trained
        value, always, no matter how wrong the configuration is. That quantity can never report
        a mismatch, which is the entire point of computing it.
        """
        hz = float(observed_hz) if observed_hz and observed_hz > 0 else self._sample_hz
        return {
            "decim": self._ratio,
            "window": self._window,
            "sample_hz": self._sample_hz,
            "target_hz": self._target_hz,
            "window_s": self._window * self._ratio / hz if hz > 0 else 0.0,
            "trained_window_s": self._window / self._target_hz,
            "nominal": not (observed_hz and observed_hz > 0),
        }

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
        """Payload for GET /cls: enabled flag, latest decision, frame entries after ``since``,
        and the health counters (``resets``/``last_reset``/``infer_ms``/``lag_s``)."""
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
            resets = self._resets
            last_reset = dict(self._last_reset) if self._last_reset else None
            infer_ms = self._infer_ms
            lag_s = self._lag_s
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
            # Window resets forced by the data (not by pause/resume/source switches), with the
            # last one's reason and step; model time per window (EMA); and how late behind
            # sample arrival the last inference ran. Together they say whether a quiet CLS
            # page is a stalled stream, a slow model or a starved thread.
            "resets": resets,
            "last_reset": last_reset,
            "infer_ms": infer_ms,
            "lag_s": lag_s,
        }
