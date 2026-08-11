"""ControlService — the fixed-rate loop that turns knee signals + a mode into motor commands.

One thread. Every tick it reads the newest control sample from the sensor source, picks the
profile for the current locomotion mode, evaluates it per leg to get a position reference, runs
the position loop against the motor feedback, and writes the resulting velocity command back
over the same serial link the samples arrived on.

**The invariant this class exists to hold: while enabled, the link never goes silent.** Every
tick writes something. When anything is wrong — no mode, no fresh frame, the device's enable
line low, a profile that failed to load, an exception in the middle of the loop — what it writes
is zero. Zero velocity means the motor holds still, which is genuinely safe here; that is why
the boundary with Simulink was drawn below the position loop rather than above it.

Three design choices worth knowing before changing anything:

**Its own thread, not the reader thread.** ``decode_frames`` sits inside ``ser.read()`` for up
to a second without yielding, and after a dropped link ``_read_loop`` waits ``REOPEN_DELAY_S``
between reopen attempts — so an emitter driven by frame arrivals goes quiet in exactly the
situations where commanding zero matters most. A separate clock is the only way to promise a
steady rate. It also keeps the bounded-but-real write cost off the read path, where it would
turn into input-buffer overflow and re-sync churn.

**Newest sample, not a cursor drain.** ``ClsService`` needs a cursor because it accumulates a
window; this loop accumulates nothing from the sensor side. Draining a backlog after a stall
would emit a burst of commands computed from stale angles, of which only the last is meaningful.
Zero-order hold on the newest sample is the standard digital-control answer and makes staleness
a direct timestamp comparison.

**Fixed rate, because the loop is stateful.** The PID's integrator and second-order damping term
both depend on the step size, so the tick — not the arrival of a frame — defines the sample
rate. ``dt`` is measured rather than assumed, and an implausible one resets the state instead of
being integrated (see ``pid.VelocityPid.step``).

Mode arrives by push (``set_mode``, called from the CLS thread once per aggregated decision) and
is treated as stale after ``mode_timeout_s``. Push rather than polling because
``ClsService.current_decision`` shares a lock with inference and with every ``/cls`` request; the
timeout is what turns "CLS paused / never started / still refilling its vote window" into the
neutral profile without this class knowing any of those states exist.
"""

from __future__ import annotations

import threading
import time

from loguru import logger

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.control.pid import VelocityPid
from jetson_imu_tui.control.profile import NEUTRAL, Profile
from jetson_imu_tui.ring_buffer import RingBuffer


class ControlService:
    """Fixed-rate assistance controller. Construct, ``start()``, and ``stop()`` at shutdown."""

    def __init__(
        self,
        source,
        *,
        label: str,
        enabled: bool = False,
        rate_hz: float = 100.0,
        mode_timeout_s: float = 2.0,
        sample_timeout_s: float = 0.15,
        default_profile: str = "",
        pid_cfg: dict | None = None,
        modes_cfg: dict | None = None,
        profiles_cfg: dict | None = None,
        log_size: int = 6000,
    ) -> None:
        """Build the controller from parsed config. Does not touch the source or start a thread.

        Args:
            source:            the active sensor source. Duck-typed: needs
                               ``latest_control_sample(label)`` and ``send_velocity(r, l)`` to do
                               anything, and is treated as inactive (but still ticked) without
                               them — that is how switching to the I2C source behaves.
            label:             str, sensor label to read control samples under.
            enabled:           bool, master switch (``[control] enabled``). False means the
                               thread never starts and nothing is ever written.
            rate_hz:           float, tick rate in Hz. Also the PID's nominal sample rate.
            mode_timeout_s:    float, seconds a pushed mode stays valid.
            sample_timeout_s:  float, seconds a control sample stays fresh.
            default_profile:   str, profile name to fall back on when no mode is valid;
                               "" means fall back to neutral (command zero).
            pid_cfg:           dict | None, ``[control.pid]`` — kp/ki/kv/wn/zt/x0/vel_limit.
            modes_cfg:         dict | None, ``[control.modes]``, class name -> profile name.
            profiles_cfg:      dict | None, ``[control.profiles]``, name -> profile table.
            log_size:          int, ticks retained in the telemetry ring buffer (6000 = 60 s
                               at 100 Hz).

        Returns:
            None.
        """
        self._source = source
        self._label = label
        self._enabled = bool(enabled)
        self._rate_hz = max(1.0, float(rate_hz))
        self._period = 1.0 / self._rate_hz
        self._mode_timeout_s = float(mode_timeout_s)
        self._sample_timeout_s = float(sample_timeout_s)

        pid_cfg = dict(pid_cfg or {})
        self._profiles = self._build_profiles(profiles_cfg or {})
        self._by_index = self._build_mode_table(modes_cfg or {}, default_profile)
        self._default_profile = self._profiles.get(default_profile, NEUTRAL) if default_profile \
            else NEUTRAL

        # One loop per leg. Gains default to the values read out of the Simulink models; the
        # shipped config deliberately starts kp lower (see the plan's serial-latency risk).
        def _pid() -> VelocityPid:
            return VelocityPid(
                kp=float(pid_cfg.get("kp", 7.5)),
                ki=float(pid_cfg.get("ki", 0.0)),
                kv=float(pid_cfg.get("kv", 0.02)),
                wn=float(pid_cfg.get("wn", 0.0)),
                zt=float(pid_cfg.get("zt", 1.0)),
                x0=float(pid_cfg.get("x0", 0.0)),
                vel_limit=float(pid_cfg.get("vel_limit", 41.87)),
                nominal_dt=self._period,
            )

        self._pid_r = _pid()
        self._pid_l = _pid()

        # Mode, pushed from the CLS thread. Guarded by its own lock so the control tick never
        # contends with inference or with an HTTP request.
        self._mode_lock = threading.Lock()
        self._mode_idx: int | None = None
        self._mode_t: float = float("-inf")

        # Swapped by the web thread on a source switch; read every tick.
        self._src_lock = threading.Lock()

        self._log = RingBuffer(maxlen=int(log_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._err_logged = False
        self._overruns = 0

        # Latest tick's status, for /data. Plain attributes: single writer (the loop), and
        # readers only ever want a consistent-enough snapshot.
        self._active = False
        self._stale = True
        self._profile_name = NEUTRAL.name
        self._last_vel = (0.0, 0.0)
        self._last_err = (0.0, 0.0)
        self._reason = "" if self._enabled else "disabled in config"

        if self._enabled and mode_timeout_s > 0:
            logger.info(
                f"control: {self._rate_hz:g} Hz, mode timeout {self._mode_timeout_s:g}s, "
                f"sample timeout {self._sample_timeout_s:g}s, "
                f"profiles={sorted(self._profiles) or 'none'}"
            )

    # --- construction helpers ------------------------------------------------
    @staticmethod
    def _build_profiles(cfg: dict) -> dict[str, Profile]:
        """Turn ``[control.profiles]`` into name -> Profile. Never raises.

        Args:
            cfg: dict, ``{profile_name: {table...}}`` straight from the TOML.

        Returns:
            dict[str, Profile]. A profile whose table is malformed is present but neutral,
            carrying its ``reason`` — keeping it in the map means a mode pointing at it still
            resolves, and the reason surfaces instead of the mode silently disappearing.
        """
        out: dict[str, Profile] = {}
        for name, table in cfg.items():
            if not isinstance(table, dict):
                logger.warning(f"control profile '{name}' ignored — not a table")
                continue
            out[str(name)] = Profile.from_config(str(name), table)
        return out

    def _build_mode_table(self, modes_cfg: dict, default_profile: str) -> list[str]:
        """Expand ``[control.modes]`` into a per-class-index profile-name lookup.

        Args:
            modes_cfg:       dict, class name -> profile name. Keyed by *name* because
                             ``cls.model.CLASSES`` is the authority on label order; an
                             index-keyed table would silently remap if a retrained checkpoint
                             reordered its labels.
            default_profile: str, profile name for classes not listed; "" means neutral.

        Returns:
            list[str] of length ``len(CLASSES)``: the profile name for each class index, so the
            hot path is a list index rather than two dict lookups. Unknown class names and
            unknown profile names are warned about once, at construction, and dropped.
        """
        table = [default_profile] * len(CLASSES)
        by_name = {name: i for i, name in enumerate(CLASSES)}
        for cls_name, prof_name in (modes_cfg or {}).items():
            idx = by_name.get(str(cls_name))
            if idx is None:
                logger.warning(
                    f"control: [control.modes] '{cls_name}' is not a known class — ignored"
                )
                continue
            if str(prof_name) not in self._profiles:
                logger.warning(
                    f"control: mode '{cls_name}' points at unknown profile "
                    f"'{prof_name}' — falling back to '{default_profile or 'neutral'}'"
                )
                continue
            table[idx] = str(prof_name)
        return table

    # --- lifecycle ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """bool — the config master switch. False means no thread and no writes, ever."""
        return self._enabled

    @property
    def running(self) -> bool:
        """bool — True while the tick thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the tick thread. No-op when disabled or already running.

        Args:    none.
        Returns: None.
        """
        if not self._enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="control")
        self._thread.start()

    def stop(self) -> None:
        """Stop the tick thread, then command zero one last time.

        Args:    none.
        Returns: None.

        The stop write happens *after* the join so it cannot race a live command from the loop.
        It is best-effort by nature: only a clean shutdown path reaches here at all — SIGTERM, a
        crash, and ``_free_port``'s SIGKILL escalation all bypass it, and every thread here is a
        daemon. The device-side receive watchdog is what actually guarantees the motors stop.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._pid_r.reset()
        self._pid_l.reset()
        self.flush_stop()

    # --- inputs from other threads -------------------------------------------
    def set_mode(self, index: int) -> None:
        """Record the locomotion mode. Called from the CLS thread, once per decision.

        Args:
            index: int, class index into ``cls.model.CLASSES``. Out-of-range values are
                   ignored rather than raising — this runs on the inference thread.

        Returns:
            None.

        Also stamps the arrival time: the loop treats a mode older than ``mode_timeout_s`` as
        absent, which is what makes "CLS stopped" and "CLS keeps saying walk" distinguishable
        without this class knowing anything about CLS's internal state.
        """
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return
        if not 0 <= idx < len(CLASSES):
            return
        with self._mode_lock:
            self._mode_idx = idx
            self._mode_t = time.monotonic()

    def set_source(self, source) -> None:
        """Re-point at a different sensor source. Called from the web thread on a switch.

        Args:
            source: the new sensor source (same duck-typed surface as the constructor's).

        Returns:
            None.

        Resets both position loops: the new source's feedback is a different signal, and
        carrying integrator state across that boundary would command a step. Call this
        *before* tearing the old source down, then ``flush_stop(old)`` — otherwise the last
        thing the old link carried is a live command that the device will hold forever.
        """
        with self._src_lock:
            self._source = source
        self._pid_r.reset()
        self._pid_l.reset()

    def flush_stop(self, source=None) -> bool:
        """Write a single zero-velocity command, bounded. Safe from any thread.

        Args:
            source: the source to write to, or None for the currently active one. Passing the
                    *old* source explicitly is what lets a switch stop the motors on a link it
                    is about to close.

        Returns:
            bool, True if the zero command reached the port. False when the source cannot
            transmit (the I2C source has no ``send_velocity``) or the write failed.
        """
        if source is None:
            with self._src_lock:
                source = self._source
        send = getattr(source, "send_velocity", None)
        if send is None:
            return False
        try:
            return bool(send(0.0, 0.0))
        except Exception as err:  # pragma: no cover - runtime safety
            logger.warning(f"control: stop command failed ({err})")
            return False

    # --- the loop -------------------------------------------------------------
    def _loop(self) -> None:
        """Tick at ``rate_hz`` until stopped. Never returns early, never raises.

        Args:    none.
        Returns: None.

        Same cadence discipline as ``ClsService._loop`` and ``Recorder._loop``: advance a
        deadline by the period, sleep the remainder on the stop event, and resync (counting an
        overrun) when the tick has already fallen behind.
        """
        next_tick = time.monotonic()
        last = next_tick
        while not self._stop.is_set():
            now = time.monotonic()
            dt, last = now - last, now
            try:
                self._tick(now, dt)
            except Exception as err:  # pragma: no cover - runtime safety
                # Never let one bad tick kill the loop: the loop is what commands zero.
                if not self._err_logged:
                    logger.warning(f"control: tick failed ({err}) — commanding zero")
                    self._err_logged = True
                self._pid_r.reset()
                self._pid_l.reset()
                self._safe_stop()
            next_tick += self._period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                self._overruns += 1
                next_tick = time.monotonic()

    def _tick(self, now: float, dt: float) -> None:
        """Run one control step and write its command.

        Args:
            now: float, ``time.monotonic()`` at the top of this tick, in seconds.
            dt:  float, measured seconds since the previous tick.

        Returns:
            None. Always writes exactly one command frame (or nothing, if the active source
            cannot transmit at all — in which case the controller reports itself inactive).
        """
        with self._src_lock:
            source = self._source
        send = getattr(source, "send_velocity", None)
        latest = getattr(source, "latest_control_sample", None)
        if send is None or latest is None:
            # I2C, or any source without the control surface. Keep ticking so a switch back
            # resumes instantly, but write nothing and say why.
            self._active = False
            self._stale = True
            self._reason = "active source has no control channel"
            self._last_vel = (0.0, 0.0)
            self._pid_r.reset()
            self._pid_l.reset()
            return
        self._active = True

        sample = latest(self._label)
        fresh = sample is not None and (now - sample["t"]) <= self._sample_timeout_s
        if sample is None:
            self._reason = (
                "layout carries no knee channels"
                if not getattr(source, "has_joints", True)
                else "no frame yet"
            )
        elif not fresh:
            self._reason = "sensor sample is stale"
        elif sample.get("pos_r") is None or sample.get("pos_l") is None:
            fresh = False
            self._reason = "layout carries no motor feedback"
        elif not sample["enable"]:
            self._reason = "device enable line is low"
        else:
            self._reason = ""

        profile = self._current_profile(now)
        self._profile_name = profile.name

        if not fresh or not sample["enable"]:
            # Nothing trustworthy to act on. Reset so the loops restart from a known state when
            # data returns, and command a stop rather than going quiet.
            self._stale = True
            self._pid_r.reset()
            self._pid_l.reset()
            vel_r = vel_l = 0.0
            ref_r = ref_l = 0.0
            sent = bool(send(0.0, 0.0))
        else:
            self._stale = False
            ref_r = profile.command(sample["ang_r"], sample["vel_r"], "r")
            ref_l = profile.command(sample["ang_l"], sample["vel_l"], "l")
            vel_r = self._pid_r.step(ref_r, sample["pos_r"], dt, enable=True)
            vel_l = self._pid_l.step(ref_l, sample["pos_l"], dt, enable=True)
            sent = bool(send(vel_r, vel_l))
            self._err_logged = False

        self._last_vel = (vel_r, vel_l)
        self._last_err = (self._pid_r.last_error, self._pid_l.last_error)
        self._log.append({
            "t": now,
            "profile": profile.name,
            "ang_r": sample["ang_r"] if sample else None,
            "vel_r": sample["vel_r"] if sample else None,
            "ang_l": sample["ang_l"] if sample else None,
            "vel_l": sample["vel_l"] if sample else None,
            "ref_r": ref_r,
            "ref_l": ref_l,
            "fb_r": sample.get("pos_r") if sample else None,
            "fb_l": sample.get("pos_l") if sample else None,
            "err_r": self._pid_r.last_error,
            "err_l": self._pid_l.last_error,
            "cmd_r": vel_r,
            "cmd_l": vel_l,
            "enable": bool(sample["enable"]) if sample else False,
            "stale": self._stale,
            "sent": sent,
        })

    def _current_profile(self, now: float) -> Profile:
        """The profile for the mode in force right now.

        Args:
            now: float, ``time.monotonic()`` for the staleness comparison.

        Returns:
            Profile. The configured default (or ``NEUTRAL``) when no mode has been pushed, or
            when the last one is older than ``mode_timeout_s``.
        """
        with self._mode_lock:
            idx, mode_t = self._mode_idx, self._mode_t
        if idx is None or (now - mode_t) > self._mode_timeout_s:
            return self._default_profile
        return self._profiles.get(self._by_index[idx], self._default_profile)

    def _safe_stop(self) -> None:
        """Best-effort zero command from inside the loop's exception handler.

        Args:    none.
        Returns: None. Swallows everything — this runs when something has already gone wrong.
        """
        try:
            self.flush_stop()
        except Exception:  # pragma: no cover - runtime safety
            pass

    # --- accessors ------------------------------------------------------------
    def entries_since(self, t: float, limit: int | None = None) -> list[dict]:
        """Telemetry rows newer than monotonic ``t``, oldest first — the recorder's source.

        Args:
            t:     float, host-monotonic cursor.
            limit: int | None, keep at most the newest ``limit`` rows.

        Returns:
            list[dict], each row as appended in ``_tick``: inputs, references, feedback,
            errors and commands for one tick. Input *and* output together, because a velocity
            trace alone cannot tell a profile problem from a loop problem.
        """
        return self._log.since(t, limit=limit)

    def snapshot(self) -> dict:
        """Status for ``GET /data``.

        Args:    none.
        Returns: dict with
                   "enabled":  bool, the config switch.
                   "active":   bool, the current source can actually be commanded.
                   "running":  bool, the tick thread is alive.
                   "profile":  str, profile name used on the last tick.
                   "mode_age": float | None, seconds since the last pushed decision; None if
                               none has ever arrived. Exposed because "CLS died and the
                               controller is quietly holding zero" is otherwise invisible.
                   "stale":    bool, last tick had no usable sample.
                   "reason":   str, "" when healthy, else why it is commanding zero.
                   "vel":      [float, float], last commanded (right, left) in rad/s.
                   "err":      [float, float], last position error (right, left) in rad.
                   "overruns": int, ticks that missed their deadline since start.
        """
        with self._mode_lock:
            mode_t = self._mode_t
        age = None if mode_t == float("-inf") else max(0.0, time.monotonic() - mode_t)
        return {
            "enabled": self._enabled,
            "active": self._active,
            "running": self.running,
            "profile": self._profile_name,
            "mode_age": age,
            "stale": self._stale,
            "reason": self._reason,
            "vel": [self._last_vel[0], self._last_vel[1]],
            "err": [self._last_err[0], self._last_err[1]],
            "overruns": self._overruns,
        }
