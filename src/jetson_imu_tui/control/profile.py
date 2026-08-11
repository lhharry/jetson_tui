"""One assistance profile: joint angle + angular velocity -> motor position reference.

This is the block the Simulink models put between the IMU and the position loop — a lookup
curve on joint angle, a velocity feed-forward term, an output gain, and a per-side saturation.
The control law implemented here is:

    ref = clamp( sign_side * (gain * curve(angle) + kv_ff * velocity),  limits_side )

which is general enough to express both shapes found in the models, so a table ported from
either lands here without restructuring:

* ``hybrid_hip_control_DEMO`` / ``HIP_FLEX_WALKING_DEMO`` scale only the curve and add the
  velocity term outside that gain:  ``-6 * LUT(ang) + 0.4 * vel``
  -> ``sign = 1, gain = -6, kv_ff = 0.4``
* ``HIP_EXT_STS_DEMO`` scales the *sum*:  ``±3.5 * (0.6 * LUT(ang) + 0.6 * vel)``
  -> ``sign = ±1, gain = 2.1, kv_ff = 2.1``

``sign_r`` / ``sign_l`` exist so one table can serve both legs, which is how the models do it —
they hold a single curve and mirror it with opposite gains. Keep them at +/-1 and put the
magnitude in ``gain`` / ``kv_ff``; nothing enforces that, but mixing the two makes a table
impossible to compare against its Simulink original.

**Stateless on purpose.** ``ControlService`` acts on the newest sample rather than replaying
every sample since its last tick, and that is only sound because evaluating a profile depends
on nothing but its arguments. Anything with memory belongs in ``pid`` (which is why the loop
runs at a fixed rate) or on the Simulink side.

**Never raises.** A malformed table produces a neutral profile carrying a ``reason`` string
instead of an exception: the control loop must keep ticking and commanding zero, and a config
typo that silently stopped the loop would be more dangerous than one that visibly does nothing.
"""

from __future__ import annotations

import math

from loguru import logger

from jetson_imu_tui.control.spline import INTERP_METHODS, Curve

# Angle-table units, mirroring [source] joint_units. Commands are never converted: they are
# motor-side references whose scale is set by the mechanism, not by an angular convention.
TABLE_UNIT_SCALE = {"rad": 1.0, "deg": math.pi / 180.0}

SIDES = ("r", "l")


class Profile:
    """A named assistance profile, ready to evaluate on either leg.

    Build with ``Profile.from_config`` (tolerant, logs and degrades) rather than the
    constructor, unless you are writing a test and want the arguments checked.
    """

    def __init__(
        self,
        name: str,
        curve: Curve | None,
        *,
        gain: float = 0.0,
        kv_ff: float = 0.0,
        sign_r: float = 1.0,
        sign_l: float = -1.0,
        limits_r: tuple[float, float] = (0.0, 0.0),
        limits_l: tuple[float, float] = (0.0, 0.0),
        reason: str = "",
    ) -> None:
        """Assemble a profile from already-validated parts.

        Args:
            name:     str, the profile's config key; appears in logs and in control.csv.
            curve:    Curve | None, the angle lookup, breakpoints already in radians.
                      None means "no usable table" and makes this profile neutral.
            gain:     float, multiplies the curve output. Dimensionless.
            kv_ff:    float, velocity feed-forward, in reference-units per rad/s.
            sign_r:   float, right-leg mirror, normally +1.
            sign_l:   float, left-leg mirror, normally -1.
            limits_r: tuple[float, float], (low, high) saturation for the right reference.
            limits_l: tuple[float, float], (low, high) saturation for the left reference.
            reason:   str, "" when the profile is usable; otherwise why it is neutral.

        Returns:
            None.
        """
        self.name = name
        self.curve = curve
        self.gain = float(gain)
        self.kv_ff = float(kv_ff)
        self.sign_r = float(sign_r)
        self.sign_l = float(sign_l)
        self.limits_r = (float(limits_r[0]), float(limits_r[1]))
        self.limits_l = (float(limits_l[0]), float(limits_l[1]))
        self.reason = reason

    # --- construction -------------------------------------------------------
    @classmethod
    def neutral(cls, name: str = "neutral", reason: str = "") -> "Profile":
        """A profile that commands exactly zero on both sides, whatever the input.

        Args:
            name:   str, name to report.
            reason: str, why this is neutral (shown in ``/data`` so a silent controller
                    explains itself instead of just looking broken).

        Returns:
            Profile with no curve and zero gains.
        """
        return cls(name, None, reason=reason)

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "Profile":
        """Build a profile from one ``[control.profiles.<name>]`` table. Never raises.

        Args:
            name: str, the profile's key in the config.
            cfg:  dict, the parsed TOML sub-table. Recognised keys, all optional:
                    table_units: str, "deg" (default) or "rad" — units of ``angles``.
                    angles:      list[float], strictly increasing breakpoints.
                    commands:    list[float], reference value at each breakpoint.
                    interp:      str, "spline" (default) or "linear".
                    gain:        float, default 0.0 (a table with no gain is inert).
                    kv_ff:       float, default 0.0.
                    sign_r:      float, default +1.0.
                    sign_l:      float, default -1.0.
                    limits_r:    [low, high], default [0.0, 0.0].
                    limits_l:    [low, high], default [0.0, 0.0].

        Returns:
            Profile. On any problem — missing table, ragged table, non-monotonic breakpoints,
            unknown interp, bad limits — returns ``Profile.neutral`` with ``reason`` set and
            logs a warning once at load. The shipped defaults (empty table, zero gain, zero
            limits) mean a profile that has not been filled in yet commands zero rather than
            something arbitrary.
        """
        try:
            units = str(cfg.get("table_units", "deg")).lower()
            if units not in TABLE_UNIT_SCALE:
                return cls._reject(name, f"unknown table_units {units!r}")
            scale = TABLE_UNIT_SCALE[units]

            angles = [float(v) for v in (cfg.get("angles") or [])]
            commands = [float(v) for v in (cfg.get("commands") or [])]
            interp = str(cfg.get("interp", "spline")).lower()
            if interp not in INTERP_METHODS:
                return cls._reject(name, f"unknown interp {interp!r}")

            # A table with fewer than two points cannot describe a curve. That is the shipped
            # placeholder state, not an error, so it degrades quietly to neutral.
            if len(angles) < 2 or len(commands) < 2:
                return cls.neutral(name, reason="no angle table configured")

            curve = Curve([a * scale for a in angles], commands, interp=interp)

            lim_r = cls._limits(cfg.get("limits_r"))
            lim_l = cls._limits(cfg.get("limits_l"))
            if lim_r is None or lim_l is None:
                return cls._reject(name, "limits must be [low, high] with low <= high")

            return cls(
                name,
                curve,
                gain=float(cfg.get("gain", 0.0)),
                kv_ff=float(cfg.get("kv_ff", 0.0)),
                sign_r=float(cfg.get("sign_r", 1.0)),
                sign_l=float(cfg.get("sign_l", -1.0)),
                limits_r=lim_r,
                limits_l=lim_l,
            )
        except (ValueError, TypeError) as err:
            return cls._reject(name, str(err))

    @staticmethod
    def _limits(raw) -> tuple[float, float] | None:
        """Parse and sanity-check one ``[low, high]`` saturation pair.

        Args:
            raw: the config value — expected to be a 2-element sequence, or None for the
                 default of (0.0, 0.0).

        Returns:
            tuple[float, float] on success, or None if it is not a 2-element pair, contains
            a non-finite value, or has low > high.
        """
        if raw is None:
            return (0.0, 0.0)
        try:
            lo, hi = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            return None
        return (lo, hi)

    @classmethod
    def _reject(cls, name: str, why: str) -> "Profile":
        """Log a bad profile once and return the neutral stand-in.

        Args:
            name: str, profile key.
            why:  str, the specific problem, included in the log and in ``reason``.

        Returns:
            Profile.neutral carrying ``why``.
        """
        logger.warning(f"control profile '{name}' disabled — {why}")
        return cls.neutral(name, reason=why)

    # --- evaluation ---------------------------------------------------------
    @property
    def usable(self) -> bool:
        """bool — True when this profile has a curve and can produce a non-zero reference."""
        return self.curve is not None

    def command(self, angle: float, velocity: float, side: str) -> float:
        """Evaluate the profile for one leg.

        Args:
            angle:    float, knee angle in rad (already unit-converted by the source).
            velocity: float, knee angular velocity in rad/s.
            side:     str, "r" or "l". Anything else is treated as "r" — the caller passes a
                      literal, so a typo is a programming error, not a runtime condition worth
                      an exception on the control thread.

        Returns:
            float, the motor-side position reference, saturated to this side's limits. Always
            finite: a non-finite input lands on the curve's endpoint clamp, and the saturation
            bounds the result regardless. Returns 0.0 when the profile is neutral.
        """
        if self.curve is None:
            return 0.0
        if side == "l":
            sign, (lo, hi) = self.sign_l, self.limits_l
        else:
            sign, (lo, hi) = self.sign_r, self.limits_r
        vel = velocity if math.isfinite(velocity) else 0.0
        raw = sign * (self.gain * self.curve(angle) + self.kv_ff * vel)
        if not math.isfinite(raw):
            return 0.0
        return lo if raw < lo else (hi if raw > hi else raw)


# The fallback used whenever no mode is valid: commands zero on both sides, always.
NEUTRAL = Profile.neutral("neutral", reason="no active mode")
