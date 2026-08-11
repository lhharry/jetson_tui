"""The position loop lifted out of Simulink's ``PID3`` / ``PID4`` subsystems.

Despite the block name, the original is **not** a textbook PID. Read off its wiring and the
sign bits on its Sum blocks (``Sum.Inputs = "|-+"``, ``Sum5.Inputs = "|+|+|-"``):

    e(t) = ref - actual
    u(t) = Kp*e + Ki*integral(e) - Kv*ydot(t)          with  y = LPF2(u),  ydot = dy/dt

The third term is not a derivative of the error. It is a negative feedback of the controller's
**own output** rate: ``u`` is fed through a second-order lowpass and the derivative of *that*
is subtracted back. It exists to damp jitter in the velocity command, and it is what makes this
loop stable enough to be worth keeping as-is rather than replacing with a stock PID.

The lowpass, from ``Subsystem2`` inside the same block:

    ydot = wn*v
    vdot = wn*[(x - y) - 2*zt*v]        =>   yddot + 2*zt*wn*ydot + wn^2*y = wn^2*x

i.e. a standard second-order lowpass with damping ratio ``zt`` and natural frequency ``wn``,
initial output ``x0``. Only ``ydot`` is used; the model terminates ``y``.

Gains as found in the models: ``Kp = 7.5``, ``Ki = 0`` (the integrator is wired but its gain
block is zero, so it is effectively off), ``Kv = 0.02``. ``wn`` / ``zt`` / ``x0`` live in the
MATLAB base workspace and are *not* recoverable from the ``.slx`` — they must be supplied.

Two differences from the original, both forced by moving the loop off the microcontroller:

* **Sampling.** Simulink runs this at a fixed 10 ms step. Here the tick is a Python thread, so
  ``dt`` is measured rather than assumed, and a tick that lands far outside its nominal period
  resets the state instead of integrating one enormous step.
* **Anti-windup.** The original has ``Ki = 0`` and therefore cannot wind up. The moment anyone
  raises ``ki`` in config, a saturated output with a live integrator would. Integration is
  frozen while the output is clamped.

The loop now closes across a serial link, which adds ~20-40 ms of round-trip delay that the
original did not have. That is a real stability hazard at ``Kp = 7.5``; see the plan's risk
section. Start low and raise it with the recorded command in view.
"""

from __future__ import annotations

import math

# When a measured tick counts as a *discontinuity* rather than merely a slow tick.
#
# The distinction matters more than it looks. Jitter is normal and harmless: the OS scheduler,
# the GIL under torch inference and Windows' ~15 ms timer granularity all routinely stretch a
# tick to several times its nominal period, and integrating that longer step is simply correct.
# What must never be integrated is a *stall* — a source switch, a GC pause, a suspended
# process — where the elapsed time bears no relation to the control problem.
#
# So the threshold is absolute, not a tight ratio. An earlier version rejected anything beyond
# 2x nominal, which at a 100-200 Hz tick meant ordinary jitter reset the loop on nearly every
# tick and the controller silently never commanded anything at all.
DT_STALL_RATIO = 10.0   # ... more than this multiple of the nominal period,
DT_STALL_S = 0.25       # ... or this many seconds, whichever comes first.

# A tick can also arrive *too early*: when the requested rate is finer than the OS timer
# granularity (Windows' ~15.6 ms against a 5-10 ms period), the loop's overrun resync fires two
# ticks back to back and the second one measures dt ~ 0. That is not a discontinuity — almost no
# time has passed — and the correct output is simply the previous one. Resetting on it instead
# made the command alternate between its true value and zero on every other tick, i.e. chatter
# the motor at half the loop rate. So a short dt is floored, not rejected: the integrator and
# the lowpass advance by a negligible amount and the output is unchanged.
DT_FLOOR_RATIO = 1e-3


class SecondOrderLowpass:
    """``yddot + 2*zt*wn*ydot + wn^2*y = wn^2*x``, integrated at a fixed step.

    State is ``(y, v)`` where ``ydot = wn*v``; this is the same state the Simulink subsystem
    carries in its two integrators, so a reset here matches a reset there.
    """

    def __init__(self, wn: float, zt: float, x0: float = 0.0) -> None:
        """Configure the filter and put it in its reset state.

        Args:
            wn: float, natural frequency in rad/s. Must be > 0; a non-positive value makes the
                filter a pass-through of zero, which is inert rather than explosive.
            zt: float, damping ratio, dimensionless. 1.0 is critically damped.
            x0: float, initial value of ``y`` on reset, in the same units as the input.

        Returns:
            None.
        """
        self.wn = float(wn)
        self.zt = float(zt)
        self.x0 = float(x0)
        self.y = self.x0
        self.v = 0.0

    def reset(self) -> None:
        """Return the filter to its initial state (``y = x0``, ``v = 0``).

        Args:    none.
        Returns: None.
        """
        self.y = self.x0
        self.v = 0.0

    def step(self, x: float, dt: float) -> float:
        """Advance one step and return the derivative of the filtered signal.

        Args:
            x:  float, the input sample (here: the controller's own output ``u``).
            dt: float, step length in seconds. Must be > 0.

        Returns:
            float, ``ydot = wn * v`` after the step — the only output the control law uses.
            Returns 0.0 if the filter is inert (``wn <= 0``) or if the state has gone
            non-finite, which also resets it: a NaN here would otherwise poison every
            subsequent command.

        Integrated with the explicit midpoint method (RK2) rather than forward Euler. Forward
        Euler on an oscillator is unconditionally growing in amplitude, so at a large ``wn``
        relative to the tick it would add energy to exactly the loop we are damping.
        """
        if self.wn <= 0.0 or dt <= 0.0:
            return 0.0

        def deriv(y: float, v: float) -> tuple[float, float]:
            return self.wn * v, self.wn * ((x - y) - 2.0 * self.zt * v)

        dy1, dv1 = deriv(self.y, self.v)
        dy2, dv2 = deriv(self.y + 0.5 * dt * dy1, self.v + 0.5 * dt * dv1)
        self.y += dt * dy2
        self.v += dt * dv2
        if not (math.isfinite(self.y) and math.isfinite(self.v)):
            self.reset()
            return 0.0
        return self.wn * self.v


class VelocityPid:
    """Position error in, motor velocity command out — the loop from ``PID3`` / ``PID4``.

    Stateful (integrator + lowpass), so it must be stepped at a steady rate; ``ControlService``
    owns that. One instance per leg.
    """

    def __init__(
        self,
        *,
        kp: float,
        ki: float,
        kv: float,
        wn: float,
        zt: float,
        x0: float = 0.0,
        vel_limit: float,
        nominal_dt: float,
    ) -> None:
        """Configure the loop and put it in its reset state.

        Args:
            kp:         float, proportional gain, (rad/s) per rad of position error.
            ki:         float, integral gain, (rad/s) per (rad*s). 0 disables the term, which
                        is how the Simulink models ship.
            kv:         float, output-rate damping gain, applied to the lowpass derivative.
            wn:         float, lowpass natural frequency in rad/s.
            zt:         float, lowpass damping ratio.
            x0:         float, lowpass initial output.
            vel_limit:  float, output saturation magnitude in rad/s (see serial_service's
                        VEL_LIMIT; this one is the controller's own, usually the same).
            nominal_dt: float, the expected tick period in seconds. Only used to judge whether
                        a measured ``dt`` is plausible.

        Returns:
            None.
        """
        self.kp = float(kp)
        self.ki = float(ki)
        self.kv = float(kv)
        self.vel_limit = abs(float(vel_limit))
        self.nominal_dt = float(nominal_dt)
        self._lp = SecondOrderLowpass(wn, zt, x0)
        self._integral = 0.0
        self._enabled_prev = False
        # Last values, for telemetry — the recorded error is how loop instability is diagnosed.
        self.last_error = 0.0
        self.last_output = 0.0

    def reset(self) -> None:
        """Clear the integrator and the lowpass, and forget the enable edge.

        Args:    none.
        Returns: None.

        Mirrors the Simulink enabled subsystem's ``StatesWhenEnabling = "reset"``. Called on a
        disable, on a stale sample, and on an implausible ``dt`` — anywhere continuity of the
        state would be a fiction.
        """
        self._lp.reset()
        self._integral = 0.0
        self._enabled_prev = False
        self.last_error = 0.0
        self.last_output = 0.0

    def step(self, ref: float, actual: float, dt: float, *, enable: bool) -> float:
        """Advance the controller one tick and return the velocity command.

        Args:
            ref:    float, position reference in rad (motor side, the profile's output).
            actual: float, measured motor position in rad, as reported after Simulink's
                    CHINESE CORRECTION unwrap — not the raw CAN value.
            dt:     float, the tick's *measured* duration in seconds. Measured, not nominal:
                    a tick that overruns makes the nominal period a lie, and both the
                    integrator and the second-order state are sensitive to dt. Ordinary jitter
                    is integrated as-is; a value beyond the stall threshold (see DT_STALL_*)
                    resets the state and returns 0.0 instead, because a multi-tick gap must
                    not be integrated as one huge step.
            enable: bool, the device's SWITCH line. A False->True edge zeroes the integrator
                    and the lowpass, mirroring StatesWhenEnabling = "reset".

        Returns:
            float, velocity command in rad/s, saturated to +/-vel_limit. Returns 0.0 while
            disabled — zero velocity means the motor holds still, which is the real safe value
            for this loop. (A zero *position reference* would instead command the leg back to
            zero, which is why the reference boundary was moved downstream of this block.)

        Anti-windup: the integrator only accumulates when the output is not clamped, so raising
        ``ki`` cannot build up a charge that has to unwind before the loop responds again.
        """
        if not enable:
            if self._enabled_prev:
                self.reset()
            self._enabled_prev = False
            self.last_error = 0.0
            self.last_output = 0.0
            return 0.0
        if not self._enabled_prev:
            self.reset()          # rising edge: start from a known state, as Simulink does
            self._enabled_prev = True

        stall = min(DT_STALL_RATIO * self.nominal_dt, DT_STALL_S)
        if not math.isfinite(dt) or dt < 0.0 or dt > stall:
            self.reset()
            self._enabled_prev = True   # still enabled; only the state is discontinuous
            return 0.0
        dt = max(dt, DT_FLOOR_RATIO * self.nominal_dt)
        if not (math.isfinite(ref) and math.isfinite(actual)):
            self.reset()
            self._enabled_prev = True
            return 0.0

        error = ref - actual
        ydot = self._lp.step(self.last_output, dt)
        raw = self.kp * error + self.ki * self._integral - self.kv * ydot

        out = raw
        clamped = False
        if out > self.vel_limit:
            out, clamped = self.vel_limit, True
        elif out < -self.vel_limit:
            out, clamped = -self.vel_limit, True
        if not math.isfinite(out):
            self.reset()
            self._enabled_prev = True
            return 0.0
        # Anti-windup: integrate only while the output has room to move.
        if not clamped:
            self._integral += error * dt

        self.last_error = error
        self.last_output = out
        return out
