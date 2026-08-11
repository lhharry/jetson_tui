"""Real-time assistance control: knee signals + a locomotion mode -> motor velocity commands.

This package holds everything the Simulink models used to do between reading the sensors and
packing a CAN frame: the per-mode assistance profile (a lookup curve on joint angle plus a
velocity feed-forward term), and the position loop whose output is the velocity command sent
back over the serial link. Simulink keeps only the CAN driver layer.

Layered so the numerics can be tested without threads, and the threading without hardware:

* ``spline``  — cubic (not-a-knot) and linear interpolation. Pure, no state.
* ``profile`` — one assistance profile: curve + feed-forward + gain + per-side saturation.
                Pure and **stateless**, which is what lets the loop act on the newest sample
                rather than replaying a backlog.
* ``pid``     — the position loop. **Stateful**, hence a fixed-rate tick.
* ``service`` — ``ControlService``: the thread that ties them together and owns the invariant
                that the link never goes silent while the controller is enabled.
"""
