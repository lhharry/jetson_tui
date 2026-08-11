"""1-D interpolation for assistance lookup curves — cubic (not-a-knot) and linear.

These curves replace the ``Lookup_n-D`` blocks in the Simulink models, which are configured
``InterpMethod = Cubic spline`` / ``ExtrapMethod = Cubic spline``. MATLAB's ``spline`` and
``interp1(..., 'spline')`` both use **not-a-knot** end conditions, not natural ones, and the two
differ visibly over the first and last intervals — precisely where an assistance table tends to
carry its steepest segment. So not-a-knot is what is implemented here, and ``"linear"`` is
offered alongside as the exactly-reproducible fallback for anyone who would rather compare
against a hand-computed table than trust a boundary condition.

**Extrapolation clamps to the endpoint value** rather than continuing the cubic. This departs
from the Simulink blocks' spline extrapolation on purpose: a cubic diverges cubically, so a
single corrupt frame reporting a 200 deg knee angle would produce an enormous command, whereas
the physical meaning of "past the end of the table" is "keep doing what the last entry says".
Per-side saturation in ``profile`` bounds it either way; this makes the intent explicit instead
of relying on a limit downstream.

No scipy. numpy is a hard dependency of this project and scipy is not — installing it on a
Jetson for one interpolator is not a trade worth making. The solve is a single small dense
system done once at construction; evaluation is a bisect plus a Horner step.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

INTERP_METHODS = ("spline", "linear")


class Curve:
    """An interpolating curve through a table of (x, y) points, clamped outside its range.

    Constructed once from config and then called on the control thread, so all the solving
    happens in ``__init__`` and ``__call__`` is O(log n) with no allocation.
    """

    def __init__(self, x, y, *, interp: str = "spline") -> None:
        """Build the curve and precompute its coefficients.

        Args:
            x:      sequence[float], breakpoints. Must be strictly increasing and the same
                    length as ``y``. Units are whatever the caller works in — ``profile``
                    converts its table to radians before constructing.
            y:      sequence[float], the value at each breakpoint.
            interp: str, ``"spline"`` (cubic, not-a-knot) or ``"linear"``.

        Returns:
            None. Raises ValueError if the table is empty, ragged, non-monotonic, or if
            ``interp`` is not one of ``INTERP_METHODS``. Raising is deliberate: the caller
            (``profile.Profile``) turns a bad table into a neutral profile with a logged
            reason, which is far safer than letting ``bisect`` index into a table whose
            ordering assumption does not hold.
        """
        if interp not in INTERP_METHODS:
            raise ValueError(f"unknown interp {interp!r}; expected one of {INTERP_METHODS}")
        xs = [float(v) for v in x]
        ys = [float(v) for v in y]
        if not xs:
            raise ValueError("interpolation table is empty")
        if len(xs) != len(ys):
            raise ValueError(f"table is ragged: {len(xs)} breakpoints, {len(ys)} values")
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("breakpoints must be strictly increasing")
        if not all(np.isfinite(v) for v in xs + ys):
            raise ValueError("table contains non-finite values")

        self.x = xs
        self.y = ys
        self.interp = interp
        self._m = self._second_derivatives() if interp == "spline" else None

    # --- construction -------------------------------------------------------
    def _second_derivatives(self) -> list[float]:
        """Solve for the spline's second derivative at each breakpoint.

        Args:    none — uses ``self.x`` / ``self.y``.
        Returns: list[float] of length ``len(self.x)``, the moments M_i = S''(x_i).

        Uses the classical moment formulation: the interior rows enforce continuity of the
        first derivative, and the two boundary rows enforce not-a-knot, i.e. continuity of the
        *third* derivative across the second and second-to-last breakpoints, which is the same
        as saying the first two pieces are one cubic and the last two are one cubic.

        Short tables are special-cased because not-a-knot is not defined for them:
          n == 1  ->  constant, no curvature
          n == 2  ->  a straight line, no curvature
          n == 3  ->  the two not-a-knot conditions collapse into one, leaving the system
                      singular; the curve that satisfies them is the single parabola through
                      all three points, whose second derivative is the constant below.
        """
        n = len(self.x)
        if n <= 2:
            return [0.0] * n
        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        slope = [(self.y[i + 1] - self.y[i]) / h[i] for i in range(n - 1)]
        if n == 3:
            m = 2.0 * (slope[1] - slope[0]) / (h[0] + h[1])
            return [m, m, m]

        a = np.zeros((n, n), dtype=float)
        rhs = np.zeros(n, dtype=float)
        for i in range(1, n - 1):
            a[i, i - 1] = h[i - 1]
            a[i, i] = 2.0 * (h[i - 1] + h[i])
            a[i, i + 1] = h[i]
            rhs[i] = 6.0 * (slope[i] - slope[i - 1])
        # Not-a-knot at x[1]:  (M1 - M0)/h0 == (M2 - M1)/h1
        a[0, 0] = h[1]
        a[0, 1] = -(h[0] + h[1])
        a[0, 2] = h[0]
        # Not-a-knot at x[n-2]: (M[n-2] - M[n-3])/h[n-3] == (M[n-1] - M[n-2])/h[n-2]
        a[n - 1, n - 3] = h[n - 2]
        a[n - 1, n - 2] = -(h[n - 3] + h[n - 2])
        a[n - 1, n - 1] = h[n - 3]
        return [float(v) for v in np.linalg.solve(a, rhs)]

    # --- evaluation ---------------------------------------------------------
    def __call__(self, xq: float) -> float:
        """Interpolate at one point.

        Args:
            xq: float, the query point, in the same units as the breakpoints.

        Returns:
            float. Inside the table's range, the interpolated value; outside it, the nearest
            endpoint value (see the module docstring on why extrapolation clamps). A non-finite
            ``xq`` also returns the nearest endpoint — treating it as "off the end" keeps a
            corrupt frame from propagating NaN into the position loop.

        Scalar-only and allocation-free: this runs on the control thread every tick. Use
        ``evaluate_many`` for arrays.
        """
        if not np.isfinite(xq):
            return self.y[0]
        if xq <= self.x[0]:
            return self.y[0]
        if xq >= self.x[-1]:
            return self.y[-1]
        i = bisect_right(self.x, xq) - 1
        dx = xq - self.x[i]
        h = self.x[i + 1] - self.x[i]
        if self._m is None:  # linear
            return self.y[i] + (self.y[i + 1] - self.y[i]) * dx / h
        mi, mj = self._m[i], self._m[i + 1]
        c = mi / 2.0
        d = (mj - mi) / (6.0 * h)
        b = (self.y[i + 1] - self.y[i]) / h - h * (2.0 * mi + mj) / 6.0
        return self.y[i] + dx * (b + dx * (c + dx * d))

    def evaluate_many(self, xs) -> np.ndarray:
        """Interpolate at many points — for tests, plots and offline analysis, never the loop.

        Args:
            xs: sequence[float] | np.ndarray, query points.

        Returns:
            np.ndarray of float64, same length as ``xs``.
        """
        return np.array([self(float(v)) for v in np.asarray(xs, dtype=float).ravel()])
