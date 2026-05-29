"""Lightweight Unicode-Braille plot canvas (one widget per panel).

Rendering is tuned for low CPU and a solid, smooth-looking line:

* Dense series (more samples than horizontal dot columns) are drawn as a
  per-column min/max **envelope** — a continuous vertical fill per column,
  bridged to its neighbour — so the trace never breaks up into scattered dots.
* Sparse series fall back to Bresenham segments between points.
* Lines are 2 dots thick so flat/diagonal runs read as a line, not a dotted row.
* The Text is emitted with run-length grouping (one append per same-style run)
  instead of one append per cell.
* The y-range is auto-scaled but *sticky* (expand instantly, contract slowly,
  snapped to round bounds) so the curve doesn't jitter every frame.
"""

from __future__ import annotations

import math
from typing import Sequence

from rich.text import Text
from textual.widget import Widget


_DOT = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)

# Fraction the displayed y-range moves toward a smaller target each frame.
_CONTRACT = 0.1


def _draw_line(x0, y0, x1, y1, color, plot):
    """Bresenham line between two dot coordinates."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        plot(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _nice_step(span: float) -> float:
    """A round 1/2/5×10^k step that divides ``span`` into ~5 parts."""
    if span <= 0:
        return 1.0
    raw = span / 5.0
    mag = 10.0 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1.0:
        nice = 1.0
    elif norm <= 2.0:
        nice = 2.0
    elif norm <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * mag


class BrailleCanvas(Widget):
    DEFAULT_CSS = """
    BrailleCanvas {
        height: 1fr;
        min-height: 5;
        border: round $accent;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._series: list[tuple[str, str, Sequence[float], Sequence[float]]] = []
        self._x_range: tuple[float, float] = (0.0, 1.0)
        self._y_target: tuple[float, float] = (0.0, 1.0)
        self._title = ""
        # Persisted (sticky) displayed y-range; None until first frame.
        self._y_lo: float | None = None
        self._y_hi: float | None = None

    def set_plot(
        self,
        series: list[tuple[str, str, Sequence[float], Sequence[float]]],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        title: str = "",
    ) -> None:
        """Update the plot. ``y_range`` is the *raw* data (min, max); the canvas
        applies its own padding and sticky smoothing on top."""
        self._series = series
        self._x_range = x_range
        self._y_target = y_range
        self._title = title
        self.refresh()

    def _sticky_yrange(self) -> tuple[float, float]:
        lo, hi = self._y_target
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, hi + 0.5
        margin = (hi - lo) * 0.08
        lo -= margin
        hi += margin
        step = _nice_step(hi - lo)
        t_lo = math.floor(lo / step) * step
        t_hi = math.ceil(hi / step) * step
        if self._y_lo is None or self._y_hi is None:
            self._y_lo, self._y_hi = t_lo, t_hi
        else:
            # Expand instantly to fit; contract slowly so the axis settles smoothly.
            self._y_lo = t_lo if t_lo < self._y_lo else self._y_lo + (t_lo - self._y_lo) * _CONTRACT
            self._y_hi = t_hi if t_hi > self._y_hi else self._y_hi + (t_hi - self._y_hi) * _CONTRACT
        return self._y_lo, self._y_hi

    def render(self) -> Text:
        w = max(8, self.size.width)
        full_h = max(2, self.size.height)
        body_h = max(1, full_h - 1)
        dots_w = w * 2
        dots_h = body_h * 4

        bits = [[0] * w for _ in range(body_h)]
        colors: list[list[str | None]] = [[None] * w for _ in range(body_h)]

        xmin, xmax = self._x_range
        ymin, ymax = self._sticky_yrange()
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or 1.0

        def plot_dot(dx: int, dy: int, color: str) -> None:
            cc = dx >> 1
            cr = dy >> 2
            bits[cr][cc] |= _DOT[dx & 1][dy & 3]
            if colors[cr][cc] is None:
                colors[cr][cc] = color

        def plot_thick(dx: int, dy: int, color: str) -> None:
            plot_dot(dx, dy, color)
            if dy + 1 < dots_h:
                plot_dot(dx, dy + 1, color)

        def x_to_dot(x: float) -> int:
            fx = (x - xmin) / xspan
            fx = 0.0 if fx < 0.0 else 1.0 if fx > 1.0 else fx
            return int(fx * (dots_w - 1))

        def y_to_dot(y: float) -> int:
            fy = (y - ymin) / yspan
            fy = 0.0 if fy < 0.0 else 1.0 if fy > 1.0 else fy
            return int((1.0 - fy) * (dots_h - 1))

        for _name, color, xs, ys in self._series:
            n = len(xs)
            if n == 0:
                continue
            if n >= dots_w:
                self._draw_envelope(xs, ys, color, dots_w, dots_h, x_to_dot, y_to_dot, plot_dot)
            else:
                prev: tuple[int, int] | None = None
                for x, y in zip(xs, ys):
                    dx, dy = x_to_dot(x), y_to_dot(y)
                    if prev is None:
                        plot_thick(dx, dy, color)
                    else:
                        _draw_line(prev[0], prev[1], dx, dy, color, plot_thick)
                    prev = (dx, dy)

        return self._emit(bits, colors, w, body_h, ymin, ymax)

    def _draw_envelope(self, xs, ys, color, dots_w, dots_h, x_to_dot, y_to_dot, plot_dot) -> None:
        """Per-column min/max fill, bridged between columns for a continuous line."""
        col_lo: list[int | None] = [None] * dots_w
        col_hi: list[int | None] = [None] * dots_w
        for x, y in zip(xs, ys):
            dx = x_to_dot(x)
            dy = y_to_dot(y)
            lo = col_lo[dx]
            if lo is None:
                col_lo[dx] = col_hi[dx] = dy
            else:
                if dy < lo:
                    col_lo[dx] = dy
                if dy > col_hi[dx]:  # type: ignore[operator]
                    col_hi[dx] = dy
        prev_lo: int | None = None
        prev_hi: int | None = None
        for dx in range(dots_w):
            lo = col_lo[dx]
            if lo is None:
                continue
            hi = col_hi[dx]
            draw_lo, draw_hi = lo, hi
            if prev_lo is not None:
                # Bridge any vertical gap to the previous column.
                if draw_hi < prev_lo:
                    draw_hi = prev_lo
                elif draw_lo > prev_hi:  # type: ignore[operator]
                    draw_lo = prev_hi
            if draw_hi == draw_lo:  # min 2-dot thickness so flat runs read as a line
                draw_hi = min(draw_hi + 1, dots_h - 1)
            for dy in range(draw_lo, draw_hi + 1):
                plot_dot(dx, dy, color)
            prev_lo, prev_hi = lo, hi

    def _emit(self, bits, colors, w, body_h, ymin, ymax) -> Text:
        out = Text(no_wrap=True, overflow="crop")
        if self._title:
            out.append(self._title, style="bold")
        seen: set[str] = set()
        for name, color, _, _ in self._series:
            if name in seen:
                continue
            seen.add(name)
            out.append("  ")
            out.append(name, style=color)
        out.append(f"   y:[{ymin:.2g}, {ymax:.2g}]", style="dim")
        out.append("\n")

        for row in range(body_h):
            row_bits = bits[row]
            row_colors = colors[row]
            col = 0
            while col < w:
                b = row_bits[col]
                if not b:
                    start = col
                    col += 1
                    while col < w and not row_bits[col]:
                        col += 1
                    out.append(" " * (col - start))
                else:
                    color = row_colors[col] or "white"
                    chars = [chr(0x2800 + b)]
                    col += 1
                    while col < w and row_bits[col] and (row_colors[col] or "white") == color:
                        chars.append(chr(0x2800 + row_bits[col]))
                        col += 1
                    out.append("".join(chars), style=color)
            if row < body_h - 1:
                out.append("\n")
        return out
