"""Lightweight Unicode-Braille plot canvas (one widget per panel)."""

from __future__ import annotations

from typing import Sequence

from rich.text import Text
from textual.widget import Widget


_DOT = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)


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
        self._y_range: tuple[float, float] = (0.0, 1.0)
        self._title = ""

    def set_plot(
        self,
        series: list[tuple[str, str, Sequence[float], Sequence[float]]],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        title: str = "",
    ) -> None:
        self._series = series
        self._x_range = x_range
        self._y_range = y_range
        self._title = title
        self.refresh()

    def render(self) -> Text:
        w = max(8, self.size.width)
        full_h = max(2, self.size.height)
        body_h = max(1, full_h - 1)
        dots_w = w * 2
        dots_h = body_h * 4

        bits = [[0] * w for _ in range(body_h)]
        colors: list[list[str | None]] = [[None] * w for _ in range(body_h)]

        xmin, xmax = self._x_range
        ymin, ymax = self._y_range
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or 1.0

        for _name, color, xs, ys in self._series:
            for x, y in zip(xs, ys):
                fx = (x - xmin) / xspan
                fy = (y - ymin) / yspan
                if fx < 0.0:
                    fx = 0.0
                elif fx > 1.0:
                    fx = 1.0
                if fy < 0.0:
                    fy = 0.0
                elif fy > 1.0:
                    fy = 1.0
                dx = int(fx * (dots_w - 1))
                dy = int((1.0 - fy) * (dots_h - 1))
                cc = dx >> 1
                cr = dy >> 2
                bits[cr][cc] |= _DOT[dx & 1][dy & 3]
                if colors[cr][cc] is None:
                    colors[cr][cc] = color

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
        out.append("\n")

        for row in range(body_h):
            row_bits = bits[row]
            row_colors = colors[row]
            for col in range(w):
                b = row_bits[col]
                if b:
                    out.append(chr(0x2800 + b), style=row_colors[col] or "white")
                else:
                    out.append(" ")
            if row < body_h - 1:
                out.append("\n")
        return out
