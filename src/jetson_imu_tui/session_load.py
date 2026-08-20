"""Read a recorded session back off disk for the browser's offline viewer.

``Recorder`` writes one folder per session, ``<log_dir>/YYYY_MM_DD/HH_MM_SS/``, holding one CSV
per signal and per telemetry group. This module turns such a folder back into the columnar
arrays uPlot draws, so a run can be reviewed after the fact with the same charts that showed it
live. Read-only: nothing here writes, deletes or renames.

Three things make this more than ``csv.reader``:

**Decimation happens here, not in the browser.** Ten minutes at 200 Hz is 120k rows across ~23
channels; shipping that as JSON would be tens of megabytes to draw a few thousand pixels. Each
window is reduced to ``max_points`` per channel by a min/max envelope, which — unlike picking
every Nth row — cannot hide the one-sample spike or dropout the viewer exists to find. Zooming
re-requests a narrower window, so detail appears as it is needed and full resolution is
reachable everywhere.

**The time column has no date.** ``Recorder`` writes ``%H:%M:%S.%f``, so a session running past
midnight wraps to 00:00 and naive parsing reports a negative duration. The folder name supplies
the date and a backward step adds a day, which is correct for any session shorter than 24 h.

**Files may be missing, and that is information.** An I2C recording has no ``knee.csv`` because
that source has no knee channels — distinct from a serial recording whose knees read zero. A
missing file is reported as a missing group rather than as an error or as zeros.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from jetson_imu_tui.imu_common import SIGNAL_UNITS, TELEMETRY_GROUPS
from jetson_imu_tui.recorder import telemetry_filename

# Signal key -> the file Recorder writes it to. Mirrors the ``layout`` dict in Recorder.__enter__;
# both sides name the same four files and neither invents a fifth.
SIGNAL_FILES: dict[str, str] = {
    "euler": "euler_angles.csv",
    "accel": "accelerometers.csv",
    "gyro": "gyroscopes.csv",
    "quat": "quaternions.csv",
}

# The file whose row count and time span describe the session as a whole. Every sample-rate file
# is written from one drain batch, so they all agree; this one is simply always present.
_SPAN_FILE = "accelerometers.csv"

TIME_FMT = "%H:%M:%S.%f"
FOLDER_DATE_FMT = "%Y_%m_%d"
FOLDER_TIME_FMT = "%H_%M_%S"

DEFAULT_MAX_POINTS = 4000


def _session_start(folder: Path) -> datetime | None:
    """Folder path -> the wall-clock instant the recording started, or None if unparseable.

    Input:  ``folder`` = Path ending in ``YYYY_MM_DD/HH_MM_SS``.
    Output: datetime (local, naive) or None.

    The folder name is the only place the date exists — the CSVs carry time of day alone.
    """
    try:
        return datetime.strptime(
            f"{folder.parent.name} {folder.name}", f"{FOLDER_DATE_FMT} {FOLDER_TIME_FMT}"
        )
    except ValueError:
        return None


def _parse_times(raw: list[str]) -> np.ndarray:
    """Time-of-day strings -> float seconds relative to the first row, monotonically increasing.

    Input:  ``raw`` = list[str] of ``%H:%M:%S.%f`` values in file order.
    Output: np.ndarray float64, shape (len(raw),), starting at 0.0. An unparseable row inherits
            the previous value so one bad line cannot shift everything after it.

    A backward step means the clock passed midnight, so a day is added from there on. That is
    the correct reading for any session shorter than 24 h, which is every session — the wrap
    otherwise shows up as a negative duration and breaks every from/to filter downstream.
    """
    out = np.zeros(len(raw), dtype=np.float64)
    day = 0.0
    prev = None
    base = None
    for i, cell in enumerate(raw):
        try:
            tod = datetime.strptime(cell.strip(), TIME_FMT)
        except ValueError:
            out[i] = out[i - 1] if i else 0.0
            continue
        secs = tod.hour * 3600 + tod.minute * 60 + tod.second + tod.microsecond / 1e6
        if prev is not None and secs < prev:
            day += 86400.0
        prev = secs
        secs += day
        if base is None:
            base = secs
        out[i] = secs - base
    return out


def _read_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """One recorded CSV -> (times, {column: values}). None if the file is absent or has no rows.

    Input:  ``path`` = Path to a Recorder-written CSV whose first column is ``Time``.
    Output: (np.ndarray float64, seconds from session start,
             dict[str, np.ndarray float64], one entry per remaining column, empty cells NaN)
            or None.

    Empty cells become NaN rather than 0.0: the recorder writes an empty cell precisely when it
    had no value, and zero is a meaningful reading for every channel in these files.
    """
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    times = _parse_times([r[0] if r else "" for r in body])
    cols: dict[str, np.ndarray] = {}
    for j, name in enumerate(header[1:], start=1):
        vals = np.full(len(body), np.nan, dtype=np.float64)
        for i, r in enumerate(body):
            if j < len(r) and r[j] != "":
                try:
                    vals[i] = float(r[j])
                except ValueError:
                    pass
        cols[name] = vals
    return times, cols


def _envelope(
    times: np.ndarray, cols: dict[str, np.ndarray], max_points: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reduce every column to at most ``max_points`` samples, preserving extremes.

    Input:  ``times`` = np.ndarray float64 (n,); ``cols`` = dict[str, np.ndarray float64 (n,)];
            ``max_points`` = int, the per-channel budget.
    Output: (np.ndarray float64 (m,), dict[str, np.ndarray float64 (m,)]), m <= max_points.
            The input is returned untouched when it already fits.

    Buckets are cut on **time**, not per channel, so every column keeps sharing one x array —
    which is what uPlot wants, and what makes a cursor readout across charts mean anything.
    Within a bucket each channel contributes its min then its max, placed at the bucket's first
    and last timestamp. That pair can therefore be ordered opposite to how the samples actually
    occurred, distorting time by less than one bucket width; it is invisible at any zoom where
    bucketing is active, and the alternative — per-channel timestamps — needs one x array per
    channel.
    """
    n = len(times)
    if n <= max_points or max_points < 2:
        return times, cols
    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    lo, hi = edges[:-1], edges[1:]
    keep = lo < hi
    lo, hi = lo[keep], hi[keep]
    t_out = np.empty(len(lo) * 2, dtype=np.float64)
    t_out[0::2] = times[lo]
    t_out[1::2] = times[hi - 1]
    out: dict[str, np.ndarray] = {}
    for name, vals in cols.items():
        v = np.empty(len(lo) * 2, dtype=np.float64)
        for k, (a, b) in enumerate(zip(lo, hi)):
            chunk = vals[a:b]
            if np.all(np.isnan(chunk)):
                v[2 * k] = v[2 * k + 1] = np.nan
            else:
                v[2 * k] = np.nanmin(chunk)
                v[2 * k + 1] = np.nanmax(chunk)
        out[name] = v
    return t_out, out


def _jsonable(vals: np.ndarray) -> list:
    """NumPy column -> a list ``JSON.parse`` accepts: NaN becomes null, the rest floats.

    ``json.dumps`` emits a bare ``NaN`` token, which is valid Python and invalid JSON — the
    browser's parser rejects the entire response over one gap. uPlot reads null as a break in
    the line, which is exactly what a missing sample is.
    """
    return [None if np.isnan(v) else float(v) for v in vals]


def _split_signal_columns(cols: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """``{"Left_x": [...]}`` -> ``{"x": {"Left": [...]}}`` — axis first, then sensor label.

    Input:  ``cols`` = dict[str, np.ndarray], keys as ``Recorder._hdr`` writes them.
    Output: dict[axis, dict[label, np.ndarray]]. A column with no underscore is skipped.

    Axis-major because that is how the charts are cut: one chart per axis, one line per sensor.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, vals in cols.items():
        label, _, axis = name.rpartition("_")
        if not label or not axis:
            continue
        out.setdefault(axis, {})[label] = vals
    return out


def _count_rows(path: Path) -> int:
    """Data rows in a CSV, header excluded; 0 if absent. Counts bytes, never parses."""
    if not path.is_file():
        return 0
    n = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            n += block.count(b"\n")
    return max(0, n - 1)


def list_sessions(log_dir, limit: int | None = None) -> list[dict]:
    """Every recorded session under ``log_dir``, newest first.

    Input:  ``log_dir`` = str | Path, the configured recording root; ``limit`` = int | None,
            keep only the newest N.
    Output: list[dict], each
            {"id": "YYYY_MM_DD/HH_MM_SS", "date": str, "time": str,
             "started": str | None, "n_rows": int, "duration_s": float | None,
             "signals": list[str], "telemetry": list[str], "has_cls": bool}

    A folder whose name is not a date/time is skipped rather than reported, so ``axis_remap.json``
    and anything else living in the log directory cannot appear as a session.
    """
    root = Path(log_dir).expanduser()
    if not root.is_dir():
        return []
    out: list[dict] = []
    for day in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        for folder in sorted((p for p in day.iterdir() if p.is_dir()), reverse=True):
            started = _session_start(folder)
            if started is None:
                continue
            span = _read_csv(folder / _SPAN_FILE)
            duration = float(span[0][-1]) if span is not None and len(span[0]) else None
            out.append({
                "id": f"{day.name}/{folder.name}",
                "date": day.name,
                "time": folder.name,
                "started": started.isoformat(sep=" ", timespec="milliseconds"),
                "n_rows": _count_rows(folder / _SPAN_FILE),
                "duration_s": duration,
                "signals": [k for k, f in SIGNAL_FILES.items() if (folder / f).is_file()],
                "telemetry": [
                    g for g, _c, _u in TELEMETRY_GROUPS
                    if (folder / telemetry_filename(g)).is_file()
                ],
                "has_cls": (folder / "cls.csv").is_file(),
            })
            if limit is not None and len(out) >= limit:
                return out
    return out


def load_session(
    log_dir,
    session_id: str,
    *,
    t_from: float | None = None,
    t_to: float | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict:
    """One recorded session as columnar arrays ready for uPlot.

    Input:
        log_dir:    str | Path, the recording root.
        session_id: str, "YYYY_MM_DD/HH_MM_SS" as ``list_sessions`` reports it.
        t_from:     float | None, window start in seconds from session start (None = beginning).
        t_to:       float | None, window end, same units (None = end).
        max_points: int, per-channel budget after decimation.

    Output: dict
        {"id": str, "started": str | None, "span": [float, float],
         "window": [float, float], "max_points": int, "labels": list[str],
         "signals":   {signal: {"unit": str, "t": list[float],
                                "axes": {axis: {label: list[float | None]}}}},
         "telemetry": {group:  {"unit": str, "t": list[float],
                                "channels": {channel: list[float | None]}}},
         "missing": {"signals": list[str], "telemetry": list[str]}}

    Raises FileNotFoundError if the folder does not exist or holds no readable session.

    Each file is windowed and decimated on its own ``t`` rather than on a session-wide one. The
    recorder does keep the sample-rate files in lockstep, but a truncated file — a session
    killed mid-write — would otherwise misalign every channel silently, and this costs nothing.
    """
    root = Path(log_dir).expanduser()
    folder = root / session_id
    if not folder.is_dir():
        raise FileNotFoundError(f"no such session: {session_id}")
    started = _session_start(folder)

    span = _read_csv(folder / _SPAN_FILE)
    if span is None:
        raise FileNotFoundError(f"session {session_id} has no readable {_SPAN_FILE}")
    span_t = span[0]
    t0, t1 = float(span_t[0]), float(span_t[-1])
    lo = t0 if t_from is None else max(t0, float(t_from))
    hi = t1 if t_to is None else min(t1, float(t_to))
    if hi <= lo:
        lo, hi = t0, t1
    budget = max(2, int(max_points))

    def window(times: np.ndarray, cols: dict[str, np.ndarray]):
        mask = (times >= lo) & (times <= hi)
        if not mask.any():
            return None
        return _envelope(times[mask], {k: v[mask] for k, v in cols.items()}, budget)

    labels: list[str] = []
    signals: dict[str, dict] = {}
    missing_signals: list[str] = []
    for key, fname in SIGNAL_FILES.items():
        parsed = _read_csv(folder / fname)
        cut = window(*parsed) if parsed is not None else None
        if cut is None:
            missing_signals.append(key)
            continue
        times, cols = cut
        axes = _split_signal_columns(cols)
        # A signal with no numbers anywhere is what the recorder writes for a channel the source
        # cannot produce (euler/quat over serial). Report it absent, rather than as a chart of
        # nulls the viewer has to squint at to recognise as empty.
        if not any(
            not np.all(np.isnan(v)) for per_label in axes.values() for v in per_label.values()
        ):
            missing_signals.append(key)
            continue
        for per_label in axes.values():
            for label in per_label:
                if label not in labels:
                    labels.append(label)
        signals[key] = {
            "unit": SIGNAL_UNITS.get(key, ""),
            "t": [float(v) for v in times],
            "axes": {
                axis: {label: _jsonable(vals) for label, vals in per_label.items()}
                for axis, per_label in axes.items()
            },
        }

    telemetry: dict[str, dict] = {}
    missing_telemetry: list[str] = []
    for group, channels, unit in TELEMETRY_GROUPS:
        parsed = _read_csv(folder / telemetry_filename(group))
        cut = window(*parsed) if parsed is not None else None
        if cut is None:
            missing_telemetry.append(group)
            continue
        times, cols = cut
        telemetry[group] = {
            "unit": unit,
            "t": [float(v) for v in times],
            # Iterate the registry, not the file: a column the file lacks is reported as an
            # empty series under its proper name instead of vanishing from the chart.
            "channels": {ch: _jsonable(cols[ch]) if ch in cols else [] for ch in channels},
        }

    return {
        "id": session_id,
        "started": started.isoformat(sep=" ", timespec="milliseconds") if started else None,
        "span": [t0, t1],
        "window": [lo, hi],
        "max_points": budget,
        "labels": labels,
        "signals": signals,
        "telemetry": telemetry,
        "missing": {"signals": missing_signals, "telemetry": missing_telemetry},
    }
