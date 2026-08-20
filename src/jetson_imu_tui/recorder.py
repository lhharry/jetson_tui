"""Threaded CSV recorder that writes the per-sensor signals and the device telemetry in lockstep.

Four per-sensor files always (quaternions / accelerometers / gyroscopes / euler_angles), plus one
file per **telemetry group** the source carries (``imu_common.TELEMETRY_GROUPS``: knee, motor,
trace, state) and the three CLS files when classification is running.

Every file that describes a *sample* is written from a single ``samples_since`` batch, so their
row counts are equal by construction rather than by luck. That is the whole reason telemetry
rides inside the sample rows instead of having its own cursor: two cursors would let frames
arrive between the two reads, and the files would silently drift apart — which is exactly the
alignment a post-hoc analysis of "what was the motor doing when the classifier said walk" needs.

The CLS files are the deliberate exception. ``model_input.csv`` and ``cls_vote.csv`` run at the
model's and the voter's own rates, so they keep their own cursors and their real timestamps
instead of being stretched onto the sample grid.

**``hz`` is the CSV row rate, not a polling cadence.** ``hz = 0`` writes every frame the source
produced. A positive ``hz`` thins the stream to about that many rows per second by *selecting
whole frames* — never averaging them, never repeating one. Every cell in a row is therefore a
number the device actually sent, and the row's ``Time`` is when it sent it; what a thinned
recording loses is only the frames in between, and the timestamps show exactly which.

Selection is gated on each sample's own timestamp rather than on a fixed "every Nth frame"
stride, because a stride has to be computed from the wire rate and the configured wire rate is
exactly the thing that tends to be wrong (see ``[source] sample_hz``). Gating on time cannot
inherit that error: ask for 50 rows/s and you get 50, whatever the device turns out to be doing,
and it stays right if the device's rate later changes. The cost is that the gap between adjacent
rows alternates between neighbouring frame counts (3 or 4 frames, at 161 Hz thinned to 50)
rather than being a constant stride.

How often the writer thread wakes is ``DRAIN_HZ``, a constant. It affects how much is batched
per write and how promptly rows reach the disk — nothing about their content — so it is not a
setting. Conflating the two is what made the old ``record_hz`` misleading: it looked like a row
rate and behaved like a wake-up interval.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from jetson_imu_tui.imu_common import TELEMETRY_CHANNELS, TELEMETRY_GROUPS

if TYPE_CHECKING:  # ImuService pulls in the Linux-only hardware stack; only needed for hints.
    from jetson_imu_tui.imu_service import ImuService


def _hdr(labels: list[str], axes: tuple[str, ...]) -> str:
    cols = ["Time"]
    for label in labels:
        cols.extend(f"{label}_{a}" for a in axes)
    return ",".join(cols) + "\n"


def telemetry_filename(group: str) -> str:
    """Group name -> its CSV filename. One place, so the loader and the writer cannot disagree."""
    return f"{group}.csv"


# How often the writer thread wakes to move buffered samples to disk. Not a row rate: each wake
# writes everything that arrived since the last one, so raising it shrinks the batches rather
# than producing more rows. 100 Hz keeps at most ~10 ms of data un-written without waking the
# thread pointlessly on a link running far slower.
DRAIN_HZ = 100.0

# Window for the reported row rate. Long enough to be steady at 1 row/s, short enough to react.
RATE_WINDOW_S = 5.0

# Slack when testing a sample against the row grid, in seconds. Covers float64 error in
# ``t - t0`` (~1e-11 s at a monotonic clock reading 1e5) with orders of magnitude to spare,
# while staying far below any real sample interval.
GRID_EPS = 1e-9


class Recorder:
    def __init__(self, service: "ImuService", log_dir: Path, hz: float, cls=None) -> None:
        self._service = service
        self._labels = service.labels
        # Target CSV rows per second. 0 (or less) means every frame — see the module docstring
        # for why this is a time gate rather than an "every Nth frame" stride.
        self._hz = max(0.0, float(hz))
        self._row_dt = 0.0 if self._hz <= 0 else 1.0 / self._hz
        self._row_t0: float | None = None   # first recorded sample = origin of the row grid
        self._row_n = 0                     # how many grid points have been consumed
        # Rows actually written in the last RATE_WINDOW_S, so the UI can show what the setting
        # produced instead of what it promised.
        self._rate_n = 0
        self._rate_t0 = time.monotonic()
        self._rows_hz: float | None = None
        # Telemetry groups this source carries, asked once. Duck-typed: only SerialImuService
        # has the method, so the I2C source simply records no telemetry files. Asking the
        # source beats inspecting a sample -- an all-zero frame must not read as "no channel".
        avail = getattr(service, "available_telemetry", None)
        self._tele_groups: tuple[str, ...] = tuple(avail()) if avail is not None else ()
        # Optional ClsService: when enabled, the held activity prediction is written to
        # cls.csv in lockstep with the IMU rows (one row per drained sample, 100 Hz).
        self._cls = cls
        now = datetime.now()
        self.folder: Path = (
            Path(log_dir).expanduser()
            / now.strftime("%Y_%m_%d")
            / now.strftime("%H_%M_%S")
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._files: dict[str, TextIOWrapper] = {}
        self._tele_files: dict[str, TextIOWrapper] = {}
        self._cls_file: TextIOWrapper | None = None
        self._model_file: TextIOWrapper | None = None
        self._vote_file: TextIOWrapper | None = None
        self._model_cursor: float = 0.0
        self._decision_cursor: float = 0.0

    def __enter__(self) -> "Recorder":
        self.folder.mkdir(parents=True, exist_ok=True)
        layout = {
            "quaternions.csv": ("quat", ("w", "x", "y", "z")),
            "accelerometers.csv": ("accel", ("x", "y", "z")),
            "gyroscopes.csv": ("gyro", ("x", "y", "z")),
            "euler_angles.csv": ("euler", ("x", "y", "z")),
        }
        for fname, (_signal, axes) in layout.items():
            fh = open(self.folder / fname, "w", encoding="utf-8", newline="")
            fh.write(_hdr(self._labels, axes))
            self._files[fname] = fh
        self._layout = layout
        # One file per telemetry group the source actually carries, columns straight from
        # imu_common.TELEMETRY_GROUPS. A group the link does not have gets no file at all
        # rather than a file of empty cells, so "this recording has no knee channels" is
        # visible from the directory listing.
        for group, channels, _unit in TELEMETRY_GROUPS:
            if group not in self._tele_groups:
                continue
            fh = open(self.folder / telemetry_filename(group), "w", encoding="utf-8", newline="")
            fh.write(",".join(["Time", *channels]) + "\n")
            self._tele_files[group] = fh
        # cls.csv: Time, cls, conf, <one column per class prob>. Only when CLS is active.
        # model_input.csv: the exact 6-channel vectors (raw accel+gyro of the CLS sensor)
        # fed to the model, at the model's own rate — enough to replay inference offline.
        # cls_vote.csv: the aggregated decisions — the service's actual output. Pairing it with
        # cls.csv is what lets frame-level and post-vote accuracy be evaluated separately.
        if self._cls is not None and self._cls.enabled:
            fh = open(self.folder / "cls.csv", "w", encoding="utf-8", newline="")
            fh.write(",".join(["Time", "cls", "conf", *self._cls.classes]) + "\n")
            self._cls_file = fh
            mf = open(self.folder / "model_input.csv", "w", encoding="utf-8", newline="")
            mf.write("Time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
            self._model_file = mf
            vf = open(self.folder / "cls_vote.csv", "w", encoding="utf-8", newline="")
            vf.write(
                ",".join(["Time", "vote_cls", "vote_conf", *self._cls.classes, "n_frames"]) + "\n"
            )
            self._vote_file = vf
        # Drain cursor + a monotonic->wall-clock reference. The ring buffer only stores
        # monotonic timestamps, so batched samples get their own wall-clock time from this
        # reference rather than all sharing datetime.now() at write time.
        self._t0_mono = time.monotonic()
        self._t0_wall = datetime.now()
        self._cursor = self._t0_mono
        self._model_cursor = self._t0_mono
        self._decision_cursor = self._t0_mono
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # One last pass now the writer thread is done. It exits from its sleep the moment _stop
        # is set, so everything that arrived since its final wake-up is still unwritten — up to
        # a whole drain period, and a whole second at hz = 1. Nothing else can be touching the
        # files at this point, so this is safe to do on the caller's thread.
        try:
            self._drain()
        except Exception as err:  # a broken source must not lose the rows already on disk
            logger.warning(f"recorder: final drain failed ({err})")
        for fh in (*self._files.values(), *self._tele_files.values()):
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()
        self._tele_files.clear()
        for attr in ("_cls_file", "_model_file", "_vote_file"):
            fh = getattr(self, attr)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    @property
    def rows_hz(self) -> float | None:
        """Rows per second actually written recently, or None before the first window closes.

        What the UI shows next to the requested rate. A setting that cannot be met (asking for
        more rows than the device produces) is then visible rather than assumed."""
        return self._rows_hz

    def _loop(self) -> None:
        period = 1.0 / DRAIN_HZ
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self._drain()
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                next_tick = time.monotonic()   # falling behind — resync rather than spin

    def _drain(self) -> int:
        """Write the buffered samples newer than the cursor, then advance it. Returns rows written.

        Reuses ``ImuService.samples_since`` (the plot's data source) and ``limit=None``, so a
        late wake-up never loses data and the advancing cursor never repeats any. At ``hz = 0``
        every sample becomes a row; otherwise the time gate below selects a subset — the same
        subset for all eight per-sample files, because the decision is taken once per sample and
        every file is written inside that decision."""
        # Model input and decisions run at the CLS rates, independent of the 100 Hz IMU drain,
        # so pull them first — even on ticks with no new IMU sample — to capture the exact
        # streams rather than only what happens to coincide with a sample.
        self._drain_model_input()
        self._drain_decisions()
        samples = self._service.samples_since(self._cursor, limit=None)
        if not samples:
            return 0
        # Snapshot the held CLS prediction once per drain: it can't change mid-drain, so every
        # sample in this batch shares it — the step-hold between consecutive inferences.
        cls_cells = self._cls_row_cells() if self._cls_file is not None else None
        written = 0
        for sample in samples:
            if not self._select(sample["t"]):
                continue
            written += 1
            ts = (self._t0_wall + timedelta(seconds=sample["t"] - self._t0_mono)).strftime(
                "%H:%M:%S.%f"
            )
            for fname, (signal, axes) in self._layout.items():
                cells = [ts]
                for label in self._labels:
                    vals = sample[signal].get(label)
                    if vals is None:
                        cells.extend("" for _ in axes)
                    else:
                        cells.extend(f"{v:.6f}" for v in vals)
                fh = self._files.get(fname)
                if fh is not None:
                    fh.write(",".join(cells) + "\n")
            tele = sample.get("telemetry") or {}
            for group, fh in self._tele_files.items():
                vals = tele.get(group)
                n = len(TELEMETRY_CHANNELS[group])
                if vals is None:
                    cells = ["" for _ in range(n)]
                else:
                    # A channel can be individually absent (the state group's enable is None
                    # on a layout without that block). Empty cell, never 0.0 -- zero is a real
                    # value for every one of these channels.
                    cells = ["" if v is None else f"{v:.6f}" for v in vals[:n]]
                    cells += ["" for _ in range(n - len(cells))]
                fh.write(",".join([ts, *cells]) + "\n")
            if self._cls_file is not None:
                self._cls_file.write(",".join([ts, *cls_cells]) + "\n")
        # Past the whole batch, including the samples the gate skipped: they were considered and
        # rejected, not deferred, so leaving the cursor behind would offer them again next time.
        self._cursor = samples[-1]["t"]
        self._count_rows(written)
        return written

    def _select(self, t: float) -> bool:
        """Should the sample at monotonic ``t`` become a row? Advances the gate when it does.

        Args:    t: float, the sample's host-monotonic timestamp.
        Returns: bool.

        ``hz = 0`` takes everything. Otherwise the first sample at or past each point on a fixed
        1/hz grid wins and the rest are dropped, yielding hz rows per second without needing to
        know the device's rate — the number the old setting got wrong. The first sample of a
        recording always passes, so a session shorter than one interval is not empty.

        The grid is ``t0 + n/hz`` recomputed from the origin, not a running "time since the last
        row". Accumulating drifts, and worse, it turns the common case into a coin flip: when the
        device rate is an exact multiple of the target (160 Hz down to 40), the due sample lands
        exactly on the boundary, and float error a few parts in 1e11 below it costs a whole
        stride — 160 Hz thinned to 40 came out at 32. ``GRID_EPS`` absorbs that error; it is
        nanoseconds, far below any real sample spacing, so it can never admit a sample that is
        not genuinely due.
        """
        if self._row_dt <= 0.0:
            return True
        if self._row_t0 is None:
            self._row_t0, self._row_n = t, 1
            return True
        elapsed = t - self._row_t0
        if elapsed + GRID_EPS < self._row_n * self._row_dt:
            return False
        self._row_n += 1
        # After a gap in the data — a stall, a reconnect — the grid can be several points
        # behind. Skip it forward instead of emitting a burst of catch-up rows for samples that
        # never existed.
        if self._row_n * self._row_dt <= elapsed:
            self._row_n = int(elapsed / self._row_dt) + 1
        return True

    def _count_rows(self, n: int) -> None:
        """Fold ``n`` newly written rows into the reported rate, closing the window when due."""
        self._rate_n += n
        elapsed = time.monotonic() - self._rate_t0
        if elapsed >= RATE_WINDOW_S:
            self._rows_hz = self._rate_n / elapsed
            self._rate_n = 0
            self._rate_t0 = time.monotonic()

    def _drain_model_input(self) -> None:
        """Write every model-input vector newer than the model cursor to model_input.csv,
        oldest first, then advance the cursor. Exactly the 6-channel data fed to the model:
        no duplication, no upsampling — one row per vector the classifier consumed."""
        if self._model_file is None:
            return
        for inp in self._cls.inputs_since(self._model_cursor):
            ts = (self._t0_wall + timedelta(seconds=inp["t"] - self._t0_mono)).strftime(
                "%H:%M:%S.%f"
            )
            vals = (*inp["acc"], *inp["gyr"])
            self._model_file.write(",".join([ts, *(f"{v:.6f}" for v in vals)]) + "\n")
            self._model_cursor = inp["t"]

    def _drain_decisions(self) -> None:
        """Write every aggregated decision newer than the decision cursor to cls_vote.csv,
        oldest first, then advance the cursor.

        One row per decision at its own rate — deliberately *not* step-held to the IMU rate the
        way cls.csv is, so the file is the decision stream with its real timestamps."""
        if self._vote_file is None:
            return
        n_probs = len(self._cls.classes)
        for d in self._cls.decisions_since(self._decision_cursor):
            ts = (self._t0_wall + timedelta(seconds=d["t"] - self._t0_mono)).strftime(
                "%H:%M:%S.%f"
            )
            probs = d.get("probs") or []
            prob_cells = [f"{p:.6f}" for p in probs[:n_probs]]
            prob_cells += ["" for _ in range(n_probs - len(prob_cells))]
            self._vote_file.write(
                ",".join(
                    [
                        ts,
                        str(d.get("cls", "")),
                        f"{d.get('conf', 0.0):.6f}",
                        *prob_cells,
                        str(d.get("n", "")),
                    ]
                )
                + "\n"
            )
            self._decision_cursor = d["t"]

    def _cls_row_cells(self) -> list[str]:
        """cls/conf/probs cells for one cls.csv row, or empty cells before the first
        prediction. The number of prob columns always matches the header."""
        n_probs = len(self._cls.classes)
        pred = self._cls.current()
        if pred is None:
            return ["", ""] + ["" for _ in range(n_probs)]
        probs = pred.get("probs") or []
        prob_cells = [f"{p:.6f}" for p in probs]
        prob_cells += ["" for _ in range(n_probs - len(prob_cells))]
        return [str(pred.get("cls", "")), f"{pred.get('conf', 0.0):.6f}", *prob_cells[:n_probs]]
