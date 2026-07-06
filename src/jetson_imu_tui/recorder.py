"""Threaded CSV recorder that writes 4 files in lockstep."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # ImuService pulls in the Linux-only hardware stack; only needed for hints.
    from jetson_imu_tui.imu_service import ImuService


def _hdr(labels: list[str], axes: tuple[str, ...]) -> str:
    cols = ["Time"]
    for label in labels:
        cols.extend(f"{label}_{a}" for a in axes)
    return ",".join(cols) + "\n"


class Recorder:
    def __init__(self, service: "ImuService", log_dir: Path, hz: float, cls=None) -> None:
        self._service = service
        self._labels = service.labels
        self._hz = float(hz)
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
        self._cls_file: TextIOWrapper | None = None
        self._model_file: TextIOWrapper | None = None
        self._model_cursor: float = 0.0

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
        # cls.csv: Time, cls, conf, <one column per class prob>. Only when CLS is active.
        # model_input.csv: the exact 6-channel vectors (raw accel+gyro of the CLS sensor)
        # fed to the model, at the model's own rate — enough to replay inference offline.
        if self._cls is not None and self._cls.enabled:
            fh = open(self.folder / "cls.csv", "w", encoding="utf-8", newline="")
            fh.write(",".join(["Time", "cls", "conf", *self._cls.classes]) + "\n")
            self._cls_file = fh
            mf = open(self.folder / "model_input.csv", "w", encoding="utf-8", newline="")
            mf.write("Time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
            self._model_file = mf
        # Drain cursor + a monotonic->wall-clock reference. The ring buffer only stores
        # monotonic timestamps, so batched samples get their own wall-clock time from this
        # reference rather than all sharing datetime.now() at write time.
        self._t0_mono = time.monotonic()
        self._t0_wall = datetime.now()
        self._cursor = self._t0_mono
        self._model_cursor = self._t0_mono
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()
        for attr in ("_cls_file", "_model_file"):
            fh = getattr(self, attr)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _loop(self) -> None:
        period = 1.0 / self._hz
        next_tick = time.monotonic()
        stat_start = next_tick
        rows = overruns = 0
        while not self._stop.is_set():
            rows += self._drain()
            now = time.monotonic()
            if now - stat_start >= 5.0:
                logger.info(
                    f"recorder: {rows / (now - stat_start):.1f} Hz (target {self._hz:.0f}) · overruns={overruns}"
                )
                stat_start = now
                rows = overruns = 0
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                # Falling behind — resync rather than spin
                overruns += 1
                next_tick = time.monotonic()

    def _drain(self) -> int:
        """Write every buffered sample newer than the cursor — the exact aligned samples the
        plot draws — then advance the cursor past them. Returns the number of rows written.

        Reuses ``ImuService.samples_since`` (the plot's data source), so CSV row == plot point
        == sampler sample by construction: no duplicated or dropped rows. ``limit=None`` means a
        late tick never drops data; the cursor advancing past written samples means no dup."""
        # Model input runs at the CLS rate (~10 Hz), independent of the 100 Hz IMU drain, so
        # pull it first — even on ticks with no new IMU sample — to capture the exact stream.
        self._drain_model_input()
        samples = self._service.samples_since(self._cursor, limit=None)
        if not samples:
            return 0
        # Snapshot the held CLS prediction once per drain: it can't change mid-drain, so every
        # sample in this batch shares it — the step-hold between the sparse ~1 s inferences.
        cls_cells = self._cls_row_cells() if self._cls_file is not None else None
        for sample in samples:
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
            if self._cls_file is not None:
                self._cls_file.write(",".join([ts, *cls_cells]) + "\n")
        self._cursor = samples[-1]["t"]
        return len(samples)

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
