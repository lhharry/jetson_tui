"""Threaded CSV recorder that writes 4 files in lockstep."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

from jetson_imu_tui.imu_service import ImuService


def _hdr(labels: list[str], axes: tuple[str, ...]) -> str:
    cols = ["Time"]
    for label in labels:
        cols.extend(f"{label}_{a}" for a in axes)
    return ",".join(cols) + "\n"


class Recorder:
    def __init__(self, service: ImuService, log_dir: Path, hz: float) -> None:
        self._service = service
        self._labels = service.labels
        self._hz = float(hz)
        now = datetime.now()
        self.folder: Path = (
            Path(log_dir).expanduser()
            / now.strftime("%Y_%m_%d")
            / now.strftime("%H_%M_%S")
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._files: dict[str, TextIOWrapper] = {}

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

    def _loop(self) -> None:
        period = 1.0 / self._hz
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self._write_row()
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            else:
                # Falling behind — resync rather than spin
                next_tick = time.monotonic()

    def _write_row(self) -> None:
        sigs = self._service.signals()
        ts = datetime.now().strftime("%H:%M:%S.%f")
        rows: dict[str, list[str]] = {fname: [ts] for fname in self._layout}
        for label in self._labels:
            sig = sigs.get(label)
            quat_vals: list[str] = ["", "", "", ""]
            accel_vals: list[str] = ["", "", ""]
            gyro_vals: list[str] = ["", "", ""]
            euler_vals: list[str] = ["", "", ""]
            if sig is not None:
                quat_vals = [f"{v:.6f}" for v in sig["quat"]]
                accel_vals = [f"{v:.6f}" for v in sig["accel"]]
                gyro_vals = [f"{v:.6f}" for v in sig["gyro"]]
                euler_vals = [f"{v:.6f}" for v in sig["euler"]]
            rows["quaternions.csv"].extend(quat_vals)
            rows["accelerometers.csv"].extend(accel_vals)
            rows["gyroscopes.csv"].extend(gyro_vals)
            rows["euler_angles.csv"].extend(euler_vals)
        for fname, cells in rows.items():
            fh = self._files.get(fname)
            if fh is None:
                continue
            fh.write(",".join(cells) + "\n")
