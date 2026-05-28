"""Threaded TSV recorder that writes 4 files in lockstep."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

from jetson_imu_tui.imu_service import ImuService
from jetson_imu_tui.ring_buffer import RAD_TO_DEG


def _hdr(labels: list[str], axes: tuple[str, ...]) -> str:
    cols = ["Time"]
    for label in labels:
        cols.extend(f"{label}_{a}" for a in axes)
    return "\t".join(cols) + "\n"


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
            "quaternions.tsv": ("quat", ("w", "x", "y", "z")),
            "accelerometers.tsv": ("accel", ("x", "y", "z")),
            "gyroscopes.tsv": ("gyro", ("x", "y", "z")),
            "euler_angles.tsv": ("euler", ("roll", "pitch", "yaw")),
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
        snap = self._service.snapshot()
        ts = datetime.now().strftime("%H:%M:%S.%f")
        rows: dict[str, list[str]] = {fname: [ts] for fname in self._layout}
        for label in self._labels:
            data = snap.get(label)
            quat_vals: list[str] = ["", "", "", ""]
            accel_vals: list[str] = ["", "", ""]
            gyro_vals: list[str] = ["", "", ""]
            euler_vals: list[str] = ["", "", ""]
            if data is not None:
                q = data.quat
                quat_vals = [f"{q.w:.6f}", f"{q.x:.6f}", f"{q.y:.6f}", f"{q.z:.6f}"]
                a = data.device_data.accel
                accel_vals = [f"{a.x:.6f}", f"{a.y:.6f}", f"{a.z:.6f}"]
                g = data.device_data.gyro
                gyro_vals = [f"{g.x:.6f}", f"{g.y:.6f}", f"{g.z:.6f}"]
                e = data.quat.to_euler("ZYX")
                euler_vals = [
                    f"{e.x * RAD_TO_DEG:.6f}",
                    f"{e.y * RAD_TO_DEG:.6f}",
                    f"{e.z * RAD_TO_DEG:.6f}",
                ]
            rows["quaternions.tsv"].extend(quat_vals)
            rows["accelerometers.tsv"].extend(accel_vals)
            rows["gyroscopes.tsv"].extend(gyro_vals)
            rows["euler_angles.tsv"].extend(euler_vals)
        for fname, cells in rows.items():
            fh = self._files.get(fname)
            if fh is None:
                continue
            fh.write("\t".join(cells) + "\n")
