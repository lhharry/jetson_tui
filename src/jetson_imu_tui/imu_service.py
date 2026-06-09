"""IMU lifecycle and snapshot helpers wrapping imu-python."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from imu_python.base_classes import IMUData
from imu_python.definitions import MOCK_NAME
from imu_python.factory import IMUFactory
from imu_python.sensor_manager import IMUManager

from jetson_imu_tui.ring_buffer import RAD_TO_DEG


@dataclass
class ImuInfo:
    label: str
    bus_id: int
    sensor_name: str
    is_mock: bool


class ImuService:
    def __init__(self, bus_labels: dict[int, str]) -> None:
        self._bus_labels = dict(bus_labels)
        self.managers: dict[str, IMUManager] = {}
        # Per-label zero offset for euler/accel/gyro (tare). None = no offset.
        self._offset: dict[str, dict[str, list[float]]] | None = None
        self._offset_lock = threading.Lock()

    @property
    def labels(self) -> list[str]:
        return [self._bus_labels[k] for k in sorted(self._bus_labels)]

    def connect(self) -> list[ImuInfo]:
        if self.managers:
            return self.info()
        # free_threading is auto-gated inside imu-python; pass False to keep things
        # predictable on stock CPython.
        managers = IMUFactory.detect_and_create(free_threading=False, log_data=False)
        labeled: dict[str, IMUManager] = {}
        for m in managers:
            bus_id = int(m.i2c_id) if m.i2c_id is not None else -1
            label = self._bus_labels.get(bus_id, f"bus_{bus_id}")
            labeled[label] = m
        for m in labeled.values():
            m.start()
        self.managers = labeled
        return self.info()

    def disconnect(self) -> None:
        for m in self.managers.values():
            m.stop()
        self.managers.clear()

    def info(self) -> list[ImuInfo]:
        out: list[ImuInfo] = []
        for label, m in self.managers.items():
            bus_id = int(m.i2c_id) if m.i2c_id is not None else -1
            name = m.imu_descriptor.name
            out.append(
                ImuInfo(label=label, bus_id=bus_id, sensor_name=name, is_mock=name == MOCK_NAME)
            )
        return out

    def is_connected(self) -> bool:
        return bool(self.managers)

    def snapshot(self) -> dict[str, IMUData | None]:
        return {label: m.get_data() for label, m in self.managers.items()}

    @staticmethod
    def _derive(data: IMUData) -> dict[str, list[float]]:
        """Raw derived signals (euler in degrees, accel, gyro, quat) for one IMU."""
        e = data.quat.to_euler("ZYX")
        a = data.device_data.accel
        g = data.device_data.gyro
        q = data.quat
        return {
            "euler": [e.x * RAD_TO_DEG, e.y * RAD_TO_DEG, e.z * RAD_TO_DEG],
            "accel": [a.x, a.y, a.z],
            "gyro": [g.x, g.y, g.z],
            "quat": [q.w, q.x, q.y, q.z],
        }

    def signals(self) -> dict[str, dict[str, list[float]] | None]:
        """Derived signals per label with the zero offset applied to euler/accel/gyro.

        Quaternion is never offset. Labels with no data return None.
        """
        offset = self._offset
        out: dict[str, dict[str, list[float]] | None] = {}
        for label, data in self.snapshot().items():
            if data is None:
                out[label] = None
                continue
            sig = self._derive(data)
            if offset is not None and label in offset:
                off = offset[label]
                for key in ("euler", "accel", "gyro"):
                    sig[key] = [v - o for v, o in zip(sig[key], off[key])]
            out[label] = sig
        return out

    @property
    def is_zeroed(self) -> bool:
        return self._offset is not None

    def zero_toggle(self) -> bool:
        """Toggle tare: capture current absolute readings as zero, or clear.

        Returns True if an offset is now active, False if cleared.
        """
        with self._offset_lock:
            if self._offset is not None:
                self._offset = None
                return False
            offset: dict[str, dict[str, list[float]]] = {}
            for label, data in self.snapshot().items():
                if data is None:
                    continue
                sig = self._derive(data)
                offset[label] = {k: list(sig[k]) for k in ("euler", "accel", "gyro")}
            self._offset = offset
            return True
