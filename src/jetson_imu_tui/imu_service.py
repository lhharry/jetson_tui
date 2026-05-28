"""IMU lifecycle and snapshot helpers wrapping imu-python."""

from __future__ import annotations

from dataclasses import dataclass

from imu_python.base_classes import IMUData
from imu_python.definitions import MOCK_NAME
from imu_python.factory import IMUFactory
from imu_python.sensor_manager import IMUManager


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
