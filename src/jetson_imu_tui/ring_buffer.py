"""Shared ring buffers for the plot screen."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from imu_python.base_classes import IMUData


SIGNAL_AXES = {
    "euler": ("x", "y", "z"),
    "accel": ("x", "y", "z"),
    "gyro": ("x", "y", "z"),
    "quat": ("w", "x", "y", "z"),
}

RAD_TO_DEG = 180.0 / math.pi


@dataclass
class RingBuffers:
    labels: list[str]
    maxlen: int
    time: dict[str, deque[float]] = field(default_factory=dict)
    data: dict[tuple[str, str, str], deque[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in self.labels:
            self.time[label] = deque(maxlen=self.maxlen)
            for signal, axes in SIGNAL_AXES.items():
                for axis in axes:
                    self.data[(label, signal, axis)] = deque(maxlen=self.maxlen)

    def append(self, label: str, imu_data: IMUData | None):
        """Append a sample and return the Euler orientation it computed.

        Returning the Euler object lets callers (e.g. the readout widget) reuse it
        instead of calling ``quat.to_euler`` a second time on the same sample.
        """
        if label not in self.time:
            return None
        if imu_data is None:
            return None
        self.time[label].append(imu_data.timestamp)
        euler = imu_data.quat.to_euler("ZYX")
        self.data[(label, "euler", "x")].append(euler.x * RAD_TO_DEG)
        self.data[(label, "euler", "y")].append(euler.y * RAD_TO_DEG)
        self.data[(label, "euler", "z")].append(euler.z * RAD_TO_DEG)
        accel = imu_data.device_data.accel
        self.data[(label, "accel", "x")].append(accel.x)
        self.data[(label, "accel", "y")].append(accel.y)
        self.data[(label, "accel", "z")].append(accel.z)
        gyro = imu_data.device_data.gyro
        self.data[(label, "gyro", "x")].append(gyro.x)
        self.data[(label, "gyro", "y")].append(gyro.y)
        self.data[(label, "gyro", "z")].append(gyro.z)
        q = imu_data.quat
        self.data[(label, "quat", "w")].append(q.w)
        self.data[(label, "quat", "x")].append(q.x)
        self.data[(label, "quat", "y")].append(q.y)
        self.data[(label, "quat", "z")].append(q.z)
        return euler
