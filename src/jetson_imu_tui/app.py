"""Jetson IMU TUI Textual app."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from textual.app import App
from textual.binding import Binding
from textual.worker import Worker

from jetson_imu_tui.config import AppConfig
from jetson_imu_tui.imu_service import ImuService
from jetson_imu_tui.recorder import Recorder
from jetson_imu_tui.ring_buffer import RingBuffers
from jetson_imu_tui.screens.folder_modal import FolderModal
from jetson_imu_tui.screens.frequency_modal import FrequencyModal
from jetson_imu_tui.screens.main_screen import MainScreen
from jetson_imu_tui.screens.plot_screen import PlotScreen


class JetsonImuApp(App):
    CSS_PATH = "styles/app.tcss"
    TITLE = "Jetson IMU TUI"

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "Quit", priority=True),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.app_config = config
        self.service = ImuService(config.bus_labels)
        self.buffers = RingBuffers(
            labels=config.labels, maxlen=config.plot_window_samples
        )
        self.recorder: Recorder | None = None
        self.streaming = False
        self._record_hz = config.record_hz
        self._log_dir = config.log_dir
        self._loguru_handle: int | None = None
        self._main_screen: MainScreen | None = None

    @property
    def main_screen(self) -> MainScreen:
        assert self._main_screen is not None, "MainScreen not mounted yet"
        return self._main_screen

    def on_mount(self) -> None:
        self._main_screen = MainScreen(self.app_config.labels)
        self.push_screen(self._main_screen)
        self.call_after_refresh(self._post_mount_init)

    def _post_mount_init(self) -> None:
        ms = self.main_screen
        ms.status.set_record_hz(self._record_hz)
        ms.status.set_log_dir(self._log_dir)
        for label in self.app_config.labels:
            ro = ms.readout(label)
            if ro is not None:
                ro.set_subtitle(f"{label} (disconnected)")
        try:
            self._loguru_handle = logger.add(
                ms.console.loguru_sink,
                level="WARNING",
                format="{time:HH:mm:ss} {level} {message}",
            )
        except Exception:
            self._loguru_handle = None
        ms.console.write_line("Ready. Press [c] to connect.")
        self.set_interval(1.0 / max(1, self.app_config.ui_refresh_hz), self._tick)

    def _tick(self) -> None:
        if not self.service.is_connected() or not self.streaming:
            return
        snap = self.service.snapshot()
        ms = self.main_screen
        for label, data in snap.items():
            self.buffers.append(label, data)
            ro = ms.readout(label)
            if ro is not None:
                ro.update_from(data)

    # ------------------------------------------------------------------ Actions

    def action_quit_app(self) -> None:
        if self.recorder is not None:
            self._stop_recording()
        if self.service.is_connected():
            self.run_worker(self._disconnect_worker, thread=True, exclusive=True)
        self.exit()

    def action_connect(self) -> None:
        ms = self.main_screen
        if self.service.is_connected():
            ms.console.write_line("Disconnecting...")
            self.streaming = False
            ms.status.set_streaming(False)
            if self.recorder is not None:
                self._stop_recording()
            self.run_worker(self._disconnect_worker, thread=True, exclusive=True)
        else:
            ms.console.write_line("Connecting to IMUs...")
            self.run_worker(self._connect_worker, thread=True, exclusive=True)

    def action_toggle_stream(self) -> None:
        ms = self.main_screen
        if not self.service.is_connected():
            ms.console.write_line("Not connected — press [c] first.")
            return
        self.streaming = not self.streaming
        ms.status.set_streaming(self.streaming)
        ms.console.write_line(f"Streaming {'ON' if self.streaming else 'OFF'}")
        if not self.streaming:
            for label in self.app_config.labels:
                ro = ms.readout(label)
                if ro is not None:
                    ro.update_from(None)

    def action_toggle_record(self) -> None:
        ms = self.main_screen
        if self.recorder is not None:
            self._stop_recording()
            return
        if not self.service.is_connected():
            ms.console.write_line("Not connected — press [c] first.")
            return
        try:
            self.recorder = Recorder(self.service, self._log_dir, self._record_hz).__enter__()
        except Exception as err:
            ms.console.write_line(f"Failed to start recorder: {err}")
            self.recorder = None
            return
        ms.status.set_recording(True)
        ms.console.write_line(f"Recording → {self.recorder.folder}")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.__exit__(None, None, None)
        finally:
            self.recorder = None
        ms = self.main_screen
        ms.status.set_recording(False)
        ms.console.write_line("Recording stopped.")

    def action_set_frequency(self) -> None:
        def _apply(result: int | None) -> None:
            if result is None:
                return
            self._record_hz = int(result)
            self.main_screen.status.set_record_hz(self._record_hz)
            self.main_screen.console.write_line(f"Recording frequency set to {self._record_hz} Hz")
            if self.recorder is not None:
                self.main_screen.console.write_line("Restarting recorder with new frequency...")
                self._stop_recording()
                self.action_toggle_record()

        self.push_screen(FrequencyModal(self._record_hz), _apply)

    def action_set_log_dir(self) -> None:
        def _apply(result: Path | None) -> None:
            if result is None:
                return
            self._log_dir = Path(result)
            self.main_screen.status.set_log_dir(self._log_dir)
            self.main_screen.console.write_line(f"Log dir set to {self._log_dir}")

        self.push_screen(FolderModal(self._log_dir), _apply)

    def action_plot(self) -> None:
        self.push_screen(PlotScreen(self.buffers))

    # ------------------------------------------------------------------ Workers

    def _connect_worker(self) -> None:
        try:
            info = self.service.connect()
        except Exception as err:
            self.call_from_thread(self._on_connect_failed, err)
            return
        self.call_from_thread(self._on_connected, info)

    def _disconnect_worker(self) -> None:
        try:
            self.service.disconnect()
        except Exception as err:
            self.call_from_thread(self._on_disconnect_failed, err)
            return
        self.call_from_thread(self._on_disconnected)

    def _on_connected(self, info: list) -> None:
        ms = self.main_screen
        if not info:
            ms.console.write_line("No IMUs detected.")
            ms.status.set_imus({})
            return
        states = {i.label: ("MOCK" if i.is_mock else "CONNECTED") for i in info}
        ms.status.set_imus(states)
        for entry in info:
            ro = ms.readout(entry.label)
            if ro is not None:
                ro.set_subtitle(f"{entry.label} (bus {entry.bus_id}, {entry.sensor_name})")
        names = ", ".join(f"{i.label}={i.sensor_name}" for i in info)
        ms.console.write_line(f"Connected {len(info)} IMUs: {names}")
        ms.console.write_line(
            "BNO055 fusion warming up — orientation may drift for ~30 s."
        )
        self.streaming = True
        ms.status.set_streaming(True)

    def _on_connect_failed(self, err: Exception) -> None:
        self.main_screen.console.write_line(f"Connect failed: {err}")

    def _on_disconnected(self) -> None:
        ms = self.main_screen
        ms.status.set_imus({})
        ms.status.set_streaming(False)
        for label in self.app_config.labels:
            ro = ms.readout(label)
            if ro is not None:
                ro.set_subtitle(f"{label} (disconnected)")
                ro.update_from(None)
        ms.console.write_line("Disconnected.")

    def _on_disconnect_failed(self, err: Exception) -> None:
        self.main_screen.console.write_line(f"Disconnect error: {err}")

    def on_unmount(self) -> None:
        if self._loguru_handle is not None:
            try:
                logger.remove(self._loguru_handle)
            except Exception:
                pass
