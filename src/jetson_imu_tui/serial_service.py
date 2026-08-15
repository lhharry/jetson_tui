"""SerialImuService — one IMU arriving as binary frames over a serial port.

A drop-in alternative to ``ImuService`` for setups where the BNO055 is read by an Arduino (or a
Simulink model) that streams ``read_serial``'s 7-float frames instead of the Jetson reading the
chip over I2C. The web server picks between them from ``[source] kind`` in the config; everything
downstream (``web_server._payload``, ``Recorder``, ``ClsService``) is unchanged because the
method surface matches.

What serial cannot provide, it reports as ``None`` rather than faking: the stream carries only
accelerometer and gyroscope, so ``euler`` and ``quat`` are always None (Euler/Quaternion plots,
the 3D cube and the two matching CSVs stay empty) and calibration status is unavailable.

The **axis remap is now writable here**, unlike in the register-based version: it is a host-side
transform (``imu_common.AxisState``, the same object ``ImuService`` uses), so it no longer
matters that the serial link cannot reach the sensor's registers. It applies to accel/gyro; the
euler/quat half of the transform is inert because this source has neither. Note this composes
*on top of* whatever mapping the transmitting device already applies in its own firmware.

The link is **bidirectional**: ``send_result`` writes one byte back per aggregated classification
decision, so the Arduino that supplies the IMU is also the consumer of the model output over the
same cable. This class owns the ``serial.Serial`` handle for that reason (``decode_frames`` takes
an open port rather than opening its own) and guards it with ``_io_lock``, which the reader thread
holds only to publish or clear the handle — never across a blocking read — so a writer waits
microseconds at most and never sees a half-closed port.

Two invariants keep the rest of the app working unchanged:

* Buffered ``"t"`` is the host's ``time.monotonic()`` at decode, exactly like
  ``ImuService._sample_loop``. The recorder's monotonic->wall-clock mapping, the browser's
  ``since`` cursor and CLS's ``MAX_RAW_GAP_S`` gap check all assume that clock; the source's own
  timestamp rides along as ``"t_src"``.
* ``gyro_scale`` is applied once, in the reader thread before buffering, so the plots, the CSVs
  and the model all see the same rad/s values. BNO055 firmware commonly emits deg/s (values
  quantized to 1/16, peaking in the hundreds) while the classifier was trained on rad/s — feeding
  deg/s through is the ~57x error the README's training/deployment contract warns about.
"""

from __future__ import annotations

import math
import threading
import time

import serial
from loguru import logger

from jetson_imu_tui.imu_common import (
    SIGNAL_KEYS,
    AxisState,
    ImuInfo,
    apply_axis_transform,
    apply_offset,
)
from jetson_imu_tui.read_serial import (
    DEFAULT_BAUD,
    DEFAULT_LAYOUT,
    DEFAULT_PORT,
    LAYOUTS,
    decode_frames,
)
from jetson_imu_tui.ring_buffer import RingBuffer

# Wait this long before re-opening the port after a failed open or a dropped link. Long enough
# not to spin on a missing device, short enough that re-plugging the Arduino recovers on its own.
REOPEN_DELAY_S = 2.0

# Measure the incoming frame rate over this long, then report it once per connection.
RATE_REPORT_S = 3.0

# Hard ceiling on a result write, and on waiting for the port lock. Both must be small: they are
# paid on the CLS inference thread, and a transmitting device that never drains its receive
# buffer would otherwise block the write *forever* (pyserial's default write_timeout is None).
# One byte at 115200 baud takes ~87us, so 50 ms is enormous headroom; exceeding it means the
# far end is not reading, and the protocol already treats a missing result as silence.
WRITE_TIMEOUT_S = 0.05

# No frame for this long means the stream has stopped, even though the port is still open.
STALE_AFTER_S = 1.0

GYRO_SCALE = {"rad": 1.0, "deg": math.pi / 180.0}


class SerialImuService:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        *,
        label: str = "Left",
        magic: str = "",
        layout: str = DEFAULT_LAYOUT,
        gyro_units: str = "deg",
        state_path=None,
        axis_ops=(),
        axis: AxisState | None = None,
    ) -> None:
        self._port = port
        self._baud = int(baud)
        self._label = label
        self._magic = magic
        if layout not in LAYOUTS:
            logger.warning(f"unknown layout '{layout}' — assuming {DEFAULT_LAYOUT}")
        self._layout = layout if layout in LAYOUTS else DEFAULT_LAYOUT
        if gyro_units not in GYRO_SCALE:
            logger.warning(f"unknown gyro_units '{gyro_units}' — assuming deg/s")
        self._gyro_scale = GYRO_SCALE.get(gyro_units, GYRO_SCALE["deg"])

        self._buf = RingBuffer()
        self._offset: dict[str, dict[str, list[float]]] | None = None
        self._offset_lock = threading.Lock()
        # Software axis remap, identical to the I2C source's — applied on read-out. ``axis``
        # lets the caller share one AxisState across both sources (the web server does).
        self._axis = axis if axis is not None else AxisState(state_path, axis_ops)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self.error: Exception | None = None  # last reader failure, for status reporting
        self.observed_hz: float | None = None  # measured wire rate, for checking sample_hz
        self._last_frame_t: float | None = None  # monotonic time of the newest decoded frame
        # None until the first result is written; False once the far end stops accepting them.
        self.tx_ok: bool | None = None

        # The open port, shared between the reader thread and ``send_result``. ``_io_lock``
        # guards publishing/clearing it and the writes themselves; it is never held across a
        # read, so the CLS thread never blocks behind the reader's 1 s read timeout.
        self._ser: serial.Serial | None = None
        self._io_lock = threading.Lock()
        # Rate limits: the reader retries every REOPEN_DELAY_S and results are written
        # continuously, so both failures log once per episode rather than on every attempt.
        self._open_err_logged = False
        self._tx_err_logged = False

    # --- lifecycle ---------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        return [self._label]

    def connect(self) -> list[ImuInfo]:
        """Probe the port so startup can report a real result; the reader thread opens it again.

        Returns [] if the port cannot be opened, mirroring ``ImuService.connect`` finding no
        sensors — the server then serves nulls instead of refusing to start."""
        if self._connected:
            return self.info()
        try:
            serial.Serial(
                self._port, self._baud, timeout=1, write_timeout=WRITE_TIMEOUT_S
            ).close()
        except Exception as err:
            self.error = err
            logger.warning(f"{self._label}: cannot open {self._port} ({err})")
            return []
        self._connected = True
        return self.info()

    def disconnect(self) -> None:
        self.stop_sampling()
        self._connected = False

    def info(self) -> list[ImuInfo]:
        if not self._connected:
            return []
        return [ImuInfo(label=self._label, bus_id=-1, sensor_name=f"BNO055 (serial {self._port})")]

    def is_connected(self) -> bool:
        """The port is open. Note this says nothing about frames actually arriving — a
        transmitter that has gone quiet still leaves an openable port. See ``receiving``."""
        return self._connected

    @property
    def receiving(self) -> bool:
        """A frame arrived recently, i.e. data is genuinely flowing.

        The distinction matters when nothing appears to work: an open port with no frames looks
        identical to a healthy one everywhere else, and the plots simply stop updating with no
        explanation. The UI reports this separately so 'no link' and 'no data' are tellable
        apart without reading the log."""
        last = self._last_frame_t
        return last is not None and (time.monotonic() - last) < STALE_AFTER_S

    # --- background sampling -----------------------------------------------
    @property
    def sampling(self) -> bool:
        return self._thread is not None

    def start_sampling(self, hz: float = 100.0) -> None:
        """Start the reader thread. ``hz`` is accepted for interface parity and ignored — the
        transmitting device sets the rate; ``sample_hz`` in the config must match what it sends,
        since that is what CLS decimates by."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"serial-imu-{self._label}"
        )
        self._thread.start()

    def stop_sampling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _read_loop(self) -> None:
        """Decode frames into the ring buffer, re-opening the port until told to stop.

        Owns the port so ``send_result`` can write to the same handle: it is published under
        ``_io_lock`` while live and cleared *before* being closed, so a concurrent writer sees
        either an open port or None."""
        while not self._stop.is_set():
            try:
                ser = serial.Serial(
                    self._port, self._baud, timeout=1, write_timeout=WRITE_TIMEOUT_S
                )
                ser.reset_output_buffer()
            except Exception as err:  # device not plugged in (yet)
                self.error = err
                self._connected = False
                if not self._open_err_logged:
                    logger.warning(f"{self._label}: cannot open {self._port} ({err}) — retrying")
                    self._open_err_logged = True
                self._stop.wait(REOPEN_DELAY_S)
                continue
            with self._io_lock:
                self._ser = ser
                self._open_err_logged = False
                self._tx_err_logged = False
            n_frames, t_first, rate_logged = 0, None, False
            try:
                for f in decode_frames(
                    ser, magic=self._magic, layout=self._layout, stop=self._stop
                ):
                    gx, gy, gz = f["gyro"]
                    now = time.monotonic()
                    self._buf.append({
                        "t": now,
                        "t_src": f["t"],
                        "accel": f["accel"],
                        "gyro": [gx * self._gyro_scale, gy * self._gyro_scale, gz * self._gyro_scale],
                    })
                    self._connected = True
                    self._last_frame_t = now
                    # Report the rate the device actually sends at, once per connection:
                    # ``sample_hz`` has to match it, and it is otherwise invisible.
                    n_frames += 1
                    if t_first is None:
                        t_first = now
                    elif not rate_logged and now - t_first >= RATE_REPORT_S:
                        self.observed_hz = n_frames / (now - t_first)
                        logger.info(
                            f"{self._label}: {self.observed_hz:.1f} Hz observed on {self._port} "
                            f"— set sample_hz to match"
                        )
                        rate_logged = True
            except Exception as err:  # port missing / unplugged mid-stream
                self.error = err
                self._connected = False
                logger.warning(f"{self._label}: serial read failed ({err}) — retrying")
            finally:
                with self._io_lock:
                    self._ser = None
                    try:
                        ser.close()
                    except Exception:
                        pass
            self._stop.wait(REOPEN_DELAY_S)

    # --- result return channel ----------------------------------------------
    def send_result(self, index: int) -> bool:
        """Write one classification-result byte back to the device. Returns True if written.

        Called from the CLS inference thread once per aggregated decision (2 Hz at the shipped
        settings), so it must never raise and never block: a closed port is simply silence — the
        reader thread is already retrying, and the agreed protocol has no 'no result' sentinel.
        One byte at 115200 baud needs no ``flush()``; the write reaches the OS buffer directly.

        Bounded twice over, because inference must not be hostage to the far end. A device that
        never reads its serial input (a pure transmitter) leaves the host's write buffer full,
        and an unbounded ``write`` would then block this thread permanently — stopping inference
        outright. ``WRITE_TIMEOUT_S`` caps the write, and the lock is acquired with the same
        timeout so a writer already stuck in one cannot hold up the next."""
        if not isinstance(index, int) or not 0 <= index <= 255:
            logger.warning(f"{self._label}: refusing to send out-of-range result {index!r}")
            return False
        if not self._io_lock.acquire(timeout=WRITE_TIMEOUT_S):
            self.tx_ok = False
            return False  # a previous write is still draining — drop this result, don't queue
        try:
            ser = self._ser
            if ser is None:
                return False
            try:
                ser.write(bytes([index]))
            except Exception as err:
                self.tx_ok = False
                # Results stream continuously; log the first failure and stay quiet until one
                # succeeds again, so an unplugged or non-reading device cannot flood the log.
                if not self._tx_err_logged:
                    logger.warning(
                        f"{self._label}: result write failed ({err}) — is the device reading "
                        f"its serial input?"
                    )
                    self._tx_err_logged = True
                return False
            self._tx_err_logged = False
            self.tx_ok = True
            return True
        finally:
            self._io_lock.release()

    # --- data --------------------------------------------------------------
    def _latest_raw(self) -> dict | None:
        return self._buf.latest()

    def _as_signal(self, sample: dict | None, tf=None) -> dict | None:
        """Ring-buffer sample -> the four-key signal dict every consumer expects, axis-remapped.

        Input: ``sample`` = buffer row ``{"t","t_src","accel","gyro"}`` or None; ``tf`` =
        AxisTransform to apply, defaulting to the current one — pass it explicitly to pin a
        single mapping across a whole batch. Output: a new
        ``{"euler","accel","gyro","quat"}`` dict of list[float], euler/quat always None
        (nothing in the stream to fill them), or None when ``sample`` is None.

        ``apply_axis_transform`` copies, so the buffer row is never exposed or mutated."""
        if sample is None:
            return None
        return apply_axis_transform(
            {"euler": None, "accel": sample["accel"], "gyro": sample["gyro"], "quat": None},
            self._axis.transform if tf is None else tf,
        )

    def signals(self) -> dict[str, dict | None]:
        """Latest signals for the single label, with the zero offset applied when active."""
        off = self._offset
        sig = self._as_signal(self._latest_raw())
        return {self._label: apply_offset(sig, off.get(self._label) if off else None)}

    def read_raw(self, label: str) -> dict | None:
        """Latest signals with the zero/tare offset NOT applied (what CLS needs: gravity in)."""
        if label != self._label:
            return None
        return self._as_signal(self._latest_raw())

    def raw_samples_since(self, label: str, t: float, limit: int | None = None) -> list[dict]:
        """Axis-remapped, tare-NOT-applied buffered samples with ``sample["t"] > t``, oldest
        first.

        The one method ``ClsService`` calls. Copies, so callers never mutate buffer state."""
        if label != self._label:
            return []
        tf = self._axis.transform  # read once: the whole batch must use one mapping
        return [
            {"t": s["t"], "t_src": s["t_src"], **self._as_signal(s, tf)}
            for s in self._buf.since(t, limit=limit)
        ]

    def samples_since(self, t: float, limit: int = 300) -> list[dict]:
        """Payload-shaped samples newer than monotonic ``t``, oldest first — the plot's and the
        recorder's data source. One sensor, so there is no cross-label alignment to do; euler and
        quat are present as None so the browser still discovers the label."""
        off = self._offset
        o = off.get(self._label) if off else None
        tf = self._axis.transform  # read once: the whole batch must use one mapping
        out: list[dict] = []
        for s in self._buf.since(t, limit=limit):
            sig = apply_offset(self._as_signal(s, tf), o)
            row: dict = {"t": s["t"]}
            for key in SIGNAL_KEYS:
                row[key] = {self._label: sig[key] if sig is not None else None}
            out.append(row)
        return out

    # --- zero / tare -------------------------------------------------------
    @property
    def is_zeroed(self) -> bool:
        return self._offset is not None

    def zero_toggle(self) -> bool:
        """Capture the current accel/gyro as the zero reference, or clear it. Returns True if
        now zeroed. CLS bypasses this (it reads ``raw_samples_since``) and keeps seeing gravity."""
        with self._offset_lock:
            if self._offset is None:
                sig = self._as_signal(self._latest_raw())
                self._offset = (
                    {self._label: {"accel": list(sig["accel"]), "gyro": list(sig["gyro"])}}
                    if sig is not None
                    else {}
                )
            else:
                self._offset = None
            return self._offset is not None

    # --- calibration (reported) / axis remap (software) ---------------------
    def calibration_status(self) -> dict[str, dict | None]:
        """No calibration registers over serial — the transmitting device owns that."""
        return {self._label: None}

    def get_axis_remap(self) -> dict:
        """Output: the current mapping as a JSON-ready dict — see ``AxisTransform.describe``."""
        return self._axis.describe()

    def set_axis_remap(self, *, ops=None, config: int | None = None, sign: int | None = None,
                       persist: bool = True) -> dict:
        """Replace the software axis remap — same contract as ``ImuService.set_axis_remap``.

        Input: ``ops`` = list of op names applied in order, or ``config``/``sign`` bytes.
        Output: ``describe()`` plus ``ok``/``message``.

        This composes on top of whatever the transmitting device already applies in firmware,
        so it corrects a mounting mismatch without reflashing — but the combination, not this
        transform alone, is what has to match the mapping the training data was captured with.
        Clears the tare on success (the zero reference belongs to the previous frame)."""
        result = self._axis.set(ops=ops, config=config, sign=sign, persist=persist)
        if result.get("ok"):
            with self._offset_lock:
                self._offset = None
        return result
