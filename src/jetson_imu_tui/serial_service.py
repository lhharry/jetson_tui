"""SerialImuService — one IMU (and, on the control layouts, both knees and the motors)
arriving as binary frames over a serial port.

A drop-in alternative to ``ImuService`` for setups where the BNO055 is read by an Arduino (or a
Simulink model) that streams ``read_serial``'s frames instead of the Jetson reading the chip over
I2C. The web server picks between them from ``[source] kind`` in the config; everything
downstream (``web_server._payload``, ``Recorder``, ``ClsService``) is unchanged because the
method surface matches.

What serial cannot provide, it reports as ``None`` rather than faking: the stream carries no
fusion output, so ``euler`` and ``quat`` are always None (Euler/Quaternion plots, the 3D cube and
the two matching CSVs stay empty), calibration status is unavailable, and the axis remap is
read-only — that mapping is configured on the Arduino, not writable from here.

The link is **bidirectional**: ``send_velocity`` writes the motor velocity command back at the
control loop's rate, so the device that supplies the sensors is also the consumer of the
controller's output over the same cable. This class owns the ``serial.Serial`` handle for that
reason (``decode_frames`` takes an open port rather than opening its own) and guards it with
``_io_lock``, which the reader thread holds only to publish or clear the handle — never across a
blocking read — so a writer waits microseconds at most and never sees a half-closed port.

Three invariants keep the rest of the app working unchanged:

* Buffered ``"t"`` is the host's ``time.monotonic()`` at decode, exactly like
  ``ImuService._sample_loop``. The recorder's monotonic->wall-clock mapping, the browser's
  ``since`` cursor, CLS's ``MAX_RAW_GAP_S`` gap check and the controller's staleness gate all
  assume that clock; the source's own timestamp rides along as ``"t_src"``.
* ``gyro_scale`` is applied once, in the reader thread before buffering, so the plots, the CSVs
  and the model all see the same rad/s values. BNO055 firmware commonly emits deg/s (values
  quantized to 1/16, peaking in the hundreds) while the classifier was trained on rad/s — feeding
  deg/s through is the ~57x error the README's training/deployment contract warns about.
* ``joint_scale`` is applied the same way and at the same place, to all four knee channels. The
  Simulink models this replaces multiply every one of them by the same +/-pi/180, so angle and
  angular velocity share one knob here too. Motor feedback is **not** scaled: its unit and zero
  are set by the CAN unpack on the far side, and guessing at them here would be exactly the kind
  of silent unit error the two rules above exist to prevent.
"""

from __future__ import annotations

import math
import struct
import threading
import time

import serial
from loguru import logger

from jetson_imu_tui.imu_common import (
    DEFAULT_CONFIG,
    DEFAULT_SIGN,
    SIGNAL_KEYS,
    ImuInfo,
    apply_offset,
    decode_axis_remap,
    is_valid_config,
    placement_for,
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

# Angular unit conversion, shared by the gyro and the knee channels: "rad" passes through,
# "deg" is scaled to radians (and deg/s to rad/s — the factor is the same for a rate).
DEG_RAD_SCALE = {"rad": 1.0, "deg": math.pi / 180.0}
GYRO_SCALE = DEG_RAD_SCALE  # kept as the historical name

# Absolute ceiling on a velocity command, matching Saturation1 inside Left/Right Motor Velocity
# in the Simulink models (12-bit CAN velocity field, +/-45 rad/s, saturated to this before pack).
# Enforced in send_velocity so no controller bug or bad profile table can exceed it on the wire.
VEL_LIMIT = 41.87

# Downlink frame header, matching the Serial Receive block's Header setting on the device.
# Different from the uplink's aa55 on purpose: a header that also appears in the other
# direction's traffic is one more way for a mis-wired loopback to look like it works.
DEFAULT_TX_MAGIC = "5aa5"


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
        joint_units: str = "deg",
        tx_magic: str = DEFAULT_TX_MAGIC,
    ) -> None:
        self._port = port
        self._baud = int(baud)
        self._label = label
        self._magic = magic
        if layout not in LAYOUTS:
            logger.warning(f"unknown layout '{layout}' — assuming {DEFAULT_LAYOUT}")
        self._layout = layout if layout in LAYOUTS else DEFAULT_LAYOUT
        if gyro_units not in DEG_RAD_SCALE:
            logger.warning(f"unknown gyro_units '{gyro_units}' — assuming deg/s")
        self._gyro_scale = DEG_RAD_SCALE.get(gyro_units, DEG_RAD_SCALE["deg"])
        if joint_units not in DEG_RAD_SCALE:
            logger.warning(f"unknown joint_units '{joint_units}' — assuming deg")
        self._joint_scale = DEG_RAD_SCALE.get(joint_units, DEG_RAD_SCALE["deg"])
        # Which optional blocks this layout actually carries. Fixed at construction because the
        # layout cannot change without rebuilding the service, and the control loop checks it
        # every tick to tell "no knee channels configured" from "no data right now".
        _blocks = LAYOUTS[self._layout].blocks
        self._has_joints = "knee4" in _blocks
        self._has_feedback = any(b in _blocks for b in ("fb2", "fb6"))
        try:
            self._tx_magic = bytes.fromhex(tx_magic) if tx_magic else b""
        except ValueError:
            logger.warning(f"invalid tx_magic '{tx_magic}' — sending unframed commands")
            self._tx_magic = b""

        self._buf = RingBuffer()
        self._offset: dict[str, dict[str, list[float]]] | None = None
        self._offset_lock = threading.Lock()
        # Axis remap is set on the Arduino; kept here only so the UI popup has something
        # coherent to render and to report what this end believes is in effect.
        self._axis_config = DEFAULT_CONFIG
        self._axis_sign = DEFAULT_SIGN

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self.error: Exception | None = None  # last reader failure, for status reporting
        self.observed_hz: float | None = None  # measured wire rate, for checking sample_hz
        self._last_frame_t: float | None = None  # monotonic time of the newest decoded frame
        # None until the first result is written; False once the far end stops accepting them.
        self.tx_ok: bool | None = None

        # The open port, shared between the reader thread and ``send_velocity``. ``_io_lock``
        # guards publishing/clearing it and the writes themselves; it is never held across a
        # read, so the CLS thread never blocks behind the reader's 1 s read timeout.
        self._ser: serial.Serial | None = None
        self._io_lock = threading.Lock()
        # Rate limits: the reader retries every REOPEN_DELAY_S and results are written
        # continuously, so both failures log once per episode rather than on every attempt.
        self._open_err_logged = False
        self._tx_err_logged = False
        self._tx_clip_logged = False

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

        Owns the port so ``send_velocity`` can write to the same handle: it is published under
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
                    joints = f["joints"]
                    self._buf.append({
                        "t": now,
                        "t_src": f["t"],
                        "accel": f["accel"],
                        "gyro": [gx * self._gyro_scale, gy * self._gyro_scale, gz * self._gyro_scale],
                        # Scaled here, once, for the same reason gyro is: every consumer must see
                        # radians, and a per-consumer conversion is a per-consumer chance to
                        # forget. Order is read_serial's canonical ang_r, vel_r, ang_l, vel_l.
                        "joints": (
                            [v * self._joint_scale for v in joints] if joints is not None else None
                        ),
                        # Motor feedback stays in the device's own units — see the module
                        # docstring: its scale and zero belong to the CAN unpack on the far side.
                        "feedback": f["feedback"],
                        "enable": f["enable"],
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

    # --- command return channel ----------------------------------------------
    def send_velocity(self, vel_r: float, vel_l: float) -> bool:
        """Write one motor velocity command frame back to the device. True if it reached the OS.

        Args:
            vel_r: float, right motor velocity command in rad/s.
            vel_l: float, left motor velocity command in rad/s.

        Returns:
            bool. True when the frame was handed to the port; False when there is no open port,
            when the port lock could not be taken within ``WRITE_TIMEOUT_S``, or when the write
            itself failed. Never raises — the caller is the control loop, and a link problem
            must degrade to silence rather than stop the loop that is also responsible for
            commanding zero.

        Wire format: ``tx_magic`` (default 5A A5) followed by two little-endian float32,
        right then left — 10 bytes, matching a Serial Receive block configured for 2 singles
        with that Header. At 100 Hz that is 1 kB/s against 11.5 kB/s of 115200 8N1.

        Two guards live here rather than in the controller, so that no caller — a bad profile
        table, an unstable loop, a future second consumer — can bypass them:

        * **Non-finite becomes 0.0.** NaN/inf survive ``struct.pack`` happily and would reach
          the CAN pack as garbage. The Simulink side does have an IsNaN guard, but a safety
          property that depends on the far end being wired correctly is not one we hold.
        * **Absolute clamp to +/-VEL_LIMIT**, the same 41.87 rad/s the model saturates to
          before packing. Defence in depth behind the profile's own per-side limits.

        Bounded twice over: a device that never drains its receive buffer would otherwise block
        this thread forever (pyserial's default ``write_timeout`` is None), and the control loop
        must keep ticking to stay safe — it is the thing responsible for commanding zero.
        """
        vals = []
        clipped = False
        for v in (vel_r, vel_l):
            try:
                f = float(v)
            except (TypeError, ValueError):
                f, clipped = 0.0, True
            if not math.isfinite(f):
                f, clipped = 0.0, True
            elif f > VEL_LIMIT:
                f, clipped = VEL_LIMIT, True
            elif f < -VEL_LIMIT:
                f, clipped = -VEL_LIMIT, True
            vals.append(f)
        if clipped and not self._tx_clip_logged:
            logger.warning(
                f"{self._label}: velocity command out of range or non-finite "
                f"({vel_r!r}, {vel_l!r}) — clamped to +/-{VEL_LIMIT}"
            )
            self._tx_clip_logged = True

        payload = self._tx_magic + struct.pack("<2f", vals[0], vals[1])
        if not self._io_lock.acquire(timeout=WRITE_TIMEOUT_S):
            self.tx_ok = False
            return False  # a previous write is still draining — drop this tick, don't queue
        try:
            ser = self._ser
            if ser is None:
                return False
            try:
                ser.write(payload)
            except Exception as err:
                self.tx_ok = False
                # Commands stream continuously; log the first failure and stay quiet until one
                # succeeds again, so an unplugged device cannot flood the log at the loop rate.
                if not self._tx_err_logged:
                    logger.warning(
                        f"{self._label}: command write failed ({err}) — is the device reading "
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

    @staticmethod
    def _as_signal(sample: dict | None) -> dict | None:
        """Ring-buffer sample -> the four-key signal dict every consumer expects."""
        if sample is None:
            return None
        return {
            "euler": None,
            "accel": list(sample["accel"]),
            "gyro": list(sample["gyro"]),
            "quat": None,
        }

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
        """Raw (tare-NOT-applied) buffered samples with ``sample["t"] > t``, oldest first.

        The one method ``ClsService`` calls. Copies, so callers never mutate buffer state."""
        if label != self._label:
            return []
        return [
            {"t": s["t"], "t_src": s["t_src"], "accel": list(s["accel"]), "gyro": list(s["gyro"])}
            for s in self._buf.since(t, limit=limit)
        ]

    # --- control channels ----------------------------------------------------
    @property
    def has_joints(self) -> bool:
        """bool — True when the configured layout carries the four knee channels.

        Lets the controller report "this link has no knee channels" instead of sitting silently
        at zero, which is indistinguishable from a dead sensor from the outside."""
        return self._has_joints

    @property
    def has_feedback(self) -> bool:
        """bool — True when the configured layout carries motor position feedback.

        Without it the position loop has nothing to close on, so the controller refuses to run
        rather than differencing against a made-up zero."""
        return self._has_feedback

    def latest_control_sample(self, label: str) -> dict | None:
        """Newest sample flattened into exactly what one control tick needs. O(1).

        Args:
            label: str, sensor label; anything but this service's own label returns None.

        Returns:
            dict, or None when the label is unknown, no frame has arrived yet, or the layout
            carries no knee channels. Keys:
              "t":       float, host ``time.monotonic()`` at decode — the clock the staleness
                         gate compares against.
              "enable":  bool, the device's SWITCH line. True when the layout has no ``enable``
                         block, since a link that cannot say otherwise must not silently
                         disable the controller.
              "ang_r", "vel_r", "ang_l", "vel_l": float, knee angle (rad) and angular
                         velocity (rad/s), already unit-converted by the reader thread.
              "pos_r", "pos_l": float | None, motor position feedback in the device's own
                         units, normalised out of either feedback block (fb2 is pos_r/pos_l;
                         fb6 is pos/speed/torque per side, so the positions are elements 0
                         and 3). None when the layout carries no feedback block.

        Copies every value out of the buffered sample. ``RingBuffer.latest`` hands back the very
        dict the reader thread appended — and will keep mutating no field of it, but the lists
        inside are shared — so returning it directly would let the control thread alias buffer
        state. ``raw_samples_since`` rebuilds its dicts for the same reason.
        """
        if label != self._label:
            return None
        s = self._buf.latest()
        if s is None:
            return None
        joints = s.get("joints")
        if joints is None or len(joints) < 4:
            return None
        fb = s.get("feedback")
        pos_r = pos_l = None
        if fb is not None:
            # fb2: (pos_r, pos_l).  fb6: (pos_r, speed_r, torque_r, pos_l, speed_l, torque_l).
            if len(fb) >= 6:
                pos_r, pos_l = float(fb[0]), float(fb[3])
            elif len(fb) >= 2:
                pos_r, pos_l = float(fb[0]), float(fb[1])
        en = s.get("enable")
        return {
            "t": s["t"],
            "enable": True if en is None else bool(en),
            "ang_r": float(joints[0]),
            "vel_r": float(joints[1]),
            "ang_l": float(joints[2]),
            "vel_l": float(joints[3]),
            "pos_r": pos_r,
            "pos_l": pos_l,
        }

    def joint_samples_since(self, label: str, t: float, limit: int | None = None) -> list[dict]:
        """Buffered knee/feedback samples with ``sample["t"] > t``, oldest first.

        Args:
            label: str, sensor label; anything else returns [].
            t:     float, host-monotonic cursor; samples strictly newer than this are returned.
            limit: int | None, keep at most the newest ``limit`` samples. None = no cap.

        Returns:
            list[dict], each ``{"t": float, "enable": float|None, "joints": list[float] len 4
            (rad, rad/s) | None, "feedback": list[float] | None}``. Copies, so callers never
            mutate buffer state.

        Not used by the control loop — that one wants the newest value, not a backlog (see
        ``ControlService``). This exists for offline analysis and any future joint plot, which
        do want every sample.
        """
        if label != self._label:
            return []
        out = []
        for s in self._buf.since(t, limit=limit):
            j, fb = s.get("joints"), s.get("feedback")
            out.append({
                "t": s["t"],
                "enable": s.get("enable"),
                "joints": list(j) if j is not None else None,
                "feedback": list(fb) if fb is not None else None,
            })
        return out

    def samples_since(self, t: float, limit: int = 300) -> list[dict]:
        """Payload-shaped samples newer than monotonic ``t``, oldest first — the plot's and the
        recorder's data source. One sensor, so there is no cross-label alignment to do; euler and
        quat are present as None so the browser still discovers the label."""
        off = self._offset
        o = off.get(self._label) if off else None
        out: list[dict] = []
        for s in self._buf.since(t, limit=limit):
            sig = apply_offset(self._as_signal(s), o)
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

    # --- calibration / axis remap (reported, not controllable) --------------
    def calibration_status(self) -> dict[str, dict | None]:
        """No calibration registers over serial — the transmitting device owns that."""
        return {self._label: None}

    def get_axis_remap(self) -> dict:
        c, s = self._axis_config, self._axis_sign
        return {
            "config": c,
            "sign": s,
            "config_hex": f"0x{c:02X}",
            "sign_hex": f"0x{s:02X}",
            "mapping": decode_axis_remap(c, s),
            "placement": placement_for(c, s),
            "valid": is_valid_config(c),
        }

    def set_axis_remap(self, config_byte: int, sign_byte: int, *, persist: bool = True) -> dict:
        """Record the mapping this end assumes; it cannot be written over the serial link.

        The transmitting device applies its own AXIS_MAP_CONFIG/SIGN, and that mapping must match
        the one used to collect the training data or the classifier degrades silently."""
        config_byte &= 0xFF
        sign_byte &= 0xFF
        valid = is_valid_config(config_byte)
        if valid:
            self._axis_config = config_byte
            self._axis_sign = sign_byte
        result = self.get_axis_remap()
        result.update({
            "config": config_byte,
            "sign": sign_byte,
            "config_hex": f"0x{config_byte:02X}",
            "sign_hex": f"0x{sign_byte:02X}",
            "mapping": decode_axis_remap(config_byte, sign_byte),
            "placement": placement_for(config_byte, sign_byte),
            "valid": valid,
            "ok": False,
            "hardware": False,
            "applied": {},
            "message": (
                "Invalid mapping: each output axis must map to a distinct source axis."
                if not valid
                else "Serial source — set the axis mapping on the transmitting device."
            ),
        })
        return result
