"""SerialImuService — one IMU (plus the rig's telemetry) arriving as binary frames over serial.

A drop-in alternative to ``ImuService`` for setups where the BNO055 is read by an Arduino (or a
Simulink model) that streams ``read_serial``'s frames instead of the Jetson reading the chip over
I2C. The web server picks between them from ``[source] kind`` in the config; everything
downstream (``web_server._payload``, ``Recorder``, ``ClsService``) is unchanged because the
method surface matches.

**Two kinds of channel travel the same frame.** The IMU half (accel/gyro) is a *sensor signal*:
it is axis-remapped, tare-able, and keyed by sensor label. The rest — knee angles, motor
feedback, the controller trace, the enable line, the device clock — is *device telemetry*: it
belongs to the rig rather than to a labelled IMU, so it has no label layer and never passes
through the axis transform (rotating a motor torque would be meaningless). The two shapes stay
separate all the way to the browser; ``imu_common.TELEMETRY_GROUPS`` is the one table that names
the telemetry channels. Which groups a link actually carries depends on its ``layout``, and
``available_telemetry`` reports that — a layout without knee channels says so, rather than
serving zeros.

What serial cannot provide, it reports as ``None`` rather than faking: the stream carries no
fusion output, so ``euler`` and ``quat`` are always None (Euler/Quaternion plots, the 3D cube
and the two matching CSVs stay empty) and calibration status is unavailable.

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
  timestamp rides along as ``"t_src"`` and gates nothing — it is there so a dropped frame can be
  measured against the device's clock rather than inferred from host arrival jitter.
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
    TELEMETRY_KEYS,
    AxisState,
    ImuInfo,
    apply_axis_transform,
    apply_offset,
)
from jetson_imu_tui.read_serial import (
    DT_MAX,
    DEFAULT_BAUD,
    DEFAULT_LAYOUT,
    DEFAULT_PORT,
    LAYOUTS,
    decode_frames,
)
from jetson_imu_tui.ring_buffer import RingBuffer

# Telemetry group -> the buffer-row key holding its values, in TELEMETRY_GROUPS order. "state"
# is assembled rather than copied (its two channels come from different blocks), so it is absent
# here and handled explicitly in ``_telemetry``.
_GROUP_SOURCE = {"knee": "joints", "motor": "feedback", "trace": "trace"}

# Which read_serial block has to be present for each group to be offered at all. A layout
# without the block reports the group as unavailable, which is not the same as reporting zeros.
# "state" keys on ``enable`` rather than on the clock: a layout carrying only a clock is the
# plain IMU forwarder, whose ``t_src`` already rides in every row and needs no group of its own.
_GROUP_BLOCK = {"knee": ("knee4",), "motor": ("fb6",), "trace": ("trace5",),
                "state": ("enable",)}

# Wait this long before re-opening the port after a failed open or a dropped link. Long enough
# not to spin on a missing device, short enough that re-plugging the Arduino recovers on its own.
REOPEN_DELAY_S = 2.0

# Length of one wire-rate measurement window. The rate is recomputed every window for as long
# as the link is up, not latched after the first one: ``sample_hz`` has to match the rate the
# device is sending *now*, and a device whose rate drifts mid-session would otherwise keep
# showing whatever it happened to be doing three seconds after connecting.
RATE_REPORT_S = 3.0

# Fraction of frames repeating the previous frame's device clock, past which the link is
# reporting stale data and says so. A healthy link repeats nothing; the exo rig currently
# repeats every sample three times (ratio 0.67) because its model overruns real time 3x.
# This is the only signal that catches it -- the wire rate stays a perfectly healthy 100 Hz.
DUP_WARN_RATIO = 0.10

# Largest magnitude any non-clock channel may carry. Real values are orders below it: knee
# velocities peak in the hundreds of deg/s, motor feedback lower still. A frame that decoded
# from misaligned bytes reads as a float32 built from unrelated bits, which is astronomically
# large far more often than it is plausible -- every corrupted frame observed in the recorded
# sessions carried at least one value above 1e11.
MAX_ABS_VALUE = 1e6

# Consecutive frames rejected *solely* by the clock rule before the current one is accepted as a
# new baseline. Without this, a device reboot (its clock restarts at zero, which reads as a
# backward jump forever after) would silently discard every subsequent frame. Layouts carrying
# ``t`` recover from a restart by re-aligning; this is the equivalent escape hatch for ``t_src``,
# which by design gates nothing in the decoder.
CLOCK_REBASELINE_FRAMES = 8

# One log line per this many seconds while frames are being dropped, however many there are.
DROP_LOG_EVERY_S = 10.0

# Hard ceiling on a result write, and on waiting for the port lock. Both must be small: they are
# paid on the CLS inference thread, and a transmitting device that never drains its receive
# buffer would otherwise block the write *forever* (pyserial's default write_timeout is None).
# One byte at 115200 baud takes ~87us, so 50 ms is enormous headroom; exceeding it means the
# far end is not reading, and the protocol already treats a missing result as silence.
WRITE_TIMEOUT_S = 0.05

# Rate-limit the TX failure log by *time*, not by state. A link sitting right at its limit
# alternates success and failure many times a second, and a flag cleared on every success turns
# "one line per episode" into a flood that buries everything else in the log.
TX_LOG_EVERY_S = 10.0

# After a failed write, stop trying for this long. A far end that is not draining its receive
# buffer cannot recover within one decision interval, and every attempt costs the inference
# thread up to WRITE_TIMEOUT_S — at several decisions a second that is real time spent blocking
# on a link already known to be dead.
TX_BACKOFF_S = 1.0

# No frame for this long means the stream has stopped, even though the port is still open.
STALE_AFTER_S = 1.0

# The port opened but nothing decoded for this long. Almost always a layout mismatch: the
# device's frame length is not the configured one, so no byte phase ever aligns and the decoder
# retries in silence forever. Worth one log line, because the symptom (plots never move) is
# identical to an unplugged cable and the decoder itself is deliberately quiet.
NO_FRAME_WARN_S = 5.0

# Serial links run faster than the I2C sensors and can be pushed to 200 Hz, where the shared
# default (2048 ≈ 10 s) is thin for the plot window plus a stalled poll. Sized here rather than
# by raising ring_buffer.DEFAULT_MAXLEN, which the I2C sensors and CLS also draw on.
BUFFER_MAXLEN = 4096

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
        expected_hz: float | None = None,
        state_path=None,
        axis_ops=(),
        axis: AxisState | None = None,
        clip: dict[str, float] | None = None,
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
        # Which telemetry groups this link carries, fixed at construction because the layout is.
        # Consumers ask once and build their columns/charts from the answer.
        blocks = set(LAYOUTS[self._layout].blocks)
        self._telemetry = tuple(
            g for g in TELEMETRY_KEYS if blocks & set(_GROUP_BLOCK.get(g, ()))
        )
        # Per-group absolute limit, applied on read-out (see ``_clean``). Empty = no clamping.
        self._clip = {k: abs(float(v)) for k, v in (clip or {}).items()}

        self._buf = RingBuffer(maxlen=BUFFER_MAXLEN)
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
        # Configured wire rate, kept only to warn when the measurement disagrees with it.
        self._expected_hz = float(expected_hz) if expected_hz else None
        # Fraction of recent frames carrying the previous frame's device clock. None until the
        # first window closes, or when the layout has no clock to compare.
        self.dup_ratio: float | None = None
        self._rate_logged = False
        # Entry filter state: the last accepted device clock, and how many frames in a row have
        # been rejected by the clock rule alone (see ``_accept``).
        self._last_good_clock: float | None = None
        self._clock_reject_streak = 0
        self.dropped_frames = 0
        self._drop_logged_at = 0.0
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
        # monotonic time of the last logged TX failure, and the earliest time to try again.
        self._tx_err_logged_at = 0.0
        self._tx_retry_at = 0.0

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
                self._tx_err_logged_at = 0.0
                self._tx_retry_at = 0.0
            # Rolling measurement window: frames, duplicate clocks, and its start time.
            win_n, win_dup, win_t0, prev_clock, first_frame = 0, 0, None, None, True
            self._rate_logged = False
            # The device may have restarted while the port was down, so the previous
            # connection's clock says nothing about this one.
            self._last_good_clock, self._clock_reject_streak = None, 0
            # An open port that never yields a frame is the one failure ``decode_frames`` cannot
            # report: it just keeps hunting for a byte phase, silently, forever. The reader
            # thread is blocked inside that generator and cannot notice, so the deadline is
            # armed here and cancelled by the first frame.
            warn_timer = threading.Timer(NO_FRAME_WARN_S, self._warn_no_frames)
            warn_timer.daemon = True
            warn_timer.start()
            try:
                for f in decode_frames(
                    ser, magic=self._magic, layout=self._layout, stop=self._stop
                ):
                    if not self._accept(f):
                        continue
                    gx, gy, gz = f["gyro"]
                    now = time.monotonic()
                    # Buffered timestamps must be *strictly* increasing. Every consumer walks
                    # this buffer with a "> cursor" cursor -- the recorder, CLS and the browser
                    # all do -- so two samples sharing a value means whichever the cursor lands
                    # on hides the rest of its tick forever. That is not hypothetical on a
                    # coarse clock: Windows' monotonic ticks at 15.6 ms, which at 160 Hz puts
                    # two or three frames on the same value. Nudging by a microsecond keeps the
                    # ordering honest; ``t`` is host arrival time and already approximate, and
                    # the device's own clock rides untouched in ``t_src``.
                    if self._last_frame_t is not None and now <= self._last_frame_t:
                        now = self._last_frame_t + 1e-6
                    # The device clock, whichever block carries it. ``t_src`` is the data block
                    # and ``t`` the sync one; a layout has at most one, and downstream neither
                    # knows nor cares which it was.
                    self._buf.append({
                        "t": now,
                        "t_src": f["t_src"] if f["t_src"] is not None else f["t"],
                        "accel": f["accel"],
                        "gyro": [gx * self._gyro_scale, gy * self._gyro_scale, gz * self._gyro_scale],
                        "joints": f["joints"],
                        "feedback": f["feedback"],
                        "trace": f["trace"],
                        "enable": f["enable"],
                    })
                    self._connected = True
                    self._last_frame_t = now
                    # Rate and staleness, measured continuously. ``sample_hz`` has to match the
                    # wire rate and nothing else validates it; duplication is invisible to the
                    # rate alone, so both are counted in the same window.
                    clock = f["t_src"] if f["t_src"] is not None else f["t"]
                    if clock is not None and prev_clock is not None and clock == prev_clock:
                        win_dup += 1
                    prev_clock = clock
                    win_n += 1
                    if first_frame:
                        first_frame = False
                        win_t0 = now
                        warn_timer.cancel()
                    elif now - win_t0 >= RATE_REPORT_S:
                        self._close_rate_window(win_n, win_dup, now - win_t0, has_clock=clock is not None)
                        win_n, win_dup, win_t0 = 0, 0, now
            except Exception as err:  # port missing / unplugged mid-stream
                self.error = err
                self._connected = False
                logger.warning(f"{self._label}: serial read failed ({err}) — retrying")
            finally:
                warn_timer.cancel()
                with self._io_lock:
                    self._ser = None
                    try:
                        ser.close()
                    except Exception:
                        pass
            self._stop.wait(REOPEN_DELAY_S)

    def _accept(self, f: dict) -> bool:
        """Is this decoded frame plausible enough to buffer? Advances the filter's state.

        Args:    f: dict, one frame as ``read_serial.decode_frames`` yields it.
        Returns: bool. False means drop this frame and keep going -- never re-sync.

        ``exo_v1`` carries no sync timestamp, so once the decoder has locked on, its only check
        is that ``aa55`` still lands on the frame boundary. A misaligned frame whose body happens
        to contain those two bytes at that offset is accepted whole, with a body of garbage.
        That is not hypothetical: the three recorded sessions hold seven such frames, and every
        one of them reached the plots and the CSVs.

        Two rules, and deliberately not a third:

        * **Clock sanity** -- the device clock must not go backwards, and must not jump more than
          ``DT_MAX``, measured against the last *accepted* frame rather than the last decoded one
          so a single bad frame cannot drag the baseline with it.
        * **Magnitude** -- any data channel beyond ``MAX_ABS_VALUE`` condemns the frame.

        **Non-finite data does not.** A NaN in the controller trace is something the device
        genuinely computed (a 0/0 in the control law), not evidence that the bytes are wrong, and
        the frame carrying it still holds perfectly good IMU and knee readings. Those are cleaned
        to None where they are read out -- ``_clean`` for telemetry, ``_as_signal`` for accel and
        gyro -- which is the existing "buffer keeps what the device sent" contract. Dropping the
        whole frame would discard real data to remove a value already handled, and it costs
        nothing in detection: every one of the seven corrupted frames in the recorded sessions
        carries a magnitude above 1e11, and across all three sessions there is not one NaN.
        A non-finite *clock* is different and is rejected below -- it is this filter's own state
        variable, and a NaN baseline would never compare true against anything again.

        The clock rule pointedly does **not** reject ``dt == 0``. Repeated timestamps are normal
        here twice over: this device sends each sample three times, and float32 quantisation
        makes adjacent timestamps compare equal once uptime is long enough (``read_serial``'s
        module docstring works the threshold out). Rejecting equality would drop two thirds of a
        healthy stream today.

        Re-syncing on a bad frame would be worse than dropping it: alignment is currently correct
        -- one frame in ~10000 is corrupt, not the phase -- so tearing down the lock would throw
        away ``SYNC_FRAMES`` good frames to fix nothing.
        """
        for key in ("accel", "gyro", "joints", "feedback", "trace"):
            vals = f.get(key)
            if vals is None:
                continue
            for v in vals:
                if abs(v) > MAX_ABS_VALUE:   # NaN compares False here, which is intended
                    return self._drop(f"{key} out of range ({v:g})")
        enable = f.get("enable")
        if enable is not None and abs(enable) > MAX_ABS_VALUE:
            return self._drop(f"enable out of range ({enable:g})")

        clock = f["t_src"] if f["t_src"] is not None else f["t"]
        if clock is None:
            return True                      # no clock in this layout: magnitude rule is all there is
        if not math.isfinite(clock):
            # Never counted toward the re-baseline streak: adopting NaN as the baseline would
            # make every later comparison false and reject the stream forever.
            return self._drop(f"clock not finite ({clock})")
        last = self._last_good_clock
        if last is not None and not (0.0 <= clock - last < DT_MAX):
            self._clock_reject_streak += 1
            if self._clock_reject_streak < CLOCK_REBASELINE_FRAMES:
                return self._drop(f"clock step {clock - last:+g}s outside [0, {DT_MAX})")
            # Long enough to be the device restarting rather than one corrupt frame. Adopt it.
            logger.info(
                f"{self._label}: device clock restarted (now {clock:g}s) — re-baselining after "
                f"{self._clock_reject_streak} rejected frames"
            )
        self._clock_reject_streak = 0
        self._last_good_clock = clock
        return True

    def _drop(self, why: str) -> bool:
        """Count one rejected frame, logging at most once per ``DROP_LOG_EVERY_S``. Returns False."""
        self.dropped_frames += 1
        now = time.monotonic()
        if now - self._drop_logged_at >= DROP_LOG_EVERY_S:
            self._drop_logged_at = now
            logger.warning(
                f"{self._label}: dropped an implausible frame ({why}); "
                f"{self.dropped_frames} total since start"
            )
        return False

    def _close_rate_window(self, n_frames: int, n_dup: int, elapsed: float,
                           *, has_clock: bool) -> None:
        """Publish one measurement window's wire rate and duplicate ratio, warning if either is bad.

        Args:
            n_frames:  int, frames decoded in this window.
            n_dup:     int, of those, how many repeated the previous frame's device clock.
            elapsed:   float, seconds the window spanned.
            has_clock: bool, whether the layout carries a device clock at all. Without one
                       duplication is unmeasurable and is reported as None, never as zero.

        Returns: None. Logs at most one line per condition per connection.

        Two independent failures, and the first hides the second: a device that resends every
        sample three times still puts a healthy 100 Hz on the wire, so the rate check passes
        while the model is being fed data a third as fresh as it thinks.
        """
        if elapsed <= 0 or n_frames <= 0:
            return
        self.observed_hz = n_frames / elapsed
        self.dup_ratio = (n_dup / n_frames) if has_clock else None
        if self._rate_logged:
            return
        self._rate_logged = True
        logger.info(
            f"{self._label}: {self.observed_hz:.1f} Hz observed on {self._port} "
            f"— set sample_hz to match"
        )
        exp = self._expected_hz
        if exp and abs(self.observed_hz - exp) / exp > 0.05:
            logger.warning(
                f"{self._label}: [source] sample_hz = {exp:g} but the device is sending "
                f"{self.observed_hz:.1f} Hz — CLS decimates by sample_hz, so every model window "
                f"is scaled by {self.observed_hz / exp:.2f}x with no other symptom"
            )
        if self.dup_ratio is not None and self.dup_ratio > DUP_WARN_RATIO:
            factor = 1.0 / max(1e-9, 1.0 - self.dup_ratio)
            logger.warning(
                f"{self._label}: {self.dup_ratio * 100:.0f}% of frames repeat the previous "
                f"device timestamp — each sample is being sent ~{factor:.1f}x, so the link "
                f"carries only ~{self.observed_hz / factor:.0f} distinct samples/s despite "
                f"{self.observed_hz:.0f} Hz on the wire. This is a device-side fix."
            )

    def _warn_no_frames(self) -> None:
        """Timer callback: the port has been open for NO_FRAME_WARN_S with nothing decoded.

        Args:    none.
        Returns: None. Logs once per connection episode.

        Names the likely cause rather than the symptom. Every silent way this fails — wrong
        ``layout`` (the device's frame length is not the configured one, so no byte phase can
        ever align), wrong ``magic``, or a device that has stopped transmitting — looks exactly
        like an unplugged cable from the outside.
        """
        n = LAYOUTS[self._layout].n_floats
        logger.warning(
            f"{self._label}: {self._port} is open but no frame decoded in {NO_FRAME_WARN_S:.0f}s "
            f"— check [source] layout ('{self._layout}', {n} floats) and magic "
            f"('{self._magic or 'none'}') against what the device sends"
        )

    # --- result return channel ----------------------------------------------
    def send_result(self, index: int) -> bool:
        """Write one classification-result byte back to the device. Returns True if written.

        Called from the CLS inference thread once per aggregated decision, so it must never
        raise and never block: a closed port is simply silence — the reader thread is already
        retrying, and the agreed protocol has no 'no result' sentinel. One byte needs no
        ``flush()``; the write reaches the OS buffer directly.

        Bounded three ways, because inference must not be hostage to the far end:

        * ``WRITE_TIMEOUT_S`` caps the write itself. A device that does not drain its receive
          buffer leaves the host's write buffer full, and an unbounded ``write`` would block
          this thread permanently — stopping inference outright.
        * The lock is acquired with the same timeout, so a writer already stuck in one cannot
          hold up the next.
        * ``TX_BACKOFF_S`` skips writes entirely for a while after a failure. Without it a
          blocked link costs ``WRITE_TIMEOUT_S`` on *every* decision, which at several decisions
          a second is a large slice of the inference thread spent waiting on a link already
          known to be refusing bytes.

        Note what a *flapping* result means, because it is the diagnostic that matters here: the
        far end is draining, just not fast enough to keep up, so its buffer fills gradually and
        the link degrades from "occasionally blocked" to "always blocked". That is a receive
        *rate* problem on the device, not a wiring problem — see the README's return-channel
        section.
        """
        if not isinstance(index, int) or not 0 <= index <= 255:
            logger.warning(f"{self._label}: refusing to send out-of-range result {index!r}")
            return False
        now = time.monotonic()
        if now < self._tx_retry_at:
            return False              # backing off; the next decision carries a fresher class
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
                self._tx_retry_at = now + TX_BACKOFF_S
                if now - self._tx_err_logged_at >= TX_LOG_EVERY_S:
                    self._tx_err_logged_at = now
                    logger.warning(
                        f"{self._label}: result write failed ({err}) — the device is not "
                        f"draining its serial input fast enough (further failures quiet for "
                        f"{TX_LOG_EVERY_S:.0f}s)"
                    )
                return False
            if self.tx_ok is False:
                logger.info(f"{self._label}: result writes recovered")
            self._tx_err_logged_at = 0.0
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

        A non-finite component becomes None, the same answer ``_clean`` gives telemetry. The
        entry filter already refuses such frames, so this is the second line rather than the
        first -- but it is the line that matters if one ever gets through: ``json.dumps`` writes
        NaN as a bare ``NaN`` token, which is valid Python and invalid JSON, so a single NaN
        makes the browser reject the whole ``/data`` response, and ``tick()``'s catch swallows
        the failure into a frozen page with nothing in the log.

        ``apply_axis_transform`` copies, so the buffer row is never exposed or mutated."""
        if sample is None:
            return None
        out = apply_axis_transform(
            {"euler": None, "accel": sample["accel"], "gyro": sample["gyro"], "quat": None},
            self._axis.transform if tf is None else tf,
        )
        for key in ("accel", "gyro"):
            vals = out.get(key)
            if vals is not None and not all(math.isfinite(v) for v in vals):
                out[key] = [v if math.isfinite(v) else None for v in vals]
        return out

    def available_telemetry(self) -> tuple[str, ...]:
        """The ``imu_common.TELEMETRY_GROUPS`` names this link carries, in table order.

        Args:    none.
        Returns: tuple[str, ...] — e.g. ("knee", "motor", "trace", "state") on ``exo_v1``,
                 ("state",) on a legacy layout that only has a clock, () on one with neither.

        Fixed by ``layout`` at construction. Consumers build their columns, files and charts
        from this rather than probing a sample, so an all-zero frame cannot be mistaken for an
        absent channel. ``ImuService`` has no such method — callers reach it through
        ``getattr(service, "available_telemetry", None)``, the same duck-typing the result
        return channel uses."""
        return self._telemetry

    def _telemetry_row(self, sample: dict | None) -> dict[str, list[float] | None] | None:
        """Buffer row -> ``{group: [values] | None}`` for the groups this layout carries.

        Input:  ``sample`` = buffer row, or None.
        Output: dict keyed by group name, values list[float] (copied) or None when the frame
                had nothing for it; None when ``sample`` is None.

        Copies, so the buffer row is never exposed. Deliberately does **not** go through
        ``apply_axis_transform``: that function emits exactly the four sensor keys and drops
        everything else, and rotating a motor torque or a class index would be meaningless
        anyway. Telemetry is attached alongside the transformed signal, never through it.

        Clamping and non-finite rejection happen here rather than in the reader thread, so the
        ring buffer always holds what the device actually sent: changing a limit re-applies it
        to the whole buffered history instead of leaving mangled samples behind it.
        """
        if sample is None:
            return None
        out: dict[str, list[float] | None] = {}
        for group in self._telemetry:
            if group == "state":
                # Assembled, not copied: its two channels come from different blocks, and
                # either may be absent on a layout that carries only the other.
                vals = [sample.get("enable"), sample.get("t_src")]
            else:
                vals = sample.get(_GROUP_SOURCE[group])
            out[group] = self._clean(group, vals) if vals is not None else None
        return out

    def _clean(self, group: str, vals) -> list:
        """One group's raw values -> what consumers may see: clamped, and finite or None.

        Args:
            group: str, a TELEMETRY_GROUPS name — selects the limit from ``self._clip``.
            vals:  sequence of float | None, the group's channels in wire order.

        Returns:
            list of float | None, same length and order as ``vals``.

        Two different problems, two different answers:

        * **Out of range** is a noise spike on a channel whose real values are small, and it is
          *clamped* to the limit. The trade is deliberate and lossy — a flat line at the limit
          is indistinguishable from the signal genuinely sitting there — but it keeps the shared
          Y axis usable, which a single 9999 does not. Only groups named in ``[telemetry.clip]``
          are clamped; anything else passes through, which is why ``state`` must stay out of
          that table (``t_src`` is a device clock that climbs into the thousands).
        * **Non-finite** is not out of range, it is invalid, so it becomes None regardless of
          any limit. This is not cosmetic: ``json.dumps`` renders NaN as a bare ``NaN`` token,
          which is valid Python and invalid JSON, so one NaN on the wire makes the browser
          reject the entire ``/data`` response — and ``tick()``'s catch swallows it, leaving a
          frozen page and nothing in the log.
        """
        lim = self._clip.get(group)
        out: list = []
        for v in vals:
            if v is None or not math.isfinite(v):
                out.append(None)
            elif lim is None:
                out.append(v)
            else:
                out.append(lim if v > lim else (-lim if v < -lim else v))
        return out

    def telemetry(self) -> dict[str, list[float] | None]:
        """Latest telemetry values, one entry per available group ({} when none)."""
        return self._telemetry_row(self._latest_raw()) or {}

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
        quat are present as None so the browser still discovers the label.

        Rows carry ``"telemetry"`` when the layout has any: ``{group: [values] | None}``, the
        device-global channels alongside the per-label signals. **It has to be this one call**,
        not a second cursor-based method — the recorder writes every CSV from one batch, and two
        cursors would let frames land between them and drift the files' row counts apart."""
        off = self._offset
        o = off.get(self._label) if off else None
        tf = self._axis.transform  # read once: the whole batch must use one mapping
        out: list[dict] = []
        for s in self._buf.since(t, limit=limit):
            sig = apply_offset(self._as_signal(s, tf), o)
            row: dict = {"t": s["t"]}
            for key in SIGNAL_KEYS:
                row[key] = {self._label: sig[key] if sig is not None else None}
            if self._telemetry:
                row["telemetry"] = self._telemetry_row(s)
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
