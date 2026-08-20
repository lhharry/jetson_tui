# jetson_tui

A headless service that streams and records two BNO055 IMUs on an NVIDIA Jetson and
serves their data to a browser for live plotting. Left thigh sits on `/dev/i2c-1`,
right thigh on `/dev/i2c-7` (both at default address `0x28`). Reading and fusion use the
official [`adafruit-circuitpython-bno055`](https://github.com/adafruit/Adafruit_CircuitPython_BNO055)
driver with the BNO055's **onboard** sensor fusion (no software filter on the host).

Rendering happens **in the browser on your laptop** (with [uPlot](https://github.com/leeoniya/uPlot)),
not on the Jetson — so the Jetson spends ~no CPU on the UI, leaving headroom for other
workloads (e.g. an AI model) on the same board.

## Features

- Auto-connects both IMUs on start and serves a single live page.
- Four signal views (Euler / Accel / Gyro / Quaternion); Left + Right overlaid.
- Switch to a **numbers** view for live numeric readouts of every signal.
- **Zero** (tare) the current Euler/Accel/Gyro readings from the page.
- **Axis** popup: remap the sensor axes by building a chain of steps — rotate an axis by
  90/180/270°, or negate one — applied **in software on the host**, for both sources, with a
  live 3D cube to confirm the result. Datasheet §3.4 placements P0–P7 remain as presets.
- **Calib** popup: live onboard calibration status (sys / gyro / accel, 0–3) with guidance.
- **IMU:** button switches between the onboard I2C IMUs and a serial (Arduino) IMU at runtime.
- **CLS** page: online activity classification, with frame predictions aggregated by soft voting
  into one stable decision — sent back to the Arduino as a class-index byte over the same link.
- **Record** toggle and an adjustable **recording frequency** from the page.
- **Load** a past recording and review it offline in the same charts, with zoom that
  re-fetches at full resolution.
- Recording writes one directory of comma-separated files per session under
  `<log_dir>/YYYY_MM_DD/HH_MM_SS/`:
  - `quaternions.csv` — `Time,Left_w,Left_x,Left_y,Left_z,Right_w,Right_x,Right_y,Right_z`
  - `accelerometers.csv` — `Time,Left_x,Left_y,Left_z,Right_x,Right_y,Right_z`
  - `gyroscopes.csv` — same columns as accel
  - `euler_angles.csv` — degrees (ZYX intrinsic: x=roll, y=pitch, z=yaw)
  - one file per **telemetry group** the source carries — `knee.csv`, `motor.csv`,
    `trace.csv`, `state.csv` (serial `exo_v1` only; a group the link lacks gets no file
    rather than a file of empty cells)
  - with CLS active: `cls.csv` (frame predictions + per-class probabilities),
    `cls_vote.csv` (one row per aggregated decision) and `model_input.csv` (the exact
    6-channel vectors fed to the model)

  Every per-sample file is written from one drain batch, so their row counts are equal by
  construction — line *n* of `knee.csv` is the same instant as line *n* of `accelerometers.csv`.

## Install

```bash
pip install -e .            # flask + pyserial + adafruit-circuitpython-bno055 + Blinka + extended-bus
```

`pyserial` is a core dependency: `[source] kind = "serial"` is a supported boot default and a
missing source at startup is a hard exit, not a fallback. The old `.[serial]` extra still
resolves, but adds nothing.

Make sure the user is in the `i2c` group and both buses are exposed:

```bash
ls /dev/i2c-1 /dev/i2c-7
i2cdetect -y 1
i2cdetect -y 7   # expect 0x28 on both
```

On Jetson Nano (legacy), bus 7 may need a device-tree overlay. On Orin Nano dev
kits, pins 3/5 already map to bus 7 out of the box.

> **Hardware required.** There is no mock fallback — the Adafruit driver talks to a real
> BNO055 over I2C. On a machine without the sensors, labels simply report `null` and the
> page shows no data.

## Run

```bash
jetson-imu-tui                          # loopback only (default 127.0.0.1:8000)
jetson-imu-tui --lan                    # serve the network instead (bind ::)
jetson-imu-tui --config path/to/my.toml
jetson-imu-tui --host 0.0.0.0 --port 8011
```

**It binds loopback by default**, so a Jetson in the field exposes nothing and the app needs
no network at all — every asset (uPlot, three.js) is vendored under `static/`. Reach the UI
over an SSH tunnel, or pass `--lan` to serve it on the network. An explicit `--host` wins over
both, so a specific address is never silently widened.

On start it auto-connects the IMUs and prints the URL(s). Open it from your laptop:

- **SSH tunnel (most reliable** — works regardless of the Jetson's LAN/IPv6/NAT, and
  reuses the connection you already have):
  ```bash
  ssh -L 8000:localhost:8000 <user>@<jetson>     # keep this open
  # then open http://localhost:8000
  ```
- **Direct** (laptop on the same network): use the IPv6/IPv4 URL the server prints,
  e.g. `http://[2001:...]:8000` (IPv6 needs the `[ ]` brackets). Note a global IPv6
  address may be a rotating privacy address — prefer the tunnel for a stable URL.

In the browser: switch signal (Euler / Accel / Gyro / Quat), toggle **Numbers** for a
text readout, **Pause** to freeze, set the **Hz** field to change recording rate, and
**Record** to write CSVs on the Jetson, and **Load** to reopen a past recording offline (see
**Reviewing a recording** below). uPlot and three.js are vendored in `static/` — nothing is
fetched from the internet, by the Jetson or the browser.

Which signal buttons appear depends on the source: the four per-sensor signals always, plus one
button per **telemetry group** the link carries (`knee`, `motor`, `trace`, `state` on the
`exo_v1` serial layout — see **Serial source** below). The status line shows the measured wire
rate next to the poll rate; it is the only visible check that `[source] sample_hz` matches
reality.

HTTP API (for scripting): `GET /` (page), `GET /data` (latest values + status JSON),
`POST /record` (toggle), `POST /freq?hz=N` (set recording rate), `POST /zero` (tare toggle),
`POST /source?kind=i2c|serial` (switch IMU source), `GET /calibration` (per-sensor calibration
levels), `GET /cls` (predictions + latest decision), `POST /cls/toggle` (stop/start inference),
`GET /axis-remap` (current mapping) and `POST /axis-remap` (apply `{"ops":["rot_x_90",...]}`,
or `placement=P0..P7`, or numeric `config`/`sign`), `GET /recordings` (list recorded sessions),
`GET /recordings/<date>/<time>` (one session as plot-ready columns; optional `from`, `to`,
`max_points`). `Ctrl-C` stops cleanly.

## Configuration

`config/default.toml`:

```toml
[buses]
1 = "Left"
7 = "Right"

[defaults]
log_dir = "./logs"
plot_fps = 15             # browser poll rate (samples/sec fetched by the page)
plot_window_seconds = 10  # rolling time window shown in the plots
record_hz = 100           # CSV drain cadence
web_host = "127.0.0.1"    # loopback only; "::" = IPv6+IPv4 dual-stack, or pass --lan
web_port = 8000
```

The bus→label table is the single source of truth — change it here to swap
Left/Right assignment or rename the IMUs.

## Axis remap (software)

The sensor axes are remapped **on the host**, applied to both sources. Open the **Axis** popup
and build a chain of steps: pick an axis, pick 90/180/270°, hit **Rotate** — or **Negate** to
flip one axis. The 3D cube follows live, so you adjust until it matches the real sensor.

```toml
[axis]
ops = ["rot_y_90"]        # applied in list order; [] = identity
```

Op names are `rot_{x,y,z}_{90,180,270}` and `flip_{x,y,z}`. **Rotations turn the readings**, not
the sensor — `rot_x_90` sends `(x, y, z) → (x, -z, y)`, right-hand rule.

* `[axis] ops` is only the **boot default**. A mapping applied from the UI is written to
  `<log_dir>/axis_remap.json`, which wins at startup; delete that file to fall back to the TOML.
* Applying a mapping **clears the tare** (the zero reference belongs to the old frame) and
  **resets the CLS window** (its buffered frames do too, and their timestamps are continuous so
  the gap check would never catch it).
* `accel` and `gyro` are permuted and signed; `quat` is re-expressed and `euler` is recomputed
  from it, so all four CSVs and the cube stay in one coordinate system. At identity the chip's
  own fused euler is passed through untouched.
* An **odd number of negations is a mirror**, not a mounting orientation. `accel`/`gyro` still
  follow it, but no rotation quaternion exists, so `quat`/`euler` pass through unchanged and the
  popup says so. Prefer rotations unless you specifically want a reflection.
* P0–P7 presets and numeric `config`/`sign` still work (same bit layout as the old registers),
  so existing scripts and old `axis_remap.json` files keep working.

> This used to write the BNO055's `AXIS_MAP_CONFIG`/`AXIS_MAP_SIGN` registers. Moving it to the
> host fixed three things: the serial source can use it at all, nothing is lost on a power cycle,
> and the UI can speak in rotations instead of packed bytes. The chip is left at its P1 identity
> default — if you previously relied on a register mapping, re-enter it here.

Covered by `others/tests/test_axis_transform.py` (61 cases, no hardware needed).

## Serial source (Arduino / Simulink instead of I2C)

**This is the shipped default.** The app runs off binary frames arriving over a serial port —
an Arduino (`others/simulink/motor_control.slx` on an MKR WiFi 1010) reading the sensors and
forwarding them — instead of the Jetson reading BNO055s over I2C. On the `exo_v1` layout the
same frame carries the rig's telemetry too: both knees, motor feedback, and the controller
trace that shows how `finalClass` turned into a motor command. Plots, recording and CLS all
work, and the link is bidirectional, with classification results going back to the Arduino
(see below). Switch back to the onboard I2C sensors at runtime with the **IMU:** button, or
change the boot default in the config:

```toml
[source]
kind = "serial"           # "serial" (default) | "i2c" — boot default; switchable at runtime
port = "/dev/ttyACM0"     # "COM17" on Windows
baud = 115200
label = "Left"            # label the IMU appears under; must match [cls] sensor
magic = "aa55"            # frame header in hex; required unless the layout carries a timestamp
layout = "exo_v1"         # frame contents and channel order — see the table below
gyro_units = "deg"        # units the device sends; "deg" is scaled to rad/s, "rad" passes through
sample_hz = 80            # rate the device actually sends at — MEASURED, not nominal
```

**Wire format** (`read_serial.py`): an optional header followed by little-endian float32. What
those floats mean is an ordered list of **channel blocks**, so a frame that grows new channels
costs one table row rather than a new decoder. Accel is m/s² **including gravity**.

| block | floats | contents |
|---|---|---|
| `enable` | 1 | the device's SWITCH line; non-zero = the controller may drive |
| `gyro` | 3 | `gx gy gz`, in the device's units (`gyro_units` converts) |
| `accel` | 3 | `ax ay az`, m/s² **including gravity** |
| `knee4` | 4 | `ang_r vel_r ang_l vel_l` — knee angle + angular velocity, both legs |
| `fb6` | 6 | `pos_r speed_r torque_r pos_l speed_l torque_l` — motor feedback |
| `trace5` | 5 | `finalClass LU_AVEL_F L_KVEL L_KWRAP MotorCom_L` — controller trace |
| `t_src` | 1 | the device's own clock in seconds, as **data** |
| `t` | 1 | the device's own clock in seconds, as a **sync anchor**; last block only |

`layout` names a combination of them:

| `layout` | floats | frame | typical source |
|---|---|---|---|
| `accel_gyro_t` | `ax ay az gx gy gz t` | 28 B + header | Simulink Serial Transmit |
| `gyro_accel_t` | `gx gy gz ax ay az t` | 28 B + header | as above, channels swapped |
| `accel_gyro` | `ax ay az gx gy gz` | 24 B + header | Arduino sketch, no clock |
| `gyro_accel` | `gx gy gz ax ay az` | 24 B + header | Arduino sketch, no clock |
| `exo_v1` (shipped) | `enable gyro accel knee4 fb6 trace5 t_src` | **92 B + header `aa55`** = 94 B | the full rig uplink |

A layout that does not carry a block reports it as `None`, never as a zero — so "this link has
no knee channels" stays distinguishable from "the knees are at zero". The table is checked at
import (block names, no duplicates, `t` last, at most one clock, declared float count), because
a misplaced block shifts every channel after it and that failure is completely silent.

**`t` vs `t_src`** — same quantity, different job, and a layout may hold at most one. `t` gates
sync: it must increase monotonically or the decoder re-aligns. `t_src` is an ordinary channel
that gates nothing. Prefer `t_src` on a long frame: float32 quantises, and once its ulp exceeds
the frame period two adjacent timestamps round to the same value, `Δt` reads 0 and a healthy
stream is declared out of sync — at 100 Hz that is `t ≥ 2¹⁷ s` (~36 h), at 200 Hz ~18 h. A
94-byte frame does not need the help: a 2-byte header matching at 9 consecutive frame boundaries
is a ~(1/65536)⁹ false lock.

**Telemetry is not a sensor signal.** The knee, motor, trace and enable channels belong to the
rig, not to a labelled IMU, so they have no label layer, they are never axis-remapped (rotating
a motor torque would be meaningless) and they get their own charts, their own Numbers card and
their own CSVs. `imu_common.TELEMETRY_GROUPS` is the one table that names them — it drives the
`/data` payload, the CSV headers, the offline loader and the charts at once.

**How a group is cut into charts** is derived from its channel names, not written out anywhere:

| group | charts | why |
|---|---|---|
| `knee` | `right` (`ang_r` `vel_r`), `left` (`ang_l` `vel_l`) | the channels pair off by side, so one leg's angle and rate sit together and read against the other's |
| `motor` | `right` (`pos_r` `speed_r` `torque_r`), `left` | same |
| `trace` | one per channel | no side to pair by, and the ranges do not compare — a 0–10 class index on the same axis as a few-hundred deg/s velocity is a flat line |
| `state` | one per channel | `enable` is 0/1, `t_src` climbs into the thousands |

Colours run by position *within* a chart, so the right chart's `pos_r` and the left chart's
`pos_l` share one — the two legs read as parallel. A group added to `TELEMETRY_GROUPS` gets a
layout from the same rule with no edit to the page.

**Chart heights are draggable.** Charts share the column evenly by default, which stops working
once a group splits into five or six plots. Drag the grip along a chart's bottom edge to pin
that one to a height; the rest keep sharing what is left, and the column scrolls if the total
overflows. Double-click the grip to hand a chart back to the shared row. Heights are stored per
`signal|chart` in `localStorage`, so they survive the rebuild that every theme change, label
change and source switch triggers.

**Category channels are labelled, not numbered.** `finalClass` is a class *index*, so its chart
is drawn differently from a measurement:

* the Y axis reads `stand` / `walk` / `sit`, not `0` / `1` / `7`;
* the range is pinned to the whole class set, so a given height on the chart always means the
  same class instead of the axis rescaling around whichever classes are on screen;
* the line is stepped — a class index does not interpolate, and a ramp between two classes
  would draw a transition through classes that were never predicted;
* how many class names fit is computed from the chart's current height, so dragging it taller
  fills in the ones a short chart had to skip.

`web_server.ENUM_TICKS` is the table (`{"finalClass": CLASSES}`); it lives there rather than in
`imu_common` so the channel registry stays free of classifier imports, and a test pins it to
`CLASSES` so the two cannot drift.

**Clamping noisy channels.** A single spike on a channel whose real values are small stretches
the shared Y axis and flattens everything beside it, so a group can be given an absolute limit:

```toml
[telemetry.clip]
trace = 100.0             # values outside ±100 are clamped to it
```

* Applied **on read-out**, so the ring buffer still holds what the device sent and raising the
  limit re-exposes the buffered history instead of finding it already flattened.
* Applied once, in `SerialImuService._clean`, which every consumer goes through — so a clamped
  value is what the plots draw *and* what `trace.csv` records.
* A group not listed passes through untouched. **Never add `state`**: `t_src` is the device
  clock, and any limit turns it into a plateau seconds after start-up.
* Lossy on purpose — a flat line at the limit cannot be told from the signal genuinely sitting
  there, so a clamped group's chart header carries a `clip ±100` badge.
* **Non-finite values (NaN/inf) are dropped for every group**, limit or no limit. That is not
  cosmetic: `json.dumps` renders NaN as a bare `NaN` token, which is valid Python and invalid
  JSON, so one NaN on the wire would make the browser reject the whole `/data` response — and
  the poll's `catch` swallows it, leaving a frozen page and nothing in the log.

**Switching to `exo_v1` is not backward compatible.** 26 and 94 are not multiples of each other,
so a device still sending the old frame never aligns and the UI shows `(no data)` forever. The
reader logs a hint after 5 s (`open but no frame decoded … check [source] layout`), and
`others/tools/serial_monitor.py` is the direct check:

```bash
python others/tools/serial_monitor.py --layout exo_v1
```

On a wide layout it prints **every channel by name**, refreshed in place. That block is the
acceptance test for the transmitting side, because a Mux wired in the wrong order is silent on
the wire — the frame length is unchanged and the header still aligns, only the values sit in the
wrong columns. Move one joint at a time and check that only its own two channels react.

**Alignment** needs a timestamp or a header, and the layout decides which. With a timestamp,
byte alignment is recovered from it increasing monotonically, so the header is optional and a
stream that loses bytes re-aligns by itself. Without one the header is the only anchor and is
therefore **required** — `decode_frames` raises rather than emit misaligned junk — and re-sync
works off the header alone.

**Channel order is declared, not detected.** Nothing in the bytes distinguishes accel from gyro,
and choosing wrong puts gravity into the gyro channels: the classifier then degrades with no
error anywhere. Verify once on the device — hold it still and watch `/data`: one *accel* axis
must read ≈ 9.8 and the gyro must read ≈ 0.

`sample_hz` must equal the rate the device actually sends, since it drives the CLS decimation.
The serial reader logs the measured rate a few seconds after connecting
(`Left: 79.7 Hz observed on /dev/ttyACM0 — set sample_hz to match`), so check the two agree.

**Gyro units are the trap.** BNO055 firmware commonly reports **deg/s** (values quantized to
1/16, peaking in the hundreds) while the classifier was trained on rad/s — leaving `gyro_units`
wrong feeds the model input ~57× too large and predictions become meaningless without any error.
Verify on the device: hold it still (gyro ≈ 0, |accel| ≈ 9.8), then rotate ~90° in one second and
watch `/data` — the peak should read ≈ 1.5, not ≈ 90.

**Not available over serial**, because the stream carries no fusion output: Euler and
Quaternion plots, the 3D cube, `euler_angles.csv` / `quaternions.csv` (written with empty
cells), and the calibration popup.

The **axis remap does work over serial** — it is a host-side transform now, so it no longer
matters that the link cannot reach the sensor's registers. It composes *on top of* whatever
mapping the transmitting device applies in its own firmware, so you can correct a mounting
mismatch without reflashing; what has to match the training capture is the combination.

### Result return channel (Jetson → Arduino)

The same cable carries the classification result back. Each aggregated decision is written as
**one raw byte** — the class index, `0x00`–`0x0A`, in `CLASSES` order:

```
0 stand   1 walk   2 turn   3 jog   4 rampascent   5 stairascent
6 stairdescent   7 sit   8 sit-to-stand   9 stand-to-sit   10 rampdescent
```

> **The sketch must read its serial input, even if it ignores the value.** A device that only
> transmits never drains its USB receive buffer; that buffer fills after a handful of results and
> every further write blocks. The host caps each write (`WRITE_TIMEOUT_S`) so inference survives,
> but from then on **no result reaches the device** — the UI shows `IMU: Serial (TX blocked)` and
> the log says `result write failed — is the device reading its serial input?`.

```c
// Arduino side — drain every loop(), not just when you feel like acting on it
while (Serial.available()) {
  uint8_t cls = Serial.read();
  last = millis();
  act(cls);
}
if (millis() - last > 1500) failsafe();   // link went quiet
```

* **Rate:** one byte per decision — **2 Hz** at the shipped settings, not per frame.
* **No sentinel.** When there is no decision (CLS stopped, window still filling, a sensor stall)
  nothing is sent; the link simply goes quiet. Time out on the Arduino if that matters.
* **Serial source only.** Switch to the I2C IMUs and the port is closed, so transmission stops.
* One `serial.Serial` handle is shared by the reader thread and the writer, so this needs no
  second port. Writes are bounded and a failure never reaches the inference thread — it is logged
  once, the result is dropped rather than queued, and the reader reconnects on its own.

The source button distinguishes the three ways this can fail: `(no link)` the port will not open,
`(no data)` it opened but no frames arrive, `(TX blocked)` frames arrive but results are not
being accepted.

## Reviewing a recording

**Load** in the toolbar lists every session under `<log_dir>` and reopens one in the same charts
the live view uses. The page stops polling while a session is open; **Live** returns to the
stream and **Full** zooms back out to the whole session.

* **Decimation is server-side, and it is a min/max envelope.** Ten minutes at 200 Hz is 120k
  rows across ~23 channels — far more than a few thousand pixels can show and far more than is
  worth serialising. Each window is reduced to `max_points` per channel by taking the minimum
  and maximum inside each time bucket, so a one-sample spike or dropout survives; plain "every
  Nth row" would hide exactly what the viewer is opened to find.
* **Zoom re-fetches.** Drag-select a range and the page requests that window again at full
  budget, so full resolution is reachable everywhere without ever shipping the whole file.
  Buckets are cut on time rather than per channel, so all channels keep one shared x axis; the
  min/max pair inside a bucket may be ordered opposite to the samples, which shifts a point by
  less than one bucket width and disappears as soon as you zoom in.
* **A missing file means a missing channel, not zeros.** An I2C recording has no `knee.csv`, and
  a serial one has `euler_angles.csv` full of empty cells. Both are reported as absent groups
  rather than drawn as a flat line at zero, which would look like a real measurement.
* **Sessions that cross midnight are handled.** The `Time` column is `%H:%M:%S.%f` with no date;
  the folder name supplies the day and a backward step adds another, so the duration does not
  come out negative.

Scriptable without the browser:

```bash
curl -s localhost:8000/recordings | python -m json.tool
curl -s 'localhost:8000/recordings/2026_08_20/14_03_11?from=12&to=18&max_points=8000'
```

Or in Python, with no server at all:

```python
from jetson_imu_tui.session_load import list_sessions, load_session
d = load_session("./logs", "2026_08_20/14_03_11", t_from=12, t_to=18)
d["telemetry"]["motor"]["channels"]["torque_l"]     # list[float | None]
```

## Switching source at runtime

The **IMU:** button in the toolbar flips between the onboard I2C IMUs and the serial one without
a restart. The plots rebuild for the new label set (two sensors ↔ one) on the next poll, and CLS
is re-pointed **without reloading the checkpoint**.

* **Session only** — `config/default.toml` is never written; a restart returns to `[source] kind`.
* An **active recording is stopped** (the CSV headers and label set are already fixed). Press
  Record again after switching to start a new session.
* The CLS window and the vote buffer are cleared, so the first new decision takes about
  `window × (1/target_hz)` + one vote window — roughly 2.5 s at the shipped settings.
* If the device is not present the switch still happens and says so; the serial reader keeps
  retrying, so plugging it in afterwards recovers on its own.
* Scriptable: `curl -X POST 'localhost:8000/source?kind=serial'` → `{"ok":true,...}`.
* The button greys out if the other source's dependency is not installed (`pyserial` for serial,
  the Adafruit I2C stack for I2C).

If the two sources run at different wire rates, set `[source] sample_hz` — it drives the CLS
decimation, and one global value cannot be right for both.

## Orientation fusion & calibration

The BNO055 runs in **IMUPLUS** mode: the chip fuses **accelerometer + gyroscope** on-board to
produce relative orientation. The **magnetometer is not used**, which deliberately avoids the
magnetic-distortion problems of a thigh mount near motors/metal — the trade-off is that yaw
(heading) is **relative** (no absolute north) and drifts slowly on the gyro. Euler, quaternion,
acceleration and gyroscope are all read straight from the chip's fused output; there is no
software filter on the host.

**Calibration** — open the **Calib** popup for live status:

- **Gyro** — set the sensor down and keep it still for a few seconds → level reaches 3.
- **Accel** — slowly tilt through a few stable positions (≈45°/90°) → level reaches 3.
- **Mag** — unused in IMUPLUS; it stays at 0 by design (no figure-8 needed).
- Calibration is not persisted across power cycles; the levels re-converge after each boot.

**Units:** Euler in degrees (`x`=roll, `y`=pitch, `z`=heading), acceleration in m/s²,
gyroscope in rad/s, quaternion `(w, x, y, z)` unitless. If gyro readings look ~57× too large,
the installed driver is reporting deg/s — set `_GYRO_TO_RADS = math.pi/180` in
`src/jetson_imu_tui/imu_service.py`.

**Axis remap** — see [Axis remap (software)](#axis-remap-software) below. It is applied on the
host, not written to the chip's `AXIS_MAP_CONFIG`/`AXIS_MAP_SIGN` registers.

> Absolute heading (9-DOF) would require **NDOF** mode + magnetometer + a per-boot figure-8.
> That was deliberately *not* chosen here because of the thigh-mount magnetic environment; it
> is a one-line change in `imu_service.py` (`FUSION_MODE`) if ever needed.

## Real-time activity classification (CLS)

An optional page runs a vendored LIMU-BERT + GRU classifier on the live IMU stream (11
classes, ids 0–10: stand / walk / turn / jog / rampascent / stairascent / stairdescent /
sit / sit-to-stand / stand-to-sit / rampdescent). It samples one IMU, **block-averages** the
100 Hz stream down to 10 Hz (matching the training `down_sample`, not plain decimation), keeps
a 20-sample (2 s) sliding window, and emits a prediction every ~100 ms. Those frame predictions
are then aggregated into one stable **decision** (see below). It self-disables if `torch` or the
checkpoint is missing.

Enable it in `config/default.toml`:

```toml
[cls]
enabled = true
model_path = "src/jetson_imu_tui/cls/model/<checkpoint>.pt"  # a BERT-finetune jetson_leg .pt
sensor = "Left"           # which IMU to classify (leg source is robust to both mounts)
target_hz = 10            # must match the training sampling rate; sample_hz must be an integer multiple, or CLS self-disables
window = 20               # 2 s @ 10 Hz
stride = 1                # a new prediction every ~100 ms
```

### Vote aggregation

A single frame prediction is noisy — sensor noise, transient poses and genuinely ambiguous
points in the gait cycle all produce one-off misclassifications. A downstream consumer (a
controller adjusting assistance) must not react to every frame or its behaviour jitters. So
frames are pooled over a short window and reduced to one decision, trading a little latency for
a large gain in stability. Same idea as the majority vote in the exosuit literature, except the
windows here are a fixed number of frames: there is no gait-phase or step-segmentation signal
available, the prediction stream is all there is.

**Soft voting**, not a majority vote: the probability vectors are averaged element-wise and the
argmax of the average wins. At five votes that matters twice over — a label count over 3+ classes
can split 2:2:1 and force an invented tie-break, and it throws away confidence, so two hesitant
51% errors outvote one confident 99% correct frame. Averaging has neither problem and costs
nothing.

```toml
[cls.vote]
enabled = true            # false = passthrough (one decision per inference)
window = 5                # frames averaged per decision
emit_every = 5            # frames between decisions
hysteresis = 0            # 0 = switch as soon as the vote does
```

**decision rate = `target_hz / (stride × emit_every)`** — the voter counts *inferences*, not raw
samples. The shipped values give **2 Hz**, one decision per 500 ms from 5 frames.

| `window` / `emit_every` | behaviour |
|---|---|
| `5` / `5` | **tumbling** — each decision from 5 fresh frames (the baseline) |
| `10` / `5` | **sliding** — same 2 Hz, averaging the last 10; steadier across class boundaries, more latency |
| `1` / `1` | passthrough — every inference is its own decision |

`hysteresis = n > 1` additionally requires a new class to win `n` consecutive windows before the
output follows; until then the previous class is re-emitted. Use it if the output flickers at
boundaries — the trade is false switches against delayed transitions, so it depends on which your
consumer tolerates less. Start with the plain 5-frame version as a baseline.

The decision is what the CLS banner shows, what goes back over serial, and what `cls_vote.csv`
records; the 10 Hz frame stream stays visible in the log below the banner, with a `◀` marker on
each frame that closed a window. A recording therefore holds both streams — `cls.csv`
(frame-level, step-held at the IMU rate) and `cls_vote.csv` (one row per decision, at its own
rate) — which is what an offline comparison of frame-level vs post-voting accuracy needs. Evaluate
windows where the true class changes mid-window as their own category: that is where this kind of
aggregation is weakest.

Any discontinuity — a sensor stall, pausing CLS, switching source — clears the partial window
rather than voting across the gap, so no decision is ever built from two sides of a break.
The aggregator itself is standalone (`cls/vote.py`: no torch, no numpy, no I/O) and injected
into `ClsService`, so it can be swapped or unit-tested on its own.

> **Checkpoint.** Use the winning finetune-high-lr jetson_leg checkpoint from
> LIMU-BERT-Public (`bench_run45`, `finetune-high-lr__lr0.3__seed42.pt`). Copy the `.pt`
> into `src/jetson_imu_tui/cls/model/` and point `model_path` at it.

**Training↔deployment contract — verify on the device or accuracy silently degrades:**

1. **Gyro units.** The model was trained on rad/s. If this driver build reports deg/s
   (readings ~57× too large), set `_GYRO_TO_RADS = math.pi/180` in `imu_service.py`.
   Pin the `adafruit-circuitpython-bno055` version to the one used during data capture.
   On the serial source the equivalent knob is `[source] gyro_units`.
2. **Axis remap.** The effective mapping — `[axis] ops`, or `<log_dir>/axis_remap.json` if
   present — must match the one used when the training data was collected (default identity
   unless changed). On the serial source this means the *combination* of the transmitting
   device's own firmware mapping and the host-side one.
3. **Tare / gravity.** CLS bypasses the tare and feeds **gravity-inclusive** accel (the
   model needs it). Confirm the training data was captured with tare OFF too.
4. **Model I/O.** Accel is normalized ÷9.8 (in `cls/preprocess.py`); the 11-class order in
   `cls/model/__init__.py` matches `dataset/jetson_leg/label_map.json`. Don't reorder — it is
   also the byte values sent back over serial.

Offline model-math parity is proven by `others/tools/parity_check_cls.py` (run from the
LIMU-BERT-Public repo); the 100 Hz→10 Hz block-average is covered by
`others/tests/test_cls_downsample.py`, serial decoding + unit conversion by
`others/tests/test_read_serial.py`, vote aggregation by `others/tests/test_vote.py`, the return
channel by `others/tests/test_serial_tx.py`, the runtime source switch by
`others/tests/test_source_switch.py`, and the axis remap by
`others/tests/test_axis_transform.py`.

> The checkpoint's output width must match `CLASSES` in `cls/model/__init__.py`. A 7-class `.pt`
> against the 11-label list fails to load and CLS self-disables with a `size mismatch` warning.

## Out of scope

CAN bus, joint-angle math, ML gait phase, calibration UI.
