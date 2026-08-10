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
- **Axis** popup: remap the BNO055 output axes (datasheet §3.4 placements P0–P7 or a manual
  axis/sign mapping), applied to the chip's `AXIS_MAP_CONFIG`/`AXIS_MAP_SIGN` registers, with a
  live 3D cube to confirm the result.
- **Calib** popup: live onboard calibration status (sys / gyro / accel, 0–3) with guidance.
- **IMU:** button switches between the onboard I2C IMUs and a serial (Arduino) IMU at runtime.
- **CLS** page: online activity classification, with frame predictions aggregated by soft voting
  into one stable decision — sent back to the Arduino as a class-index byte over the same link.
- **Record** toggle and an adjustable **recording frequency** (1–200 Hz) from the page.
- Recording writes one directory of comma-separated files per session under
  `<log_dir>/YYYY_MM_DD/HH_MM_SS/`:
  - `quaternions.csv` — `Time,Left_w,Left_x,Left_y,Left_z,Right_w,Right_x,Right_y,Right_z`
  - `accelerometers.csv` — `Time,Left_x,Left_y,Left_z,Right_x,Right_y,Right_z`
  - `gyroscopes.csv` — same columns as accel
  - `euler_angles.csv` — degrees (ZYX intrinsic: x=roll, y=pitch, z=yaw)
  - with CLS active: `cls.csv` (frame predictions + per-class probabilities),
    `cls_vote.csv` (one row per aggregated decision) and `model_input.csv` (the exact
    6-channel vectors fed to the model)

## Install

```bash
pip install -e .            # flask + adafruit-circuitpython-bno055 + Adafruit-Blinka + adafruit-extended-bus
pip install -e ".[serial]"  # add pyserial if the IMU arrives over serial (see below)
```

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
jetson-imu-tui                          # bind/port from config (default [::]:8000)
jetson-imu-tui --config path/to/my.toml
jetson-imu-tui --host 127.0.0.1 --port 8011
```

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
**Record** to write TSVs on the Jetson. uPlot is loaded from a CDN, so the *browser*
needs internet for the library (the Jetson does not).

HTTP API (for scripting): `GET /` (page), `GET /data` (latest values + status JSON),
`POST /record` (toggle), `POST /freq?hz=N` (set recording rate), `POST /zero` (tare toggle),
`POST /source?kind=i2c|serial` (switch IMU source), `GET /calibration` (per-sensor calibration
levels), `GET /cls` (predictions + latest decision), `POST /cls/toggle` (stop/start inference),
`GET /axis-remap` (current mapping) and `POST /axis-remap` (apply `placement=P0..P7` or numeric
`config`/`sign`). `Ctrl-C` stops cleanly.

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
record_hz = 100           # TSV recording rate
web_host = "::"           # "::" = IPv6+IPv4 dual-stack; "0.0.0.0" = IPv4 only
web_port = 8000
```

The bus→label table is the single source of truth — change it here to swap
Left/Right assignment or rename the IMUs.

## Serial source (Arduino / Simulink instead of I2C)

The same app can run off **one IMU streaming binary frames over a serial port** — an Arduino (or
a Simulink model) reading the BNO055 and forwarding it — instead of the Jetson reading the chips
over I2C. Plots, recording and CLS all work; the link is bidirectional, with classification
results going back to the Arduino (see below). The boot default is chosen in the config:

```toml
[source]
kind = "serial"           # "i2c" (default) | "serial" — boot default; switchable at runtime
port = "/dev/ttyACM0"     # "COM17" on Windows
baud = 115200
label = "Left"            # label the IMU appears under; must match [cls] sensor
magic = "aa55"            # frame header in hex; required unless the layout carries a timestamp
layout = "gyro_accel"     # frame contents and channel order — see the table below
gyro_units = "rad"        # units the device sends; "deg" is scaled to rad/s, "rad" passes through
sample_hz = 80            # rate the device sends at; omit to inherit [defaults] sample_hz
```

**Wire format** (`read_serial.py`): an optional header followed by 6 or 7 little-endian float32.
Accel is m/s² **including gravity**; `t`, where present, is the source's own clock in seconds.
Pick the frame contents with `layout`:

| `layout` | floats | frame | typical source |
|---|---|---|---|
| `accel_gyro_t` (default) | `ax ay az gx gy gz t` | 28 B + header | Simulink Serial Transmit, header `5aa5` |
| `gyro_accel_t` | `gx gy gz ax ay az t` | 28 B + header | as above, channels swapped |
| `accel_gyro` | `ax ay az gx gy gz` | 24 B + header | Arduino sketch, no clock |
| `gyro_accel` | `gx gy gz ax ay az` | 24 B + header | Arduino sketch, no clock |

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

**Not available over serial**, because the stream carries only accelerometer and gyroscope:
Euler and Quaternion plots, the 3D cube, `euler_angles.csv` / `quaternions.csv` (written with
empty cells), and the calibration popup. The axis remap is *reported* but not writable — that
mapping is configured on the transmitting device, and it must match the one used to collect the
training data (see the contract below).

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

**Axis remap** — the **Axis** popup writes `AXIS_MAP_CONFIG`/`AXIS_MAP_SIGN` on the chip. These
registers are **volatile** (lost on power cycle); the chosen mapping is saved to
`<log_dir>/axis_remap.json` and re-applied automatically on connect.

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
2. **Axis remap.** The persisted `<log_dir>/axis_remap.json` mapping must match the
   mapping used when the training data was collected (default P1 identity unless changed).
3. **Tare / gravity.** CLS bypasses the tare and feeds **gravity-inclusive** accel (the
   model needs it). Confirm the training data was captured with tare OFF too.
4. **Model I/O.** Accel is normalized ÷9.8 (in `cls/preprocess.py`); the 11-class order in
   `cls/model/__init__.py` matches `dataset/jetson_leg/label_map.json`. Don't reorder — it is
   also the byte values sent back over serial.

Offline model-math parity is proven by `others/tools/parity_check_cls.py` (run from the
LIMU-BERT-Public repo); the 100 Hz→10 Hz block-average is covered by
`others/tests/test_cls_downsample.py`, serial decoding + unit conversion by
`others/tests/test_read_serial.py`, vote aggregation by `others/tests/test_vote.py`, the return
channel by `others/tests/test_serial_tx.py`, and the runtime source switch by
`others/tests/test_source_switch.py`.

> The checkpoint's output width must match `CLASSES` in `cls/model/__init__.py`. A 7-class `.pt`
> against the 11-label list fails to load and CLS self-disables with a `size mismatch` warning.

## Out of scope

CAN bus, joint-angle math, ML gait phase, calibration UI.
