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
- **Record** toggle and an adjustable **recording frequency** (1–200 Hz) from the page.
- Recording writes four comma-separated files per session under
  `<log_dir>/YYYY_MM_DD/HH_MM_SS/`:
  - `quaternions.csv` — `Time,Left_w,Left_x,Left_y,Left_z,Right_w,Right_x,Right_y,Right_z`
  - `accelerometers.csv` — `Time,Left_x,Left_y,Left_z,Right_x,Right_y,Right_z`
  - `gyroscopes.csv` — same columns as accel
  - `euler_angles.csv` — degrees (ZYX intrinsic: x=roll, y=pitch, z=yaw)

## Install

```bash
pip install -e .   # flask + adafruit-circuitpython-bno055 + Adafruit-Blinka + adafruit-extended-bus
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
`GET /calibration` (per-sensor calibration levels), `GET /axis-remap` (current mapping) and
`POST /axis-remap` (apply `placement=P0..P7` or numeric `config`/`sign`). `Ctrl-C` stops cleanly.

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

## Out of scope

CAN bus, joint-angle math, ML gait phase, calibration UI.
