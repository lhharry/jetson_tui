# jetson_tui

A headless service that streams and records two BNO055 IMUs on an NVIDIA Jetson and
serves their data to a browser for live plotting. Left thigh sits on `/dev/i2c-1`,
right thigh on `/dev/i2c-7` (both at default address `0x28`). Reading and on-board
fusion are handled by the [`imu-python`](https://pypi.org/project/imu-python/) package.

Rendering happens **in the browser on your laptop** (with [uPlot](https://github.com/leeoniya/uPlot)),
not on the Jetson — so the Jetson spends ~no CPU on the UI, leaving headroom for other
workloads (e.g. an AI model) on the same board.

## Features

- Auto-connects both IMUs on start and serves a single live page.
- Four signal views (Euler / Accel / Gyro / Quaternion); Left + Right overlaid.
- Switch to a **numbers** view for live numeric readouts of every signal.
- **Record** toggle and an adjustable **recording frequency** (1–200 Hz) from the page.
- Recording writes four comma-separated files per session under
  `<log_dir>/YYYY_MM_DD/HH_MM_SS/`:
  - `quaternions.csv` — `Time,Left_w,Left_x,Left_y,Left_z,Right_w,Right_x,Right_y,Right_z`
  - `accelerometers.csv` — `Time,Left_x,Left_y,Left_z,Right_x,Right_y,Right_z`
  - `gyroscopes.csv` — same columns as accel
  - `euler_angles.csv` — degrees (ZYX intrinsic: x=roll, y=pitch, z=yaw)

## Install

### Jetson hardware

```bash
pip install -e '.[hw]'   # hw = BNO055 driver + jetson-gpio
```

Make sure the user is in the `i2c` group and both buses are exposed:

```bash
ls /dev/i2c-1 /dev/i2c-7
i2cdetect -y 1
i2cdetect -y 7   # expect 0x28 on both
```

On Jetson Nano (legacy), bus 7 may need a device-tree overlay. On Orin Nano dev
kits, pins 3/5 already map to bus 7 out of the box.

### Dev host (laptop, no Jetson hardware)

`imu-python` falls back to mock IMUs when no I2C bus is reachable, so everything runs
end-to-end with synthetic data:

```bash
pip install -e .
```

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
`POST /record` (toggle), `POST /freq?hz=N` (set recording rate). `Ctrl-C` stops cleanly.

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

## Orientation fusion & calibration (notes)

**Default (current):** `imu_python` runs the BNO055 in **AMG mode** (raw accel/gyro/mag)
and computes orientation itself with a software **Madgwick** filter. The **magnetometer
is disabled** (no calibration file present), so:

- orientation is **relative**; yaw (heading) drifts slowly with no magnetic reference;
- there is **no per-boot calibration** — the filter just converges over ~30 s.

The true sensor rate (~100–115 Hz) is set by the sensor bandwidth + I2C, not by config;
`freq_hz=100` in `imu_python` is only a Madgwick fallback constant.

Two optional upgrades (not enabled here):

**A) Magnetometer-aided (keep software fusion).** Run the upstream calibration **once** per
sensor/environment, then restart:
```bash
python -m imu_python.calibration.calibration   # rotate each IMU through all orientations, Ctrl+C
```
It fits hard/soft-iron params and writes `calibration.json` under imu_python's data dir
(e.g. `.venv/lib/pythonX.Y/data/calibration/`). On next start the log shows
"Loaded magnetometer calibration …" and yaw becomes more stable. (Note: that path lives
inside the venv and is lost on reinstall; a Pi calibration file is **not** reusable — params
are per physical sensor + mounting + local field.)

**B) Onboard NDOF fusion (chip does the fusion + per-boot figure-8).** Before connecting,
override `imu_python.registry.IMU_DEVICES["BNO055"]` with a config that (1) ends `pre_config`
in `NDOF_MODE`, (2) maps `IMUSensorTypes.quat: IMU0` (so `wrapper.get_imu_data` returns the
chip quaternion directly, bypassing Madgwick), and (3) sets `scalar_first=True`. Then
figure-8 each boot until `.calibration_status` reaches 3. Fusion is locked to 100 Hz.

**⚠️ Magnetic-environment caveat:** both A and B rely on the magnetometer. Mounted on a thigh
near motors/metal, the field is disturbed — NDOF calibration may never reach 3 and heading
can be *worse* than gyro/accel-only. That is why the default is AMG + software fusion with
the magnetometer off.

## Out of scope

CAN bus, joint-angle math, ML gait phase, calibration UI.
