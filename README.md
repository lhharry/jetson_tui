# jetson_tui

A Textual TUI for streaming and recording two BNO055 IMUs on an NVIDIA Jetson.
Left thigh sits on `/dev/i2c-1`, right thigh on `/dev/i2c-7` (both at default
address `0x28`). Reading and on-board fusion are handled by the
[`imu-python`](https://pypi.org/project/imu-python/) package; this project
provides the operator-facing UI: status panel, per-IMU live readouts, TSV
recorder, frequency / log-folder pickers, and an in-terminal plot via
[`textual-plotext`](https://pypi.org/project/textual-plotext/).

## Features

- Connect / disconnect both IMUs in a background worker (the asyncio loop never blocks on I2C).
- Toggle streaming, set the recording frequency (1–200 Hz), pick the log directory.
- Recording writes four tab-separated files per session under
  `<log_dir>/YYYY_MM_DD/HH_MM_SS/`:
  - `quaternions.tsv` — `Time Left_w Left_x Left_y Left_z Right_w Right_x Right_y Right_z`
  - `accelerometers.tsv` — `Time Left_x Left_y Left_z Right_x Right_y Right_z`
  - `gyroscopes.tsv` — same columns as accel
  - `euler_angles.tsv` — degrees, `Time Left_roll Left_pitch Left_yaw Right_roll Right_pitch Right_yaw`
- Plot screen with four signal modes (Euler / Accel / Gyro / Quaternion) and
  pause toggle. Both IMUs are overlaid in the same panel.
- Loguru warnings from `imu-python` (e.g. uncalibrated magnetometer) surface in
  the TUI console panel.

## Install

### Dev host (Windows / Linux laptop, no Jetson hardware)

`imu-python` falls back to mock IMUs when no I2C bus is reachable, so the TUI
runs end-to-end with synthetic data:

```bash
pip install -e .
jetson-imu-tui
```

### Jetson hardware

Add the `hw` extra to pull in the Adafruit BNO055 driver and `jetson-gpio`:

```bash
pip install -e .[hw]
jetson-imu-tui
```

Make sure the user is in the `i2c` group and that both buses are exposed:

```bash
ls /dev/i2c-1 /dev/i2c-7
i2cdetect -y 1
i2cdetect -y 7   # expect 0x28 on both
```

On Jetson Nano (legacy), bus 7 may need a device-tree overlay. On Orin Nano
dev kits, pins 3/5 already map to bus 7 out of the box.

## Run

```bash
jetson-imu-tui              # use config/default.toml
jetson-imu-tui --config path/to/my.toml
```

Key bindings on the main screen:

| Key | Action |
|-----|--------|
| `c` | Connect / disconnect |
| `s` | Toggle streaming |
| `r` | Toggle recording |
| `f` | Set recording frequency (modal, 1–200 Hz) |
| `l` | Pick log directory (modal) |
| `p` | Open the live plot screen |
| `q` | Quit |

Plot screen: `1` Euler · `2` Accel · `3` Gyro · `4` Quat · `space` pause · `q` back.

## Configuration

`config/default.toml`:

```toml
[buses]
1 = "Left"
7 = "Right"

[defaults]
log_dir = "./logs"
ui_refresh_hz = 30
record_hz = 100
plot_window_samples = 500
```

The bus→label table is the single source of truth — change it here to swap
Left/Right assignment or rename the IMUs.

## Out of scope

CAN bus, joint-angle math, ML gait phase, Flask/HTML plotting, calibration UI.
For magnetometer calibration use the upstream tool: `python -m imu_python`
with `calibration_mode=True`.
