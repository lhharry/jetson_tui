"""Headless web server: serve IMU samples for a browser uPlot frontend.

Flask + a few routes (`GET /`, `GET /data`, `POST /record`, `POST /freq`). Acquisition is
decoupled from serving: ``ImuService.start_sampling`` fills one ring buffer per sensor at
``sample_hz``, and every consumer (this server, the recorder, CLS) reads the buffers — a
`/data` request costs no I2C. The browser polls `/data?since=<t>` at a modest rate and
receives the full batch of samples since its last poll, so the plots show the complete
``sample_hz`` stream while staying robust on lossy networks (a dropped poll self-heals on
the next one). No websocket, no async; rendering happens in the browser on the laptop.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request
from loguru import logger

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.cls.service import ClsService
from jetson_imu_tui.cls.vote import SoftVoter
from jetson_imu_tui.config import AppConfig
from jetson_imu_tui.imu_common import (
    PLACEMENTS,
    SIGNAL_AXES,
    SIGNAL_UNITS,
    TELEMETRY_CHARTS,
    TELEMETRY_GROUPS,
    AxisState,
)
from jetson_imu_tui.recorder import Recorder
from jetson_imu_tui.session_load import DEFAULT_MAX_POINTS, list_sessions, load_session

# Both sources are optional *at import time*: the I2C stack (Blinka / adafruit-bno055) is
# Linux-only and pyserial is an extra, so a machine set up for one source need not have the
# other installed. The failure is only reported if that source is the one being asked for.
try:
    from jetson_imu_tui.imu_service import ImuService
except Exception as err:  # pragma: no cover - depends on what is installed
    ImuService, _I2C_IMPORT_ERR = None, err
else:
    _I2C_IMPORT_ERR = None
try:
    from jetson_imu_tui.serial_service import SerialImuService
except Exception as err:  # pragma: no cover - depends on what is installed
    SerialImuService, _SERIAL_IMPORT_ERR = None, err
else:
    _SERIAL_IMPORT_ERR = None


def get_local_ip() -> str | None:
    """Best-effort LAN IPv4 (borrowed from the Pi tool). None if there's no IPv4 route."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        sock.close()


def get_local_ip6() -> str | None:
    """Best-effort global IPv6. None if there's no IPv6 route."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        sock.connect(("2001:4860:4860::8888", 80))
        ip = sock.getsockname()[0]
        return None if ip.startswith(("::1", "fe80")) else ip
    except OSError:
        return None
    finally:
        sock.close()


def _pids_listening_on(port: int) -> list[int]:
    """PIDs holding a LISTEN socket on TCP ``port``, found via ``/proc`` (Linux only, no
    external tools). Returns [] on non-Linux, or when nothing is found / not permitted."""
    if not Path("/proc/net/tcp").exists():
        return []  # not Linux (e.g. the Windows dev box) — nothing to reclaim
    want = f"{port:04X}"  # /proc encodes the local port as uppercase hex
    inodes: set[str] = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]  # drop the header row
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            # cols: sl local_address rem_address st ... inode; st 0A == TCP_LISTEN
            if len(parts) < 10 or parts[3] != "0A":
                continue
            if parts[1].rsplit(":", 1)[-1].upper() == want:  # "IPHEX:PORTHEX" -> PORTHEX
                inodes.add(parts[9])
    if not inodes:
        return []
    pids: set[int] = set()
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            fds = list((pid_dir / "fd").iterdir())
        except OSError:
            continue  # process vanished, or another user's fds we cannot read
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                pids.add(int(pid_dir.name))
                break
    return sorted(pids)


def _free_port(host: str, port: int, *, timeout: float = 5.0) -> None:
    """Reclaim ``port`` if a process is already listening on it, so ``make_server`` can bind.

    A stale instance of this server also holds the I2C buses and CUDA, so callers free the
    port *before* connecting hardware. SIGINT first (the server catches KeyboardInterrupt and
    shuts down gracefully), escalating to SIGKILL if it does not release within ``timeout``."""
    pids = _pids_listening_on(port)
    if not pids:
        return
    print(f"Port {port} in use by PID {', '.join(map(str, pids))} — terminating to reclaim it")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pids_listening_on(port):
            return
        time.sleep(0.1)
    for pid in _pids_listening_on(port):  # graceful stop timed out — force it
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.2)


class ServerState:
    """Holds the active IMU service and the optional recorder; toggled from the web UI."""

    def __init__(
        self,
        cfg: AppConfig,
        service: "ImuService | SerialImuService",
        log_dir: Path,
        record_hz: int,
        axis: AxisState | None = None,
    ) -> None:
        self.cfg = cfg
        self.service = service
        self.source_kind = cfg.source_kind
        # The axis remap describes the physical mounting, so it is shared by every source
        # rather than cached per-source; ``_make_source`` hands the same object to each.
        self.axis = axis if axis is not None else AxisState(
            Path(cfg.log_dir) / "axis_remap.json", cfg.axis_ops
        )
        # Built sources are kept so switching back is instant and the per-source tare offset
        # survives a round trip.
        self._sources: dict[str, "ImuService | SerialImuService"] = {cfg.source_kind: service}
        self.log_dir = log_dir
        self.record_hz = record_hz
        self.recorder: Recorder | None = None
        self.cls: "ClsService | None" = None
        self._lock = threading.Lock()

    def toggle_zero(self) -> bool:
        return self.service.zero_toggle()

    def emit_cls_result(self, index: int) -> None:
        """Sink for aggregated CLS decisions: hand the class index to the active source.

        Duck-typed on purpose — only ``SerialImuService`` has ``send_result``, so this is
        silently a no-op on I2C without ``cls/`` knowing that serial sources exist."""
        send = getattr(self.service, "send_result", None)
        if send is not None:
            send(index)

    def _stop_recorder_locked(self) -> bool:
        """Stop an active recording. Caller must hold ``_lock`` (which is not reentrant)."""
        if self.recorder is None:
            return False
        try:
            self.recorder.__exit__(None, None, None)
        finally:
            self.recorder = None
        return True

    def toggle_record(self) -> bool:
        with self._lock:
            if self.recorder is None:
                self.recorder = Recorder(
                    self.service, self.log_dir, self.record_hz, cls=self.cls
                ).__enter__()
                return True
            self._stop_recorder_locked()
            return False

    def switch_source(self, kind: str) -> dict:
        """Swap the live sample source. Returns ``{"ok", "kind", "message"}``.

        The new source is connected and sampling *before* ``self.service`` is repointed and the
        old one is torn down, so a concurrent ``/data`` poll never sees a dead source. An active
        recording is stopped first: ``Recorder`` caches the label list and has already written
        its CSV headers, and the label set changes between the two-sensor and one-sensor sources.
        """
        kind = str(kind).lower()
        with self._lock:
            if kind == self.source_kind:
                return {"ok": True, "kind": kind, "message": f"already using {kind}"}
            new = self._sources.get(kind)
            if new is None:
                new, err = _make_source(self.cfg, kind, self.axis)
                if err is not None:
                    return {"ok": False, "kind": self.source_kind, "message": err}
                self._sources[kind] = new

            notes: list[str] = []
            if self._stop_recorder_locked():
                notes.append("recording stopped")

            hz = self.cfg.sample_hz_for(kind)
            try:
                info = new.connect()
            except Exception as err:  # pragma: no cover - hardware dependent
                info = []
                logger.warning(f"switch to {kind}: connect failed ({err})")
            # Start sampling even when nothing was detected: switch anyway, matching startup's
            # "serve nulls rather than refuse" behaviour. This is what makes recovery work —
            # the serial reader thread retries the open on its own, so plugging the device in
            # afterwards picks it up. (I2C's start_sampling is a no-op with no sensors.)
            new.start_sampling(hz)
            if not info:
                notes.append("no sensor detected yet — will keep retrying")

            old, self.service, self.source_kind = self.service, new, kind
            if self.cls is not None:
                # Re-points CLS without reloading the checkpoint; the loop thread clears the
                # inference window and the vote buffer on its next tick.
                err = self.cls.set_source(new, sample_hz=hz)
                if err is not None:
                    notes.append(f"CLS paused: {err}")
            try:
                old.stop_sampling()
                old.disconnect()
            except Exception as err:  # pragma: no cover - hardware dependent
                logger.warning(f"switch to {kind}: releasing previous source failed ({err})")

            msg = f"switched to {kind}"
            if notes:
                msg += " (" + "; ".join(notes) + ")"
            logger.info(msg)
            return {"ok": True, "kind": kind, "message": msg}

    def set_record_hz(self, hz) -> int:
        """Set the recording rate (1–200 Hz); restart an active recorder to apply it."""
        try:
            hz = max(1, min(100, int(hz)))
        except (TypeError, ValueError):
            return self.record_hz
        with self._lock:
            self.record_hz = hz
            if self.recorder is not None:
                try:
                    self.recorder.__exit__(None, None, None)
                except Exception:
                    pass
                self.recorder = Recorder(
                    self.service, self.log_dir, self.record_hz, cls=self.cls
                ).__enter__()
        return self.record_hz

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def shutdown(self) -> None:
        if self.cls is not None:
            try:
                self.cls.stop()
            except Exception:
                pass
            self.cls = None
        if self.recorder is not None:
            try:
                self.recorder.__exit__(None, None, None)
            except Exception:
                pass
            self.recorder = None
        try:
            self.service.disconnect()
        except Exception:
            pass


def _payload(state: ServerState, since: float | None = None) -> dict:
    out: dict = {
        "t": time.monotonic(),
        "recording": state.recording,
        "zeroed": state.service.is_zeroed,
        "hz": state.record_hz,
        "source": state.source_kind,
        "source_connected": state.service.is_connected(),
        # Open port != data arriving. Only the serial source can tell the two apart; the I2C
        # sampler threads run whenever sensors are connected, so being connected is enough there.
        "source_streaming": bool(
            getattr(state.service, "receiving", state.service.is_connected())
        ),
        # Return channel health: None = nothing sent yet, False = the device is not accepting
        # results (usually because its sketch never reads its serial input).
        "source_tx": getattr(state.service, "tx_ok", None),
        # The wire rate actually measured, so the page can show it against the configured
        # sample_hz. A mismatch silently rescales every CLS window, and is invisible otherwise.
        "observed_hz": getattr(state.service, "observed_hz", None),
        "sources": _source_available(),
        # Which device-global telemetry groups this source carries. Empty on I2C. The page
        # builds its telemetry charts from this, so a group the link lacks gets no chart
        # rather than a flat line at zero.
        "telemetry_groups": list(_available_telemetry(state.service)),
        "euler": {},
        "accel": {},
        "gyro": {},
        "quat": {},
        "telemetry": {},
    }
    for label, sig in state.service.signals().items():
        for key in ("euler", "accel", "gyro", "quat"):
            out[key][label] = sig[key] if sig is not None else None
    tele = getattr(state.service, "telemetry", None)
    if tele is not None:
        out["telemetry"] = tele()
    if since is not None:
        # Batch of buffered samples newer than the client's cursor (memory read, no I2C).
        # Telemetry rides inside these rows -- see SerialImuService.samples_since for why it
        # must not be a second cursor.
        out["samples"] = state.service.samples_since(since)
    return out


def _available_telemetry(service) -> tuple[str, ...]:
    """Telemetry groups a source carries, () for one that has none.

    Input:  ``service`` = the active sensor source.
    Output: tuple[str, ...] of ``imu_common.TELEMETRY_GROUPS`` names.

    Duck-typed like ``send_result``: only ``SerialImuService`` defines the method, so the I2C
    source needs no knowledge that device telemetry exists.
    """
    fn = getattr(service, "available_telemetry", None)
    return tuple(fn()) if fn is not None else ()


def create_app(state: ServerState, window_s: float, poll_ms: int) -> Flask:
    app = Flask(__name__, static_folder="static")
    html = (
        _HTML.replace("__WINDOW_S__", str(float(window_s)))
        .replace("__POLL_MS__", str(int(poll_ms)))
        # Index -> name for the decision markers; injected so it can never drift from CLASSES.
        .replace("__CLASSES__", json.dumps(list(CLASSES)))
        # Channel tables, injected rather than written out in the page: imu_common owns the
        # names, units and axis order, and the payload, the CSV headers and these charts all
        # read that one table. A channel added there appears here with no JS edit.
        .replace("__SIGNALS__", json.dumps(
            {k: {"axes": list(SIGNAL_AXES[k]), "unit": SIGNAL_UNITS.get(k, "")}
             for k in SIGNAL_AXES}
        ))
        # ``charts`` is how a group is cut into plots (imu_common._chart_split); ``clip`` is
        # the limit its values were clamped to, shown in the header so a flat line at the limit
        # is not mistaken for the signal genuinely sitting there.
        .replace("__TELEMETRY__", json.dumps(
            [{"key": g, "channels": list(ch), "unit": u,
              "clip": state.cfg.telemetry_clip.get(g),
              "charts": [{"title": t, "channels": list(cs)} for t, cs in TELEMETRY_CHARTS[g]]}
             for g, ch, u in TELEMETRY_GROUPS]
        ))
    )

    @app.route("/")
    def index() -> Response:
        return Response(html, mimetype="text/html")

    @app.route("/data")
    def data() -> Response:
        try:
            since = float(request.args["since"])
        except (KeyError, TypeError, ValueError):
            since = None
        return jsonify(_payload(state, since))

    @app.route("/record", methods=["POST"])
    def record() -> Response:
        return jsonify({"recording": state.toggle_record()})

    @app.route("/zero", methods=["POST"])
    def zero() -> Response:
        return jsonify({"zeroed": state.toggle_zero()})

    @app.route("/freq", methods=["POST"])
    def freq() -> Response:
        hz = request.args.get("hz") or (request.get_json(silent=True) or {}).get("hz")
        return jsonify({"hz": state.set_record_hz(hz)})

    @app.route("/source", methods=["POST"])
    def source() -> Response:
        kind = request.args.get("kind") or (request.get_json(silent=True) or {}).get("kind")
        if not kind:
            return jsonify({"ok": False, "kind": state.source_kind, "message": "missing 'kind'"})
        return jsonify(state.switch_source(kind))

    @app.route("/calibration", methods=["GET"])
    def calibration() -> Response:
        return jsonify(state.service.calibration_status())

    @app.route("/cls", methods=["GET"])
    def cls() -> Response:
        if state.cls is None:
            return jsonify({"enabled": False, "reason": "not configured", "current": None, "entries": []})
        try:
            since = int(request.args.get("since", 0))
        except (TypeError, ValueError):
            since = 0
        return jsonify(state.cls.snapshot(since))

    @app.route("/cls/toggle", methods=["POST"])
    def cls_toggle() -> Response:
        if state.cls is None or not state.cls.enabled:
            return jsonify({"running": False, "reason": "not configured"})
        return jsonify({"running": state.cls.toggle_running()})

    @app.route("/recordings")
    def recordings() -> Response:
        """Every recorded session under the log directory, newest first — the Load picker."""
        try:
            limit = int(request.args["limit"])
        except (KeyError, TypeError, ValueError):
            limit = None
        return jsonify({"sessions": list_sessions(state.cfg.log_dir, limit=limit)})

    @app.route("/recordings/<path:session_id>")
    def recording(session_id: str):
        """One recorded session as uPlot-ready columns.

        Query: ``from``/``to`` in seconds from the session start (both optional), and
        ``max_points`` per channel. Zooming re-requests a narrower window, which is how full
        resolution stays reachable without ever shipping the whole file.
        """
        def _f(name):
            try:
                return float(request.args[name])
            except (KeyError, TypeError, ValueError):
                return None

        try:
            max_points = int(request.args.get("max_points", DEFAULT_MAX_POINTS))
        except (TypeError, ValueError):
            max_points = DEFAULT_MAX_POINTS
        try:
            return jsonify(load_session(
                state.cfg.log_dir, session_id,
                t_from=_f("from"), t_to=_f("to"),
                max_points=max(2, min(max_points, 20000)),
            ))
        except FileNotFoundError as err:
            return jsonify({"error": str(err)}), 404
        except Exception as err:  # a truncated or hand-edited CSV must not 500 the server
            logger.warning(f"cannot load session {session_id}: {err}")
            return jsonify({"error": f"cannot read session: {err}"}), 400

    @app.route("/axis-remap", methods=["GET"])
    def axis_remap_get() -> Response:
        return jsonify(state.service.get_axis_remap())

    @app.route("/axis-remap", methods=["POST"])
    def axis_remap_post():
        """Apply a host-side axis remap. Accepts, in precedence order:

        * ``ops``   — list of op names (``rot_x_90``, ``flip_z``, ...) applied in list order
        * ``placement`` — ``P0``..``P7``, the datasheet mounting presets
        * ``config`` + ``sign`` — the two bytes, kept so existing scripts keep working

        On success the tare is dropped by the service and the CLS window is cleared here: the
        buffered frames carry the old coordinate frame, and their timestamps are continuous so
        the gap check would never notice."""
        body = request.get_json(silent=True) or {}
        ops = body.get("ops")
        placement = request.args.get("placement") or body.get("placement")
        kwargs: dict
        if ops is not None:
            if not isinstance(ops, list):
                return jsonify({"ok": False, "message": "'ops' must be a list of op names"}), 400
            kwargs = {"ops": ops}
        elif placement and str(placement).upper() in PLACEMENTS:
            cfg_b, sgn_b = PLACEMENTS[str(placement).upper()]
            kwargs = {"config": cfg_b, "sign": sgn_b}
        else:
            raw_cfg = request.args.get("config", body.get("config"))
            raw_sgn = request.args.get("sign", body.get("sign"))
            try:
                cfg_b = int(raw_cfg, 0) if isinstance(raw_cfg, str) else int(raw_cfg)
                sgn_b = int(raw_sgn, 0) if isinstance(raw_sgn, str) else int(raw_sgn)
            except (TypeError, ValueError):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "valid": False,
                            "message": "provide 'ops', 'placement' (P0-P7), or numeric 'config' and 'sign'",
                        }
                    ),
                    400,
                )
            kwargs = {"config": cfg_b, "sign": sgn_b}
        result = state.service.set_axis_remap(**kwargs)
        if result.get("ok") and state.cls is not None:
            state.cls.reset_window()
        return jsonify(result)

    return app


def _source_available() -> dict[str, bool]:
    """Which source kinds this install can actually build — the UI greys out the rest."""
    return {"i2c": ImuService is not None, "serial": SerialImuService is not None}


def _make_source(
    cfg: AppConfig, kind: str, axis: AxisState
) -> tuple["ImuService | SerialImuService | None", str | None]:
    """Build one sample source. Returns ``(service, None)`` or ``(None, error_message)``.

    Both kinds expose the same method surface, so nothing downstream (payload, recorder, CLS)
    cares which one it gets. A missing dependency is returned rather than raised: at startup the
    caller turns it into a ``SystemExit`` (a silent fallback to the other source would be worse
    than not starting), but a runtime switch just reports it and stays where it is.

    ``axis`` is the **one shared** ``AxisState`` — both sources get the same object, not a copy.
    The remap describes how the sensor is physically mounted, so it cannot be per-source: built
    sources are cached across switches, and independent copies would let a mapping applied on
    one source silently not apply to the other."""
    if kind == "serial":
        if SerialImuService is None:
            return None, (
                f'[source] kind = "serial" needs pyserial: {_SERIAL_IMPORT_ERR}\n'
                f'  install it with:  pip install -e ".[serial]"'
            )
        return SerialImuService(
            cfg.serial_port,
            cfg.serial_baud,
            label=cfg.serial_label,
            magic=cfg.serial_magic,
            layout=cfg.serial_layout,
            gyro_units=cfg.serial_gyro_units,
            axis=axis,
            clip=cfg.telemetry_clip,
        ), None
    if kind != "i2c":
        return None, f'Unknown source kind "{kind}" — expected "i2c" or "serial"'
    if ImuService is None:
        return None, f'[source] kind = "i2c" needs the Adafruit I2C stack: {_I2C_IMPORT_ERR}'
    return ImuService(cfg.bus_labels, axis=axis), None


def run_server(cfg: AppConfig, host: str | None = None, port: int | None = None) -> None:
    host = host or cfg.web_host
    port = int(port or cfg.web_port)

    # Quiet werkzeug; keep loguru at INFO on stderr for rate/overrun telemetry
    # (sampler threads, recorder) — stdout stays reserved for the startup banner.
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Reclaim the port before touching hardware: a stale server instance would still hold
    # this port *and* the I2C buses + CUDA, so killing it first lets us grab everything cleanly.
    _free_port(host, port)

    # One axis remap for the whole process: <log_dir>/axis_remap.json if it exists, else the
    # [axis] ops default from the TOML. Every source built below shares this object.
    axis = AxisState(Path(cfg.log_dir) / "axis_remap.json", cfg.axis_ops)
    service, err = _make_source(cfg, cfg.source_kind, axis)
    if err is not None:
        # The *boot* source is a hard requirement; a runtime switch reports instead.
        raise SystemExit(err)
    if cfg.source_kind == "serial":
        print(f"Opening serial IMU on {cfg.serial_port} @ {cfg.serial_baud}...")
    else:
        print("Connecting to IMUs...")
    try:
        info = service.connect()
    except Exception as err:  # pragma: no cover - hardware dependent
        print(f"Connect failed: {err}")
        info = []
    if info:
        print("Connected: " + ", ".join(f"{i.label}={i.sensor_name}" for i in info))
    else:
        print("No IMUs detected — serving anyway (values will be null).")
    # Started either way: the serial reader retries the open on its own, so a device plugged in
    # after startup is picked up. With no I2C sensors this is a no-op.
    service.start_sampling(cfg.sample_hz_for(cfg.source_kind))

    state = ServerState(cfg, service, cfg.log_dir, cfg.record_hz, axis=axis)
    if cfg.cls_enabled and cfg.cls_model_path:
        # Aggregation is always present; disabling it is a window of 1, i.e. one decision per
        # inference, which keeps ClsService to a single code path.
        voter = (
            SoftVoter(
                window=cfg.vote_window,
                emit_every=cfg.vote_emit_every,
                hysteresis=cfg.vote_hysteresis,
            )
            if cfg.vote_enabled
            else SoftVoter(window=1, emit_every=1)
        )
        state.cls = ClsService(
            service,
            cfg.cls_model_path,
            sensor=cfg.cls_sensor,
            sample_hz=cfg.sample_hz_for(cfg.source_kind),
            target_hz=cfg.cls_target_hz,
            window=cfg.cls_window,
            stride=cfg.cls_stride,
            aggregator=voter,
            on_result=state.emit_cls_result,
        )
        state.cls.start()
    poll_ms = max(20, int(1000 / max(1, cfg.plot_fps)))
    app = create_app(state, cfg.plot_window_seconds, poll_ms)

    print(f"\nServing on {host}:{port}   (Ctrl-C to stop)")
    if host in ("0.0.0.0", "::"):
        ip6 = get_local_ip6() if host == "::" else None
        ip4 = get_local_ip()
        if ip6:
            print(f"  IPv6:   http://[{ip6}]:{port}")
        if ip4:
            print(f"  IPv4:   http://{ip4}:{port}")
    else:
        print(f"  URL:    http://{host}:{port}")
    if host in ("127.0.0.1", "::1", "localhost"):
        # The shipped default. Nothing is reachable off the machine, which is the point
        # on a field Jetson with no network; the tunnel below is how a laptop still gets
        # a browser onto it, and --lan is the deliberate opt-out.
        print("  loopback only — use the tunnel below, or --lan to serve the network")
    print(f"  tunnel: ssh -L {port}:localhost:{port} <user>@<jetson>   then open http://localhost:{port}\n")

    from werkzeug.serving import make_server

    srv = make_server(host, port, app, threaded=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown()
        print("\nStopped.")


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Jetson IMU Live</title>
<link rel="stylesheet" href="/static/uPlot.min.css">
<script src="/static/uPlot.iife.min.js"></script>
<script src="/static/three.min.js"></script>
<style>
  :root{--bg:#0e1014;--panel:#161922;--panel2:#1d212c;--border:#2a2f3a;--fg:#e5e7eb;--muted:#9aa4b2;--accent:#3b82f6}
  :root.light{--bg:#f5f7fa;--panel:#ffffff;--panel2:#eef1f6;--border:#d6dce6;--fg:#1b1f27;--muted:#5b6472;--accent:#2563eb}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--fg)}
  #app{display:flex;flex-direction:column;height:100%}
  #bar{display:flex;gap:9px;align-items:center;padding:9px 14px;background:var(--panel);border-bottom:1px solid var(--border);flex-wrap:wrap}
  .seg{display:inline-flex;background:var(--panel2);border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .seg button{border:0;background:transparent;color:var(--muted);padding:7px 14px;font-size:13px;cursor:pointer}
  .seg button:hover{background:rgba(127,127,127,.15);color:var(--fg)}
  .seg button.active{background:var(--accent);color:#fff}
  .btn{border:1px solid var(--border);background:var(--panel2);color:var(--fg);padding:7px 13px;border-radius:8px;font-size:13px;cursor:pointer}
  .btn:hover{filter:brightness(1.08)}
  .btn.rec-on{background:#ef4444;border-color:#ef4444;color:#fff}
  .btn.pause-on{background:#f59e0b;border-color:#f59e0b;color:#111}
  .btn.src-serial{background:#0ea5e9;border-color:#0ea5e9;color:#fff}
  .btn.src-warn{background:#f59e0b;border-color:#f59e0b;color:#111}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .srcmsg{font-size:12px;color:var(--muted);max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .reclabel{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
  .num{width:64px;background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px}
  #yman{align-items:center;gap:5px}
  .grow{flex:1}
  #status{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;white-space:nowrap}
  #dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:7px;vertical-align:middle}
  /* Charts never shrink below MIN_CHART_PX: a group split into 5 or 6 plots would otherwise
     squeeze each one flat. Few charts still grow to fill; too many scroll instead. */
  #charts{flex:1;min-height:0;display:flex;flex-direction:column;gap:8px;padding:8px;overflow-y:auto}
  .chart{flex:1 0 160px;min-height:160px;display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:6px 10px}
  .chead{display:flex;align-items:center;gap:16px;padding:1px 2px 5px;font-size:12px;color:var(--muted)}
  .ctitle{font-weight:700;color:var(--fg);text-transform:uppercase;letter-spacing:.05em}
  /* Channel names keep their own case so they match the Simulink signal they came from —
     finalClass, not FINALCLASS. */
  .cname{text-transform:none;letter-spacing:0}
  .cval{display:inline-flex;align-items:center;gap:6px;font-variant-numeric:tabular-nums}
  .cval i{width:10px;height:10px;border-radius:3px;display:inline-block}
  .cval b{color:var(--fg);min-width:60px;display:inline-block}
  .canvas{flex:1;min-height:0}
  #readout{display:none;flex:1;min-height:0;overflow:auto;padding:14px;gap:14px;
           grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
  .rcard{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 18px}
  .rtitle{font-size:15px;font-weight:700;margin-bottom:6px}
  .rgroup{display:flex;align-items:center;gap:12px;padding:9px 0;border-top:1px solid var(--border)}
  .rgname{width:58px;color:var(--muted);font-size:11px;text-transform:uppercase;line-height:1.2}
  .runit{display:block;font-size:10px;color:var(--muted);opacity:.8}
  .rvals{display:flex;gap:20px;flex-wrap:wrap;font-variant-numeric:tabular-nums}
  .rax{display:inline-flex;gap:7px;align-items:baseline}
  .rax i{font-style:normal;font-weight:700;width:11px}
  .rax b{font-size:19px;color:var(--fg);min-width:90px;text-align:right;display:inline-block}
  /* ---- axis-remap modal ---- */
  .overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:50}
  .overlay.open{display:flex}
  .loadrow{padding:8px 10px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px;cursor:pointer}
  .loadrow:hover{border-color:var(--accent,#60a5fa);background:var(--hover,rgba(127,127,127,.08))}
  .modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;width:min(820px,94vw);max-height:92vh;overflow:auto;box-shadow:0 18px 50px rgba(0,0,0,.45)}
  .mhead{display:flex;align-items:center;gap:12px;padding:13px 16px;border-bottom:1px solid var(--border)}
  .mtitle{font-weight:700;font-size:15px}
  .mhead .grow{flex:1}
  .mbody{display:flex;gap:18px;padding:16px;flex-wrap:wrap}
  .mcol{flex:1;min-width:300px;display:flex;flex-direction:column;gap:10px}
  .mlabel{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);display:flex;align-items:center;gap:10px}
  .presets{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}
  .presets button{border:1px solid var(--border);background:var(--panel2);color:var(--fg);padding:8px 0;border-radius:8px;font-size:13px;cursor:pointer}
  .presets button:hover{filter:brightness(1.1)}
  .presets button.active{background:var(--accent);border-color:var(--accent);color:#fff}
  .opsrow{display:flex;align-items:center;gap:7px}
  .opsrow select{flex:1;background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:7px}
  .chain{display:flex;flex-wrap:wrap;gap:6px;align-items:center;min-height:28px}
  .chip{display:inline-flex;align-items:center;gap:7px;background:var(--panel2);border:1px solid var(--border);border-radius:999px;padding:4px 10px;font-size:12px}
  .chip.mir{border-color:#f59e0b}
  .chip button{background:none;border:0;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:0}
  .chip button:hover{color:#f87171}
  .mapout{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--fg);line-height:1.7}
  .warn{color:#f87171;font-size:12px;font-weight:600}
  .mfoot{display:flex;align-items:center;gap:10px;padding-top:4px}
  .mfoot .grow{flex:1}
  #axisApply[disabled]{opacity:.45;cursor:not-allowed}
  .muted{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
  #cubeWrap{height:240px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;overflow:hidden}
  #cube{display:block;width:100%;height:100%}
  .calrow{display:flex;align-items:center;gap:10px;padding:6px 0}
  .calname{width:52px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .calbars{display:flex;gap:5px}
  .calseg{width:30px;height:12px;border-radius:3px;background:var(--panel2);border:1px solid var(--border)}
  .calseg.on{background:#22c55e;border-color:#22c55e}
  .calready{font-size:12px;font-weight:700;margin-left:8px}
  /* ---- CLS page ---- */
  #clsview{display:none;flex:1;min-height:0;flex-direction:column;padding:12px;gap:10px}
  .clstools{display:flex;justify-content:flex-end}
  .clsbanner{display:flex;align-items:center;justify-content:center;gap:18px;min-height:96px;
             background:var(--panel);border:1px solid var(--border);border-radius:12px}
  .clsbanner.on{border-color:var(--accent)}
  .clscls{font-size:40px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}
  .clsconf{font-size:22px;font-weight:700;color:var(--muted);font-variant-numeric:tabular-nums}
  .clshead{display:flex;gap:12px;padding:0 12px;font-size:11px;text-transform:uppercase;
           letter-spacing:.05em;color:var(--muted)}
  .clshcol{width:78px}.clshcol.grow{flex:1;width:auto}
  #clsLog{flex:1;min-height:0;overflow:auto;background:var(--panel);border:1px solid var(--border);
          border-radius:12px;padding:4px 0}
  .clsrow{display:flex;align-items:center;gap:12px;padding:7px 12px;border-top:1px solid var(--border);
          font-variant-numeric:tabular-nums}
  .clsrow:first-child{border-top:0}
  .clstime{width:78px;color:var(--muted);font-size:12px}
  .clsname{flex:1;font-weight:700;text-transform:capitalize}
  .clsbar{width:120px;height:8px;background:var(--panel2);border-radius:4px;overflow:hidden}
  .clsbar i{display:block;height:100%}
  .clspct{width:44px;text-align:right;color:var(--muted);font-size:12px}
  .clsvote{font-size:12px;color:var(--muted);text-transform:none;letter-spacing:0}
  .clsdeccol{width:150px}
  .clsdec{font-size:12px;font-weight:700;text-transform:capitalize}
</style>
</head>
<body>
  <div id="app">
    <div id="bar">
      <div class="seg" id="sigseg"></div>
      <button id="viewBtn" class="btn" onclick="toggleView()">Numbers</button>
      <button id="pauseBtn" class="btn" onclick="togglePause()">Pause</button>
      <button id="loadBtn" class="btn" onclick="openLoad()"
              title="Load a recorded session from the log folder and review it offline">Load</button>
      <button id="axisBtn" class="btn" onclick="openAxis()">Axis</button>
      <button id="calibBtn" class="btn" onclick="openCalib()">Calib</button>
      <button id="yBtn" class="btn" onclick="toggleYMode()">Y: Auto</button>
      <span id="yman" style="display:none">
        <input id="ymin" class="num" type="number" step="any" title="Y min">
        <span style="color:var(--muted)">–</span>
        <input id="ymax" class="num" type="number" step="any" title="Y max">
      </span>
      <span class="grow"></span>
      <span id="srcMsg" class="srcmsg"></span>
      <button id="srcBtn" class="btn" onclick="toggleSource()"
              title="Switch between the onboard I2C IMUs and the serial (Arduino) IMU">IMU: —</button>
      <button id="themeBtn" class="btn" onclick="toggleTheme()">Light</button>
      <button id="zeroBtn" class="btn" onclick="toggleZero()" title="Zero out current Euler/Accel/Gyro readings (tare)">Zero</button>
      <button id="recBtn" class="btn" onclick="toggleRecord()">Record</button>
      <label class="reclabel" title="Recording rate — only affects logging to disk, not the plot">
        Rec Hz <input id="freq" class="num" type="number" min="1" max="100" step="1"></label>
      <span id="status"><span id="dot"></span>connecting…</span>
    </div>
    <div id="charts"></div>
    <div id="readout"></div>
    <div id="clsview">
      <div class="clstools">
        <button id="clsRunBtn" class="btn" onclick="toggleCls()" style="display:none"
                title="Stop/start online inference (pausing frees CPU for sampling)">Stop</button>
      </div>
      <div id="clsBanner" class="clsbanner"><span class="muted">connecting…</span></div>
      <div class="clshead">
        <span class="clshcol">time</span><span class="clshcol grow">activity (frame)</span>
        <span class="clshcol">conf</span><span class="clshcol" style="width:150px">decision</span>
      </div>
      <div id="clsLog"></div>
    </div>
  </div>

  <div id="axisOverlay" class="overlay" onclick="if(event.target===this)closeAxis()">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Axis remap">
      <div class="mhead">
        <span class="mtitle">Axis Remap &nbsp;<span class="muted">software · shared by all sensors</span></span>
        <span class="grow"></span>
        <button class="btn" onclick="closeAxis()">Close</button>
      </div>
      <div class="mbody">
        <div class="mcol">
          <div class="mlabel">Add a step</div>
          <div class="opsrow">
            <select id="opAxis"><option value="x">X</option><option value="y">Y</option><option value="z">Z</option></select>
            <select id="opDeg"><option value="90">90°</option><option value="180">180°</option><option value="270">270°</option></select>
            <button class="btn" onclick="addRot()">Rotate</button>
            <button class="btn" onclick="addFlip()">Negate</button>
          </div>
          <div class="mlabel">Steps <span class="muted">applied top to bottom</span><span style="flex:1"></span>
            <button class="btn" onclick="undoOp()">Undo</button>
            <button class="btn" onclick="clearOps()">Clear</button>
          </div>
          <div class="chain" id="opChain"></div>
          <div class="mlabel">Result <span class="muted">output ← source · sign</span></div>
          <div class="mapout" id="axisMap"></div>
          <div id="axisWarn" class="warn" style="display:none"></div>
          <div class="mfoot">
            <span id="axisBytes" class="muted">CONFIG 0x24 · SIGN 0x00</span>
            <span class="grow"></span>
            <button id="axisApply" class="btn" onclick="applyAxis()">Apply</button>
          </div>
          <div id="axisMsg" class="muted"></div>
          <div class="mlabel">Datasheet presets <span class="muted">P0–P7 · replaces the steps</span></div>
          <div class="presets" id="presets"></div>
        </div>
        <div class="mcol">
          <div class="mlabel">Live orientation
            <select id="cubeSensor" class="num" style="width:auto" onchange="cubeLabel=this.value"></select>
          </div>
          <div id="cubeWrap"><canvas id="cube"></canvas></div>
          <div class="muted" style="font-size:11px">Rotate the physical sensor — the cube follows the (remapped) reported orientation.</div>
        </div>
      </div>
    </div>
  </div>

  <div id="loadOverlay" class="overlay" onclick="if(event.target===this)closeLoad()">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Load a recording">
      <div class="mhead">
        <span class="mtitle">Recordings &nbsp;<span class="muted">from the log folder</span></span>
        <span class="grow"></span>
        <button class="btn" onclick="closeLoad()">Close</button>
      </div>
      <div class="mbody">
        <div class="mcol" id="loadBody" style="min-width:520px">
          <div class="muted">Loading…</div>
        </div>
      </div>
    </div>
  </div>

  <div id="calibOverlay" class="overlay" onclick="if(event.target===this)closeCalib()">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Calibration status">
      <div class="mhead">
        <span class="mtitle">Calibration &nbsp;<span class="muted">BNO055 onboard · IMUPLUS (no magnetometer)</span></span>
        <span class="grow"></span>
        <button class="btn" onclick="closeCalib()">Close</button>
      </div>
      <div class="mbody">
        <div class="mcol" id="calibBody" style="min-width:340px"></div>
        <div class="mcol" style="min-width:240px">
          <div class="mlabel">How to calibrate</div>
          <div class="muted" style="font-size:12px;line-height:1.6">
            • <b>Gyro</b> — set the sensor down and keep it still for a few seconds.<br>
            • <b>Accel</b> — slowly tilt it through a few stable positions (≈45°/90°).<br>
            • <b>Mag</b> — unused in IMUPLUS mode; it stays at 0 by design.<br>
            • Each level goes 0→3; <b>ready</b> = gyro and accel both at 3.
          </div>
        </div>
      </div>
    </div>
  </div>
<script>
const WINDOW_S = __WINDOW_S__;
const POLL_MS  = __POLL_MS__;
// Channel tables, injected from imu_common so the page owns no channel names of its own.
// SIGNALS: per-sensor, one chart per axis and one line per sensor label.
// TELEMETRY: device-global, one chart per group and one line per channel — no label layer.
const SIGNALS = __SIGNALS__;
const TELEMETRY = __TELEMETRY__;
const TELE_BY_KEY = Object.fromEntries(TELEMETRY.map(g => [g.key, g]));
const UNITS = Object.fromEntries(Object.entries(SIGNALS).map(([k,v]) => [k, v.unit]));
const isTele = k => !!TELE_BY_KEY[k];
const unitOf = k => isTele(k) ? TELE_BY_KEY[k].unit : (SIGNALS[k] ? SIGNALS[k].unit : '');
const THEMES = {
  dark:  { axis:'#8b93a7', grid:'#222a38', series:['#e879f9','#22d3ee'], ax:{x:'#f87171',y:'#4ade80',z:'#60a5fa',w:'#fbbf24'},
           multi:['#e879f9','#22d3ee','#4ade80','#fbbf24','#f87171','#a78bfa'] },
  light: { axis:'#5b6472', grid:'#e2e6ee', series:['#c026d3','#0891b2'], ax:{x:'#dc2626',y:'#16a34a',z:'#2563eb',w:'#d97706'},
           multi:['#c026d3','#0891b2','#16a34a','#d97706','#dc2626','#7c3aed'] },
};
const theme = () => document.documentElement.classList.contains('light') ? THEMES.light : THEMES.dark;

let labels = ['Left','Right'];
let signal = 'euler';
let view = 'plot';
let paused = false;
let samples = [];
let charts = [], heads = [], ro = null, specs = [];
let latestT = 0;
let yBySignal = {};               // signal -> {auto:true} | {auto:false, min, max}
let teleAvail = [];               // telemetry groups the live source carries
// Offline viewing: null when live, otherwise the loaded session from /recordings/<id>.
// While set, tick() does not fetch and the charts read their columns from here instead.
let offline = null;
let offlineBusy = false, zoomTimer = null;
let clsMode = false;              // CLS page active (a pseudo-signal, not a plot)
let clsSince = 0;                 // highest CLS entry id already shown
let clsTimer = null;
const CLS_NAMES = __CLASSES__;    // index -> label, injected from cls/model/__init__.py CLASSES
const CLS_COLORS = { stand:'#9aa4b2', walk:'#22c55e', turn:'#eab308', jog:'#ef4444',
                     rampascent:'#3b82f6', stairascent:'#a855f7', stairdescent:'#ec4899',
                     sit:'#14b8a6', 'sit-to-stand':'#f97316', 'stand-to-sit':'#8b5cf6',
                     rampdescent:'#06b6d4' };
const clsColor = c => CLS_COLORS[c] || '#60a5fa';

const fmt = (sig, v) => v == null ? '--' : v.toFixed(sig === 'quat' ? 3 : 2);

// Decimals for an x tick, from the tick spacing uPlot settled on. Enough to tell adjacent
// ticks apart and no more: at 1 s spacing milliseconds are noise, but at 20 ms spacing
// dropping them makes every label on the axis read the same number.
function tDecimals(incr){
  if(!(incr > 0)) return 1;
  if(incr >= 1) return 0;
  if(incr >= 0.1) return 1;
  if(incr >= 0.01) return 2;
  return 3;
}

// One chart spec per plot the current selection needs. Per-sensor signals cut one chart per
// axis with a line per sensor; a telemetry group is cut by imu_common._chart_split — right/left
// where the channels pair up, one chart per channel where they do not. Both shapes come out as
// {title, key, axIdx, series:[{name, idx, color}]}, so everything downstream is shared.
//
// Colours run by position *within the chart*, not by wire position, so the right chart's pos_r
// and the left chart's pos_l share a colour and the two legs read as parallel.
function chartSpecs(){
  const T = theme();
  if(isTele(signal)){
    const g = TELE_BY_KEY[signal];
    return (g.charts || []).map(ch => ({
      title: signal + ' <span class="cname" style="color:' + T.multi[0] + '">' + ch.title + '</span>',
      key: ch.title,
      axIdx: 0,
      series: ch.channels.map((name,k) => ({
        name, idx: g.channels.indexOf(name), color: T.multi[k % T.multi.length],
      })),
    }));
  }
  const sg = SIGNALS[signal];
  if(!sg) return [];
  return sg.axes.map((ax, ai) => ({
    title: signal + ' <span class="cname" style="color:' + (T.ax[ax] || T.axis) + '">' + ax + '</span>',
    key: ax,
    axIdx: ai,
    series: labels.map((lab,k) => ({ name:lab, idx:k, color: T.series[k % T.series.length] })),
  }));
}

// Columns for one chart in uPlot order [xs, ...series]. The only place that knows where the
// numbers come from, so live polling and offline replay differ here and nowhere else.
function chartCols(spec){
  if(offline){
    if(isTele(signal)){
      const g = offline.telemetry[signal];
      if(!g) return [[], ...spec.series.map(()=>[])];
      return [g.t, ...spec.series.map(sr => g.channels[sr.name] || [])];
    }
    const sg = offline.signals[signal];
    if(!sg) return [[], ...spec.series.map(()=>[])];
    const per = sg.axes[spec.key] || {};
    return [sg.t, ...spec.series.map(sr => per[sr.name] || [])];
  }
  const ts = samples.map(s => s.t);
  if(isTele(signal)){
    // sr.idx is the channel's position in the group's wire order — NOT its position in this
    // chart. With one chart per channel the two differ, and using the chart-local index would
    // draw every plot from the group's first channel.
    return [ts, ...spec.series.map(sr =>
      samples.map(s => { const v = s.telemetry && s.telemetry[signal]; return v ? v[sr.idx] : null; }))];
  }
  return [ts, ...spec.series.map(sr =>
    samples.map(s => { const v = s[signal] && s[signal][sr.name]; return v ? v[spec.axIdx] : null; }))];
}

function chartOpts(w, h, spec){
  const T = theme();
  const series = [{}];
  spec.series.forEach(sr => series.push({ stroke: sr.color, width: 2, points:{show:false} }));
  const ym = yBySignal[signal];
  const yscale = (ym && !ym.auto && ym.max > ym.min) ? { range: [ym.min, ym.max] } : {};
  // Live: a rolling window pinned to the newest sample, ticks labelled as seconds ago.
  // Offline: the window the server returned, ticks labelled as seconds into the session.
  const xscale = offline ? { time:false }
                         : { time:false, range: () => [latestT - WINDOW_S, latestT] };
  const xvalues = offline ? (u,vs,ai,sp,incr) => vs.map(v => v.toFixed(tDecimals(incr)))
                          : (u,vs,ai,sp,incr) => vs.map(v => (v - latestT).toFixed(tDecimals(incr)));
  return {
    width: w, height: h,
    legend: { show:false },
    cursor: { drag:{ x:true, y:false }, points:{ show:false } },
    scales: { x: xscale, y: yscale },
    axes: [
      { stroke:T.axis, grid:{ stroke:T.grid }, ticks:{ stroke:T.grid }, values:xvalues },
      { stroke:T.axis, grid:{ stroke:T.grid }, ticks:{ stroke:T.grid }, size:54 },
    ],
    // Offline only: a drag-zoom asks for more detail than the decimated window holds, so it
    // re-fetches that range at full budget instead of magnifying the envelope.
    hooks: offline ? { setScale: [ (u, key) => { if(key === 'x') onOfflineZoom(u); } ] } : {},
    series,
  };
}

function rebuildCharts(){
  if(ro) ro.disconnect();
  charts.forEach(u => u.destroy());
  charts = []; heads = [];
  const wrap = document.getElementById('charts');
  wrap.innerHTML = '';
  if(typeof uPlot === 'undefined'){
    wrap.innerHTML = '<div class="muted" style="padding:14px">uPlot failed to load — Plots unavailable; the Numbers view still works.</div>';
    return;
  }
  const T = theme();
  ro = new ResizeObserver(entries => {
    for(const e of entries){
      const u = e.target.__u;
      if(u && e.contentRect.width > 0 && e.contentRect.height > 0)
        u.setSize({ width: e.contentRect.width, height: e.contentRect.height });
    }
  });
  const unit = unitOf(signal);
  // Clamped groups say so: a flat line at the limit reads as a real measurement otherwise.
  const clip = isTele(signal) ? TELE_BY_KEY[signal].clip : null;
  const badge = (unit ? ' <span class="runit">' + unit + '</span>' : '')
    + (clip != null ? ' <span class="runit">clip ±' + clip + '</span>' : '');
  specs = chartSpecs();
  specs.forEach((spec) => {
    const card = document.createElement('div'); card.className = 'chart';
    const head = document.createElement('div'); head.className = 'chead';
    head.innerHTML = '<span class="ctitle">' + spec.title + badge + '</span>'
      + spec.series.map((sr,k)=>'<span class="cval"><i style="background:' + sr.color + '"></i>'
          + sr.name + ' <b data-v="' + k + '">--</b></span>').join('');
    const body = document.createElement('div'); body.className = 'canvas';
    card.appendChild(head); card.appendChild(body); wrap.appendChild(card);
    const u = new uPlot(chartOpts(body.clientWidth || 300, body.clientHeight || 140, spec),
                        [[], ...spec.series.map(()=>[])], body);
    body.__u = u; charts.push(u); heads.push(head); ro.observe(body);
  });
  redraw();
}

function redraw(){
  charts.forEach((u, i) => {
    const spec = specs[i];
    if(!spec) return;
    const cols = chartCols(spec);
    u.setData(cols);
    // Header value: the newest sample live, the last point of the window offline.
    const n = cols[0].length;
    spec.series.forEach((sr,k) => {
      const el = heads[i].querySelector('b[data-v="' + k + '"]');
      if(!el) return;
      const col = cols[k+1];
      const v = (n && col && col.length >= n) ? col[n-1] : null;
      el.textContent = (v == null) ? '--' : fmt(signal, v);
    });
  });
}

// Numbers view: one card per sensor label for the per-sensor signals, then one card holding
// every telemetry group the source carries. Telemetry has no label layer, hence its own card
// rather than a column inside each sensor's.
function buildReadout(){
  const T = theme();
  const wrap = document.getElementById('readout');
  const sigKeys = Object.keys(SIGNALS);
  let html = labels.map(lab =>
    '<div class="rcard"><div class="rtitle">' + lab + '</div>'
    + sigKeys.map(sig =>
        '<div class="rgroup"><div class="rgname">' + sig
          + (UNITS[sig] ? '<span class="runit">' + UNITS[sig] + '</span>' : '') + '</div>'
        + '<div class="rvals">'
        + SIGNALS[sig].axes.map((ax,i) => '<span class="rax"><i style="color:' + (T.ax[ax] || T.axis) + '">' + ax
            + '</i><b data-k="' + lab + '|' + sig + '|' + i + '">--</b></span>').join('')
        + '</div></div>').join('')
    + '</div>').join('');
  const groups = TELEMETRY.filter(g => teleAvail.indexOf(g.key) >= 0);
  if(groups.length){
    html += '<div class="rcard"><div class="rtitle">Telemetry</div>'
      + groups.map(g =>
          '<div class="rgroup"><div class="rgname">' + g.key
            + (g.unit ? '<span class="runit">' + g.unit + '</span>' : '') + '</div>'
          + '<div class="rvals">'
          + g.channels.map((ch,i) => '<span class="rax"><i style="color:' + T.multi[i % T.multi.length] + '">'
              + ch + '</i><b data-k="tele|' + g.key + '|' + i + '">--</b></span>').join('')
          + '</div></div>').join('')
      + '</div>';
  }
  wrap.innerHTML = html;
}

function updateReadout(d){
  for(const lab of labels) for(const sig of Object.keys(SIGNALS)) SIGNALS[sig].axes.forEach((ax,i) => {
    const el = document.querySelector('b[data-k="' + lab + '|' + sig + '|' + i + '"]');
    if(!el) return;
    const v = d[sig] && d[sig][lab];
    el.textContent = v ? fmt(sig, v[i]) : '--';
  });
  const tele = d.telemetry || {};
  for(const g of TELEMETRY) g.channels.forEach((ch,i) => {
    const el = document.querySelector('b[data-k="tele|' + g.key + '|' + i + '"]');
    if(!el) return;
    const v = tele[g.key];
    el.textContent = (v && v[i] != null) ? fmt(g.key, v[i]) : '--';
  });
}

function setSignal(s){
  if(s === 'cls'){ enterCls(); return; }
  if(clsMode) exitCls();
  signal = s;
  document.querySelectorAll('.sigbtn').forEach(b => b.classList.toggle('active', b.dataset.sig === s));
  if(view === 'plot') rebuildCharts();
  syncYControls();
}
// ---- signal / telemetry buttons ------------------------------------------
// Built from the injected tables rather than written into the markup, and filtered by what the
// active source actually carries — so the page never offers a chart that could only be a flat
// line, and a channel added in imu_common appears here with no edit.
function buildSigButtons(){
  const sigKeys = Object.keys(SIGNALS);
  const tele = TELEMETRY.filter(g => teleAvail.indexOf(g.key) >= 0).map(g => g.key);
  const want = sigKeys.concat(tele).concat(offline ? [] : ['cls']);
  const seg = document.getElementById('sigseg');
  if(seg.dataset.keys === want.join(',')) return;   // unchanged: leave the DOM alone
  seg.dataset.keys = want.join(',');
  seg.innerHTML = want.map(k => {
    const on = clsMode ? (k === 'cls') : (k === signal);
    const cap = k.charAt(0).toUpperCase() + k.slice(1);
    return '<button class="sigbtn' + (on ? ' active' : '') + '" data-sig="' + k
         + '" onclick="setSignal(&quot;' + k + '&quot;)">' + cap + '</button>';
  }).join('');
}

// ---- offline viewing -----------------------------------------------------
// `offline` holds one loaded session; while it is set, tick() does not fetch and every chart
// reads its columns from it. The live cursor is reset on the way out so resuming cannot replay
// a stale window.
function offlineBanner(){
  if(!offline) return;
  const w = offline.window, sp = offline.span;
  document.getElementById('status').innerHTML =
    '<span id="dot"></span>offline · ' + offline.id
    + ' · ' + w[0].toFixed(1) + '–' + w[1].toFixed(1) + ' s of ' + sp[1].toFixed(1) + ' s'
    + ' <button class="btn" style="margin-left:8px" onclick="offlineFull()">Full</button>'
    + ' <button class="btn" onclick="exitOffline()">Live</button>';
}

async function openLoad(){
  document.getElementById('loadOverlay').classList.add('open');
  const body = document.getElementById('loadBody');
  body.innerHTML = '<div class="muted">Loading…</div>';
  let d;
  try { d = await (await fetch('/recordings')).json(); }
  catch(e){ body.innerHTML = '<div class="muted">Could not list recordings.</div>'; return; }
  const rows = d.sessions || [];
  if(!rows.length){
    body.innerHTML = '<div class="muted">No recordings yet — press Record to make one.</div>';
    return;
  }
  body.innerHTML = rows.map(r => {
    const dur = (r.duration_s == null) ? '—' : r.duration_s.toFixed(1) + ' s';
    const groups = (r.signals || []).concat(r.telemetry || []).concat(r.has_cls ? ['cls'] : []);
    return '<div class="loadrow" onclick="loadSession(&quot;' + r.id + '&quot;)">'
      + '<b>' + r.started + '</b>'
      + '<span class="muted"> · ' + r.n_rows + ' rows · ' + dur + '</span>'
      + '<div class="muted" style="font-size:11px">' + groups.join(' · ') + '</div>'
      + '</div>';
  }).join('');
}
function closeLoad(){ document.getElementById('loadOverlay').classList.remove('open'); }

async function loadSession(id, from, to){
  if(offlineBusy) return;
  offlineBusy = true;
  let q = '/recordings/' + id + '?max_points=4000';
  if(from != null && to != null) q += '&from=' + from + '&to=' + to;
  let d;
  try { d = await (await fetch(q)).json(); }
  catch(e){ offlineBusy = false; return; }
  if(d.error){ offlineBusy = false; alert(d.error); return; }
  offline = d;
  if(d.labels && d.labels.length) labels = d.labels;
  teleAvail = Object.keys(d.telemetry || {});
  closeLoad();
  if(clsMode) exitCls();
  // A recording need not hold whatever is selected — serial sessions have no euler/quat, an
  // I2C one has no telemetry. Fall back to something the file actually contains.
  const have = Object.keys(d.signals || {}).concat(teleAvail);
  if(have.length && have.indexOf(signal) < 0) signal = have[0];
  buildSigButtons();
  document.querySelectorAll('.sigbtn').forEach(b => b.classList.toggle('active', b.dataset.sig === signal));
  if(view === 'plot') rebuildCharts(); else { buildReadout(); fillReadout(); }
  offlineBanner();
  offlineBusy = false;
}

function offlineFull(){ if(offline && !offlineBusy) loadSession(offline.id); }

function exitOffline(){
  offline = null;
  samples = [];
  sinceT = 0;              // do not replay the pre-offline window against the new cursor
  teleAvail = [];          // the next poll reports what the live source carries
  buildSigButtons();
  if(view === 'plot') rebuildCharts(); else buildReadout();
}

// Drag-zoom while offline: the visible envelope is decimated, so a zoom is a request for
// detail the browser does not have. Re-fetch that range at full budget instead of magnifying
// what is already drawn. Debounced, because uPlot fires setScale once per chart on the page.
function onOfflineZoom(u){
  if(!offline || offlineBusy) return;
  const sc = u.scales.x;
  if(sc.min == null || sc.max == null) return;
  const w = offline.window;
  if(Math.abs(sc.min - w[0]) < 1e-6 && Math.abs(sc.max - w[1]) < 1e-6) return;
  const lo = sc.min, hi = sc.max;
  if(zoomTimer) clearTimeout(zoomTimer);
  zoomTimer = setTimeout(() => loadSession(offline.id, lo, hi), 250);
}

function togglePause(){
  paused = !paused;
  const b = document.getElementById('pauseBtn');
  b.textContent = paused ? 'Resume' : 'Pause';
  b.classList.toggle('pause-on', paused);
}
async function toggleRecord(){ try { await fetch('/record', {method:'POST'}); } catch(e) {} }
async function toggleZero(){ try { await fetch('/zero', {method:'POST'}); } catch(e) {} }
async function toggleCls(){ try { await fetch('/cls/toggle', {method:'POST'}); pollCls(); } catch(e) {} }

// ---- IMU source switch ----------------------------------------------------
let srcKind = null, srcSources = null, srcBusy = false, srcMsgTimer = null;

function showSrcMsg(text, ms){
  const el = document.getElementById('srcMsg');
  el.textContent = text || '';
  if(srcMsgTimer) clearTimeout(srcMsgTimer);
  if(text && ms) srcMsgTimer = setTimeout(() => { el.textContent = ''; }, ms);
}

async function toggleSource(){
  if(srcBusy || !srcKind) return;
  // Post the explicit target rather than a server-side toggle, so a double click can't
  // ping-pong the source while the first switch is still connecting hardware.
  const want = srcKind === 'serial' ? 'i2c' : 'serial';
  srcBusy = true;
  const btn = document.getElementById('srcBtn');
  btn.disabled = true;
  showSrcMsg('switching to ' + want + '…');
  try {
    const r = await (await fetch('/source?kind=' + want, {method:'POST'})).json();
    showSrcMsg((r.ok ? '' : '✗ ') + (r.message || ''), r.ok ? 5000 : 12000);
  } catch(e) {
    showSrcMsg('✗ switch failed', 12000);
  } finally {
    srcBusy = false;
    btn.disabled = false;
  }
}

function syncSourceBtn(d){
  srcKind = d.source || srcKind;
  srcSources = d.sources || srcSources;
  const btn = document.getElementById('srcBtn');
  if(srcBusy) return;                       // don't fight an in-flight switch
  const serial = srcKind === 'serial';
  const linked = d.source_connected !== false;
  // "no link" = the port will not open; "no data" = it opened but nothing is arriving. Without
  // the second, a silent transmitter is indistinguishable from a healthy one.
  const streaming = d.source_streaming !== false;
  // Results not being accepted is its own failure: data flows in fine, but the device is not
  // reading its serial input, so nothing the model decides ever reaches it.
  const txBad = d.source_tx === false;
  const bad = !linked ? ' (no link)' : (!streaming ? ' (no data)' : (txBad ? ' (TX blocked)' : ''));
  btn.textContent = 'IMU: ' + (serial ? 'Serial' : 'I2C') + bad;
  btn.classList.toggle('src-serial', serial && linked && streaming && !txBad);
  btn.classList.toggle('src-warn', !!bad);
  const other = serial ? 'i2c' : 'serial';
  const canSwitch = !srcSources || srcSources[other] !== false;
  btn.disabled = !canSwitch;
  btn.title = txBad
    ? 'Results are not being accepted — the device must read its serial input (Serial.read())'
    : canSwitch
      ? 'Switch to the ' + (serial ? 'onboard I2C IMUs' : 'serial (Arduino) IMU')
      : other + ' source unavailable on this install';
}

function toggleView(){
  if(clsMode) return;   // CLS is its own page; Numbers/Plots toggle doesn't apply
  view = (view === 'plot') ? 'numbers' : 'plot';
  document.getElementById('viewBtn').textContent = (view === 'plot') ? 'Numbers' : 'Plots';
  document.getElementById('charts').style.display = (view === 'plot') ? 'flex' : 'none';
  document.getElementById('readout').style.display = (view === 'plot') ? 'none' : 'grid';
  if(view === 'plot'){ rebuildCharts(); }
  else { buildReadout(); fillReadout(); }
}

// Feed the Numbers view whatever the current mode has. Live, that is the newest polled sample;
// offline, the last point of the loaded window — without this the view would sit at '--' while
// a session is open, because tick() (its only other caller) does not run then.
function fillReadout(){
  if(offline){
    const d = { telemetry: {} };
    for(const [k, sg] of Object.entries(offline.signals || {})){
      d[k] = {};
      for(const lab of labels){
        const vals = SIGNALS[k].axes.map(ax => ((sg.axes[ax] || {})[lab] || []).slice(-1)[0]);
        d[k][lab] = vals.every(v => v == null) ? null : vals;
      }
    }
    for(const [k, g] of Object.entries(offline.telemetry || {})){
      d.telemetry[k] = (TELE_BY_KEY[k] ? TELE_BY_KEY[k].channels : [])
        .map(ch => (g.channels[ch] || []).slice(-1)[0] ?? null);
    }
    updateReadout(d);
  } else if(samples.length){
    updateReadout(samples[samples.length - 1]);
  }
}

// ---- manual Y range -------------------------------------------------------
function syncYControls(){
  const ym = yBySignal[signal] || { auto:true };
  document.getElementById('yBtn').textContent = ym.auto ? 'Y: Auto' : 'Y: Manual';
  const man = document.getElementById('yman');
  man.style.display = ym.auto ? 'none' : 'inline-flex';
  if(!ym.auto){ document.getElementById('ymin').value = ym.min; document.getElementById('ymax').value = ym.max; }
}
function toggleYMode(){
  const cur = yBySignal[signal] || { auto:true };
  if(cur.auto){
    let mn = 0, mx = 1;
    if(charts.length && charts[0].scales && charts[0].scales.y && charts[0].scales.y.min != null){
      mn = charts[0].scales.y.min; mx = charts[0].scales.y.max;
    }
    const dec = (mx - mn) < 1 ? 3 : 2;
    yBySignal[signal] = { auto:false, min:+mn.toFixed(dec), max:+mx.toFixed(dec) };
  } else {
    yBySignal[signal] = { auto:true };
  }
  syncYControls();
  if(view === 'plot') rebuildCharts();
}
function applyYInput(){
  const mn = parseFloat(document.getElementById('ymin').value);
  const mx = parseFloat(document.getElementById('ymax').value);
  if(isFinite(mn) && isFinite(mx) && mx > mn){
    yBySignal[signal] = { auto:false, min:mn, max:mx };
    if(view === 'plot') rebuildCharts();
  }
}

// ---- theme ----------------------------------------------------------------
function applyTheme(light){
  document.documentElement.classList.toggle('light', light);
  document.getElementById('themeBtn').textContent = light ? 'Dark' : 'Light';
  try { localStorage.setItem('theme', light ? 'light' : 'dark'); } catch(_) {}
  if(view === 'plot'){ if(charts.length) rebuildCharts(); }
  else { buildReadout(); if(samples.length) updateReadout(samples[samples.length-1]); }
}
function toggleTheme(){ applyTheme(!document.documentElement.classList.contains('light')); }

// ---- axis remap modal -----------------------------------------------------
// The remap is a host-side transform now, so this panel builds a chain of steps
// (rotate an axis by 90/180/270, or negate one) instead of hand-packing register bytes.
// The op matrices mirror imu_common.OPS so the resulting mapping can be previewed without a
// round-trip; the server recomputes it from the op names and stays the authority.
const AXIS_PRESETS = {
  P0:[0x21,0x04], P1:[0x24,0x00], P2:[0x24,0x06], P3:[0x21,0x02],
  P4:[0x24,0x03], P5:[0x21,0x01], P6:[0x21,0x07], P7:[0x24,0x05],
};
const AXIS_OPS = {
  rot_x_90:[[1,0,0],[0,0,-1],[0,1,0]],   rot_x_180:[[1,0,0],[0,-1,0],[0,0,-1]],
  rot_x_270:[[1,0,0],[0,0,1],[0,-1,0]],  rot_y_90:[[0,0,1],[0,1,0],[-1,0,0]],
  rot_y_180:[[-1,0,0],[0,1,0],[0,0,-1]], rot_y_270:[[0,0,-1],[0,1,0],[1,0,0]],
  rot_z_90:[[0,-1,0],[1,0,0],[0,0,1]],   rot_z_180:[[-1,0,0],[0,-1,0],[0,0,1]],
  rot_z_270:[[0,1,0],[-1,0,0],[0,0,1]],
  flip_x:[[-1,0,0],[0,1,0],[0,0,1]], flip_y:[[1,0,0],[0,-1,0],[0,0,1]], flip_z:[[1,0,0],[0,1,0],[0,0,-1]],
};
const AXIS_NAMES = ['X','Y','Z'];
let axisOps = [];          // the step chain, applied in order
let axisPreset = null;     // [config, sign] when a P0-P7 preset is staged instead of a chain
let axisDirty = false;     // nothing to Apply until the user actually changes something
let axisServer = null;     // last state fetched from the server
let cubeLabel = null;

function buildAxisControls(){
  document.getElementById('presets').innerHTML = Object.keys(AXIS_PRESETS).map(p =>
    '<button data-p="' + p + '" onclick="applyPreset(\\'' + p + '\\')">' + p + (p==='P1'?' •':'') + '</button>').join('');
}
const hx = b => '0x' + b.toString(16).toUpperCase().padStart(2,'0');
function matmul(a,b){ return [0,1,2].map(i=>[0,1,2].map(j=>a[i][0]*b[0][j]+a[i][1]*b[1][j]+a[i][2]*b[2][j])); }
function composeOps(ops){
  let m=[[1,0,0],[0,1,0],[0,0,1]];
  for(const o of ops){ if(AXIS_OPS[o]) m=matmul(AXIS_OPS[o],m); }   // later ops on top
  return m;
}
function permSigns(m){
  const p=[],s=[];
  for(const row of m){ const j=row.findIndex(v=>v!==0); p.push(j); s.push(row[j]); }
  return [p,s];
}
function bytesFromPermSigns(p,s){
  return [p[0]|(p[1]<<2)|(p[2]<<4), ((s[0]<0?1:0)<<2)|((s[1]<0?1:0)<<1)|(s[2]<0?1:0)];
}
function permSignsFromBytes(c,sg){
  return [[c&3,(c>>2)&3,(c>>4)&3], [(sg>>2)&1?-1:1, (sg>>1)&1?-1:1, sg&1?-1:1]];
}
// Determinant of a signed permutation = product of the signs x the parity of the permutation.
// For 3 elements the Vandermonde product is +2 for an even permutation and -2 for an odd one.
function detOf(p,s){
  return s[0]*s[1]*s[2] * (((p[0]-p[1])*(p[1]-p[2])*(p[2]-p[0]))/2);
}
function opLabel(o){
  if(o.startsWith('flip_')) return '−' + o.slice(5).toUpperCase();
  const parts=o.split('_');     // rot_x_90
  return parts[1].toUpperCase() + ' +' + parts[2] + '°';
}
function addRot(){
  const a=document.getElementById('opAxis').value, d=document.getElementById('opDeg').value;
  axisOps.push('rot_'+a+'_'+d); axisPreset=null; axisDirty=true; recomputeAxis();
}
function addFlip(){
  axisOps.push('flip_'+document.getElementById('opAxis').value);
  axisPreset=null; axisDirty=true; recomputeAxis();
}
function removeOp(i){ axisOps.splice(i,1); axisPreset=null; axisDirty=true; recomputeAxis(); }
function undoOp(){ if(axisOps.length){ axisOps.pop(); axisPreset=null; axisDirty=true; recomputeAxis(); } }
function clearOps(){ axisOps=[]; axisPreset=null; axisDirty=true; recomputeAxis(); }
function applyPreset(p){ axisOps=[]; axisPreset=AXIS_PRESETS[p]; axisDirty=true; recomputeAxis(); }
// Current mapping as [perm, signs]: the staged preset, else the step chain, else — while
// nothing has been touched — whatever the server reports (which may have come from either).
function stagedPermSigns(){
  if(axisPreset) return permSignsFromBytes(axisPreset[0],axisPreset[1]);
  if(!axisDirty && axisServer) return permSignsFromBytes(axisServer.config,axisServer.sign);
  return permSigns(composeOps(axisOps));
}
function recomputeAxis(){
  document.getElementById('opChain').innerHTML = axisOps.length
    ? axisOps.map((o,i)=>'<span class="chip'+(o.startsWith('flip_')?' mir':'')+'">'+opLabel(o)
        +'<button title="remove" onclick="removeOp('+i+')">×</button></span>').join('')
    : '<span class="muted" style="font-size:12px">'
      + (axisPreset ? 'preset staged — applying replaces the steps' : 'none — identity') + '</span>';
  const [p,s] = stagedPermSigns();
  const [c,sg] = bytesFromPermSigns(p,s);
  document.getElementById('axisMap').innerHTML = ['x','y','z'].map((o,i)=>
    o.toUpperCase()+' out  ←  '+(s[i]<0?'−':'+')+AXIS_NAMES[p[i]]).join('<br>');
  document.getElementById('axisBytes').textContent='CONFIG '+hx(c)+' · SIGN '+hx(sg);
  const warn=document.getElementById('axisWarn');
  const det = detOf(p,s);   // -1 = mirror, not a mounting orientation
  warn.style.display = det<0 ? 'block' : 'none';
  warn.textContent = 'Mirrored mapping — accel/gyro follow it, but quaternion/euler cannot '
    + 'and are passed through unchanged (the 3D cube will not match).';
  document.getElementById('axisApply').disabled = !axisDirty;
  let match=null;
  for(const q in AXIS_PRESETS){ const v=AXIS_PRESETS[q]; if(v[0]===c&&v[1]===sg) match=q; }
  document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('active', b.dataset.p===match));
}
async function applyAxis(){
  const msg=document.getElementById('axisMsg');
  msg.textContent='Applying…';
  const body = axisPreset ? {config:axisPreset[0], sign:axisPreset[1]} : {ops:axisOps};
  try{
    const r=await fetch('/axis-remap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.ok){
      axisServer=d; axisOps=(d.ops||[]).slice(); axisPreset=null; axisDirty=false;
      msg.textContent='✓ '+(d.message||'Applied')+' ('+hx(d.config)+'/'+hx(d.sign)+')'
        + (d.placement ? ' · '+d.placement : '');
      recomputeAxis();
    } else { msg.textContent='✗ '+(d.message||'Failed'); }
  }catch(e){ msg.textContent='✗ request failed'; }
}
async function openAxis(){
  document.getElementById('axisOverlay').classList.add('open');
  const sel=document.getElementById('cubeSensor');
  sel.innerHTML=labels.map(l=>'<option>'+l+'</option>').join('');
  if(!cubeLabel || labels.indexOf(cubeLabel)<0) cubeLabel=labels[0];
  sel.value=cubeLabel;
  document.getElementById('axisMsg').textContent='';
  axisPreset=null; axisDirty=false;
  try{
    const d=await (await fetch('/axis-remap')).json();
    axisServer=d; axisOps=(d.ops||[]).slice();
  }catch(e){ axisServer=null; axisOps=[]; }
  recomputeAxis();
  startCube();
}
function closeAxis(){ document.getElementById('axisOverlay').classList.remove('open'); stopCube(); }

// ---- three.js live cube ---------------------------------------------------
let cube={ on:false, renderer:null, scene:null, camera:null, mesh:null, raf:0, q:null };
// A camera-facing text label (canvas texture on a sprite) placed at an axis tip.
function makeAxisLabel(text,color,pos){
  const c=document.createElement('canvas'); c.width=c.height=64;
  const ctx=c.getContext('2d');
  ctx.font='bold 48px system-ui,-apple-system,sans-serif';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillStyle=color; ctx.fillText(text,32,34);
  const tex=new THREE.CanvasTexture(c); tex.minFilter=THREE.LinearFilter;
  const spr=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
  spr.position.copy(pos); spr.scale.set(0.6,0.6,0.6);
  return spr;
}
function startCube(){
  const wrap=document.getElementById('cubeWrap'), canvas=document.getElementById('cube');
  if(typeof THREE==='undefined'){ wrap.innerHTML='<div class="muted" style="padding:14px">3D library unavailable (static/three.min.js failed to load).</div>'; return; }
  if(!cube.renderer){
    cube.renderer=new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
    cube.scene=new THREE.Scene();
    cube.camera=new THREE.PerspectiveCamera(45,1,0.1,100);
    cube.camera.position.set(3.2,2.4,3.2); cube.camera.lookAt(0,0,0);
    const g=new THREE.Group();
    // Object space = sensor body frame: the board is thin along sensor Z, so the slab's
    // small dimension is the group's Z.
    g.add(new THREE.Mesh(new THREE.BoxGeometry(1.6,1.1,0.35), new THREE.MeshNormalMaterial()));
    g.add(new THREE.AxesHelper(1.6));
    // X/Y/Z tip labels, colour-matched to the AxesHelper lines (red/green/blue).
    g.add(makeAxisLabel('X','#ff4d4d',new THREE.Vector3(1.95,0,0)));
    g.add(makeAxisLabel('Y','#4ade80',new THREE.Vector3(0,1.95,0)));
    g.add(makeAxisLabel('Z','#60a5fa',new THREE.Vector3(0,0,1.95)));
    // Basis change BNO055 Z-up -> three.js Y-up: R_x(-90°) maps (x,y,z) to (x,z,-y).
    // The sensor quaternion is applied verbatim to `g`, so its labelled axes stay the
    // sensor's body axes; the parent tilts that whole world so sensor "up" renders up.
    const world=new THREE.Group();
    world.rotation.x=-Math.PI/2;
    world.add(g);
    cube.scene.add(world); cube.mesh=g; cube.q=new THREE.Quaternion();
  }
  resizeCube(); cube.on=true; renderCube();
}
function resizeCube(){
  if(!cube.renderer) return;
  const wrap=document.getElementById('cubeWrap');
  const w=wrap.clientWidth||300, h=wrap.clientHeight||240;
  cube.renderer.setPixelRatio(window.devicePixelRatio||1);
  cube.renderer.setSize(w,h,false);
  cube.camera.aspect=w/h; cube.camera.updateProjectionMatrix();
}
function renderCube(){
  if(!cube.on) return;
  const last=samples[samples.length-1];
  const q=last && last.quat && last.quat[cubeLabel];
  if(q && cube.q){ cube.q.set(q[1],q[2],q[3],q[0]); cube.mesh.quaternion.copy(cube.q); }  // [w,x,y,z] -> (x,y,z,w)
  cube.renderer.render(cube.scene,cube.camera);
  cube.raf=requestAnimationFrame(renderCube);
}
function stopCube(){ cube.on=false; if(cube.raf) cancelAnimationFrame(cube.raf); cube.raf=0; }
window.addEventListener('resize', ()=>{ if(cube.on) resizeCube(); });

// ---- calibration status modal --------------------------------------------
let calibTimer = null;
function openCalib(){
  document.getElementById('calibOverlay').classList.add('open');
  pollCalib();
  calibTimer = setInterval(pollCalib, 400);
}
function closeCalib(){
  document.getElementById('calibOverlay').classList.remove('open');
  if(calibTimer){ clearInterval(calibTimer); calibTimer=null; }
}
function calBars(v){
  let s='';
  for(let i=1;i<=3;i++) s+='<span class="calseg'+((v>=i)?' on':'')+'"></span>';
  return '<span class="calbars">'+s+'</span>';
}
async function pollCalib(){
  let d={};
  try{ d = await (await fetch('/calibration')).json(); }catch(e){ return; }
  const body=document.getElementById('calibBody');
  const labs=Object.keys(d);
  if(!labs.length){ body.innerHTML='<div class="muted">No sensors connected.</div>'; return; }
  body.innerHTML=labs.map(lab=>{
    const c=d[lab];
    if(!c) return '<div class="rtitle" style="margin-top:8px">'+lab+' <span class="muted">no data</span></div>';
    const ready=c.ready
      ? '<span class="calready" style="color:#22c55e">● ready</span>'
      : '<span class="calready muted">converging…</span>';
    return '<div class="rtitle" style="margin-top:8px">'+lab+' '+ready+'</div>'
      +'<div class="calrow"><span class="calname">Sys</span>'+calBars(c.sys)+'<span class="muted">'+c.sys+'/3</span></div>'
      +'<div class="calrow"><span class="calname">Gyro</span>'+calBars(c.gyro)+'<span class="muted">'+c.gyro+'/3</span></div>'
      +'<div class="calrow"><span class="calname">Accel</span>'+calBars(c.accel)+'<span class="muted">'+c.accel+'/3</span></div>'
      +'<div class="calrow"><span class="calname">Mag</span><span class="muted">n/a (IMUPLUS)</span></div>';
  }).join('');
}
window.addEventListener('keydown', e=>{ if(e.key==='Escape'){ closeAxis(); closeCalib(); } });

// ---- CLS (activity classification) page -----------------------------------
function enterCls(){
  clsMode = true;
  document.querySelectorAll('.sigbtn').forEach(b => b.classList.toggle('active', b.dataset.sig === 'cls'));
  document.getElementById('charts').style.display = 'none';
  document.getElementById('readout').style.display = 'none';
  document.getElementById('clsview').style.display = 'flex';
  if(!clsTimer){ pollCls(); clsTimer = setInterval(pollCls, 100); }
}
function exitCls(){
  clsMode = false;
  document.getElementById('clsview').style.display = 'none';
  if(clsTimer){ clearInterval(clsTimer); clsTimer = null; }
  document.getElementById('charts').style.display = (view === 'plot') ? 'flex' : 'none';
  document.getElementById('readout').style.display = (view === 'plot') ? 'none' : 'grid';
}
async function pollCls(){
  let d;
  try { d = await (await fetch('/cls?since=' + clsSince)).json(); } catch(e) { return; }
  const banner = document.getElementById('clsBanner');
  const runBtn = document.getElementById('clsRunBtn');
  if(!d.enabled){
    banner.className = 'clsbanner';
    banner.innerHTML = '<span class="muted">CLS disabled — ' + (d.reason || 'model not loaded') + '</span>';
    runBtn.style.display = 'none';
    return;
  }
  const running = d.running !== false;
  runBtn.style.display = '';
  runBtn.textContent = running ? 'Stop' : 'Start';
  runBtn.classList.toggle('rec-on', running);
  const v = d.vote || {};
  // The banner shows the aggregated decision — the service's actual output, and the only thing
  // sent back over serial. The raw 10 Hz stream stays visible in the log below.
  const voteNote = v.window > 1
      ? 'voted · ' + v.window + ' frames every ' + v.emit_every
        + (v.hysteresis > 1 ? ' · hysteresis ' + v.hysteresis : '')
      : 'per frame (voting off)';
  banner.title = voteNote;
  if(!running){
    banner.className = 'clsbanner';
    const why = d.reason && d.reason !== 'ok' ? ' — ' + d.reason : ' — press Start to resume';
    banner.innerHTML = '<span class="muted">inference stopped' + why + '</span>';
  } else if(d.decision){
    banner.className = 'clsbanner on';
    banner.innerHTML = '<span class="clscls" style="color:' + clsColor(d.decision.cls) + '">'
        + d.decision.cls + '</span>'
        + '<span class="clsconf">' + (d.decision.conf * 100).toFixed(0) + '%</span>'
        + '<span class="clsvote">' + voteNote + (d.decision.held ? ' · held' : '') + '</span>';
  } else {
    banner.className = 'clsbanner';
    banner.innerHTML = '<span class="muted">waiting for data (' + (d.sensor || '') + ')…</span>';
  }
  const log = document.getElementById('clsLog');
  for(const e of (d.entries || [])){
    clsSince = Math.max(clsSince, e.id);
    const pct = (e.conf * 100).toFixed(0);
    const row = document.createElement('div'); row.className = 'clsrow';
    // Mark the frame that closed a vote window, so an outvoted noisy frame is visible.
    const dec = (e.decision === null || e.decision === undefined) ? '' :
      '<span class="clsdec" style="color:' + clsColor(CLS_NAMES[e.decision]) + '">◀ '
        + (CLS_NAMES[e.decision] || e.decision) + '</span>';
    row.innerHTML = '<span class="clstime">' + e.clock + '</span>'
      + '<span class="clsname" style="color:' + clsColor(e.cls) + '">' + e.cls + '</span>'
      + '<span class="clsbar"><i style="width:' + pct + '%;background:' + clsColor(e.cls) + '"></i></span>'
      + '<span class="clspct">' + pct + '%</span>'
      + '<span class="clsdeccol">' + dec + '</span>';
    log.insertBefore(row, log.firstChild);
  }
  while(log.childNodes.length > 3000) log.removeChild(log.lastChild);
}

let sinceT = 0;                        // cursor: newest buffered-sample t already fetched
let statPolls = 0, statSamples = 0, statStart = performance.now(), rateStr = '';

async function tick(){
  // Offline viewing owns the charts; polling would fight it for setData and the x scale.
  if(!paused && !offline){
    try {
      const d = await (await fetch('/data?since=' + sinceT)).json();
      if(d.t < sinceT){ sinceT = 0; samples = []; }   // server restarted (monotonic reset)
      latestT = d.t;
      const ks = Object.keys(d.euler || {});
      if(ks.length && JSON.stringify(ks) !== JSON.stringify(labels)){
        labels = ks;
        if(view === 'plot') rebuildCharts(); else buildReadout();
      }
      // Telemetry groups follow the source: switching to I2C drops all of them, and the
      // selected one has to go with them or the charts would draw from a key that is gone.
      const tg = d.telemetry_groups || [];
      if(JSON.stringify(tg) !== JSON.stringify(teleAvail)){
        teleAvail = tg;
        if(isTele(signal) && tg.indexOf(signal) < 0) signal = Object.keys(SIGNALS)[0];
        buildSigButtons();
        if(view === 'plot') rebuildCharts(); else buildReadout();
      }
      const batch = (d.samples && d.samples.length) ? d.samples : [d];
      if(d.samples && d.samples.length) sinceT = d.samples[d.samples.length - 1].t;
      for(const s of batch) samples.push(s);
      const cutoff = latestT - WINDOW_S;
      while(samples.length && samples[0].t < cutoff) samples.shift();

      statPolls++; statSamples += (d.samples ? d.samples.length : 0);
      const elapsed = performance.now() - statStart;
      if(elapsed >= 2000){
        rateStr = ' · poll ' + (statPolls * 1000 / elapsed).toFixed(0)
                + ' Hz · data ' + (statSamples * 1000 / elapsed).toFixed(0) + ' Hz';
        statPolls = 0; statSamples = 0; statStart = performance.now();
      }
      // The measured wire rate next to the poll rate: it is the only visible check that
      // [source] sample_hz matches reality, and a mismatch rescales every CLS window silently.
      const obs = (d.observed_hz != null) ? ' · wire ' + d.observed_hz.toFixed(1) + ' Hz' : '';
      document.getElementById('status').innerHTML = '<span id="dot"></span>live · t=' + d.t.toFixed(1) + 's' + rateStr + obs;
      const rb = document.getElementById('recBtn');
      rb.textContent = d.recording ? 'Recording' : 'Record';
      rb.classList.toggle('rec-on', !!d.recording);
      const zb = document.getElementById('zeroBtn');
      zb.textContent = d.zeroed ? 'Zeroed' : 'Zero';
      zb.classList.toggle('rec-on', !!d.zeroed);
      syncSourceBtn(d);
      const f = document.getElementById('freq');
      if(document.activeElement !== f) f.value = d.hz;

      if(view === 'plot') redraw(); else updateReadout(d);
    } catch(e) { /* skip dropped poll */ }
  }
  setTimeout(tick, POLL_MS);
}

window.addEventListener('DOMContentLoaded', () => {
  let saved = 'light';
  try { saved = localStorage.getItem('theme') || 'light'; } catch(_) {}
  applyTheme(saved === 'light');
  document.getElementById('freq').addEventListener('change', async (e) => {
    const v = parseInt(e.target.value, 10);
    if(v >= 1 && v <= 100){ try { await fetch('/freq?hz=' + v, {method:'POST'}); } catch(_) {} }
  });
  document.getElementById('ymin').addEventListener('change', applyYInput);
  document.getElementById('ymax').addEventListener('change', applyYInput);
  buildAxisControls();
  buildSigButtons();
  rebuildCharts();
  syncYControls();
  tick();
});
</script>
</body>
</html>
"""
