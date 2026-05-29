"""Headless web server: serve latest IMU values for a browser uPlot frontend.

Deliberately minimal (mirrors a proven Raspberry-Pi design): Flask + three routes
(`GET /`, `GET /data`, `POST /record`). No websocket, no ring buffer, no async — the
browser polls `/data`, accumulates points, and draws with uPlot. All rendering happens
in the browser on the laptop, so the Jetson spends ~zero CPU on the UI.

Run via `jetson-imu-tui --serve`.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request
from loguru import logger

from jetson_imu_tui.config import AppConfig
from jetson_imu_tui.imu_service import ImuService
from jetson_imu_tui.recorder import Recorder
from jetson_imu_tui.ring_buffer import RAD_TO_DEG


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


class ServerState:
    """Holds the IMU service and the optional recorder; toggled from the web UI."""

    def __init__(self, service: ImuService, log_dir: Path, record_hz: int) -> None:
        self.service = service
        self.log_dir = log_dir
        self.record_hz = record_hz
        self.recorder: Recorder | None = None
        self._lock = threading.Lock()

    def toggle_record(self) -> bool:
        with self._lock:
            if self.recorder is None:
                self.recorder = Recorder(self.service, self.log_dir, self.record_hz).__enter__()
                return True
            try:
                self.recorder.__exit__(None, None, None)
            finally:
                self.recorder = None
            return False

    def set_record_hz(self, hz) -> int:
        """Set the recording rate (1–200 Hz); restart an active recorder to apply it."""
        try:
            hz = max(1, min(200, int(hz)))
        except (TypeError, ValueError):
            return self.record_hz
        with self._lock:
            self.record_hz = hz
            if self.recorder is not None:
                try:
                    self.recorder.__exit__(None, None, None)
                except Exception:
                    pass
                self.recorder = Recorder(self.service, self.log_dir, self.record_hz).__enter__()
        return self.record_hz

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def shutdown(self) -> None:
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


def _payload(state: ServerState) -> dict:
    snap = state.service.snapshot()
    out: dict = {
        "t": time.monotonic(),
        "recording": state.recording,
        "hz": state.record_hz,
        "euler": {},
        "accel": {},
        "gyro": {},
        "quat": {},
    }
    for label, data in snap.items():
        if data is None:
            out["euler"][label] = None
            out["accel"][label] = None
            out["gyro"][label] = None
            out["quat"][label] = None
            continue
        e = data.quat.to_euler("ZYX")
        out["euler"][label] = [e.x * RAD_TO_DEG, e.y * RAD_TO_DEG, e.z * RAD_TO_DEG]
        a = data.device_data.accel
        out["accel"][label] = [a.x, a.y, a.z]
        g = data.device_data.gyro
        out["gyro"][label] = [g.x, g.y, g.z]
        q = data.quat
        out["quat"][label] = [q.w, q.x, q.y, q.z]
    return out


def create_app(state: ServerState, window_s: float, poll_ms: int) -> Flask:
    app = Flask(__name__)
    html = _HTML.replace("__WINDOW_S__", str(float(window_s))).replace("__POLL_MS__", str(int(poll_ms)))

    @app.route("/")
    def index() -> Response:
        return Response(html, mimetype="text/html")

    @app.route("/data")
    def data() -> Response:
        return jsonify(_payload(state))

    @app.route("/record", methods=["POST"])
    def record() -> Response:
        return jsonify({"recording": state.toggle_record()})

    @app.route("/freq", methods=["POST"])
    def freq() -> Response:
        hz = request.args.get("hz") or (request.get_json(silent=True) or {}).get("hz")
        return jsonify({"hz": state.set_record_hz(hz)})

    return app


def run_server(cfg: AppConfig, host: str | None = None, port: int | None = None) -> None:
    host = host or cfg.web_host
    port = int(port or cfg.web_port)

    # Quiet down imu_python (loguru) and werkzeug so stdout stays clean.
    logger.remove()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    service = ImuService(cfg.bus_labels)
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

    state = ServerState(service, cfg.log_dir, cfg.record_hz)
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
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,Arial,sans-serif;background:#0f1115;color:#ddd}
  #bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#171a21;border-bottom:1px solid #2a2f3a}
  #bar .grow{flex:1}
  button{padding:6px 12px;font-size:14px;border-radius:6px;border:1px solid #3a3f4b;background:#222733;color:#ddd;cursor:pointer}
  button:hover{background:#2c3340}
  button.active{background:#2563eb;border-color:#2563eb;color:#fff}
  #status{font-variant-numeric:tabular-nums;color:#9aa4b2;font-size:13px}
  #charts{height:calc(100% - 49px);padding:6px;box-sizing:border-box;display:flex;flex-direction:column;gap:6px}
  .chart{background:#fff;border-radius:6px;padding:2px 4px}
  .uplot, .u-wrap{width:100% !important}
  #freq{width:56px;background:#222733;color:#ddd;border:1px solid #3a3f4b;border-radius:5px;padding:4px 6px}
  #readout{display:none;height:calc(100% - 49px);margin:0;padding:16px 20px;box-sizing:border-box;
           overflow:auto;font:15px/1.7 ui-monospace,Menlo,Consolas,monospace;color:#e5e7eb}
</style>
</head>
<body>
  <div id="bar">
    <button class="sigbtn active" data-sig="euler" onclick="setSignal('euler')">Euler</button>
    <button class="sigbtn" data-sig="accel" onclick="setSignal('accel')">Accel</button>
    <button class="sigbtn" data-sig="gyro" onclick="setSignal('gyro')">Gyro</button>
    <button class="sigbtn" data-sig="quat" onclick="setSignal('quat')">Quat</button>
    <button id="viewBtn" onclick="toggleView()">Numbers</button>
    <button id="pauseBtn" onclick="togglePause()">Pause</button>
    <button id="recBtn" onclick="toggleRecord()">Record</button>
    <label style="font-size:13px;color:#9aa4b2">Hz <input id="freq" type="number" min="1" max="200" step="1"></label>
    <span class="grow"></span>
    <span id="status">connecting...</span>
  </div>
  <div id="charts"></div>
  <pre id="readout"></pre>
<script>
const WINDOW_S = __WINDOW_S__;
const POLL_MS  = __POLL_MS__;
const SIGNALS = { euler:['x','y','z'], accel:['x','y','z'], gyro:['x','y','z'], quat:['w','x','y','z'] };
const COLORS  = ['#d946ef', '#06b6d4'];  // Left magenta, Right cyan

let labels = ['Left','Right'];
let signal = 'euler';
let view = 'plot';        // 'plot' | 'numbers'
let paused = false;
let samples = [];     // each: {t, euler:{Left:[...],...}, accel:..., gyro:..., quat:...}
let charts = [];
let latestT = 0;

function rebuildCharts(){
  charts.forEach(u => u.destroy());
  charts = [];
  const wrap = document.getElementById('charts');
  wrap.innerHTML = '';
  const axes = SIGNALS[signal];
  const w = wrap.clientWidth - 12;
  const h = Math.max(110, Math.floor((wrap.clientHeight - 6*(axes.length-1)) / axes.length) - 8);
  axes.forEach((ax, i) => {
    const div = document.createElement('div');
    div.className = 'chart';
    wrap.appendChild(div);
    const series = [{}];
    labels.forEach((lab, k) => series.push({ label: lab, stroke: COLORS[k % COLORS.length], width: 2, points:{show:false} }));
    const opts = {
      width: w, height: h, title: signal + '  ' + ax,
      cursor: { drag: { x:true, y:false } },
      legend: { live: true },
      scales: { x: { time:false, range: (u,_min,_max)=>[latestT - WINDOW_S, latestT] } },
      series,
      axes: [ { values:(u,vals)=>vals.map(v=>(v - latestT).toFixed(1)) }, {} ],
    };
    const init = [[]]; labels.forEach(()=>init.push([]));
    charts.push(new uPlot(opts, init, div));
  });
  redraw();
}

function redraw(){
  const axes = SIGNALS[signal];
  const ts = samples.map(s => s.t);
  charts.forEach((u, i) => {
    const cols = [ts];
    labels.forEach(lab => {
      cols.push(samples.map(s => { const v = s[signal] && s[signal][lab]; return v ? v[i] : null; }));
    });
    u.setData(cols);
  });
}

function setSignal(s){
  signal = s;
  document.querySelectorAll('.sigbtn').forEach(b => b.classList.toggle('active', b.dataset.sig === s));
  if(view === 'plot') rebuildCharts();   // history kept: samples hold all signals
}
function togglePause(){
  paused = !paused;
  document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
}
async function toggleRecord(){
  try { await fetch('/record', {method:'POST'}); } catch(e) {}
}
function toggleView(){
  view = (view === 'plot') ? 'numbers' : 'plot';
  document.getElementById('viewBtn').textContent = (view === 'plot') ? 'Numbers' : 'Plots';
  document.getElementById('charts').style.display = (view === 'plot') ? 'flex' : 'none';
  document.getElementById('readout').style.display = (view === 'plot') ? 'none' : 'block';
  if(view === 'plot') rebuildCharts();
}

function fmtVals(arr, axes){
  return axes.map((a, i) => a + ' ' + (arr ? arr[i].toFixed(3) : '--').padStart(9)).join('   ');
}
function updateReadout(d){
  let s = '';
  for(const sig of ['euler','accel','gyro','quat']){
    const axes = SIGNALS[sig];
    s += sig.toUpperCase().padEnd(7);
    labels.forEach(lab => { s += lab + ':  ' + fmtVals(d[sig] && d[sig][lab], axes) + '      '; });
    s += '\\n';
  }
  document.getElementById('readout').textContent = s;
}

async function tick(){
  if(!paused){
    try {
      const d = await (await fetch('/data')).json();
      latestT = d.t;
      const ks = Object.keys(d.euler || {});
      if(ks.length && JSON.stringify(ks) !== JSON.stringify(labels)){ labels = ks; if(view==='plot') rebuildCharts(); }
      samples.push(d);
      const cutoff = latestT - WINDOW_S;
      while(samples.length && samples[0].t < cutoff) samples.shift();
      document.getElementById('status').textContent =
        'rec: ' + (d.recording ? 'ON' : 'off') + ' | ' + d.hz + ' Hz | t=' + d.t.toFixed(1);
      document.getElementById('recBtn').textContent = d.recording ? 'Stop Rec' : 'Record';
      document.getElementById('recBtn').classList.toggle('active', !!d.recording);
      const f = document.getElementById('freq');
      if(document.activeElement !== f) f.value = d.hz;
      if(view === 'plot') redraw(); else updateReadout(d);
    } catch(e) { /* skip dropped poll */ }
  }
  setTimeout(tick, POLL_MS);
}

window.addEventListener('resize', () => { if(view === 'plot') rebuildCharts(); });
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('freq').addEventListener('change', async (e) => {
    const v = parseInt(e.target.value, 10);
    if(v >= 1 && v <= 200){ try { await fetch('/freq?hz=' + v, {method:'POST'}); } catch(_) {} }
  });
  rebuildCharts();
  tick();
});
</script>
</body>
</html>
"""
