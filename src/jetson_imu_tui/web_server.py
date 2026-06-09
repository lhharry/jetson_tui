"""Headless web server: serve latest IMU values for a browser uPlot frontend.

Deliberately minimal (mirrors a proven Raspberry-Pi design): Flask + a few routes
(`GET /`, `GET /data`, `POST /record`, `POST /freq`). No websocket, no ring buffer,
no async — the browser polls `/data`, accumulates points, and draws with uPlot. All
rendering happens in the browser on the laptop, so the Jetson spends ~zero CPU on the UI.
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
from jetson_imu_tui.imu_service import PLACEMENTS, ImuService
from jetson_imu_tui.recorder import Recorder


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

    def toggle_zero(self) -> bool:
        return self.service.zero_toggle()

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
    out: dict = {
        "t": time.monotonic(),
        "recording": state.recording,
        "zeroed": state.service.is_zeroed,
        "hz": state.record_hz,
        "euler": {},
        "accel": {},
        "gyro": {},
        "quat": {},
    }
    for label, sig in state.service.signals().items():
        for key in ("euler", "accel", "gyro", "quat"):
            out[key][label] = sig[key] if sig is not None else None
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

    @app.route("/zero", methods=["POST"])
    def zero() -> Response:
        return jsonify({"zeroed": state.toggle_zero()})

    @app.route("/freq", methods=["POST"])
    def freq() -> Response:
        hz = request.args.get("hz") or (request.get_json(silent=True) or {}).get("hz")
        return jsonify({"hz": state.set_record_hz(hz)})

    @app.route("/axis-remap", methods=["GET"])
    def axis_remap_get() -> Response:
        return jsonify(state.service.get_axis_remap())

    @app.route("/axis-remap", methods=["POST"])
    def axis_remap_post():
        body = request.get_json(silent=True) or {}
        placement = request.args.get("placement") or body.get("placement")
        if placement and str(placement).upper() in PLACEMENTS:
            cfg_b, sgn_b = PLACEMENTS[str(placement).upper()]
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
                            "message": "provide 'placement' (P0-P7) or numeric 'config' and 'sign'",
                        }
                    ),
                    400,
                )
        return jsonify(state.service.set_axis_remap(cfg_b, sgn_b))

    return app


def run_server(cfg: AppConfig, host: str | None = None, port: int | None = None) -> None:
    host = host or cfg.web_host
    port = int(port or cfg.web_port)

    # Quiet down imu_python (loguru) and werkzeug so stdout stays clean.
    logger.remove()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    service = ImuService(cfg.bus_labels, state_path=Path(cfg.log_dir) / "axis_remap.json")
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
<script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>
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
  .reclabel{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
  .num{width:64px;background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px}
  #yman{align-items:center;gap:5px}
  .grow{flex:1}
  #status{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;white-space:nowrap}
  #dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:7px;vertical-align:middle}
  #charts{flex:1;min-height:0;display:flex;flex-direction:column;gap:8px;padding:8px}
  .chart{flex:1;min-height:0;display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:6px 10px}
  .chead{display:flex;align-items:center;gap:16px;padding:1px 2px 5px;font-size:12px;color:var(--muted)}
  .ctitle{font-weight:700;color:var(--fg);text-transform:uppercase;letter-spacing:.05em}
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
  .axisrow{display:flex;align-items:center;gap:10px}
  .axisrow>span.albl{width:74px;color:var(--muted);font-size:12px}
  .axisrow select{flex:1;background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:7px}
  .signbtn{width:42px;border:1px solid var(--border);background:var(--panel2);color:var(--fg);border-radius:6px;padding:7px 0;font-weight:700;cursor:pointer}
  .signbtn.neg{background:#ef4444;border-color:#ef4444;color:#fff}
  .warn{color:#f87171;font-size:12px;font-weight:600}
  .mfoot{display:flex;align-items:center;gap:10px;padding-top:4px}
  .mfoot .grow{flex:1}
  #axisApply[disabled]{opacity:.45;cursor:not-allowed}
  .muted{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
  #cubeWrap{height:240px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;overflow:hidden}
  #cube{display:block;width:100%;height:100%}
</style>
</head>
<body>
  <div id="app">
    <div id="bar">
      <div class="seg" id="sigseg">
        <button class="sigbtn active" data-sig="euler" onclick="setSignal('euler')">Euler</button>
        <button class="sigbtn" data-sig="accel" onclick="setSignal('accel')">Accel</button>
        <button class="sigbtn" data-sig="gyro" onclick="setSignal('gyro')">Gyro</button>
        <button class="sigbtn" data-sig="quat" onclick="setSignal('quat')">Quat</button>
      </div>
      <button id="viewBtn" class="btn" onclick="toggleView()">Numbers</button>
      <button id="pauseBtn" class="btn" onclick="togglePause()">Pause</button>
      <button id="axisBtn" class="btn" onclick="openAxis()">Axis</button>
      <button id="yBtn" class="btn" onclick="toggleYMode()">Y: Auto</button>
      <span id="yman" style="display:none">
        <input id="ymin" class="num" type="number" step="any" title="Y min">
        <span style="color:var(--muted)">–</span>
        <input id="ymax" class="num" type="number" step="any" title="Y max">
      </span>
      <span class="grow"></span>
      <button id="themeBtn" class="btn" onclick="toggleTheme()">Light</button>
      <button id="zeroBtn" class="btn" onclick="toggleZero()" title="Zero out current Euler/Accel/Gyro readings (tare)">Zero</button>
      <button id="recBtn" class="btn" onclick="toggleRecord()">Record</button>
      <label class="reclabel" title="Recording rate — only affects logging to disk, not the plot">
        Rec Hz <input id="freq" class="num" type="number" min="1" max="200" step="1"></label>
      <span id="status"><span id="dot"></span>connecting…</span>
    </div>
    <div id="charts"></div>
    <div id="readout"></div>
  </div>

  <div id="axisOverlay" class="overlay" onclick="if(event.target===this)closeAxis()">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Axis remap">
      <div class="mhead">
        <span class="mtitle">Axis Remap &nbsp;<span class="muted">BNO055 §3.4 · shared by all sensors</span></span>
        <span class="grow"></span>
        <button class="btn" onclick="closeAxis()">Close</button>
      </div>
      <div class="mbody">
        <div class="mcol">
          <div class="mlabel">Mounting presets</div>
          <div class="presets" id="presets"></div>
          <div class="mlabel">Manual mapping &nbsp;<span class="muted">output ← source · sign</span></div>
          <div class="axisrow"><span class="albl">X out</span><select id="ax-x"></select><button class="signbtn" id="sg-x" data-axis="x" onclick="toggleSign('x')">+</button></div>
          <div class="axisrow"><span class="albl">Y out</span><select id="ax-y"></select><button class="signbtn" id="sg-y" data-axis="y" onclick="toggleSign('y')">+</button></div>
          <div class="axisrow"><span class="albl">Z out</span><select id="ax-z"></select><button class="signbtn" id="sg-z" data-axis="z" onclick="toggleSign('z')">+</button></div>
          <div id="axisWarn" class="warn" style="display:none">Each output must map to a distinct source axis (invalid mapping is rejected by the chip).</div>
          <div class="mfoot">
            <span id="axisBytes" class="muted">CONFIG 0x24 · SIGN 0x00</span>
            <span class="grow"></span>
            <button id="axisApply" class="btn" onclick="applyAxis()">Apply</button>
          </div>
          <div id="axisMsg" class="muted"></div>
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
<script>
const WINDOW_S = __WINDOW_S__;
const POLL_MS  = __POLL_MS__;
const SIGNALS = { euler:['x','y','z'], accel:['x','y','z'], gyro:['x','y','z'], quat:['w','x','y','z'] };
const UNITS   = { euler:'deg', accel:'m/s^2', gyro:'rad/s', quat:'' };
const THEMES = {
  dark:  { axis:'#8b93a7', grid:'#222a38', series:['#e879f9','#22d3ee'], ax:{x:'#f87171',y:'#4ade80',z:'#60a5fa',w:'#fbbf24'} },
  light: { axis:'#5b6472', grid:'#e2e6ee', series:['#c026d3','#0891b2'], ax:{x:'#dc2626',y:'#16a34a',z:'#2563eb',w:'#d97706'} },
};
const theme = () => document.documentElement.classList.contains('light') ? THEMES.light : THEMES.dark;

let labels = ['Left','Right'];
let signal = 'euler';
let view = 'plot';
let paused = false;
let samples = [];
let charts = [], heads = [], ro = null;
let latestT = 0;
let yBySignal = {};               // signal -> {auto:true} | {auto:false, min, max}

const fmt = (sig, v) => v == null ? '--' : v.toFixed(sig === 'quat' ? 3 : 2);

function chartOpts(w, h){
  const T = theme();
  const series = [{}];
  labels.forEach((lab, k) => series.push({ stroke: T.series[k % T.series.length], width: 2, points:{show:false} }));
  const ym = yBySignal[signal];
  const yscale = (ym && !ym.auto && ym.max > ym.min) ? { range: [ym.min, ym.max] } : {};
  return {
    width: w, height: h,
    legend: { show:false },
    cursor: { drag:{ x:true, y:false }, points:{ show:false } },
    scales: { x: { time:false, range: () => [latestT - WINDOW_S, latestT] }, y: yscale },
    axes: [
      { stroke:T.axis, grid:{ stroke:T.grid }, ticks:{ stroke:T.grid },
        values:(u,vs)=>vs.map(v=>(v - latestT).toFixed(0)) },
      { stroke:T.axis, grid:{ stroke:T.grid }, ticks:{ stroke:T.grid }, size:54 },
    ],
    series,
  };
}

function rebuildCharts(){
  if(ro) ro.disconnect();
  charts.forEach(u => u.destroy());
  charts = []; heads = [];
  const wrap = document.getElementById('charts');
  wrap.innerHTML = '';
  const T = theme();
  ro = new ResizeObserver(entries => {
    for(const e of entries){
      const u = e.target.__u;
      if(u && e.contentRect.width > 0 && e.contentRect.height > 0)
        u.setSize({ width: e.contentRect.width, height: e.contentRect.height });
    }
  });
  SIGNALS[signal].forEach((ax) => {
    const card = document.createElement('div'); card.className = 'chart';
    const head = document.createElement('div'); head.className = 'chead';
    head.innerHTML = '<span class="ctitle">' + signal + ' <span style="color:' + T.ax[ax] + '">' + ax + '</span></span>'
      + labels.map((lab,k)=>'<span class="cval"><i style="background:' + T.series[k%T.series.length] + '"></i>'
          + lab + ' <b data-v="' + k + '">--</b></span>').join('');
    const body = document.createElement('div'); body.className = 'canvas';
    card.appendChild(head); card.appendChild(body); wrap.appendChild(card);
    const u = new uPlot(chartOpts(body.clientWidth || 300, body.clientHeight || 140),
                        [[], ...labels.map(()=>[])], body);
    body.__u = u; charts.push(u); heads.push(head); ro.observe(body);
  });
  redraw();
}

function redraw(){
  const ts = samples.map(s => s.t);
  const last = samples[samples.length - 1];
  charts.forEach((u, i) => {
    const cols = [ts];
    labels.forEach(lab => cols.push(samples.map(s => { const v = s[signal] && s[signal][lab]; return v ? v[i] : null; })));
    u.setData(cols);
    if(last) labels.forEach((lab,k) => {
      const el = heads[i].querySelector('b[data-v="' + k + '"]');
      const v = last[signal] && last[signal][lab];
      if(el) el.textContent = v ? fmt(signal, v[i]) : '--';
    });
  });
}

function buildReadout(){
  const T = theme();
  const wrap = document.getElementById('readout');
  wrap.innerHTML = labels.map(lab =>
    '<div class="rcard"><div class="rtitle">' + lab + '</div>'
    + ['euler','accel','gyro','quat'].map(sig =>
        '<div class="rgroup"><div class="rgname">' + sig
          + (UNITS[sig] ? '<span class="runit">' + UNITS[sig] + '</span>' : '') + '</div>'
        + '<div class="rvals">'
        + SIGNALS[sig].map((ax,i) => '<span class="rax"><i style="color:' + T.ax[ax] + '">' + ax
            + '</i><b data-k="' + lab + '|' + sig + '|' + i + '">--</b></span>').join('')
        + '</div></div>').join('')
    + '</div>').join('');
}

function updateReadout(d){
  for(const lab of labels) for(const sig of ['euler','accel','gyro','quat']) SIGNALS[sig].forEach((ax,i) => {
    const el = document.querySelector('b[data-k="' + lab + '|' + sig + '|' + i + '"]');
    if(!el) return;
    const v = d[sig] && d[sig][lab];
    el.textContent = v ? fmt(sig, v[i]) : '--';
  });
}

function setSignal(s){
  signal = s;
  document.querySelectorAll('.sigbtn').forEach(b => b.classList.toggle('active', b.dataset.sig === s));
  if(view === 'plot') rebuildCharts();
  syncYControls();
}
function togglePause(){
  paused = !paused;
  const b = document.getElementById('pauseBtn');
  b.textContent = paused ? 'Resume' : 'Pause';
  b.classList.toggle('pause-on', paused);
}
async function toggleRecord(){ try { await fetch('/record', {method:'POST'}); } catch(e) {} }
async function toggleZero(){ try { await fetch('/zero', {method:'POST'}); } catch(e) {} }

function toggleView(){
  view = (view === 'plot') ? 'numbers' : 'plot';
  document.getElementById('viewBtn').textContent = (view === 'plot') ? 'Numbers' : 'Plots';
  document.getElementById('charts').style.display = (view === 'plot') ? 'flex' : 'none';
  document.getElementById('readout').style.display = (view === 'plot') ? 'none' : 'grid';
  if(view === 'plot'){ rebuildCharts(); }
  else { buildReadout(); if(samples.length) updateReadout(samples[samples.length-1]); }
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
const AXIS_PRESETS = {
  P0:[0x21,0x04], P1:[0x24,0x00], P2:[0x24,0x06], P3:[0x21,0x02],
  P4:[0x24,0x03], P5:[0x21,0x01], P6:[0x21,0x07], P7:[0x24,0x05],
};
const AXIS_NAMES = ['X','Y','Z'];
let axisSign = { x:0, y:0, z:0 };   // 0 = +, 1 = -
let cubeLabel = null;

function buildAxisControls(){
  document.getElementById('presets').innerHTML = Object.keys(AXIS_PRESETS).map(p =>
    '<button data-p="' + p + '" onclick="applyPreset(\\'' + p + '\\')">' + p + (p==='P1'?' •':'') + '</button>').join('');
  ['x','y','z'].forEach(out => {
    const sel = document.getElementById('ax-' + out);
    sel.innerHTML = AXIS_NAMES.map((n,i)=>'<option value="' + i + '">' + n + '</option>').join('');
    sel.onchange = recomputeAxis;
  });
}
const hx = b => '0x' + b.toString(16).toUpperCase().padStart(2,'0');
const configByte = () => (+document.getElementById('ax-x').value)
  | ((+document.getElementById('ax-y').value)<<2) | ((+document.getElementById('ax-z').value)<<4);
const signByte = () => (axisSign.x<<2)|(axisSign.y<<1)|(axisSign.z);
function axisValid(c){ const f=[c&3,(c>>2)&3,(c>>4)&3].sort(); return f[0]===0&&f[1]===1&&f[2]===2; }
function updateSignBtn(out){
  const b=document.getElementById('sg-'+out);
  b.textContent = axisSign[out] ? '−' : '+';
  b.classList.toggle('neg', !!axisSign[out]);
}
function toggleSign(out){ axisSign[out]=axisSign[out]?0:1; updateSignBtn(out); recomputeAxis(); }
function applyPreset(p){ const v=AXIS_PRESETS[p]; setControls(v[0],v[1]); }
function setControls(cfg, sgn){
  document.getElementById('ax-x').value = cfg & 3;
  document.getElementById('ax-y').value = (cfg>>2)&3;
  document.getElementById('ax-z').value = (cfg>>4)&3;
  axisSign = { x:(sgn>>2)&1, y:(sgn>>1)&1, z:sgn&1 };
  ['x','y','z'].forEach(updateSignBtn);
  recomputeAxis();
}
function recomputeAxis(){
  const c=configByte(), s=signByte(), ok=axisValid(c);
  document.getElementById('axisBytes').textContent='CONFIG '+hx(c)+' · SIGN '+hx(s);
  document.getElementById('axisWarn').style.display = ok ? 'none' : 'block';
  document.getElementById('axisApply').disabled = !ok;
  let match=null;
  for(const p in AXIS_PRESETS){ const v=AXIS_PRESETS[p]; if(v[0]===c&&v[1]===s) match=p; }
  document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('active', b.dataset.p===match));
}
async function applyAxis(){
  const c=configByte(), s=signByte();
  const msg=document.getElementById('axisMsg');
  msg.textContent='Applying…';
  try{
    const r=await fetch('/axis-remap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:c,sign:s})});
    const d=await r.json();
    if(d.ok && d.hardware) msg.textContent='✓ '+(d.message||'Applied')+' ('+hx(d.config)+'/'+hx(d.sign)+')';
    else if(d.ok) msg.textContent='✓ '+(d.message||'Stored');
    else msg.textContent='✗ '+(d.message||'Failed');
  }catch(e){ msg.textContent='✗ request failed'; }
}
async function openAxis(){
  document.getElementById('axisOverlay').classList.add('open');
  const sel=document.getElementById('cubeSensor');
  sel.innerHTML=labels.map(l=>'<option>'+l+'</option>').join('');
  if(!cubeLabel || labels.indexOf(cubeLabel)<0) cubeLabel=labels[0];
  sel.value=cubeLabel;
  try{ const d=await (await fetch('/axis-remap')).json(); setControls(d.config,d.sign); }
  catch(e){ setControls(0x24,0x00); }
  startCube();
}
function closeAxis(){ document.getElementById('axisOverlay').classList.remove('open'); stopCube(); }

// ---- three.js live cube ---------------------------------------------------
let cube={ on:false, renderer:null, scene:null, camera:null, mesh:null, raf:0, q:null };
function startCube(){
  const wrap=document.getElementById('cubeWrap'), canvas=document.getElementById('cube');
  if(typeof THREE==='undefined'){ wrap.innerHTML='<div class="muted" style="padding:14px">3D library unavailable (the browser needs internet for the CDN).</div>'; return; }
  if(!cube.renderer){
    cube.renderer=new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
    cube.scene=new THREE.Scene();
    cube.camera=new THREE.PerspectiveCamera(45,1,0.1,100);
    cube.camera.position.set(3.2,2.4,3.2); cube.camera.lookAt(0,0,0);
    const g=new THREE.Group();
    g.add(new THREE.Mesh(new THREE.BoxGeometry(1.6,0.35,1.1), new THREE.MeshNormalMaterial()));
    g.add(new THREE.AxesHelper(1.6));
    cube.scene.add(g); cube.mesh=g; cube.q=new THREE.Quaternion();
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
window.addEventListener('keydown', e=>{ if(e.key==='Escape') closeAxis(); });

async function tick(){
  if(!paused){
    try {
      const d = await (await fetch('/data')).json();
      latestT = d.t;
      const ks = Object.keys(d.euler || {});
      if(ks.length && JSON.stringify(ks) !== JSON.stringify(labels)){
        labels = ks;
        if(view === 'plot') rebuildCharts(); else buildReadout();
      }
      samples.push(d);
      const cutoff = latestT - WINDOW_S;
      while(samples.length && samples[0].t < cutoff) samples.shift();

      document.getElementById('status').innerHTML = '<span id="dot"></span>live · t=' + d.t.toFixed(1) + 's';
      const rb = document.getElementById('recBtn');
      rb.textContent = d.recording ? 'Recording' : 'Record';
      rb.classList.toggle('rec-on', !!d.recording);
      const zb = document.getElementById('zeroBtn');
      zb.textContent = d.zeroed ? 'Zeroed' : 'Zero';
      zb.classList.toggle('rec-on', !!d.zeroed);
      const f = document.getElementById('freq');
      if(document.activeElement !== f) f.value = d.hz;

      if(view === 'plot') redraw(); else updateReadout(d);
    } catch(e) { /* skip dropped poll */ }
  }
  setTimeout(tick, POLL_MS);
}

window.addEventListener('DOMContentLoaded', () => {
  let saved = 'dark';
  try { saved = localStorage.getItem('theme') || 'dark'; } catch(_) {}
  applyTheme(saved === 'light');
  document.getElementById('freq').addEventListener('change', async (e) => {
    const v = parseInt(e.target.value, 10);
    if(v >= 1 && v <= 200){ try { await fetch('/freq?hz=' + v, {method:'POST'}); } catch(_) {} }
  });
  document.getElementById('ymin').addEventListener('change', applyYInput);
  document.getElementById('ymax').addEventListener('change', applyYInput);
  buildAxisControls();
  rebuildCharts();
  syncYControls();
  tick();
});
</script>
</body>
</html>
"""
