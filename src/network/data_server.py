#!/usr/bin/env python3
"""
Live Data Server for Liquid Stillness

Runs a lightweight HTTP server on localhost that serves real-time
tracking data as JSON. Optionally sends data over WebSocket to a
remote relay server (e.g., VPS for TouchDesigner).

Endpoints:
  GET /       - Live dashboard (auto-updating browser page)
  GET /data   - Raw JSON (for TouchDesigner, scripts, anything)

Usage:
    server = DataServer(port=8765, ws_url="ws://82.112.226.90:3000")
    server.start()

    # In your tracking loop:
    server.update({
        "group": {"person_count": 3, "jitter": 0.34},
        "primary": {"breathing_phase": "inhale", ...},
        ...
    })

    # When done:
    server.stop()
"""

import asyncio
import json
import queue
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional


# Shared state between server thread and main thread
_current_data: Dict[str, Any] = {}
_data_lock = threading.Lock()


class _DataHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves live JSON data."""

    def do_GET(self):
        if self.path == "/data":
            self._serve_json()
        elif self.path == "/":
            self._serve_dashboard()
        else:
            self.send_error(404)

    def _serve_json(self):
        with _data_lock:
            payload = json.dumps(_current_data, indent=2)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload.encode())

    def _serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode())

    def log_message(self, format, *args):
        # Suppress request logs
        pass


DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<title>Liquid Stillness - Live Data</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }
  h1 { color: #00e5bf; font-size: 18px; margin-bottom: 15px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }
  .card { background: #14141e; border: 1px solid #2a2a3a; border-radius: 8px; padding: 15px; }
  .card h2 { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .metric { margin-bottom: 8px; }
  .metric .label { color: #666; font-size: 11px; }
  .metric .value { font-size: 22px; font-weight: bold; }
  .bar-bg { background: #1a1a2a; height: 8px; border-radius: 4px; margin-top: 4px; }
  .bar-fill { height: 8px; border-radius: 4px; transition: width 0.15s; }
  .green { color: #00ff88; }
  .cyan { color: #00e5bf; }
  .yellow { color: #ffdd44; }
  .orange { color: #ff9944; }
  .red { color: #ff4444; }
  .dim { color: #555; }
  .person-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #1a1a2a; }
  .person-id { font-weight: bold; width: 40px; }
  .primary-badge { background: #00e5bf22; color: #00e5bf; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
  .status { position: fixed; top: 10px; right: 20px; font-size: 11px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  .dot.live { background: #00ff88; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  #raw { background: #0d0d14; border: 1px solid #1a1a2a; border-radius: 6px; padding: 12px;
         font-size: 11px; color: #668; max-height: 300px; overflow-y: auto; white-space: pre; }
</style>
</head><body>
<h1>LIQUID STILLNESS - LIVE DATA</h1>
<div class="status"><span class="dot live"></span><span id="fps">--</span> FPS | <span id="rate">--</span> updates/s</div>

<div class="grid">
  <div class="card">
    <h2>Group</h2>
    <div class="metric"><div class="label">People</div><div class="value cyan" id="people">--</div></div>
    <div class="metric">
      <div class="label">Group Jitter</div>
      <div class="value" id="jitter">--</div>
      <div class="bar-bg"><div class="bar-fill" id="jitter-bar" style="width:0%;background:#00ff88;"></div></div>
    </div>
    <div class="metric"><div class="label">Audio Layers</div><div class="value yellow" id="layers">--</div></div>
  </div>

  <div class="card">
    <h2>Primary Person</h2>
    <div class="metric"><div class="label">ID</div><div class="value cyan" id="pid">--</div></div>
    <div class="metric"><div class="label">Breathing</div><div class="value" id="phase">--</div></div>
    <div class="metric">
      <div class="label">Signal</div>
      <div class="value cyan" id="signal">--</div>
      <div class="bar-bg"><div class="bar-fill" id="signal-bar" style="width:50%;background:#00e5bf;"></div></div>
    </div>
    <div class="metric"><div class="label">BPM</div><div class="value yellow" id="bpm">--</div></div>
    <div class="metric"><div class="label">Chest Depth</div><div class="value dim" id="depth">--</div></div>
  </div>

  <div class="card">
    <h2>All Persons</h2>
    <div id="persons">
      <div class="dim" style="font-size:12px;">Waiting for data...</div>
    </div>
  </div>
</div>

<h2 style="color:#444;font-size:11px;margin-bottom:8px;">RAW JSON (GET /data)</h2>
<div id="raw">Connecting...</div>

<script>
let updateCount = 0;
let lastCountReset = Date.now();

function jitterColor(v) {
  if (v < 0.2) return '#00ff88';
  if (v < 0.5) return '#ffdd44';
  if (v < 0.8) return '#ff9944';
  return '#ff4444';
}

function phaseColor(p) {
  if (p === 'inhale') return '#c8ff64';
  if (p === 'exhale') return '#b48cff';
  return '#c8c8b4';
}

async function poll() {
  try {
    const res = await fetch('/data');
    const d = await res.json();
    updateCount++;

    // FPS
    document.getElementById('fps').textContent = (d.fps || 0).toFixed(0);

    // Update rate
    const now = Date.now();
    if (now - lastCountReset > 1000) {
      document.getElementById('rate').textContent = updateCount;
      updateCount = 0;
      lastCountReset = now;
    }

    // Group
    const g = d.group || {};
    document.getElementById('people').textContent = g.person_count ?? '--';
    const j = g.jitter ?? 0;
    const jEl = document.getElementById('jitter');
    jEl.textContent = j.toFixed(3);
    jEl.style.color = jitterColor(j);
    const jBar = document.getElementById('jitter-bar');
    jBar.style.width = (j * 100) + '%';
    jBar.style.background = jitterColor(j);
    document.getElementById('layers').textContent = g.audio_layers ?? '--';

    // Primary
    const p = d.primary || {};
    document.getElementById('pid').textContent = p.id != null ? 'P' + p.id : '--';
    const phEl = document.getElementById('phase');
    phEl.textContent = (p.breathing_phase || '--').toUpperCase();
    phEl.style.color = phaseColor(p.breathing_phase);
    const sig = p.breathing_signal ?? 0;
    document.getElementById('signal').textContent = sig.toFixed(3);
    document.getElementById('signal-bar').style.width = ((sig + 1) / 2 * 100) + '%';
    document.getElementById('bpm').textContent = p.bpm ? p.bpm.toFixed(1) : '--';
    document.getElementById('depth').textContent = p.chest_depth_mm ? p.chest_depth_mm.toFixed(0) + 'mm' : '--';

    // Persons
    const persons = d.persons || [];
    let html = '';
    for (const pr of persons) {
      const c = jitterColor(pr.jitter || 0);
      html += '<div class="person-row">';
      html += '<span class="person-id" style="color:' + c + '">P' + pr.id + '</span>';
      html += '<span style="flex:1"><div class="bar-bg"><div class="bar-fill" style="width:' + ((pr.jitter||0)*100) + '%;background:' + c + ';"></div></div></span>';
      html += '<span style="width:40px;font-size:12px;color:' + c + '">' + (pr.jitter||0).toFixed(2) + '</span>';
      html += '<span style="width:55px;font-size:11px;color:#555">' + (pr.depth_mm||0).toFixed(0) + 'mm</span>';
      if (pr.is_primary) html += '<span class="primary-badge">PRIMARY</span>';
      html += '</div>';
    }
    document.getElementById('persons').innerHTML = html || '<div class="dim" style="font-size:12px;">No people detected</div>';

    // Raw JSON
    document.getElementById('raw').textContent = JSON.stringify(d, null, 2);

  } catch(e) {
    document.getElementById('raw').textContent = 'Connection error: ' + e.message;
  }
}

setInterval(poll, 100);
</script>
</body></html>"""


class DataServer:
    """
    Lightweight HTTP server that exposes live tracking data as JSON.
    Optionally sends data over WebSocket to a remote relay server.

    Runs in a background thread. Call update() from your main loop
    to push new data. Anyone can GET /data for the latest JSON.
    """

    def __init__(self, port: int = 8765, ws_url: Optional[str] = None, ws_rate: int = 30):
        self.port = port
        self._server = None
        self._thread = None

        # WebSocket sender
        self._ws_url = ws_url
        self._ws_rate = ws_rate
        self._ws_queue: Optional[queue.Queue] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop = threading.Event()

    def start(self):
        """Start the HTTP server (and WebSocket sender if configured)."""
        self._server = HTTPServer(("0.0.0.0", self.port), _DataHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[DataServer] Live data at http://localhost:{self.port}")
        print(f"[DataServer] JSON endpoint: http://localhost:{self.port}/data")

        if self._ws_url:
            self._ws_queue = queue.Queue(maxsize=5)
            self._ws_stop.clear()
            self._ws_thread = threading.Thread(target=self._ws_sender_loop, daemon=True)
            self._ws_thread.start()
            print(f"[DataServer] WebSocket sender → {self._ws_url} ({self._ws_rate} Hz max)")

    def update(self, data: Dict[str, Any]):
        """Push new data (called from main tracking loop)."""
        global _current_data
        with _data_lock:
            _current_data = data

        # Non-blocking push to WebSocket queue
        if self._ws_queue is not None:
            try:
                self._ws_queue.put_nowait(data)
            except queue.Full:
                # Drop oldest, keep latest (real-time data)
                try:
                    self._ws_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._ws_queue.put_nowait(data)
                except queue.Full:
                    pass

    def stop(self):
        """Stop the HTTP server and WebSocket sender."""
        self._ws_stop.set()
        if self._server:
            self._server.shutdown()
        if self._ws_thread:
            self._ws_thread.join(timeout=3)
        print("[DataServer] Stopped")

    # ------------------------------------------------------------------
    #  WebSocket sender (runs in background thread with its own asyncio)
    # ------------------------------------------------------------------

    def _ws_sender_loop(self):
        """Background thread: run asyncio event loop for WebSocket."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_sender_async())
        except Exception as e:
            print(f"[WS] Sender loop exited: {e}")
        finally:
            loop.close()

    async def _ws_sender_async(self):
        """Connect to relay, send queued data, auto-reconnect on failure."""
        try:
            import websockets
        except ImportError:
            print("[WS] ERROR: 'websockets' not installed. Run: pip install websockets")
            return

        backoff = 1.0
        min_interval = 1.0 / self._ws_rate

        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    print(f"[WS] Connected to {self._ws_url}")
                    backoff = 1.0  # Reset on successful connect

                    while not self._ws_stop.is_set():
                        try:
                            data = self._ws_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        t0 = time.monotonic()
                        try:
                            await ws.send(json.dumps(data))
                        except Exception:
                            # Connection lost, break to reconnect
                            break

                        # Throttle to ws_rate Hz
                        elapsed = time.monotonic() - t0
                        if elapsed < min_interval:
                            await asyncio.sleep(min_interval - elapsed)

            except Exception as e:
                if self._ws_stop.is_set():
                    break
                print(f"[WS] Connection failed: {e} — retrying in {backoff:.0f}s")
                # Wait with periodic stop-check
                waited = 0.0
                while waited < backoff and not self._ws_stop.is_set():
                    time.sleep(0.25)
                    waited += 0.25
                backoff = min(backoff * 2, 10.0)
