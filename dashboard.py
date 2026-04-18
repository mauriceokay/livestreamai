"""
AI Streamer Dashboard — manage up to 50 concurrent AI streamers locally.

Run:
  python dashboard.py
  Open http://localhost:8080

Each streamer is an independent subprocess running pipeline.py with its own
set of environment variables. The dashboard tracks status, streams logs, and
lets you start/stop any individual streamer with one click.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

log = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

CONFIG_FILE = Path("streamers.json")
PROJECT_ROOT = Path(__file__).parent
PLATFORMS = ("tiktok", "youtube", "facebook", "instagram")
PLATFORM_DEFAULTS = {
    "tiktok":    "rtmps://live-push.tiktok.com/live/",
    "youtube":   "rtmp://a.rtmp.youtube.com/live2/",
    "facebook":  "rtmps://live-api-s.facebook.com:443/rtmp/",
    "instagram": "rtmps://edgetee-upload-lax5-1.instagram.com:443/rtmp/",
}

app = FastAPI(title="AI Streamer Dashboard", docs_url=None, redoc_url=None)


# ─────────────────────────────────────────────────────────────────────────────
#  Config models
# ─────────────────────────────────────────────────────────────────────────────

class PlatformKeys(BaseModel):
    stream_key: str = ""
    rtmp_url: str = ""      # leave empty to use built-in default


class StreamerConfig(BaseModel):
    id: str = ""
    label: str = ""         # display name in dashboard
    persona_name: str = "Maya"
    app_name: str = "MyApp"
    app_description: str = "a new chat app"
    tiktok_username: str = ""
    elevenlabs_voice_id: str = ""
    face_image: str = "assets/face.jpg"
    idle_interval_s: float = 12.0
    tiktok: PlatformKeys = PlatformKeys()
    youtube: PlatformKeys = PlatformKeys()
    facebook: PlatformKeys = PlatformKeys()
    instagram: PlatformKeys = PlatformKeys()


class GlobalConfig(BaseModel):
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    tts_backend: str = "elevenlabs"
    llm_backend: str = "claude"
    lipsync_model: str = "latsync"
    lipsync_device: str = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-streamer subprocess wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StreamerProcess:
    MAX_LOGS = 120

    def __init__(self, cfg: StreamerConfig, global_cfg: GlobalConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.proc: subprocess.Popen | None = None
        self.logs: deque[str] = deque(maxlen=self.MAX_LOGS)
        self.status: str = "stopped"   # stopped|starting|running|stopping|error
        self.started_at: float | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self.is_alive():
            return
        env = self._build_env()
        try:
            self.proc = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "pipeline.py")],
                env=env,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.status = "error"
            self.logs.append(f"[DASHBOARD] Launch failed: {exc}")
            return
        self.status = "starting"
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._tail_logs, daemon=True)
        self._thread.start()
        log.info("Started %s (pid %d)", self._name, self.proc.pid)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.status = "stopping"
            self.proc.terminate()
            log.info("Stopped %s", self._name)

    def force_kill(self) -> None:
        if self.proc:
            self.proc.kill()
        self.status = "stopped"
        self.started_at = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def to_dict(self) -> dict:
        self._sync_status()
        active_platforms = [
            p for p in PLATFORMS if getattr(self.cfg, p).stream_key
        ]
        uptime = (
            int(time.time() - self.started_at)
            if self.started_at and self.is_alive()
            else 0
        )
        return {
            "id": self.cfg.id,
            "label": self.cfg.label or self.cfg.persona_name,
            "persona_name": self.cfg.persona_name,
            "app_name": self.cfg.app_name,
            "platforms": active_platforms,
            "status": self.status,
            "uptime": uptime,
            "logs": list(self.logs)[-12:],
            "pid": self.proc.pid if self.is_alive() else None,
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _name(self) -> str:
        return self.cfg.label or self.cfg.persona_name or self.cfg.id

    def _sync_status(self) -> None:
        if self.proc and self.proc.poll() is not None:
            if self.status not in ("stopped", "error"):
                rc = self.proc.returncode
                self.status = "stopped" if rc == 0 else "error"
                self.started_at = None

    def _tail_logs(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for raw in self.proc.stdout:
            line = raw.rstrip()
            self.logs.append(line)
            if "Live on:" in line or "Streaming to:" in line:
                self.status = "running"
        rc = self.proc.wait()
        self.status = "stopped" if (rc == 0 or self.status == "stopping") else "error"
        self.started_at = None

    def _build_env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        g, c = self.global_cfg, self.cfg

        # Global API keys
        if g.anthropic_api_key:  env["ANTHROPIC_API_KEY"]  = g.anthropic_api_key
        if g.elevenlabs_api_key: env["ELEVENLABS_API_KEY"] = g.elevenlabs_api_key
        env.update({
            "TTS_BACKEND":   g.tts_backend,
            "LLM_BACKEND":   g.llm_backend,
            "LIPSYNC_MODEL": g.lipsync_model,
            "LIPSYNC_DEVICE": g.lipsync_device,
        })

        # Streamer persona
        env.update({
            "PERSONA_NAME":    c.persona_name,
            "APP_NAME":        c.app_name,
            "APP_DESCRIPTION": c.app_description,
        })
        if c.tiktok_username:     env["TIKTOK_USERNAME"]      = c.tiktok_username
        if c.elevenlabs_voice_id: env["ELEVENLABS_VOICE_ID"]  = c.elevenlabs_voice_id
        env["IDLE_INTERVAL_S"] = str(c.idle_interval_s)

        # Platform stream keys
        for plat in PLATFORMS:
            pk: PlatformKeys = getattr(c, plat)
            if pk.stream_key:
                env[f"{plat.upper()}_STREAM_KEY"] = pk.stream_key
                url = pk.rtmp_url or PLATFORM_DEFAULTS.get(plat, "")
                if url:
                    env[f"{plat.upper()}_RTMP_URL"] = url

        return env


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard state (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self.global_cfg = GlobalConfig()
        self.streamers: dict[str, StreamerConfig] = {}
        self.procs: dict[str, StreamerProcess] = {}
        self._load()

    def _load(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            data = json.loads(CONFIG_FILE.read_text())
            self.global_cfg = GlobalConfig(**data.get("global", {}))
            for raw in data.get("streamers", []):
                cfg = StreamerConfig(**raw)
                if not cfg.id:
                    cfg.id = _new_id()
                self.streamers[cfg.id] = cfg
        except Exception as exc:
            log.error("Config load error: %s", exc)

    def save(self) -> None:
        data = {
            "global": self.global_cfg.model_dump(),
            "streamers": [s.model_dump() for s in self.streamers.values()],
        }
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    def get_or_create_proc(self, sid: str) -> StreamerProcess:
        if sid not in self.procs:
            cfg = self.streamers.get(sid)
            if not cfg:
                raise KeyError(sid)
            self.procs[sid] = StreamerProcess(cfg, self.global_cfg)
        return self.procs[sid]

    def status_list(self) -> list[dict]:
        out = []
        for sid, cfg in self.streamers.items():
            if sid in self.procs:
                out.append(self.procs[sid].to_dict())
            else:
                active = [p for p in PLATFORMS if getattr(cfg, p).stream_key]
                out.append({
                    "id": sid,
                    "label": cfg.label or cfg.persona_name,
                    "persona_name": cfg.persona_name,
                    "app_name": cfg.app_name,
                    "platforms": active,
                    "status": "stopped",
                    "uptime": 0,
                    "logs": [],
                    "pid": None,
                })
        return out


state = DashboardState()


# ─────────────────────────────────────────────────────────────────────────────
#  API routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = PROJECT_ROOT / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@app.get("/api/streamers")
async def list_streamers():
    rows = state.status_list()
    running = sum(1 for r in rows if r["status"] == "running")
    return {"streamers": rows, "running": running, "total": len(rows)}


@app.post("/api/streamers")
async def create_streamer(cfg: StreamerConfig):
    if not cfg.id:
        cfg.id = _new_id()
    state.streamers[cfg.id] = cfg
    state.save()
    return {"id": cfg.id}


@app.put("/api/streamers/{sid}")
async def update_streamer(sid: str, cfg: StreamerConfig):
    if sid not in state.streamers:
        raise HTTPException(404, "Not found")
    cfg.id = sid
    state.streamers[sid] = cfg
    if sid in state.procs:
        state.procs[sid].cfg = cfg
        state.procs[sid].global_cfg = state.global_cfg
    state.save()
    return {"ok": True}


@app.delete("/api/streamers/{sid}")
async def delete_streamer(sid: str):
    if sid not in state.streamers:
        raise HTTPException(404, "Not found")
    if sid in state.procs:
        state.procs[sid].stop()
        del state.procs[sid]
    del state.streamers[sid]
    state.save()
    return {"ok": True}


@app.post("/api/streamers/{sid}/start")
async def start_streamer(sid: str):
    if sid not in state.streamers:
        raise HTTPException(404, "Not found")
    proc = state.get_or_create_proc(sid)
    proc.start()
    return {"status": proc.status}


@app.post("/api/streamers/{sid}/stop")
async def stop_streamer(sid: str):
    if sid not in state.procs:
        return {"status": "stopped"}
    state.procs[sid].stop()
    return {"status": "stopping"}


@app.post("/api/streamers/{sid}/kill")
async def kill_streamer(sid: str):
    if sid in state.procs:
        state.procs[sid].force_kill()
    return {"status": "stopped"}


@app.get("/api/streamers/{sid}/logs")
async def get_logs(sid: str):
    if sid not in state.procs:
        return {"logs": []}
    return {"logs": list(state.procs[sid].logs)}


@app.get("/api/streamers/{sid}/config")
async def get_streamer_config(sid: str):
    if sid not in state.streamers:
        raise HTTPException(404, "Not found")
    return state.streamers[sid].model_dump()


@app.get("/api/config/global")
async def get_global():
    return state.global_cfg.model_dump()


@app.put("/api/config/global")
async def update_global(cfg: GlobalConfig):
    state.global_cfg = cfg
    # Propagate to running processes on next start
    for proc in state.procs.values():
        proc.global_cfg = cfg
    state.save()
    return {"ok": True}


@app.post("/api/stop-all")
async def stop_all():
    for proc in state.procs.values():
        proc.stop()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers + entry point
# ─────────────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())[:8]


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 8080))
    log.info("Dashboard → http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
