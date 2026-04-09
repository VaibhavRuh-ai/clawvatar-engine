"""Clawvatar WebSocket server — audio in, animated avatar out."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from clawvatar.agent_pipeline import AgentPipeline
from clawvatar.config import ClawvatarConfig
from clawvatar.pipeline import ClawvatarPipeline

logger = logging.getLogger(__name__)

# Upload directory for avatars
UPLOAD_DIR = Path.home() / ".clawvatar" / "avatars"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Clawvatar Engine",
    description="Real-time 3D avatar animation — audio in, lip-synced video out",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[ClawvatarPipeline] = None
_agent: Optional[AgentPipeline] = None
_config: Optional[ClawvatarConfig] = None


def create_app(config: ClawvatarConfig) -> FastAPI:
    global _config
    _config = config

    @app.on_event("startup")
    async def startup():
        global _pipeline, _agent
        _pipeline = ClawvatarPipeline(config)
        _pipeline.setup()
        _agent = AgentPipeline(fps=config.render.fps)
        logger.info("Clawvatar server started")

        if config.avatar.model_path:
            try:
                info = _pipeline.load_avatar(config.avatar.model_path)
                logger.info(f"Auto-loaded avatar: {info}")
            except Exception as e:
                logger.error(f"Failed to auto-load avatar: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        if _pipeline:
            _pipeline.cleanup()

    return app


# --- Static UI ---
@app.get("/")
async def ui_root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/test")
async def test_page():
    return FileResponse(STATIC_DIR / "test.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- File upload ---
@app.post("/upload")
async def upload_avatar(file: UploadFile = File(...)):
    """Upload a GLB/VRM avatar file."""
    ext = Path(file.filename).suffix.lower()
    if ext not in (".glb", ".vrm", ".gltf"):
        return {"error": f"Unsupported format: {ext}. Use .glb, .vrm, or .gltf"}

    # Save to upload dir
    save_path = UPLOAD_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    logger.info(f"Avatar uploaded: {save_path} ({len(content)} bytes)")
    return {"path": str(save_path), "size": len(content), "name": file.filename}


# --- TTS (Piper) ---
_piper_voice = None
PIPER_MODEL = "/tmp/piper-models/en_US-lessac-medium.onnx"


def _get_piper():
    global _piper_voice
    if _piper_voice is None:
        try:
            from piper import PiperVoice
            _piper_voice = PiperVoice.load(PIPER_MODEL)
            logger.info(f"Piper TTS loaded: {PIPER_MODEL}")
        except Exception as e:
            logger.error(f"Piper TTS not available: {e}")
    return _piper_voice


def _synthesize(text: str) -> tuple[bytes, int]:
    voice = _get_piper()
    if voice is None:
        raise RuntimeError("Piper TTS not available")
    sr = voice.config.sample_rate
    pcm_chunks = []
    for chunk in voice.synthesize(text):
        pcm_chunks.append(chunk.audio_int16_bytes)
    pcm = b"".join(pcm_chunks)
    logger.info(f"TTS: {len(pcm)} bytes, {sr}Hz, {len(pcm)/2/sr:.2f}s")
    return pcm, sr


# --- Gemini ---
_gemini_client = None


def _get_gemini(api_key: str):
    global _gemini_client
    if _gemini_client is None or api_key:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized")
    return _gemini_client


GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-04-17",
    "gemini-1.5-flash",
]


async def _gemini_respond(api_key: str, user_text: str, system_prompt: str = "") -> str:
    """Get Gemini response. Tries multiple model names as fallback."""
    client = _get_gemini(api_key)
    prompt = user_text
    if system_prompt:
        prompt = f"{system_prompt}\n\nUser: {user_text}\nAssistant:"

    last_error = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            logger.info(f"Gemini model used: {model}")
            return response.text.strip()
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini model {model} failed: {e}")
            continue

    raise last_error


@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Simple chat endpoint: send user text, get agent response + animation."""
    api_key = request.get("api_key", "")
    user_text = request.get("text", "")
    system_prompt = request.get("system_prompt", "You are a friendly AI assistant. Keep responses concise, 1-2 sentences.")

    if not api_key:
        return {"error": "API key required"}
    if not user_text:
        return {"error": "Text required"}

    try:
        # Get Gemini response
        agent_text = await _gemini_respond(api_key, user_text, system_prompt)
        # Synthesize TTS
        pcm, sr = _synthesize(agent_text)
        # Build animation
        result = _agent.prepare(agent_text, pcm, sr)
        result["agent_text"] = agent_text
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": _pipeline.is_ready if _pipeline else False,
        "version": "0.1.0",
        "engine": "3d-blendshape",
    }


@app.get("/metrics")
async def metrics():
    if not _pipeline:
        return {"error": "pipeline not initialized"}
    m = _pipeline.metrics
    return {
        "frame_count": m.frame_count,
        "avg_latency_ms": round(m.avg_latency_ms, 2),
        "last_latency_ms": round(m.last_latency_ms, 2),
        "fps": round(m.fps, 1),
    }


@app.get("/avatar")
async def avatar_info():
    if not _pipeline or not _pipeline.avatar:
        return {"loaded": False}
    a = _pipeline.avatar
    return {
        "loaded": True,
        "name": a.name,
        "meshes": len(a.meshes),
        "vertices": a.vertex_count,
        "blend_shapes": a.blend_shape_names,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket endpoint with continuous idle streaming.

    Modes:
        - idle: sends idle weights at ~10fps when no audio (blink, breathe, sway)
        - audio: processes audio chunks, sends weights in real-time
        - audio.batch: processes entire audio at once, returns all weights
        - video: same as above but returns rendered JPEG frames instead of weights

    Client → Server:
        {"type": "audio", "data": "<b64 PCM16>", "sample_rate": 16000}
        {"type": "audio.batch", "data": "<b64 PCM16>", "sample_rate": 16000, "chunk_size": 1024}
        {"type": "avatar.load", "model_path": "/path/to/avatar.vrm"}
        {"type": "config", "idle_fps": 10, "mode": "weights"}
        {"type": "ping"}

    Server → Client:
        {"type": "weights", ...}           — blend shape weights + head pose
        {"type": "batch_weights", ...}     — all weights for batch audio
        {"type": "avatar.ready", ...}
        {"type": "error", "message": ...}
        {"type": "pong"}
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    if not _pipeline:
        await ws.send_json({"type": "error", "message": "Pipeline not initialized"})
        await ws.close()
        return

    # Per-connection state
    idle_streaming = True
    idle_fps = 10
    last_audio_time = 0.0  # timestamp of last audio message
    AUDIO_COOLDOWN = 1.5   # seconds after last audio before idle resumes
    disconnected = False

    async def idle_loop():
        """Background task: send idle weights when not processing audio."""
        while not disconnected:
            # Only send idle if no audio was received recently
            elapsed = time.time() - last_audio_time
            if idle_streaming and elapsed > AUDIO_COOLDOWN:
                try:
                    w = _pipeline.get_idle_weights()
                    if w:
                        await ws.send_json(w)
                except Exception:
                    break
            await asyncio.sleep(1.0 / idle_fps)

    # Start idle loop in background
    idle_task = asyncio.create_task(idle_loop())

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "config":
                # Configure connection
                idle_fps = msg.get("idle_fps", idle_fps)
                idle_streaming = msg.get("idle", idle_streaming)
                await ws.send_json({"type": "config.ok", "idle_fps": idle_fps, "idle": idle_streaming})

            elif msg_type == "avatar.load":
                model_path = msg.get("model_path", "")
                try:
                    info = _pipeline.load_avatar(model_path)
                    await ws.send_json({"type": "avatar.ready", "info": info})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif msg_type == "audio.batch":
                last_audio_time = time.time() + 9999  # block idle during batch
                audio_b64 = msg.get("data", "")
                sr = msg.get("sample_rate", 16000)
                chunk_size = msg.get("chunk_size", 1024)

                try:
                    pcm_bytes = base64.b64decode(audio_b64)
                    full_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    start_t = time.time()
                    frames = []
                    for i in range(0, len(full_audio), chunk_size):
                        chunk = full_audio[i:i + chunk_size]
                        if len(chunk) < 64:
                            continue
                        w = _pipeline.process_audio_weights(chunk)
                        if w:
                            frames.append({
                                "w": w.get("weights", {}),
                                "h": w.get("head", {}),
                                "v": w.get("viseme", "REST"),
                                "s": w.get("is_speaking", False),
                            })

                    elapsed = (time.time() - start_t) * 1000
                    duration = len(full_audio) / sr
                    logger.info(
                        f"Batch: {len(frames)} frames, {duration:.1f}s audio "
                        f"in {elapsed:.0f}ms ({elapsed/max(duration*1000,1)*100:.0f}% realtime)"
                    )

                    await ws.send_json({
                        "type": "batch_weights",
                        "frames": frames,
                        "duration": round(duration, 3),
                        "compute_ms": round(elapsed, 1),
                    })
                except Exception as e:
                    logger.error(f"Batch error: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "message": str(e)})
                finally:
                    # Resume idle after audio duration + cooldown
                    last_audio_time = time.time()

            elif msg_type == "audio":
                last_audio_time = time.time()
                audio_b64 = msg.get("data", "")
                try:
                    pcm_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    weights_data = _pipeline.process_audio_weights(audio)
                    if weights_data:
                        await ws.send_json(weights_data)
                except Exception as e:
                    logger.error(f"Audio error: {e}")
                    try:
                        await ws.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass

            else:
                await ws.send_json({"type": "error", "message": f"Unknown: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        disconnected = True
        idle_task.cancel()
