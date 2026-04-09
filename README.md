# Clawvatar Engine

Real-time 3D avatar lip-sync engine. Send audio, get a talking avatar. Built for AI agents on video calls.

**Audio in → Lip-synced 3D avatar out. Perfect sync. Runs on CPU.**

https://github.com/ruh-ai/clawvatar-engine

## How It Works

```
Audio chunks ──▶ [Clawvatar Server] ──▶ Blend shape weights ──▶ [Browser: Three.js + VRM] ──▶ Animated 3D avatar
                  (viseme detection,                              (60fps smooth rendering,
                   prosody analysis,                               head movement, blinks,
                   expression engine)                              emotion expressions)
```

1. You send **audio** (file upload, mic stream, or TTS output from your agent)
2. Server detects **visemes** (mouth shapes) + **prosody** (pitch, energy, rhythm) + **emotions**
3. Server returns **blend shape weights** + head pose
4. Browser renders the **VRM avatar** with Three.js in perfect sync with the audio

The same audio plays back untouched — Clawvatar only adds the visual animation.

## Features

- **Viseme lip-sync** — energy-based + optional Rhubarb phoneme detection
- **Prosody-driven expressions** — eyebrow raises on pitch peaks, head nods on syllable beats, emotions from speech patterns
- **Smooth animation** — server-side eased transitions + client-side 60fps interpolation
- **Batch processing** — send entire audio file, get all animation frames back in one shot (<1s for 5s audio)
- **Streaming mode** — mic input for real-time lip-sync
- **3D VRM avatars** — full texture, skeleton, materials via Three.js + three-vrm
- **Expression engine** — blinks, breathing, idle head movement, speech-driven emotions
- **Runs on CPU** — no GPU required, works on a 2-vCPU VPS
- **WebSocket protocol** — low-latency streaming
- **Test web UI included** — upload avatar + audio, see it work instantly

## Quick Start

```bash
# Clone
git clone https://github.com/ruh-ai/clawvatar-engine
cd clawvatar-engine

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Install system deps (Linux)
sudo apt-get install libegl1-mesa-dev libosmesa6-dev

# Start server
clawvatar serve --host 0.0.0.0 --port 8765

# Open test UI
# http://localhost:8765
```

### With HTTPS (required for mic access in browser)

```bash
clawvatar serve --host 0.0.0.0 --port 8765 \
  --ssl-cert /path/to/cert.pem \
  --ssl-key /path/to/key.pem
```

## Test Web UI

The engine ships with a built-in test UI at `/` when the server is running.

1. **Upload VRM avatar** — drag & drop or click to upload
2. **Connect** to the WebSocket server
3. **Upload audio file** (.wav/.mp3) — avatar speaks with the audio in perfect sync
4. **Use microphone** — real-time lip-sync from your mic

Get free VRM avatars:
- [Open Source Avatars](https://opensourceavatars.com) — 300+ CC0 models
- [VRoid Studio](https://vroid.com/en/studio) — create your own
- [Avaturn](https://avaturn.me) — photo to 3D

## WebSocket Protocol

Connect to `ws://localhost:8765/ws`

### Batch Mode (recommended for pre-recorded/TTS audio)

```json
// Send entire audio at once
{
  "type": "audio.batch",
  "data": "<base64 PCM int16>",
  "sample_rate": 16000,
  "chunk_size": 1024
}

// Receive all animation frames at once
{
  "type": "batch_weights",
  "frames": [
    {"w": {"aa": 0.7, "oh": 0.2}, "h": {"yaw": 1.2, "pitch": -0.5, "roll": 0.1}, "v": "A", "s": true},
    ...
  ],
  "duration": 5.0,
  "compute_ms": 450
}
```

### Streaming Mode (for real-time mic/live audio)

```json
// Send audio chunk
{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}

// Receive weights per chunk
{"type": "weights", "weights": {"aa": 0.7}, "head": {"yaw": 1.2, "pitch": -0.5, "roll": 0.1}, "viseme": "A", "is_speaking": true}
```

### Avatar Management

```json
// Upload avatar first via POST /upload, then load it
{"type": "avatar.load", "model_path": "/path/to/avatar.vrm"}

// Response
{"type": "avatar.ready", "info": {"name": "avatar", "blend_shape_count": 11}}
```

## Architecture

```
clawvatar-engine/
├── clawvatar/
│   ├── server.py              # FastAPI + WebSocket server
│   ├── pipeline.py            # Audio → viseme → expression → weights
│   ├── agent_pipeline.py      # Text + audio → pre-computed animation timeline
│   ├── config.py              # YAML config system
│   ├── cli.py                 # CLI: serve, init, validate
│   ├── audio/
│   │   ├── prosody.py         # Pitch, energy, rhythm, emotion detection
│   │   ├── vad.py             # Silero voice activity detection
│   │   └── stream.py          # Audio chunk buffer
│   ├── lipsync/
│   │   ├── visemes.py         # Viseme definitions + VRM/ARKit mappings
│   │   ├── energy.py          # Energy-based viseme detection (zero deps)
│   │   ├── smooth.py          # Smooth transitions, coarticulation, jaw oscillation
│   │   ├── phoneme.py         # Text → phoneme → viseme (gruut)
│   │   └── rhubarb.py         # Rhubarb Lip Sync integration (optional)
│   ├── animation/
│   │   ├── blendshape.py      # Blend shape animator with VRM/ARKit support
│   │   ├── expression_engine.py  # Prosody → eyebrows, eyes, emotions, head
│   │   ├── expression_planner.py # Text → emotion/gesture plan
│   │   └── timeline.py        # Pre-computed animation timeline builder
│   ├── avatar/
│   │   └── loader.py          # GLB/VRM/GLTF parser with blend shape extraction
│   ├── renderer/
│   │   └── engine.py          # moderngl 3D renderer (headless, for server-side)
│   ├── encoding/
│   │   └── video.py           # Frame encoding (JPEG/PNG)
│   ├── idle/
│   │   └── animator.py        # Blink, breathe, head sway
│   └── static/
│       └── index.html          # Test web UI (Three.js + three-vrm)
├── tests/                      # Unit tests
├── docs/
│   ├── SETUP.md
│   └── PROVIDERS.md
├── Dockerfile
├── pyproject.toml
└── avatars/README.md
```

## Avatar Requirements

Your VRM model needs **expressions** (blend shapes) for lip-sync. Standard VRM expressions:

| Expression | Used for |
|---|---|
| `aa`, `ee`, `ih`, `oh`, `ou` | Mouth shapes (vowels) |
| `blinkLeft`, `blinkRight` | Eye blinks |
| `happy`, `angry`, `sad`, `surprised`, `relaxed` | Emotions |

Validate: `clawvatar validate my-avatar.vrm`

## Configuration

```yaml
# clawvatar.yaml
server:
  host: "0.0.0.0"
  port: 8765
lipsync:
  provider: "energy"      # "energy" (zero deps) or "rhubarb" (better quality)
  smoothing: 0.3
render:
  width: 512
  height: 512
  fps: 30
```

## Integration with AI Agents

Clawvatar is designed to give AI agents a visual presence on video calls. The typical integration:

```python
# Your agent generates a response
agent_text = "Hello! I'd be happy to help you with that."
agent_audio = your_tts_engine.synthesize(agent_text)  # PCM int16 bytes

# Send to Clawvatar via WebSocket
import websockets, json, base64
async with websockets.connect("ws://localhost:8765/ws") as ws:
    ws.send(json.dumps({
        "type": "audio.batch",
        "data": base64.b64encode(agent_audio).decode(),
        "sample_rate": 16000,
        "chunk_size": 1024,
    }))
    result = json.loads(await ws.recv())
    # result["frames"] = animation weights synced to audio
    # Play audio + apply weights to avatar in your frontend
```

## Performance

| Component | Latency |
|---|---|
| VAD (Silero) | ~3ms |
| Viseme detection | ~2ms |
| Prosody analysis | ~1ms |
| Expression engine | <1ms |
| **Total per chunk** | **~7ms** |
| **Batch: 5s audio** | **~400ms** |

## Docker

```bash
docker build -t clawvatar .
docker run -p 8765:8765 clawvatar
```

## License

Apache 2.0

## Credits

Built by [Ruh AI](https://ruh.ai) for the [OpenClaw](https://github.com/open-claw/open-claw) multi-agent platform.

Open-source components:
- [Three.js](https://threejs.org) + [three-vrm](https://github.com/pixiv/three-vrm) — 3D rendering
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [gruut](https://github.com/rhasspy/gruut) — text to phonemes
- [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) — optional phoneme detection
