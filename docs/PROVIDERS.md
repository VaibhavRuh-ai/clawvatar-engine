# Provider Upgrade Guide

Clawvatar ships with free, open-source components. You can swap providers for better quality.

## Current Free Defaults

| Component | Provider | How it works |
|---|---|---|
| Lip-sync | Rhubarb Lip Sync | Audio → phonemes → visemes (C++ binary) |
| Lip-sync fallback | Energy-based | Audio spectral analysis → visemes (pure Python) |
| VAD | Silero VAD | Neural voice activity detection |
| 3D Rendering | moderngl | OpenGL offscreen rendering |
| Avatar format | GLB/VRM | Standard 3D web formats |
| Avatar creation | Avaturn / VRoid Studio / Open Source Avatars | Free online tools |

## Paid Upgrade Options

### Better Lip-Sync

| Provider | Pricing | Quality | How to integrate |
|---|---|---|---|
| **Azure Speech SDK** | $1/hr audio | Excellent, real-time viseme events | Implement `LipSyncProvider`, use their WebSocket API |
| **OVR LipSync** (Meta) | Free (Oculus SDK) | Very good | Implement `LipSyncProvider`, call native library |
| **NVIDIA Audio2Face** | Free (Omniverse) | Excellent | Heavy, needs NVIDIA GPU |

### Better Avatars

| Provider | Pricing | Quality |
|---|---|---|
| **Avaturn** | Free tier / API available | Good, from photo |
| **Reallusion Character Creator** | $199+ | Professional quality |
| **MetaHuman (Unreal)** | Free | Photorealistic |

### Alternative Renderers

| Renderer | Pros | Cons |
|---|---|---|
| **Godot headless** | Better shading, animation | Heavier setup |
| **Three.js (headless-gl)** | Web ecosystem, huge community | Node.js sidecar |
| **Filament** | Google's PBR renderer | C++ integration needed |

## Writing a Custom Lip-Sync Provider

Implement the interface in `clawvatar/providers/base.py`:

```python
from clawvatar.providers.base import LipSyncProvider

class AzureVisemeLipSync(LipSyncProvider):
    def initialize(self) -> None:
        # Set up Azure Speech SDK client
        pass

    def detect_viseme(self, audio_chunk, sample_rate=16000) -> str:
        # Use Azure's real-time viseme API
        return "A"

    def detect_viseme_weights(self, audio_chunk, sample_rate=16000) -> dict:
        # Return blend shape weights
        return {"jawOpen": 0.7}
```

## STT/TTS (External)

Clawvatar does NOT handle STT/TTS — that's the calling system's responsibility (e.g., OpenClaw).

If building a full pipeline:
- **Free STT**: faster-whisper, Whisper.cpp, Vosk
- **Paid STT**: Deepgram ($0.0043/min), Google STT, AssemblyAI
- **Free TTS**: Piper, Coqui TTS
- **Paid TTS**: ElevenLabs ($5/mo+), Cartesia, OpenAI TTS
