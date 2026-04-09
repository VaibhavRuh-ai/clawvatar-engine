# Clawvatar Engine Setup Guide

## Quick Start

```bash
# Install from source
git clone https://github.com/ruh-ai/clawvatar-engine
cd clawvatar-engine
pip install -e .

# Generate config
clawvatar init

# Start server (energy-based lip-sync, no external deps)
clawvatar serve --avatar path/to/avatar.glb
```

## Getting a 3D Avatar

You need a GLB or VRM model with blend shapes (morph targets). Free options:

### Avaturn (recommended)
1. Go to https://avaturn.me
2. Create avatar from a photo or customize
3. Export as .glb (includes ARKit blend shapes + visemes)

### Open Source Avatars
1. Go to https://opensourceavatars.com/en/gallery
2. Browse and download (300+ CC0 licensed models)
3. Use the .vrm file directly

### VRoid Studio
1. Download from https://vroid.com/en/studio
2. Create and customize your avatar
3. Export as .vrm (includes VRM blend shapes)

### Mixamo
1. Go to https://mixamo.com
2. Upload a character or use provided ones
3. Download as .glb/.fbx

### Validate Your Avatar
```bash
clawvatar validate my-avatar.glb
```
This checks the model loads correctly and has blend shapes for lip-sync.

## Rhubarb Lip Sync (Optional, Better Quality)

The engine works without Rhubarb (uses energy-based detection), but Rhubarb gives much better lip-sync accuracy.

### Install Rhubarb

**Linux:**
```bash
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip
unzip Rhubarb-Lip-Sync-1.13.0-Linux.zip
sudo mv rhubarb /usr/local/bin/
```

**macOS:**
```bash
brew install rhubarb-lip-sync
```

**Windows:**
Download from https://github.com/DanielSWolf/rhubarb-lip-sync/releases

### Verify
```bash
rhubarb --version
```

## System Dependencies

### Linux (Debian/Ubuntu)
```bash
# For headless 3D rendering
sudo apt-get install libegl1-mesa-dev libgl1-mesa-dri libosmesa6-dev

# For audio processing
sudo apt-get install libsndfile1
```

### macOS
```bash
# Most deps come with the system
brew install mesa
```

### Windows
Most dependencies work out of the box with pip install.

## Docker

```bash
# Build
docker build -t clawvatar .

# Run
docker run -p 8765:8765 -v ./avatars:/app/avatars clawvatar --avatar /app/avatars/my-avatar.glb
```

## Configuration

```yaml
# clawvatar.yaml
server:
  host: "0.0.0.0"
  port: 8765

avatar:
  model_path: "avatars/my-avatar.glb"
  idle_blink_interval: 3.5
  idle_movement_scale: 0.3
  camera_distance: 0.6
  camera_fov: 30.0

audio:
  sample_rate: 16000
  chunk_duration_ms: 40

lipsync:
  provider: "rhubarb"       # "rhubarb" or "energy"
  rhubarb_path: "rhubarb"   # path to binary
  smoothing: 0.3
  min_hold_ms: 60

render:
  width: 512
  height: 512
  fps: 30
  format: "jpeg"
  jpeg_quality: 85
  background_color: [0.15, 0.15, 0.18, 1.0]
```

## Troubleshooting

### "Failed to create OpenGL context"
Install EGL/osmesa headers:
```bash
sudo apt-get install libegl1-mesa-dev libosmesa6-dev
```

### "Rhubarb binary not found"
Engine falls back to energy-based lip-sync automatically. Install Rhubarb for better quality.

### Avatar has no blend shapes
Your model needs morph targets. Avaturn, VRoid Studio, and Open Source Avatars include them by default. Check with:
```bash
clawvatar validate my-avatar.glb
```
