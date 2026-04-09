# Avatar Models

Place your 3D avatar models (.glb or .vrm) in this directory.

## Get a Free Avatar

### Avaturn (recommended)
1. Visit https://avaturn.me
2. Create from a photo or customize
3. Export as .glb — includes ARKit blend shapes + visemes

### Open Source Avatars
1. Visit https://opensourceavatars.com/en/gallery
2. Download any avatar (300+ CC0 models)
3. Use the .vrm file directly

### VRoid Studio
1. Download from https://vroid.com/en/studio
2. Create and customize
3. Export as .vrm

## Validate

```bash
clawvatar validate avatars/my-avatar.glb
```

## Requirements

Your avatar must have **blend shapes** (morph targets) for lip-sync.
ARKit-compatible shapes work best (52 standard shapes for face tracking).
