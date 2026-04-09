"""Clawvatar configuration — 3D avatar engine with viseme lip-sync."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 5


class AvatarConfig(BaseModel):
    """Path to a GLB/VRM avatar model."""
    model_path: str = ""
    # Idle animation
    idle_blink_interval: float = 3.5
    idle_movement_scale: float = 0.3
    # Camera
    camera_distance: float = 0.6  # distance from face
    camera_fov: float = 30.0  # field of view in degrees


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_duration_ms: int = 40


class LipSyncConfig(BaseModel):
    # "rhubarb" (default) or "energy" (simple fallback)
    provider: str = "rhubarb"
    # Path to rhubarb binary (if not in PATH)
    rhubarb_path: str = "rhubarb"
    # Smoothing factor for viseme transitions (0=instant, 1=no change)
    smoothing: float = 0.3
    # Viseme hold time in ms (minimum time a viseme is shown)
    min_hold_ms: int = 60


class RenderConfig(BaseModel):
    width: int = 512
    height: int = 512
    fps: int = 30
    background_color: list[float] = Field(default_factory=lambda: [0.15, 0.15, 0.18, 1.0])
    # Output
    format: Literal["jpeg", "png", "raw"] = "jpeg"
    jpeg_quality: int = 85


class ClawvatarConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    lipsync: LipSyncConfig = Field(default_factory=LipSyncConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClawvatarConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
