"""Clawvatar Engine — public Python API for integration into other projects.

Usage:
    from clawvatar import ClawvatarEngine

    engine = ClawvatarEngine(avatar_path="avatar.vrm")

    # Process audio → get animation weights
    weights = engine.process_audio(audio_chunk)

    # Process entire audio file → get all weights at once
    result = engine.process_batch(audio_bytes, sample_rate=16000)

    # Get idle frame (no audio)
    idle = engine.get_idle()

    # Start WebSocket server
    engine.serve(host="0.0.0.0", port=8765)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from clawvatar.config import ClawvatarConfig
from clawvatar.pipeline import ClawvatarPipeline

logger = logging.getLogger(__name__)


class ClawvatarEngine:
    """Main entry point for using Clawvatar as a library.

    Args:
        avatar_path: Path to VRM/GLB avatar model (optional, can load later).
        config: ClawvatarConfig (optional, uses defaults if not provided).
        config_path: Path to clawvatar.yaml config file.
    """

    def __init__(
        self,
        avatar_path: str | None = None,
        config: ClawvatarConfig | None = None,
        config_path: str | None = None,
    ):
        if config:
            self.config = config
        elif config_path:
            self.config = ClawvatarConfig.from_yaml(config_path)
        else:
            self.config = ClawvatarConfig()

        if avatar_path:
            self.config.avatar.model_path = avatar_path

        self._pipeline: Optional[ClawvatarPipeline] = None

    def setup(self) -> None:
        """Initialize the pipeline. Call this before processing audio."""
        self._pipeline = ClawvatarPipeline(self.config)
        self._pipeline.setup()
        if self.config.avatar.model_path:
            self.load_avatar(self.config.avatar.model_path)

    def load_avatar(self, path: str) -> dict:
        """Load a VRM/GLB avatar model.

        Returns:
            Avatar info dict with name, blend shapes, etc.
        """
        if not self._pipeline:
            self.setup()
        return self._pipeline.load_avatar(path)

    def process_audio(self, audio: np.ndarray | bytes, sample_rate: int = 16000) -> dict:
        """Process a single audio chunk → get animation weights.

        Args:
            audio: float32 numpy array or PCM int16 bytes.
            sample_rate: Audio sample rate.

        Returns:
            Dict with "weights", "head", "viseme", "is_speaking", "emotion".
        """
        if not self._pipeline:
            self.setup()

        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        return self._pipeline.process_audio_weights(audio)

    def process_batch(
        self,
        audio: np.ndarray | bytes,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ) -> dict:
        """Process entire audio → get all animation weights at once.

        Args:
            audio: float32 numpy array or PCM int16 bytes.
            sample_rate: Audio sample rate.
            chunk_size: Samples per chunk.

        Returns:
            Dict with "frames" (list), "duration", "compute_ms".
        """
        if not self._pipeline:
            self.setup()

        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        import time
        start = time.time()
        frames = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < 64:
                continue
            w = self._pipeline.process_audio_weights(chunk)
            if w:
                frames.append({
                    "w": w.get("weights", {}),
                    "h": w.get("head", {}),
                    "v": w.get("viseme", "REST"),
                    "s": w.get("is_speaking", False),
                })

        elapsed = (time.time() - start) * 1000
        duration = len(audio) / sample_rate

        return {
            "frames": frames,
            "duration": round(duration, 3),
            "compute_ms": round(elapsed, 1),
            "frame_count": len(frames),
        }

    def get_idle(self) -> dict:
        """Get idle animation weights (blink, breathe, head sway)."""
        if not self._pipeline:
            self.setup()
        return self._pipeline.get_idle_weights() or {}

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        ssl_cert: str | None = None,
        ssl_key: str | None = None,
    ) -> None:
        """Start the WebSocket server.

        This blocks — call from your main thread or run in a separate process.
        """
        import uvicorn
        from clawvatar.server import create_app

        self.config.server.host = host
        self.config.server.port = port
        create_app(self.config)

        kwargs = {"host": host, "port": port, "log_level": "info"}
        if ssl_cert and ssl_key:
            kwargs["ssl_certfile"] = ssl_cert
            kwargs["ssl_keyfile"] = ssl_key

        uvicorn.run("clawvatar.server:app", **kwargs)

    def cleanup(self) -> None:
        """Release resources."""
        if self._pipeline:
            self._pipeline.cleanup()
