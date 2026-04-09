"""Clawvatar Pipeline v2 — 3D avatar with viseme lip-sync on CPU.

Audio chunks → Viseme detection → Blend shape animation → 3D render → Encoded frame
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from clawvatar.animation.blendshape import BlendShapeAnimator
from clawvatar.animation.expression_engine import ExpressionEngine
from clawvatar.audio.prosody import ProsodyAnalyzer
from clawvatar.audio.stream import AudioStreamBuffer
from clawvatar.audio.vad import SileroVAD
from clawvatar.avatar.loader import AvatarLoader, AvatarModel
from clawvatar.config import ClawvatarConfig
from clawvatar.encoding.video import FrameEncoder
from clawvatar.lipsync.smooth import SmoothLipSync
from clawvatar.renderer.engine import Renderer3D

logger = logging.getLogger(__name__)


def _create_lipsync(config: ClawvatarConfig):
    """Create lip-sync provider based on config."""
    if config.lipsync.provider == "rhubarb":
        from clawvatar.lipsync.rhubarb import RhubarbLipSync

        ls = RhubarbLipSync(rhubarb_path=config.lipsync.rhubarb_path)
        ls.initialize()
        if not ls.available:
            logger.info("Rhubarb not available, falling back to energy-based lip-sync")
            from clawvatar.lipsync.energy import EnergyLipSync

            ls = EnergyLipSync()
            ls.initialize()
        return ls
    else:
        from clawvatar.lipsync.energy import EnergyLipSync

        ls = EnergyLipSync()
        ls.initialize()
        return ls


class PipelineMetrics:
    def __init__(self):
        self.frame_count = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0
        self.avg_latency_ms = 0.0
        self.fps = 0.0
        self._fps_start = time.time()
        self._fps_count = 0

    def record(self, latency_ms: float) -> None:
        self.frame_count += 1
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.frame_count
        self._fps_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_start = time.time()


class ClawvatarPipeline:
    """Main pipeline: audio chunks in → encoded 3D avatar video frames out."""

    def __init__(self, config: ClawvatarConfig):
        self.config = config

        self.renderer: Optional[Renderer3D] = None
        self.lipsync = None
        self.vad: Optional[SileroVAD] = None
        self.animator: Optional[BlendShapeAnimator] = None
        self.prosody: Optional[ProsodyAnalyzer] = None
        self.expression: Optional[ExpressionEngine] = None
        self.smooth_lipsync: Optional[SmoothLipSync] = None
        self.audio_buffer: Optional[AudioStreamBuffer] = None
        self.frame_encoder: Optional[FrameEncoder] = None
        self.avatar: Optional[AvatarModel] = None
        self.metrics = PipelineMetrics()
        self._ready = False

    def setup(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Clawvatar 3D pipeline...")

        # Renderer
        self.renderer = Renderer3D(
            width=self.config.render.width,
            height=self.config.render.height,
        )
        self.renderer.camera_distance = self.config.avatar.camera_distance
        self.renderer.camera_fov = self.config.avatar.camera_fov
        self.renderer.bg_color = tuple(self.config.render.background_color)
        self.renderer.initialize()

        # Lip-sync
        self.lipsync = _create_lipsync(self.config)

        # VAD
        self.vad = SileroVAD(sample_rate=self.config.audio.sample_rate)
        self.vad.initialize()

        # Animator
        self.animator = BlendShapeAnimator(
            smoothing=self.config.lipsync.smoothing,
            min_hold_ms=self.config.lipsync.min_hold_ms,
            idle_blink_interval=self.config.avatar.idle_blink_interval,
            idle_movement_scale=self.config.avatar.idle_movement_scale,
        )

        # Audio buffer
        self.audio_buffer = AudioStreamBuffer(
            sample_rate=self.config.audio.sample_rate,
            chunk_duration_ms=self.config.audio.chunk_duration_ms,
        )

        # Prosody analyzer
        self.prosody = ProsodyAnalyzer(sample_rate=self.config.audio.sample_rate)

        # Expression engine
        self.expression = ExpressionEngine()

        # Smooth lip-sync processor
        self.smooth_lipsync = SmoothLipSync(format="vrm")

        # Frame encoder
        self.frame_encoder = FrameEncoder(
            format=self.config.render.format,
            jpeg_quality=self.config.render.jpeg_quality,
            output_size=(self.config.render.width, self.config.render.height),
        )

        self._ready = True
        logger.info("Pipeline initialized")

    def load_avatar(self, model_path: str) -> dict:
        """Load a 3D avatar model (GLB/VRM/GLTF)."""
        if not self._ready:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        loader = AvatarLoader()
        self.avatar = loader.load(model_path)
        self.renderer.load_avatar(self.avatar)

        # Auto-detect blend shape format (ARKit vs VRM)
        from clawvatar.lipsync.visemes import detect_blendshape_format
        bs_format = detect_blendshape_format(self.avatar.blend_shape_names)
        self.animator.blendshape_format = bs_format
        logger.info(f"Detected blend shape format: {bs_format}")

        info = {
            "name": self.avatar.name,
            "meshes": len(self.avatar.meshes),
            "vertices": self.avatar.vertex_count,
            "faces": self.avatar.face_count,
            "blend_shapes": self.avatar.blend_shape_names,
            "blend_shape_count": len(self.avatar.blend_shapes),
            "blend_shape_format": bs_format,
        }
        logger.info(f"Avatar loaded: {info['name']}")
        return info

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Process an audio chunk → return base64 encoded video frame.

        Args:
            audio_chunk: float32 PCM audio, 16kHz.

        Returns:
            Base64-encoded frame, or None if avatar not loaded.
        """
        if self.avatar is None:
            return None

        start = time.time()

        # Push to buffer
        self.audio_buffer.push(audio_chunk)

        # Check VAD
        is_speech = self.vad.is_speech(audio_chunk)
        self.audio_buffer.is_speaking = is_speech

        # Get viseme from lip-sync
        if is_speech:
            viseme = self.lipsync.detect_viseme(
                audio_chunk, self.config.audio.sample_rate
            )
            logger.debug(f"Speech detected, viseme={viseme}")
        else:
            viseme = None

        # Update animator (it handles format-aware blend shape mapping)
        anim = self.animator.update(
            viseme=viseme,
            is_speaking=is_speech,
        )
        if is_speech:
            active = {k: round(v, 3) for k, v in anim["blend_shapes"].items() if v > 0.01}
            logger.debug(f"Active blend shapes: {active}")

        # Render 3D frame
        frame = self.renderer.render(
            blend_weights=anim["blend_shapes"],
            head_yaw=anim["head_yaw"],
            head_pitch=anim["head_pitch"],
            head_roll=anim["head_roll"],
        )

        # Encode
        encoded = self.frame_encoder.encode_base64(frame)

        latency = (time.time() - start) * 1000
        self.metrics.record(latency)

        return encoded

    def process_audio_weights(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """Process audio → return full expression weights (for client-side rendering).

        Pipeline: Audio → VAD + Prosody + Viseme → Expression Engine → Weights
        """
        start = time.time()

        self.audio_buffer.push(audio_chunk)

        # Dual speech detection: VAD + energy threshold
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        try:
            is_speech_vad = self.vad.is_speech(audio_chunk)
        except Exception:
            is_speech_vad = False
        is_speech_energy = rms > 0.01  # very low threshold
        is_speech = is_speech_vad or is_speech_energy
        self.audio_buffer.is_speaking = is_speech

        # Prosody analysis (pitch, energy, rhythm, emotion)
        prosody = self.prosody.analyze(audio_chunk)

        # Viseme detection
        if is_speech:
            viseme = self.lipsync.detect_viseme(
                audio_chunk, self.config.audio.sample_rate
            )
        else:
            viseme = None

        # Smooth lip-sync: viseme + energy → smooth mouth shapes
        mouth_weights = self.smooth_lipsync.update(
            viseme=viseme,
            energy=prosody.rms,
            is_speaking=is_speech,
            dt=1 / 30,
        )

        # Expression engine: prosody → eyebrows, eyes, emotions, head motion
        expr_state = self.expression.update(
            prosody if is_speech else None,
            dt=1 / 30,
        )

        # Merge: smooth mouth + expression weights
        weights = dict(mouth_weights)

        # Expression weights (emotions, blinks)
        expr_weights = self.expression.to_vrm_weights(expr_state)
        for k, v in expr_weights.items():
            weights[k] = round(max(weights.get(k, 0), v), 4)

        # Head pose: idle + expression-driven offsets
        idle_anim = self.animator.update(viseme=None, is_speaking=is_speech)
        expr_head = self.expression.to_head_pose(expr_state)
        head = {
            "yaw": round(idle_anim["head_yaw"] + expr_head["yaw"], 3),
            "pitch": round(idle_anim["head_pitch"] + expr_head["pitch"], 3),
            "roll": round(idle_anim["head_roll"] + expr_head["roll"], 3),
        }

        # Filter near-zero
        weights = {k: v for k, v in weights.items() if v > 0.005}

        latency = (time.time() - start) * 1000
        self.metrics.record(latency)

        return {
            "type": "weights",
            "weights": weights,
            "viseme": viseme or "REST",
            "is_speaking": is_speech,
            "emotion": prosody.emotion,
            "emotion_intensity": round(prosody.emotion_intensity, 3),
            "head": head,
            "latency_ms": round(latency, 2),
        }

    def get_idle_weights(self) -> Optional[dict]:
        """Get idle animation weights (no audio)."""
        # Run expression engine in idle mode
        expr_state = self.expression.update(None, dt=1 / 30)
        expr_weights = self.expression.to_vrm_weights(expr_state)
        expr_head = self.expression.to_head_pose(expr_state)
        # Idle blend shapes from animator
        anim = self.animator.update(is_speaking=False)

        weights = dict(anim["blend_shapes"])
        for k, v in expr_weights.items():
            weights[k] = round(max(weights.get(k, 0), v), 4)
        weights = {k: round(v, 4) for k, v in weights.items() if v > 0.005}

        return {
            "type": "weights",
            "weights": weights,
            "viseme": "REST",
            "is_speaking": False,
            "emotion": "neutral",
            "emotion_intensity": 0.0,
            "head": {
                "yaw": round(anim["head_yaw"] + expr_head["yaw"], 3),
                "pitch": round(anim["head_pitch"] + expr_head["pitch"], 3),
                "roll": round(anim["head_roll"] + expr_head["roll"], 3),
            },
        }

    def get_idle_frame(self) -> Optional[str]:
        """Generate a single idle frame (no audio) — for server-side rendering."""
        if self.avatar is None:
            return None

        anim = self.animator.update(is_speaking=False)
        frame = self.renderer.render(
            blend_weights=anim["blend_shapes"],
            head_yaw=anim["head_yaw"],
            head_pitch=anim["head_pitch"],
            head_roll=anim["head_roll"],
        )
        return self.frame_encoder.encode_base64(frame)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def cleanup(self) -> None:
        if self.renderer:
            self.renderer.cleanup()
