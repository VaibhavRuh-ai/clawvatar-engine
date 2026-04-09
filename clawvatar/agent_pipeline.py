"""Agent speech pipeline — text + audio in, pre-computed animation timeline out.

This is the main pipeline for the agent speaking use case:
1. Receive agent's text + TTS audio
2. Text → phonemes → visemes (gruut)
3. Text → expression plan (rule-based)
4. Build animation timeline (frame-by-frame)
5. Stream animation frames synced with audio chunks

The key insight: we know what the agent will say,
so we can plan perfect animation BEFORE playback.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import asdict

import numpy as np

from clawvatar.animation.expression_planner import plan_expressions
from clawvatar.animation.timeline import AnimationFrame, build_animation_timeline
from clawvatar.lipsync.phoneme import build_timeline, phonemes_to_visemes, text_to_phonemes

logger = logging.getLogger(__name__)


class AgentPipeline:
    """Processes agent text + audio into pre-computed animation."""

    def __init__(self, fps: int = 30):
        self.fps = fps

    def prepare(
        self,
        text: str,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        audio_format: str = "pcm16",
    ) -> dict:
        """Prepare full animation from text + audio.

        Args:
            text: Agent's text to speak.
            audio_bytes: Raw TTS audio (PCM int16 or base64).
            sample_rate: Audio sample rate.
            audio_format: "pcm16" or "base64_pcm16".

        Returns:
            Dict with "frames" (list of frame dicts) and "audio_b64" (base64 audio).
        """
        start = time.time()

        # Decode audio
        if audio_format == "base64_pcm16":
            audio_raw = base64.b64decode(audio_bytes)
        else:
            audio_raw = audio_bytes

        audio_array = np.frombuffer(audio_raw, dtype=np.int16)
        total_duration = len(audio_array) / sample_rate

        logger.info(f"Agent pipeline: text='{text[:50]}...' duration={total_duration:.2f}s")

        # 1. Text → phonemes → visemes
        phoneme_data = text_to_phonemes(text)
        visemes = phonemes_to_visemes(phoneme_data)
        visemes = build_timeline(visemes, total_duration)

        # 2. Text → expression plan
        expression_plan = plan_expressions(text, total_duration)

        # 3. Build animation timeline
        frames = build_animation_timeline(
            visemes=visemes,
            expression_plan=expression_plan,
            total_duration=total_duration,
            fps=self.fps,
        )

        # 4. Convert to serializable format
        frame_dicts = []
        for f in frames:
            frame_dicts.append({
                "t": f.time,
                "w": f.weights,
                "h": {"y": f.head_yaw, "p": f.head_pitch, "r": f.head_roll},
            })

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"Agent pipeline done: {len(frames)} frames, "
            f"{total_duration:.2f}s audio, computed in {elapsed_ms:.0f}ms"
        )

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_raw).decode("ascii")

        return {
            "frames": frame_dicts,
            "fps": self.fps,
            "duration": round(total_duration, 4),
            "frame_count": len(frames),
            "emotion": expression_plan.overall_emotion,
            "audio_b64": audio_b64,
            "sample_rate": sample_rate,
            "compute_ms": round(elapsed_ms, 1),
        }

    def prepare_streaming(
        self,
        text: str,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
    ) -> list[dict]:
        """Prepare animation as synced chunks for streaming.

        Each chunk contains audio + animation frames for that time window.
        Client plays them sequentially.

        Returns:
            List of chunk dicts, each with "audio_b64" and "frames".
        """
        result = self.prepare(text, audio_bytes, sample_rate)
        frames = result["frames"]
        fps = result["fps"]
        total_duration = result["duration"]

        audio_raw = base64.b64decode(result["audio_b64"])
        audio_array = np.frombuffer(audio_raw, dtype=np.int16)

        # Split into chunks
        chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        frames_per_chunk = max(1, int(fps * chunk_duration_ms / 1000))

        chunks = []
        audio_offset = 0
        frame_offset = 0

        while audio_offset < len(audio_array):
            # Audio chunk
            chunk_end = min(audio_offset + chunk_samples, len(audio_array))
            audio_chunk = audio_array[audio_offset:chunk_end]
            audio_chunk_b64 = base64.b64encode(audio_chunk.tobytes()).decode("ascii")

            # Corresponding animation frames
            frame_end = min(frame_offset + frames_per_chunk, len(frames))
            chunk_frames = frames[frame_offset:frame_end]

            chunks.append({
                "type": "agent_chunk",
                "audio_b64": audio_chunk_b64,
                "sample_rate": sample_rate,
                "frames": chunk_frames,
                "chunk_index": len(chunks),
                "is_last": chunk_end >= len(audio_array),
            })

            audio_offset = chunk_end
            frame_offset = frame_end

        return chunks
