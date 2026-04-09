"""Rhubarb Lip Sync integration — audio to visemes on CPU.

Rhubarb Lip Sync: https://github.com/DanielSWolf/rhubarb-lip-sync
Outputs timed viseme sequences from audio. Open source, C++, very fast.

Install: Download binary from GitHub releases, or build from source.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from clawvatar.lipsync.visemes import VISEME_CODES, get_blendshape_weights
from clawvatar.providers.base import LipSyncProvider

logger = logging.getLogger(__name__)


class RhubarbLipSync(LipSyncProvider):
    """Audio → viseme detection using Rhubarb Lip Sync binary."""

    def __init__(self, rhubarb_path: str = "rhubarb"):
        self._rhubarb_path = rhubarb_path
        self._available = False

    def initialize(self) -> None:
        # Check if rhubarb binary is available
        path = shutil.which(self._rhubarb_path)
        if path:
            self._rhubarb_path = path
            self._available = True
            logger.info(f"Rhubarb Lip Sync found at: {path}")
        else:
            self._available = False
            logger.warning(
                f"Rhubarb binary not found at '{self._rhubarb_path}'. "
                "Download from: https://github.com/DanielSWolf/rhubarb-lip-sync/releases "
                "Falling back to energy-based detection."
            )

    @property
    def available(self) -> bool:
        return self._available

    def process_audio_file(self, wav_path: str) -> list[dict]:
        """Run Rhubarb on a WAV file and get timed viseme sequence.

        Returns:
            List of {"start": float, "end": float, "viseme": str}
        """
        if not self._available:
            raise RuntimeError("Rhubarb not available")

        result = subprocess.run(
            [self._rhubarb_path, "-f", "json", "--quiet", wav_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Rhubarb failed: {result.stderr}")

        data = json.loads(result.stdout)
        visemes = []
        for cue in data.get("mouthCues", []):
            visemes.append({
                "start": cue["start"],
                "end": cue["end"],
                "viseme": cue["value"],
            })
        return visemes

    def detect_viseme(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """Detect viseme from a short audio chunk.

        For real-time use, writes chunk to temp WAV and runs Rhubarb.
        Falls back to energy-based detection if Rhubarb unavailable.
        """
        if not self._available:
            return self._energy_detect(audio_chunk)

        # Write temp WAV
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            pcm16 = (audio_chunk * 32767).astype(np.int16)
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())

            visemes = self.process_audio_file(tmp_path)
            if visemes:
                return visemes[-1]["viseme"]
            return "X"
        except Exception as e:
            logger.debug(f"Rhubarb chunk detection failed: {e}")
            return self._energy_detect(audio_chunk)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def detect_viseme_weights(
        self, audio_chunk: np.ndarray, sample_rate: int = 16000
    ) -> dict[str, float]:
        viseme = self.detect_viseme(audio_chunk, sample_rate)
        return get_blendshape_weights(viseme)

    def _energy_detect(self, audio_chunk: np.ndarray) -> str:
        """Simple energy-based viseme detection fallback."""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        if energy < 0.01:
            return "X"
        elif energy < 0.05:
            return "G"
        elif energy < 0.1:
            return "B"
        elif energy < 0.2:
            return "C"
        elif energy < 0.4:
            return "D"
        else:
            return "A"
