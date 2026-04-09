"""Audio prosody analyzer — extract pitch, energy, rhythm, and emotion from audio in real-time.

All pure signal processing, no AI models. Runs in <5ms per chunk on CPU.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class ProsodyFeatures:
    """Extracted prosody features from an audio chunk."""

    __slots__ = [
        "rms", "rms_db", "pitch_hz", "pitch_confidence",
        "spectral_centroid", "zero_crossing_rate",
        "pitch_delta", "energy_delta",
        "is_voiced", "is_pause",
        "syllable_beat", "speech_rate",
        "emotion", "emotion_intensity",
    ]

    def __init__(self):
        self.rms: float = 0.0
        self.rms_db: float = -80.0
        self.pitch_hz: float = 0.0
        self.pitch_confidence: float = 0.0
        self.spectral_centroid: float = 0.0
        self.zero_crossing_rate: float = 0.0
        self.pitch_delta: float = 0.0  # pitch change from previous chunk
        self.energy_delta: float = 0.0  # energy change from previous chunk
        self.is_voiced: bool = False
        self.is_pause: bool = True
        self.syllable_beat: bool = False  # true on syllable onset
        self.speech_rate: float = 0.0  # syllables per second estimate
        self.emotion: str = "neutral"
        self.emotion_intensity: float = 0.0


class ProsodyAnalyzer:
    """Real-time audio prosody analysis — pitch, energy, rhythm, emotion."""

    def __init__(self, sample_rate: int = 16000, history_seconds: float = 3.0):
        self.sample_rate = sample_rate
        self._history_len = int(history_seconds * sample_rate / 640)  # chunks

        # History buffers
        self._pitch_history: deque[float] = deque(maxlen=self._history_len)
        self._energy_history: deque[float] = deque(maxlen=self._history_len)
        self._beat_history: deque[float] = deque(maxlen=self._history_len)

        # State
        self._prev_pitch: float = 0.0
        self._prev_rms: float = 0.0
        self._prev_energy_smooth: float = 0.0
        self._pause_frames: int = 0
        self._voiced_frames: int = 0
        self._syllable_count: int = 0
        self._syllable_timer: float = 0.0
        self._chunk_duration: float = 0.04  # ~640 samples at 16kHz

    def analyze(self, audio_chunk: np.ndarray) -> ProsodyFeatures:
        """Analyze a single audio chunk and return prosody features.

        Args:
            audio_chunk: float32 PCM audio, 16kHz.

        Returns:
            ProsodyFeatures with all extracted features.
        """
        f = ProsodyFeatures()
        chunk = audio_chunk.astype(np.float32).flatten()

        # --- RMS Energy ---
        f.rms = float(np.sqrt(np.mean(chunk ** 2)))
        f.rms_db = 20 * math.log10(max(f.rms, 1e-10))
        f.energy_delta = f.rms - self._prev_rms

        # --- Zero Crossing Rate ---
        signs = np.sign(chunk)
        sign_changes = np.abs(np.diff(signs))
        f.zero_crossing_rate = float(np.mean(sign_changes > 0))

        # --- Spectral Centroid ---
        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.sample_rate)
        fft_sum = np.sum(fft_mag)
        if fft_sum > 0:
            f.spectral_centroid = float(np.sum(freqs * fft_mag) / fft_sum)

        # --- Pitch (autocorrelation method) ---
        f.pitch_hz, f.pitch_confidence = self._estimate_pitch(chunk)
        f.is_voiced = f.pitch_confidence > 0.5 and f.pitch_hz > 60
        if f.is_voiced:
            f.pitch_delta = f.pitch_hz - self._prev_pitch if self._prev_pitch > 0 else 0.0
            self._prev_pitch = f.pitch_hz
        else:
            f.pitch_delta = 0.0

        # --- Pause detection ---
        if f.rms < 0.015:
            self._pause_frames += 1
            f.is_pause = self._pause_frames > 5  # ~200ms silence = pause
            self._voiced_frames = 0
        else:
            self._pause_frames = 0
            self._voiced_frames += 1
            f.is_pause = False

        # --- Syllable beat detection (energy envelope onset) ---
        smooth = 0.7 * self._prev_energy_smooth + 0.3 * f.rms
        if smooth > self._prev_energy_smooth * 1.3 and f.rms > 0.03:
            f.syllable_beat = True
            self._syllable_count += 1
        self._prev_energy_smooth = smooth

        # --- Speech rate (syllables per second) ---
        self._syllable_timer += self._chunk_duration
        if self._syllable_timer >= 1.0:
            f.speech_rate = self._syllable_count / self._syllable_timer
            self._syllable_count = 0
            self._syllable_timer = 0.0
        else:
            f.speech_rate = self._syllable_count / max(self._syllable_timer, 0.1)

        # --- Emotion detection from prosody ---
        f.emotion, f.emotion_intensity = self._detect_emotion(f)

        # Update history
        self._pitch_history.append(f.pitch_hz)
        self._energy_history.append(f.rms)
        self._prev_rms = f.rms

        return f

    def _estimate_pitch(self, chunk: np.ndarray) -> tuple[float, float]:
        """Estimate fundamental frequency using autocorrelation."""
        # Only process if there's enough energy
        if np.sqrt(np.mean(chunk ** 2)) < 0.01:
            return 0.0, 0.0

        # Autocorrelation
        n = len(chunk)
        # Search range: 60Hz to 500Hz
        min_lag = self.sample_rate // 500
        max_lag = self.sample_rate // 60

        if max_lag >= n:
            max_lag = n - 1

        # Normalized autocorrelation
        chunk_centered = chunk - np.mean(chunk)
        corr = np.correlate(chunk_centered, chunk_centered, mode='full')
        corr = corr[n - 1:]  # positive lags only

        if corr[0] == 0:
            return 0.0, 0.0

        corr = corr / corr[0]  # normalize

        # Find the first peak after min_lag
        search = corr[min_lag:max_lag]
        if len(search) < 3:
            return 0.0, 0.0

        # Find peaks
        peaks = []
        for i in range(1, len(search) - 1):
            if search[i] > search[i - 1] and search[i] > search[i + 1]:
                peaks.append((i + min_lag, search[i]))

        if not peaks:
            return 0.0, 0.0

        # Best peak (highest correlation)
        best_lag, confidence = max(peaks, key=lambda x: x[1])
        pitch_hz = self.sample_rate / best_lag

        return float(pitch_hz), float(confidence)

    def _detect_emotion(self, f: ProsodyFeatures) -> tuple[str, float]:
        """Detect emotional tone from prosody features.

        Returns:
            (emotion_name, intensity 0-1)
        """
        if f.is_pause:
            return "neutral", 0.0

        if not f.is_voiced:
            return "neutral", 0.1

        # Compute scores for each emotion
        scores = {
            "excited": 0.0,
            "happy": 0.0,
            "calm": 0.0,
            "emphasis": 0.0,
            "question": 0.0,
            "neutral": 0.3,
        }

        # High pitch + high energy = excited
        if f.pitch_hz > 200 and f.rms > 0.15:
            scores["excited"] = min(1.0, (f.pitch_hz - 200) / 200 + f.rms * 2)

        # Rising pitch = question or happy
        if f.pitch_delta > 20:
            scores["question"] = min(1.0, f.pitch_delta / 80)
            scores["happy"] = min(0.7, f.pitch_delta / 100)

        # Moderate pitch, moderate energy = happy
        if 150 < f.pitch_hz < 300 and f.rms > 0.08:
            scores["happy"] = max(scores["happy"], 0.4)

        # Low pitch + steady energy = calm
        if f.pitch_hz < 150 and f.rms < 0.12 and f.is_voiced:
            scores["calm"] = 0.5

        # Energy spike = emphasis
        if f.energy_delta > 0.08:
            scores["emphasis"] = min(1.0, f.energy_delta / 0.15)

        # Fast speech rate = more animated
        if f.speech_rate > 4:
            scores["excited"] = max(scores["excited"], 0.3)

        best = max(scores, key=scores.get)
        return best, min(1.0, scores[best])

    def get_pitch_trend(self, lookback: int = 10) -> float:
        """Get pitch trend over recent history. Positive = rising, negative = falling."""
        if len(self._pitch_history) < lookback:
            return 0.0
        recent = list(self._pitch_history)[-lookback:]
        voiced = [p for p in recent if p > 0]
        if len(voiced) < 3:
            return 0.0
        # Simple linear trend
        x = np.arange(len(voiced), dtype=np.float32)
        y = np.array(voiced, dtype=np.float32)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def get_energy_trend(self, lookback: int = 10) -> float:
        """Get energy trend. Positive = getting louder."""
        if len(self._energy_history) < lookback:
            return 0.0
        recent = list(self._energy_history)[-lookback:]
        x = np.arange(len(recent), dtype=np.float32)
        y = np.array(recent, dtype=np.float32)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
