"""Tests for energy-based lip-sync fallback."""

import numpy as np

from clawvatar.lipsync.energy import EnergyLipSync


def test_silence_returns_rest():
    ls = EnergyLipSync()
    ls.initialize()
    silence = np.zeros(640, dtype=np.float32)
    viseme = ls.detect_viseme(silence, 16000)
    assert viseme == "X"


def test_loud_audio_returns_open():
    ls = EnergyLipSync()
    ls.initialize()
    loud = np.random.randn(640).astype(np.float32) * 0.5
    viseme = ls.detect_viseme(loud, 16000)
    assert viseme in ["A", "C", "D", "B", "E"]  # Some open mouth shape


def test_weights_scale_with_energy():
    ls = EnergyLipSync()
    ls.initialize()
    quiet = np.random.randn(640).astype(np.float32) * 0.05
    loud = np.random.randn(640).astype(np.float32) * 0.4

    w_quiet = ls.detect_viseme_weights(quiet, 16000)
    w_loud = ls.detect_viseme_weights(loud, 16000)

    # Loud should have higher total weight
    total_quiet = sum(w_quiet.values()) if w_quiet else 0
    total_loud = sum(w_loud.values()) if w_loud else 0
    assert total_loud >= total_quiet
