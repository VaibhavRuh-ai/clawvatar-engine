"""Tests for audio stream buffer."""

import numpy as np

from clawvatar.audio.stream import AudioStreamBuffer


def test_push_and_get():
    buf = AudioStreamBuffer(sample_rate=16000, chunk_duration_ms=40, context_chunks=3)
    chunk = np.zeros(640, dtype=np.float32)
    buf.push(chunk)
    result = buf.get_context_window()
    assert result is not None
    assert len(result) == 640


def test_context_accumulates():
    buf = AudioStreamBuffer(sample_rate=16000, chunk_duration_ms=40, context_chunks=3)
    for i in range(3):
        buf.push(np.ones(640, dtype=np.float32) * i)
    result = buf.get_context_window()
    assert len(result) == 640 * 3


def test_ring_buffer_drops_old():
    buf = AudioStreamBuffer(sample_rate=16000, chunk_duration_ms=40, context_chunks=2)
    for i in range(5):
        buf.push(np.ones(640, dtype=np.float32) * i)
    result = buf.get_context_window()
    # context_chunks=2, maxlen=3, so 3 chunks retained
    assert len(result) == 640 * 3


def test_empty_buffer():
    buf = AudioStreamBuffer()
    assert buf.get_context_window() is None
    assert buf.get_latest_chunk() is None


def test_speaking_state():
    buf = AudioStreamBuffer()
    assert not buf.is_speaking
    buf.is_speaking = True
    assert buf.is_speaking
    buf.clear()
    assert not buf.is_speaking
