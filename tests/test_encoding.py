"""Tests for frame encoding."""

import numpy as np

from clawvatar.encoding.video import FrameEncoder


def test_jpeg_encode():
    encoder = FrameEncoder(format="jpeg", output_size=(256, 256))
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    data = encoder.encode(frame)
    assert len(data) > 0
    # JPEG magic bytes
    assert data[:2] == b"\xff\xd8"


def test_png_encode():
    encoder = FrameEncoder(format="png", output_size=(256, 256))
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    data = encoder.encode(frame)
    assert data[:4] == b"\x89PNG"


def test_base64_encode():
    encoder = FrameEncoder(format="jpeg", output_size=(128, 128))
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    b64 = encoder.encode_base64(frame)
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_resize():
    encoder = FrameEncoder(format="raw", output_size=(64, 64))
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    data = encoder.encode(frame)
    assert len(data) == 64 * 64 * 3
