"""Tests for 3D avatar loader."""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from clawvatar.avatar.loader import AvatarLoader


def _create_minimal_glb(tmp_path: Path) -> Path:
    """Create a minimal valid GLB file for testing."""
    # Minimal GLTF JSON
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "name": "test-mesh",
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
            }],
        }],
        "accessors": [
            {  # Positions
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": 3,
                "type": "VEC3",
                "max": [1, 1, 0],
                "min": [-1, -1, 0],
            },
            {  # Indices
                "bufferView": 1,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},  # 3 vertices * 3 * 4
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},   # 3 indices * 2
        ],
        "buffers": [{"byteLength": 44}],
    }

    # Binary data: 3 vertices + 3 indices
    vertices = np.array([
        [-1, -1, 0], [1, -1, 0], [0, 1, 0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)

    bin_data = vertices.tobytes() + indices.tobytes()
    # Pad to 4-byte alignment
    while len(bin_data) % 4 != 0:
        bin_data += b"\x00"

    json_str = json.dumps(gltf).encode("utf-8")
    # Pad JSON to 4-byte alignment
    while len(json_str) % 4 != 0:
        json_str += b" "

    # Build GLB
    total = 12 + 8 + len(json_str) + 8 + len(bin_data)
    glb = bytearray()
    glb += struct.pack("<III", 0x46546C67, 2, total)  # Header
    glb += struct.pack("<II", len(json_str), 0x4E4F534A)  # JSON chunk
    glb += json_str
    glb += struct.pack("<II", len(bin_data), 0x004E4942)  # BIN chunk
    glb += bin_data

    path = tmp_path / "test.glb"
    path.write_bytes(bytes(glb))
    return path


def test_load_glb(tmp_path):
    glb_path = _create_minimal_glb(tmp_path)
    loader = AvatarLoader()
    avatar = loader.load(glb_path)
    assert avatar.name == "test"
    assert len(avatar.meshes) == 1
    assert avatar.vertex_count == 3


def test_missing_file():
    loader = AvatarLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/model.glb")


def test_unsupported_format(tmp_path):
    path = tmp_path / "model.obj"
    path.write_text("v 0 0 0")
    loader = AvatarLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load(path)
