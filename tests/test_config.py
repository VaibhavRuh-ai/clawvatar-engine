"""Tests for config loading and defaults."""

import tempfile
from pathlib import Path

from clawvatar.config import ClawvatarConfig


def test_default_config():
    config = ClawvatarConfig()
    assert config.server.port == 8765
    assert config.lipsync.provider == "rhubarb"
    assert config.render.format == "jpeg"
    assert config.render.width == 512


def test_yaml_roundtrip():
    config = ClawvatarConfig()
    config.server.port = 9999
    config.lipsync.provider = "energy"

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name

    config.to_yaml(path)
    loaded = ClawvatarConfig.from_yaml(path)

    assert loaded.server.port == 9999
    assert loaded.lipsync.provider == "energy"
    Path(path).unlink()


def test_missing_yaml_returns_defaults():
    config = ClawvatarConfig.from_yaml("/nonexistent/path.yaml")
    assert config.server.port == 8765
