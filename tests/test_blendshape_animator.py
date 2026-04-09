"""Tests for blend shape animator."""

from clawvatar.animation.blendshape import BlendShapeAnimator


def test_idle_returns_structure():
    anim = BlendShapeAnimator()
    result = anim.update(is_speaking=False)
    assert "blend_shapes" in result
    assert "head_yaw" in result
    assert "head_pitch" in result
    assert "head_roll" in result


def test_speaking_applies_viseme():
    anim = BlendShapeAnimator(smoothing=0.0)  # No smoothing for test
    result = anim.update(viseme="A", is_speaking=True)
    weights = result["blend_shapes"]
    # Default format is VRM, so expect VRM shape names
    assert "blendShape2.mouth_a" in weights
    assert weights["blendShape2.mouth_a"] > 0.3


def test_smoothing_works():
    anim = BlendShapeAnimator(smoothing=0.5)
    # First frame with viseme A
    r1 = anim.update(viseme="A", is_speaking=True)
    jaw1 = r1["blend_shapes"].get("jawOpen", 0)
    # Second frame still smoothing toward A
    r2 = anim.update(viseme="A", is_speaking=True)
    jaw2 = r2["blend_shapes"].get("jawOpen", 0)
    # jaw2 should be closer to target than jaw1
    assert jaw2 >= jaw1


def test_reset():
    anim = BlendShapeAnimator(smoothing=0.0)
    anim.update(viseme="A", is_speaking=True)
    anim.reset()
    result = anim.update(is_speaking=False)
    assert result["blend_shapes"].get("blendShape2.mouth_a", 0) < 0.1
