"""Tests for idle animation system."""

import time

from clawvatar.idle.animator import IdleAnimator


def test_idle_params_structure():
    animator = IdleAnimator()
    params = animator.get_idle_params()
    assert "blink" in params
    assert "head_yaw" in params
    assert "head_pitch" in params
    assert "head_roll" in params
    assert "breathing" in params


def test_idle_values_in_range():
    animator = IdleAnimator()
    params = animator.get_idle_params()
    assert 0.0 <= params["blink"] <= 1.0
    assert 0.0 <= params["breathing"] <= 1.0
    assert -10.0 <= params["head_yaw"] <= 10.0
    assert -10.0 <= params["head_pitch"] <= 10.0


def test_idle_varies_over_time():
    animator = IdleAnimator(movement_scale=1.0)
    p1 = animator.get_idle_params()
    # Manually advance time by modifying start
    animator._start_time -= 2.0
    p2 = animator.get_idle_params()
    # Head movement should differ
    assert p1["head_yaw"] != p2["head_yaw"]
