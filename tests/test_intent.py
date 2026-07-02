"""Unit tests for observed-intent classification on synthetic kinematics."""

import numpy as np

from foresee.intent import estimate_intent

DT = 0.1
T = 30


def _valid():
    return np.ones(T, dtype=bool)


def test_constant_speed_straight_is_cruising():
    t = np.arange(T) * DT
    track = np.stack([10.0 * t, np.zeros(T)], axis=1)
    it = estimate_intent(track, _valid(), DT)
    assert it.longitudinal == "cruising"
    assert it.lateral == "straight"


def test_stationary_is_stopped():
    track = np.tile([5.0, 5.0], (T, 1))
    it = estimate_intent(track, _valid(), DT)
    assert it.longitudinal == "stopped"


def test_hard_deceleration_is_braking():
    t = np.arange(T) * DT
    speeds = np.clip(12.0 - 4.0 * t, 0.5, None)          # -4 m/s^2
    x = np.concatenate([[0.0], np.cumsum(speeds[:-1] * DT)])
    it = estimate_intent(np.stack([x, np.zeros(T)], axis=1), _valid(), DT)
    assert it.longitudinal == "braking"


def test_arc_is_turning():
    theta = np.linspace(0, np.pi / 2, T)                 # 90 degrees over the window
    track = 20.0 * np.stack([np.sin(theta), 1 - np.cos(theta)], axis=1)
    it = estimate_intent(track, _valid(), DT)
    assert it.is_turning()
    assert it.lateral == "turning left"


def test_completed_lane_change_is_detected():
    # 1.2 s smoothstep shift of 4 m to the left, then settled travel in the new lane.
    n_shift, n_hold = 12, 8
    t = np.arange(n_shift + n_hold) * DT
    x = 14.0 * t
    s = np.clip(np.arange(n_shift + n_hold) / (n_shift - 1), 0, 1)
    y = 4.0 * (3 * s**2 - 2 * s**3)
    it = estimate_intent(np.stack([x, y], axis=1), np.ones(len(t), dtype=bool), DT)
    assert it.is_lane_changing()
    assert it.lateral == "changing lanes left"


def test_too_few_points_is_invalid():
    it = estimate_intent(np.zeros((2, 2)), np.ones(2, dtype=bool), DT)
    assert not it.valid
