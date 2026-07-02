"""Geometry unit tests for the agent-centric frame transforms."""

import numpy as np

from foresee.data.features import (
    resample_polyline,
    rotation_world_to_agent,
    to_agent_frame,
    to_world_frame,
)


def test_rotation_aligns_heading_with_plus_x():
    for heading in (0.0, 0.5, np.pi / 2, -2.4):
        R = rotation_world_to_agent(heading)
        aligned = R @ np.array([np.cos(heading), np.sin(heading)])
        np.testing.assert_allclose(aligned, [1.0, 0.0], atol=1e-6)


def test_agent_world_round_trip():
    rng = np.random.default_rng(0)
    origin = np.array([12.0, -7.0], dtype=np.float32)
    R = rotation_world_to_agent(0.8)
    pts = rng.normal(size=(50, 2)).astype(np.float32) * 20
    back = to_world_frame(to_agent_frame(pts, origin, R), origin, R)
    np.testing.assert_allclose(back, pts, atol=1e-4)


def test_resample_polyline_preserves_endpoints_and_count():
    line = np.array([[0, 0], [1, 0], [3, 0], [10, 0]], dtype=np.float32)
    out = resample_polyline(line, 8)
    assert out.shape == (8, 2)
    np.testing.assert_allclose(out[0], [0, 0], atol=1e-6)
    np.testing.assert_allclose(out[-1], [10, 0], atol=1e-6)
    # Equal arc-length spacing on a straight line means equal x steps.
    np.testing.assert_allclose(np.diff(out[:, 0]), 10 / 7, atol=1e-5)


def test_resample_polyline_degenerate_input():
    point = np.array([[2.0, 3.0]])
    out = resample_polyline(point, 5)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out, np.tile([2.0, 3.0], (5, 1)), atol=1e-6)
