"""Deterministic synthetic scenarios in the same Sample format as the real pipeline.
Used by the smoke test and as a fallback when no dataset is available.
"""

from __future__ import annotations

import numpy as np

from ..config import FeatureConfig
from . import features as F


def _integrate_constant_curvature(rng, T: int, dt: float, speed_range=(2.0, 15.0)):
    """Integrate one agent under a (slowly varying) constant-curvature motion model.

    Returns world-frame ``positions (T,2)``, ``headings (T,)``, ``velocities (T,2)``.
    The constant-turn-rate-and-velocity model is the standard kinematic prior for vehicles.
    """
    x = rng.uniform(-30.0, 30.0)
    y = rng.uniform(-30.0, 30.0)
    heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(*speed_range)           # m/s
    yaw_rate = rng.normal(0.0, 0.15)            # rad/s (gentle curving)
    accel = rng.normal(0.0, 0.4)                # m/s^2

    positions = np.zeros((T, 2), np.float32)
    headings = np.zeros((T,), np.float32)
    velocities = np.zeros((T, 2), np.float32)
    for t in range(T):
        positions[t] = (x, y)
        headings[t] = heading
        velocities[t] = (speed * np.cos(heading), speed * np.sin(heading))
        # Forward-Euler integration of the kinematic state.
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        heading += yaw_rate * dt
        speed = max(0.0, speed + accel * dt)
    return positions, headings, velocities


def _make_lanes(rng, origin, R, cfg: FeatureConfig):
    """Generate a few plausible lane centerlines near the focal agent (agent frame)."""
    L, P, Fl = cfg.max_lanes, cfg.lane_num_points, cfg.lane_feature_dim
    lanes = np.zeros((L, P, Fl), np.float32)
    lane_mask = np.zeros((L,), dtype=bool)

    num_real = int(rng.integers(4, min(12, L)))
    for slot in range(num_real):
        # A lane centerline as a gently curved arc in the world, near the origin.
        theta = rng.uniform(-np.pi, np.pi)
        offset = rng.uniform(-cfg.map_radius_m * 0.6, cfg.map_radius_m * 0.6)
        s = np.linspace(-cfg.map_radius_m, cfg.map_radius_m, P)
        curv = rng.normal(0.0, 0.01)
        # Parametric arc: advance along `theta`, drift laterally by curvature.
        cx = origin[0] + np.cos(theta) * s - np.sin(theta) * (offset + 0.5 * curv * s**2)
        cy = origin[1] + np.sin(theta) * s + np.cos(theta) * (offset + 0.5 * curv * s**2)
        centerline_world = np.stack([cx, cy], axis=1).astype(np.float32)
        cl_agent = F.to_agent_frame(centerline_world, origin, R)
        lanes[slot] = F.polyline_to_features(cl_agent)
        lane_mask[slot] = True
    return lanes, lane_mask


def _raw_world_arrays(index: int, cfg: FeatureConfig):
    """Deterministically generate raw world-frame arrays for one synthetic scenario.

    Returns ``(positions, headings, velocities, valid, type_ids, is_av, order, focal_idx)`` -
    the common substrate for both the model Sample and the animated playback.
    """
    rng = np.random.default_rng(index + 1)
    T = cfg.num_total_steps
    dt = 1.0 / cfg.sample_rate_hz

    num_agents = int(rng.integers(3, cfg.max_agents))

    # Assign a semantic class to each agent and move it at a class-appropriate speed, so the
    # synthetic scenes resemble real mixed traffic (cars, pedestrians, cyclists, a bus).
    speed_by_type = {
        "vehicle": (3.0, 15.0), "bus": (2.0, 10.0), "motorcyclist": (3.0, 16.0),
        "cyclist": (1.5, 7.0), "pedestrian": (0.5, 2.0),
    }
    classes = ["vehicle", "vehicle", "vehicle", "pedestrian", "cyclist", "bus", "motorcyclist"]
    type_ids = np.empty((num_agents,), dtype=np.int64)
    positions = np.zeros((num_agents, T, 2), np.float32)
    headings = np.zeros((num_agents, T), np.float32)
    velocities = np.zeros((num_agents, T, 2), np.float32)
    for i in range(num_agents):
        name = "vehicle" if i == 0 else classes[int(rng.integers(0, len(classes)))]
        type_ids[i] = F.NAME_TO_ID[name]
        positions[i], headings[i], velocities[i] = _integrate_constant_curvature(
            rng, T, dt, speed_range=speed_by_type[name])
    valid = np.ones((num_agents, T), dtype=bool)
    # Designate one non-focal agent as the AV (ego) when present.
    is_av = np.zeros((num_agents,), dtype=bool)
    if num_agents > 1:
        is_av[1] = True

    focal_idx = 0
    last_obs = cfg.num_history_steps - 1
    order = F._select_agents(positions, valid, focal_idx, last_obs, cfg.max_agents, is_av=is_av)
    return positions, headings, velocities, valid, type_ids, is_av, order, focal_idx


def generate_sample(index: int, cfg: FeatureConfig) -> F.Sample:
    """Deterministically generate one synthetic :data:`features.Sample` from ``index``."""
    (positions, headings, velocities, valid,
     type_ids, is_av, order, focal_idx) = _raw_world_arrays(index, cfg)
    rng = np.random.default_rng(index + 1)  # reused only for lane geometry below
    last_obs = cfg.num_history_steps - 1
    origin = positions[focal_idx, last_obs].astype(np.float32)
    R = F.rotation_world_to_agent(float(headings[focal_idx, last_obs]))

    sample = F._assemble_sample(positions, headings, velocities, valid, order, origin, R, cfg,
                                type_ids=type_ids, is_av=is_av)
    sample["lanes"], sample["lane_mask"] = _make_lanes(rng, origin, R, cfg)
    sample["scenario_id"] = f"synthetic-{index:06d}"
    return sample


def raw_world_scene(index: int, cfg: FeatureConfig) -> dict:
    """Full world-frame agent trajectories for synthetic scenario ``index`` (for playback)."""
    positions, _, _, valid, type_ids, is_av, order, _ = _raw_world_arrays(index, cfg)
    return F.assemble_playback(positions, valid, type_ids, is_av, order, cfg)
