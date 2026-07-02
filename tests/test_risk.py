"""Unit tests for conflict-risk logic on hand-constructed scenes."""

import numpy as np

from foresee.risk import assess_conflict, maneuver_label

RATE = 10.0
TF = 60


def test_maneuver_label_directions():
    straight = np.linspace([0, 0], [30, 0], TF)
    left = np.linspace([0, 0], [20, 20], TF)
    right = np.linspace([0, 0], [20, -20], TF)
    stop = np.linspace([0, 0], [1.0, 0], TF)
    assert maneuver_label(straight) == "Continuing straight"
    assert maneuver_label(left) == "Turning left"
    assert maneuver_label(right) == "Turning right"
    assert maneuver_label(stop) == "Slowing / stopping"


def _playback(pred_modes, probs, ego_path):
    """Minimal playback dict for assess_conflict (focal = agent 0, ego = agent 1)."""
    return {
        "pred_world": np.asarray(pred_modes, dtype=np.float32),
        "pred_agent": np.asarray(pred_modes, dtype=np.float32),
        "probabilities": np.asarray(probs, dtype=np.float32),
        "sample_rate_hz": RATE,
        "object_types": np.array([0, 10]),          # vehicle, ego
        "is_ego": np.array([False, True]),
        "focal_xy": np.zeros(2, dtype=np.float32),
        "n_model_steps": TF,
        "ego_future_full": np.asarray(ego_path, dtype=np.float32),
        "obs_len": 50,
        "tracks_world": np.zeros((2, 110, 2), dtype=np.float32),
        "valid": np.ones((2, 110), dtype=bool),
    }


def test_driving_into_stopped_ego_is_high_risk():
    t = np.arange(TF) / RATE
    toward_ego = np.stack([10.0 * t, np.zeros(TF)], axis=1)   # focal heads +x at 10 m/s
    veer_away = np.stack([10.0 * t, 8.0 * t], axis=1)
    ego = np.tile([20.0, 0.0], (TF, 1))                       # ego parked 20 m ahead
    c = assess_conflict(_playback([toward_ego, veer_away], [0.7, 0.3], ego))
    assert c["level"] == "HIGH"
    assert c["ttc_s"] is not None and c["ttc_s"] < 3.0
    assert abs(c["risk"] - 0.7) < 1e-5                        # only the 0.7 mode conflicts


def test_parallel_traffic_is_low_risk():
    t = np.arange(TF) / RATE
    lane_a = np.stack([10.0 * t, np.zeros(TF)], axis=1)
    ego = np.stack([10.0 * t, np.full(TF, 15.0)], axis=1)     # ego in a lane 15 m away
    c = assess_conflict(_playback([lane_a, lane_a], [0.6, 0.4], ego))
    assert c["level"] == "LOW"
    assert c["ttc_s"] is None


def test_preexisting_proximity_is_not_a_conflict():
    """Car-following at a constant close gap must not be flagged (no convergence)."""
    t = np.arange(TF) / RATE
    focal = np.stack([10.0 * t, np.zeros(TF)], axis=1)
    ego = np.stack([10.0 * t + 3.0, np.zeros(TF)], axis=1)    # constant 3 m gap
    c = assess_conflict(_playback([focal, focal], [0.5, 0.5], ego))
    assert c["level"] == "LOW"


def test_no_ego_returns_not_applicable():
    pb = _playback([np.zeros((TF, 2))], [1.0], np.zeros((TF, 2)))
    pb["is_ego"] = np.array([False, False])
    c = assess_conflict(pb)
    assert c["level"] == "NO_EGO" and c["risk"] == 0.0
