"""Build world-frame scenes for the dashboard: full 110-step tracks for animation plus the
model's predictions, extended a few seconds past the supervised horizon at constant
velocity so the conflict check can tell a sustained approach from a brief crossing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .data import features as F
from .data import synthetic
from .inference import run_inference

# Seconds of constant-velocity extrapolation appended beyond the model's supervised horizon.
# AV2 has only 6 s of ground truth, so we cannot *train* a longer prediction - but extending
# each mode (and the AV) kinematically lets the conflict logic see whether a close approach is
# sustained or whether the agents simply pass and separate.
EXTRAPOLATE_SECONDS = 4.0


def kinematic_extend(path: np.ndarray, n_extra: int, smooth: int = 5) -> np.ndarray:
    """Extend a trajectory by ``n_extra`` constant-velocity steps (velocity = recent average)."""
    if n_extra <= 0 or path.shape[0] < 2:
        return path
    k = min(smooth, path.shape[0] - 1)
    step = np.diff(path[-(k + 1):], axis=0).mean(axis=0)          # average recent per-step delta
    tail = path[-1] + step[None, :] * np.arange(1, n_extra + 1)[:, None]
    return np.concatenate([path, tail], axis=0)


def _ego_full_future(raw, obs_len: int, n_total: int):
    """The AV's future over ``n_total`` steps: real where logged, extrapolated beyond. Or None."""
    is_ego = np.asarray(raw["is_ego"])
    if not is_ego.any():
        return None
    es = int(np.where(is_ego)[0][0])
    ev = raw["valid"][es, obs_len:]
    ea = raw["tracks_world"][es, obs_len:]
    if not ev.any():
        return None
    last = int(np.where(ev)[0][-1])
    known = ea[: last + 1]                                        # logged portion (contiguous)
    return kinematic_extend(known, n_total - known.shape[0])[:n_total]


def raw_scene(dataset, idx: int) -> Dict[str, object]:
    """Lightweight world-frame scene for precursor mining - no model / forecast needed."""
    if dataset.is_synthetic:
        raw = synthetic.raw_world_scene(dataset.synthetic_offset + idx, dataset.cfg)
        sid = f"synthetic-{dataset.synthetic_offset + idx:06d}"
    else:
        parquet, _ = dataset.scenario_paths[idx]
        raw = F.raw_world_scene(parquet, dataset.cfg)
        sid = parquet.parent.name
    return {
        "tracks_world": raw["tracks_world"],
        "valid": raw["valid"],
        "object_types": raw["object_types"],
        "is_ego": raw["is_ego"],
        "obs_len": dataset.cfg.num_history_steps,
        "sample_rate_hz": dataset.cfg.sample_rate_hz,
        "scenario_id": sid,
    }


def build_playback(dataset, idx: int, model, device: str = "cpu") -> Dict[str, object]:
    """Return a playback scene dict for ``dataset[idx]`` predicted by ``model``."""
    sample = dataset[idx]
    scene = run_inference(model, sample, device=device)  # world-frame preds + lanes

    render_map = None
    if dataset.is_synthetic:
        raw = synthetic.raw_world_scene(dataset.synthetic_offset + idx, dataset.cfg)
    else:
        parquet, map_json = dataset.scenario_paths[idx]
        raw = F.raw_world_scene(parquet, dataset.cfg)
        try:
            static_map = F.load_static_map(map_json)
            render_map = F.extract_render_map(static_map, scene["focal_xy"])
        except Exception:
            render_map = None

    # --- Extend predictions (and the AV) beyond the 6 s model horizon ---
    rate = dataset.cfg.sample_rate_hz
    obs_len = dataset.cfg.num_history_steps
    pred_model = scene["pred_trajectories"]                       # (K, Tf, 2)
    n_model = pred_model.shape[1]
    n_extra = int(EXTRAPOLATE_SECONDS * rate)
    pred_full = np.stack([kinematic_extend(pred_model[k], n_extra) for k in range(pred_model.shape[0])])
    ego_full = _ego_full_future(raw, obs_len, n_model + n_extra)

    return {
        # Full world-frame trajectories for animation.
        "tracks_world": raw["tracks_world"],        # (A, T_total, 2)  NaN where unobserved
        "valid": raw["valid"],                      # (A, T_total) bool
        "object_types": raw["object_types"],        # (A,) class ids
        "is_ego": raw["is_ego"],                    # (A,) bool
        "obs_len": obs_len,                         # observation cutoff (prediction time)
        "total_len": dataset.cfg.num_total_steps,
        "sample_rate_hz": rate,
        # Static map + model prediction (fixed; computed at the observation cutoff).
        "lanes": scene["lanes"],                    # (L, P, 2) world (centerlines fallback)
        "render_map": render_map,                   # rich HD-map geometry for rendering (or None)
        "pred_world": pred_full,                    # (K, Tf+E, 2) world (model + extrapolation)
        "n_model_steps": n_model,                   # first n_model steps are the supervised model
        "ego_future_full": ego_full,                # (Tf+E, 2) AV path over the full horizon
        "pred_agent": scene["pred_agent"],          # (K, Tf, 2) agent frame (for maneuvers)
        "probabilities": scene["probabilities"],    # (K,)
        "focal_xy": scene["focal_xy"],              # (2,) focal position at the cutoff
        "scenario_id": sample["scenario_id"],
    }


def view_bounds(playback: Dict[str, object], margin: float = 8.0):
    """Stable (xmin, xmax, ymin, ymax) covering all tracks/lanes/preds, so the camera is fixed."""
    pts = []
    tw = playback["tracks_world"][playback["valid"]]
    if tw.size:
        pts.append(tw.reshape(-1, 2))
    lanes = playback["lanes"].reshape(-1, 2)
    lanes = lanes[~np.isnan(lanes).any(axis=1)]
    if lanes.size:
        pts.append(lanes)
    pts.append(playback["pred_world"].reshape(-1, 2))
    allpts = np.concatenate(pts, axis=0)
    xmin, ymin = allpts.min(axis=0) - margin
    xmax, ymax = allpts.max(axis=0) + margin
    # Keep aspect square so motion isn't visually distorted.
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2
    return cx - half, cx + half, cy - half, cy + half
