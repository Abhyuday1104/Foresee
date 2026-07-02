"""Run the model on one sample and convert the outputs back to world coordinates for
rendering.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _to_world(points_agent: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Agent-frame -> world: ``world = agent @ R + origin`` (inverse of the framing)."""
    return points_agent @ R + origin


@torch.no_grad()
def run_inference(model, sample: Dict[str, torch.Tensor], device: str = "cpu") -> Dict[str, np.ndarray]:
    """Run ``model`` on one (unbatched) sample dict of tensors.

    Returns a dict of NumPy arrays in world coordinates ready for plotting:
        pred_trajectories (K, T_fut, 2)   top-K predictions, sorted by descending probability
        probabilities     (K,)            mode probabilities (sorted to match)
        observed_tracks   (A, T_obs, 2)   observed agent histories (NaN where not observed)
        focal_future_gt   (T_fut, 2)      ground-truth future (NaN where unobserved)
        lanes             (L, P, 2)        lane centerlines (NaN where slot empty)
        focal_xy          (2,)            focal agent position at the anchor step
    """
    model.eval()
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
    pred = model(batch)

    logits = pred["logits"][0]                                   # (K,)
    temp = float(getattr(model, "temperature", torch.tensor(1.0)))  # calibrated confidence
    probs = torch.softmax(logits / max(temp, 1e-3), dim=-1).cpu().numpy()
    trajs_agent = pred["trajectories"][0].cpu().numpy()          # (K, T_fut, 2)

    origin = sample["origin"].cpu().numpy()
    R = sample["rotation"].cpu().numpy()

    # Sort modes by descending probability so "mode 1" is the most likely.
    order = np.argsort(-probs)
    probs = probs[order]
    trajs_agent = trajs_agent[order]
    pred_world = np.stack([_to_world(t, origin, R) for t in trajs_agent], axis=0)

    # --- Observed agent tracks (use the validity flag, last feature channel) ---
    hist = sample["hist"].cpu().numpy()                         # (A, T_obs, Fa)
    valid = hist[..., -1] > 0.5                                 # (A, T_obs)
    obs_xy = hist[..., :2].copy()
    obs_world = np.full_like(obs_xy, np.nan)
    A, T_obs, _ = obs_xy.shape
    for a in range(A):
        if valid[a].any():
            obs_world[a, valid[a]] = _to_world(obs_xy[a, valid[a]], origin, R)

    # --- Ground-truth future ---
    fut = sample["future"].cpu().numpy()                       # (T_fut, 2)
    fmask = sample["future_mask"].cpu().numpy()                # (T_fut,)
    fut_world = np.full_like(fut, np.nan)
    if fmask.any():
        fut_world[fmask] = _to_world(fut[fmask], origin, R)

    # --- Lanes ---
    lanes = sample["lanes"].cpu().numpy()                      # (L, P, Fl)
    lane_mask = sample["lane_mask"].cpu().numpy()              # (L,)
    lanes_world = np.full((lanes.shape[0], lanes.shape[1], 2), np.nan, np.float32)
    for li in range(lanes.shape[0]):
        if lane_mask[li]:
            lanes_world[li] = _to_world(lanes[li, :, :2], origin, R)

    return {
        "pred_trajectories": pred_world,
        "pred_agent": trajs_agent,                              # (K, T_fut, 2) agent frame
        "probabilities": probs,
        "observed_tracks": obs_world,
        "object_types": sample["object_types"].cpu().numpy(),   # (A,) class ids
        "is_ego": sample["is_ego"].cpu().numpy(),               # (A,) bool
        "focal_future_gt": fut_world,
        "lanes": lanes_world,
        "focal_xy": origin,
    }
