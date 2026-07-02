"""Predict intent for surrounding vehicles by re-framing the scene around each one and
re-running the forecaster. Different from intent.py, which only reads current motion.
One model call per agent, so it is run for a handful of nearby vehicles at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .data import features as F
from .data import synthetic
from .data.dataset import _sample_to_tensors
from .data.features import OBJECT_TYPE_NAMES
from .inference import run_inference
from .risk import maneuver_label

_VEHICLE_LIKE = {"vehicle", "bus", "motorcyclist", "cyclist"}


@dataclass
class AgentIntent:
    agent_idx: int            # index within the raw scene
    world_pos: np.ndarray     # (2,) position at the anchor step
    maneuver: str             # predicted maneuver ("Turning left", ...)
    prob: float               # probability mass on that maneuver
    type_name: str


def _raw_arrays(dataset, idx):
    """Return (positions, headings, velocities, valid, type_ids, is_av, focal_idx, static_map)."""
    cfg = dataset.cfg
    if dataset.is_synthetic:
        pos, head, vel, valid, tids, is_av, _order, focal = synthetic._raw_world_arrays(
            dataset.synthetic_offset + idx, cfg)
        return pos, head, vel, valid, tids, is_av, focal, None
    parquet, map_json = dataset.scenario_paths[idx]
    scenario = F.load_scenario(parquet)
    pos, head, vel, valid, tids, is_av, focal = F._raw_tracks_from_scenario(scenario, cfg)
    static_map = F.load_static_map(map_json)
    return pos, head, vel, valid, tids, is_av, focal, static_map


def predict_agent_intents(dataset, idx: int, model, n_agents: int = 5,
                          device: str = "cpu") -> List[AgentIntent]:
    """Predict the intent of the nearest ``n_agents`` vehicles around the AV in scenario ``idx``."""
    cfg = dataset.cfg
    pos, head, vel, valid, tids, is_av, focal, static_map = _raw_arrays(dataset, idx)
    last = cfg.num_history_steps - 1

    ego = int(np.where(is_av)[0][0]) if is_av.any() else focal
    ref = pos[ego, last] if valid[ego, last] else pos[focal, last]

    cands = []
    for i in range(pos.shape[0]):
        if i in (focal, ego) or not valid[i, last]:
            continue
        if OBJECT_TYPE_NAMES[int(tids[i])] not in _VEHICLE_LIKE:
            continue
        cands.append((float(np.linalg.norm(pos[i, last] - ref)), i))
    cands.sort()

    out: List[AgentIntent] = []
    for _, t in cands[:n_agents]:
        try:
            sample = F.sample_from_raw(pos, head, vel, valid, tids, is_av, t, static_map, cfg)
            scene = run_inference(model, _sample_to_tensors(sample), device=device)
        except Exception:
            continue
        probs, pred_agent = scene["probabilities"], scene["pred_agent"]
        mass = {}
        for k in range(len(probs)):
            m = maneuver_label(pred_agent[k])
            mass[m] = mass.get(m, 0.0) + float(probs[k])
        maneuver, prob = max(mass.items(), key=lambda kv: kv[1])
        out.append(AgentIntent(t, pos[t, last].astype(np.float32), maneuver, prob,
                               OBJECT_TYPE_NAMES[int(tids[t])]))
    return out
