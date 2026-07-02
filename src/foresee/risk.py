"""Turn predictions into a conflict verdict.

assess_conflict compares the focal agent's predicted futures against the ego vehicle's path
and reports a risk level, time-to-conflict, closest approach and a plain-language reason.
assess_intent measures how much probability sits on competing maneuvers. Thresholds are
stated constants; analysis/safety_scoreboard.py measures how they hold up against ground
truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data.features import OBJECT_TYPE_NAMES

# Approximate physical radius (m) per object class, used to size the danger zone. The conflict
# threshold between two agents is the sum of their radii plus a safety margin. These are tuned
# so the threshold sits *below* a lane width (~3.7 m) - otherwise two vehicles merely driving in
# adjacent lanes (a normal, safe situation) would be flagged. A "conflict" here means a genuine
# near-collision: predicted paths coming within roughly a car-length-and-a-half of each other.
_RADIUS = {
    "vehicle": 1.0, "bus": 2.0, "motorcyclist": 0.8, "cyclist": 0.6, "pedestrian": 0.4,
    "riderless_bicycle": 0.6, "ego": 1.0,
}
_DEFAULT_RADIUS = 0.8
SAFETY_MARGIN_M = 0.5          # extra buffer added to the summed radii
# A conflict must persist for at least this many consecutive steps (0.1 s each) - filters out
# transient clips from wandering low-probability modes - and the agents must start at least
# APPROACH_MARGIN_M beyond the danger zone, so we flag genuine *approaches*, not pre-existing
# proximity (car-following / side-by-side).
MIN_CONFLICT_STEPS = 4
APPROACH_MARGIN_M = 3.0
HIGH_RISK, MED_RISK = 0.30, 0.10
# Ambiguity thresholds on the probability mass held by the *second* distinct maneuver.
HIGH_AMBIG, MED_AMBIG = 0.35, 0.20


def agent_radius(type_id: int) -> float:
    name = OBJECT_TYPE_NAMES[type_id] if 0 <= type_id < len(OBJECT_TYPE_NAMES) else "unknown"
    return _RADIUS.get(name, _DEFAULT_RADIUS)


def maneuver_label(pred_agent_traj: np.ndarray) -> str:
    """Plain-language maneuver for one predicted mode (agent frame, +x = current heading)."""
    end = pred_agent_traj[-1]
    dist = float(np.hypot(end[0], end[1]))
    if dist < 3.0:
        return "Slowing / stopping"
    angle = float(np.degrees(np.arctan2(end[1], end[0])))  # +y is left of heading
    if angle > 20:
        return "Turning left"
    if angle < -20:
        return "Turning right"
    return "Continuing straight"


def _ego_future(playback: Dict[str, object]):
    """Return (ego_xy, ego_valid, ego_type_id) aligned with ``pred_world``, or (None, None, None).

    Prefers the extrapolated full-horizon AV path (``ego_future_full``) so conflicts can be
    judged over the same extended window as the predictions; falls back to the logged path.
    """
    is_ego = np.asarray(playback["is_ego"])
    if not is_ego.any():
        return None, None, None
    ego = int(np.where(is_ego)[0][0])
    ego_type = int(playback["object_types"][ego])

    full = playback.get("ego_future_full")
    if full is not None:
        full = np.asarray(full)
        return full, np.ones(full.shape[0], dtype=bool), ego_type

    obs_len = playback["obs_len"]
    Tf = playback["pred_world"].shape[1]
    tracks = np.asarray(playback["tracks_world"])
    valid = np.asarray(playback["valid"])
    return tracks[ego, obs_len:obs_len + Tf], valid[ego, obs_len:obs_len + Tf], ego_type


def _conflict_reason(path: np.ndarray, focal_xy: np.ndarray, ego_point, n_model: int) -> str:
    """Classify *why* a predicted mode conflicts: lane change, swerve, turn-across, or closing.

    Works in the agent's own heading frame: lateral deviation from a straight continuation tells
    a lane change / swerve apart from simply closing the gap on the AV ahead; a large heading
    change is a turn across the AV's path. "toward the AV" is decided by which side the AV is on.
    """
    p = path[:n_model]
    if p.shape[0] < 4:
        return "moving onto the AV's path"
    v0 = p[min(4, p.shape[0] - 1)] - p[0]                  # initial heading direction
    n = np.linalg.norm(v0)
    if n < 1e-3:
        return "pulling out toward the AV"
    v0 = v0 / n
    perp = np.array([-v0[1], v0[0]])                       # +left of heading
    lat = (p - focal_xy) @ perp                            # signed lateral deviation per step
    j = int(np.argmax(np.abs(lat)))
    max_lat = float(lat[j])
    final_dir = p[-1] - p[max(0, p.shape[0] - 4)]
    heading_change = float(np.degrees(np.arccos(
        np.clip(np.dot(v0, final_dir) / (np.linalg.norm(final_dir) + 1e-6), -1, 1))))
    toward = False
    if ego_point is not None:
        av_lat = float((np.asarray(ego_point) - focal_xy) @ perp)
        toward = (np.sign(max_lat) == np.sign(av_lat)) and abs(av_lat) > 1.0
    side = "left" if max_lat > 0 else "right"

    if heading_change > 35:
        return "turning across the AV's path"
    if abs(max_lat) > 4.0:
        return f"swerving {side}" + (" toward the AV" if toward else "")
    if abs(max_lat) > 1.5:
        return f"changing lanes {side}" + (" into the AV" if toward else "")
    return "closing on the AV ahead"


def assess_conflict(playback: Dict[str, object]) -> Dict[str, object]:
    """Probability-weighted conflict risk of the focal agent against the ego (AV) path."""
    preds = np.asarray(playback["pred_world"])             # (K, Tf, 2)
    probs = np.asarray(playback["probabilities"])          # (K,)
    rate = playback["sample_rate_hz"]
    focal_type = int(playback["object_types"][0])

    ego_xy, ego_valid, ego_type = _ego_future(playback)
    if ego_xy is None or ego_valid.sum() < 1:
        return {"has_ego": False, "risk": 0.0, "level": "NO_EGO", "ttc_s": None,
                "closest_m": float("nan"), "closest_t_s": None,
                "threat_type": OBJECT_TYPE_NAMES[focal_type], "action": "No ego vehicle in scene",
                "conflict_point": None, "ego_point": None, "primary_mode": None,
                "per_mode": [], "reason": None}

    threshold = agent_radius(focal_type) + agent_radius(ego_type) + SAFETY_MARGIN_M

    K, Tf, _ = preds.shape
    # t (seconds from "now") for each future step j: the j-th future step is at +(j+1)/rate.
    t_of_step = (np.arange(Tf) + 1) / rate
    valid_idx = np.where(ego_valid)[0]

    risk = 0.0
    overall_closest = (float("inf"), None, None)           # (dist, mode, step)
    conflicts = []                                         # (prob, ttc_s, mode, step, point)
    per_mode = []                                          # per-mode {conflicts, ttc_s, closest_m}
    for k in range(K):
        d = np.linalg.norm(preds[k] - ego_xy, axis=1)      # (Tf,)
        d = np.where(ego_valid, d, np.inf)
        j_min = int(np.argmin(d))
        if d[j_min] < overall_closest[0]:
            overall_closest = (float(d[j_min]), k, j_min)

        # A real collision course is SUSTAINED (the mode stays within the danger zone for
        # several consecutive steps, not a 1-frame clip of a wandering low-quality mode) and
        # CONVERGING (the agents approach - they aren't already side-by-side / car-following).
        d0 = float(d[valid_idx[0]])
        below = d < threshold
        j_breach = None
        for j in range(Tf - MIN_CONFLICT_STEPS + 1):
            if below[j:j + MIN_CONFLICT_STEPS].all():
                j_breach = j
                break
        hit = j_breach is not None and d0 >= threshold + APPROACH_MARGIN_M
        per_mode.append({"conflicts": bool(hit),
                         "ttc_s": float(t_of_step[j_breach]) if hit else None,
                         "closest_m": float(d[j_min])})
        if hit:
            risk += float(probs[k])
            conflicts.append((float(probs[k]), float(t_of_step[j_breach]), k, j_breach,
                              (float(preds[k, j_breach, 0]), float(preds[k, j_breach, 1]))))

    # Modes are probability-sorted (index 0 = most likely). HIGH requires the *most likely*
    # future to be a conflict - a hedge-mode brushing the AV is only worth MEDIUM. This sharply
    # improves precision: it rejects scenarios the model only weakly/uncertainly flags.
    conflicting_modes = {c[2] for c in conflicts}
    top_conflicts = 0 in conflicting_modes
    if risk >= HIGH_RISK and top_conflicts:
        level = "HIGH"
    elif risk >= MED_RISK:
        level = "MEDIUM"
    else:
        level = "LOW"
    action = {"HIGH": "AV should brake / yield",
              "MEDIUM": "AV should monitor closely",
              "LOW": "No action needed"}[level]

    closest_d, c_mode, c_step = overall_closest
    if conflicts:
        conflicts.sort(key=lambda c: -c[0])                # primary = most probable conflict
        _, ttc_s, prim_mode, prim_step, point = conflicts[0]
        ego_point = (float(ego_xy[prim_step, 0]), float(ego_xy[prim_step, 1]))
        n_model = int(playback.get("n_model_steps", Tf))
        reason = _conflict_reason(preds[prim_mode], np.asarray(playback["focal_xy"]),
                                  ego_point, n_model)
    else:
        ttc_s, prim_mode, point, ego_point, reason = None, c_mode, None, None, None

    return {
        "has_ego": True,
        "risk": float(risk),
        "level": level,
        "ttc_s": ttc_s,
        "closest_m": float(closest_d),
        "closest_t_s": float((c_step + 1) / rate),
        "threat_type": OBJECT_TYPE_NAMES[focal_type],
        "action": action,
        "conflict_point": point,        # threat agent's predicted position at the conflict
        "ego_point": ego_point,         # the AV's position at the same instant
        "primary_mode": int(prim_mode) if prim_mode is not None else None,
        "per_mode": per_mode,           # per-mode {conflicts, ttc_s, closest_m}
        "reason": reason,               # why: "changing lanes left into the AV", "swerving", ...
    }


def assess_intent(playback: Dict[str, object]) -> Dict[str, object]:
    """Quantify intent ambiguity from disagreement among predicted *maneuvers*.

    Ambiguity that matters for a planner is "could turn OR go straight", not "two modes both go
    straight with slightly different curvature". So we aggregate probability by maneuver class
    and judge ambiguity from how much mass the *second* distinct maneuver holds.
    """
    probs = np.asarray(playback["probabilities"], dtype=np.float64)
    pred_agent = np.asarray(playback["pred_agent"])
    K = len(probs)
    maneuvers = [maneuver_label(pred_agent[k]) for k in range(K)]

    # Probability mass per distinct maneuver class.
    mass: Dict[str, float] = {}
    for k, m in enumerate(maneuvers):
        mass[m] = mass.get(m, 0.0) + float(probs[k])
    ranked = sorted(mass.items(), key=lambda kv: -kv[1])
    m1, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0

    # Level is driven by how strong the competing maneuver is.
    level = "HIGH" if p2 >= HIGH_AMBIG else "MEDIUM" if p2 >= MED_AMBIG else "LOW"
    # Meter value: entropy over the maneuver mass, normalised by log(4) maneuver classes.
    q = np.array([v for _, v in ranked])
    entropy = float(-(q * np.log(q + 1e-12)).sum())
    entropy_norm = float(np.clip(entropy / np.log(4), 0.0, 1.0))

    if p2 >= MED_AMBIG:
        m2 = ranked[1][0]
        message = f"Could {m1.lower()} ({p1:.0%}) or {m2.lower()} ({p2:.0%})"
    else:
        message = f"Likely {m1.lower()} ({p1:.0%})"
    recommendation = {
        "HIGH": "Intent unclear - AV should be cautious and prepare to yield",
        "MEDIUM": "Some uncertainty - AV should monitor",
        "LOW": "Intent is clear",
    }[level]
    return {"entropy_norm": entropy_norm, "level": level, "message": message,
            "maneuvers": maneuvers, "recommendation": recommendation}


def describe_modes(playback: Dict[str, object], report: Optional[Dict] = None) -> List[str]:
    """One human-readable sentence per predicted mode: maneuver, speed, and AV relationship.

    Uses the model's supervised horizon for the maneuver/speed words, and the conflict report
    (if given) to say explicitly which mode is the flagged collision course.
    """
    pred_agent = np.asarray(playback["pred_agent"])         # (K, Tf, 2) - model horizon only
    probs = np.asarray(playback["probabilities"])
    rate = playback["sample_rate_hz"]
    horizon_s = pred_agent.shape[1] / rate
    per_mode = (report or {}).get("conflict", {}).get("per_mode") if report else None

    out = []
    for k in range(len(probs)):
        maneuver = maneuver_label(pred_agent[k])
        # Average speed along the supervised predicted path (km/h).
        path_len = float(np.linalg.norm(np.diff(pred_agent[k], axis=0), axis=1).sum())
        kmh = (path_len / horizon_s) * 3.6
        speed_word = ("stopping" if kmh < 3 else "slow" if kmh < 18 else
                      "moderate" if kmh < 40 else "fast")
        sentence = f"{maneuver}, {speed_word} (~{kmh:.0f} km/h)"
        if per_mode is not None and k < len(per_mode) and per_mode[k]["conflicts"]:
            sentence += f" - conflicts with AV in {per_mode[k]['ttc_s']:.1f}s"
        elif per_mode is not None and k < len(per_mode):
            sentence += f" - clear of AV (nearest {per_mode[k]['closest_m']:.0f} m)"
        out.append(sentence)
    return out


def assess(playback: Dict[str, object]) -> Dict[str, object]:
    """Full risk report = conflict + intent for one scenario."""
    return {"conflict": assess_conflict(playback), "intent": assess_intent(playback)}
