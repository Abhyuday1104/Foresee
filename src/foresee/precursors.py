"""Detect near-miss precursors in observed trajectories.

The detectors mirror the crash types found in the NHTSA data (see INSIGHTS.md): closing on
a slower lead without braking, lane departure, a neighbour cutting in, a vehicle turning
across the AV's path. Kinematics only, no learned model involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .intent import scene_intents
from .risk import agent_radius

# Detector thresholds (SI). Legible assumptions, not fitted parameters.
TTC_THRESHOLD_S = 4.0        # time-to-collision below which "closing" is a precursor
FORWARD_CONE_COS = 0.906     # cos(25°): how "ahead" a lead must be
MAX_LEAD_DIST_M = 45.0
LANE_HALF_WIDTH_M = 2.0      # lead must be within ~half a lane laterally (in the AV's path)
SWERVE_LAT_ACCEL = 3.5       # m/s^2 lateral acceleration that counts as a swerve / departure
CUTIN_DIST_M = 25.0
TURN_CROSS_DIST_M = 30.0

# NHTSA real-crash mix (from INSIGHTS.md), for comparison reporting.
NHTSA_MIX = {
    "Closing on lead / fixed object": 0.47,
    "Lane departure / swerve": 0.38,
    "Turn across path": 0.14,
    "Lane change / merge": 0.01,
}


@dataclass
class Precursor:
    category: str                 # matches the NHTSA / Foresee conflict taxonomy
    severity: float               # 0..1 screening score
    protagonist: int              # the AV / focal agent index
    threat: Optional[int]         # the other agent involved (or None for self-departure)
    ttc_s: Optional[float]
    description: str              # plain-language, includes the other vehicle's intent


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


def _ego_kinematics(tracks, valid, ego, anchor, dt):
    """Return (pos, heading_dir, speed) for the protagonist at the anchor step."""
    idx = np.where(valid[ego, : anchor + 1])[0]
    pos = tracks[ego, idx[-1]]
    v = np.diff(tracks[ego, idx[-4:]], axis=0) / dt if idx.size >= 2 else np.zeros((1, 2))
    vmean = v.mean(axis=0)
    return pos, _unit(vmean), float(np.linalg.norm(vmean))


def _lateral_accel(tracks, valid, a, anchor, dt, window=12) -> float:
    """Peak lateral acceleration magnitude over the recent window (swerve detector)."""
    idx = np.where(valid[a, : anchor + 1])[0][-window:]
    if idx.size < 5:
        return 0.0
    p = tracks[a, idx]
    v = np.diff(p, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    head = np.arctan2(v[:, 1], v[:, 0])
    dhead = np.diff(np.unwrap(head)) / dt                # yaw rate
    lat = np.abs(dhead) * speed[1:]                      # a_lat = v * yaw_rate
    return float(np.max(lat)) if lat.size else 0.0


def find_precursors(scene: Dict[str, object]) -> List[Precursor]:
    """Detect crash-precursor setups in one scenario at its observation cutoff.

    ``scene`` keys: tracks (A,T,2), valid (A,T), object_types (A,), is_ego (A,), obs_len, rate.
    The protagonist is the AV (ego) if present, else the focal agent (index 0).
    """
    tracks = np.asarray(scene["tracks_world"])
    valid = np.asarray(scene["valid"])
    types = np.asarray(scene["object_types"])
    is_ego = np.asarray(scene["is_ego"])
    anchor = scene["obs_len"] - 1
    dt = 1.0 / scene["sample_rate_hz"]

    ego = int(np.where(is_ego)[0][0]) if is_ego.any() else 0
    if not valid[ego, anchor]:
        return []
    intents = scene_intents(tracks, valid, anchor, dt)
    ego_intent = intents[ego]
    ego_pos, ego_dir, ego_speed = _ego_kinematics(tracks, valid, ego, anchor, dt)
    out: List[Precursor] = []

    # --- 1. Closing on a lead / fixed object (the #1 real crash type) ---
    # The dangerous pattern is not "approaching a lead" (that is normal car-following) - it is the
    # AV *failing to react*: closing on a slower / braking / stopped lead while the AV itself is
    # NOT braking. Requiring the AV to be non-braking separates genuine near-misses from the
    # everyday following that floods a naive TTC detector. Uses both vehicles' intent.
    av_reacting = ego_intent.is_braking()
    best = None
    for a in range(tracks.shape[0]):
        if a == ego or not valid[a, anchor] or not intents[a].valid:
            continue
        rel = tracks[a, anchor] - ego_pos
        dist = float(np.linalg.norm(rel))
        if dist < 1e-2 or dist > MAX_LEAD_DIST_M:
            continue
        if float(_unit(rel) @ ego_dir) < FORWARD_CONE_COS:    # must be ahead of the AV
            continue
        # In-lane gate: the lead must be roughly in the AV's path, not a parked / adjacent-lane
        # vehicle that merely falls inside the forward cone. Lateral offset < ~half a lane.
        perp_dir = np.array([-ego_dir[1], ego_dir[0]])
        if abs(float(rel @ perp_dir)) > LANE_HALF_WIDTH_M:
            continue
        lead_vec = intents[a].speed * np.array([np.cos(intents[a].heading), np.sin(intents[a].heading)])
        closing = ego_speed - float(lead_vec @ ego_dir)       # closing speed along AV heading
        if closing < 0.5:
            continue
        gap = max(dist - agent_radius(int(types[a])) - agent_radius(int(types[ego])), 0.1)
        ttc = gap / closing
        if ttc > TTC_THRESHOLD_S:
            continue
        # Failure-to-react gate: skip if the AV is already braking for this situation.
        if av_reacting:
            continue
        slower_lead = intents[a].is_braking() or intents[a].speed < 0.6 * ego_speed
        if not slower_lead:                                   # only a real hazard vs a slower lead
            continue
        sev = float(np.clip(1 - ttc / TTC_THRESHOLD_S, 0, 1))
        if intents[a].is_braking():
            sev = min(1.0, sev + 0.2)
        if best is None or sev > best.severity:
            best = Precursor("Closing on lead / fixed object", sev, ego, a, round(ttc, 2),
                             f"AV not braking, closing on a {intents[a].label} lead at "
                             f"~{ego_speed*3.6:.0f} km/h (TTC {ttc:.1f}s)")
    if best:
        out.append(best)

    # --- 2. AV lane departure / swerve (the #2 real crash type) ---
    lat = _lateral_accel(tracks, valid, ego, anchor, dt)
    if lat > SWERVE_LAT_ACCEL and ego_speed > 3.0:
        sev = float(np.clip((lat - SWERVE_LAT_ACCEL) / 6.0, 0, 1))
        out.append(Precursor("Lane departure / swerve", sev, ego, None, None,
                             f"AV swerving / departing lane (lateral accel {lat:.1f} m/s2)"))

    # --- 3. Neighbour cutting in (lane change / merge) ---
    for a in range(tracks.shape[0]):
        if a == ego or not intents[a].valid or not intents[a].is_lane_changing():
            continue
        rel = tracks[a, anchor] - ego_pos
        dist = float(np.linalg.norm(rel))
        if dist > CUTIN_DIST_M or float(_unit(rel) @ ego_dir) < 0.3:
            continue                                           # roughly ahead / to the side-front
        # lateral velocity of the neighbour toward the AV's path?
        perp = np.array([-ego_dir[1], ego_dir[0]])
        side = float(rel @ perp)                               # which side the neighbour is on
        nv = intents[a].speed * np.array([np.cos(intents[a].heading), np.sin(intents[a].heading)])
        closing_lat = -np.sign(side) * float(nv @ perp)        # +ve => moving toward AV lane
        if closing_lat > 0.5:
            sev = float(np.clip(closing_lat / 3.0, 0, 1)) * float(np.clip(1 - dist / CUTIN_DIST_M, 0, 1))
            out.append(Precursor("Lane change / merge", sev, ego, a, None,
                                 f"{intents[a].label} neighbour cutting toward the AV's lane "
                                 f"({dist:.0f} m ahead)"))
            break

    # --- 4. Vehicle turning across the AV's path ---
    for a in range(tracks.shape[0]):
        if a == ego or not intents[a].valid or not intents[a].is_turning():
            continue
        rel = tracks[a, anchor] - ego_pos
        dist = float(np.linalg.norm(rel))
        if dist > TURN_CROSS_DIST_M or float(_unit(rel) @ ego_dir) < 0.2:
            continue
        sev = float(np.clip(1 - dist / TURN_CROSS_DIST_M, 0, 1)) * float(np.clip(ego_speed / 15, 0, 1))
        if sev > 0.05:
            out.append(Precursor("Turn across path", sev, ego, a, None,
                                 f"{intents[a].label} vehicle turning across the AV's path "
                                 f"({dist:.0f} m ahead)"))
            break

    return out

