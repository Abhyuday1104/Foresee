"""Classify what each agent is currently doing from its observed track: accelerating,
braking, stopped, turning, changing lanes. Rule-based on purpose, since it feeds the
precursor detectors and should stay easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Thresholds (SI units). Tightened so only clear maneuvers are flagged - a stated assumption set.
STOP_SPEED = 0.5          # m/s below which an agent is "stopped"
BRAKE_ACCEL = -1.6        # m/s^2 below which it is "braking" (stricter -> fewer false brakes)
ACCEL_ACCEL = 1.6         # m/s^2 above which it is "accelerating"
TURN_DEG = 28.0           # net heading change (deg) over the window => "turning"
LANE_CHANGE_LAT = 2.5     # lateral drift (m) perpendicular to heading => "lane change"
LANE_CHANGE_MAX_TURN = 12.0  # ...but only if heading stays ~straight (else it's a curve, not a LC)
MIN_MANEUVER_SPEED = 2.0  # below this, motion is too slow/noisy to call a turn or lane change


@dataclass
class Intent:
    valid: bool
    speed: float                  # m/s at the anchor
    heading: float                # rad
    longitudinal: str             # accelerating | cruising | braking | stopped | unknown
    lateral: str                  # straight | turning left | turning right | changing lanes L/R | unknown
    label: str                    # short human-readable summary

    def is_braking(self) -> bool:
        return self.longitudinal in ("braking", "stopped")

    def is_lane_changing(self) -> bool:
        return self.lateral.startswith("changing lanes")

    def is_turning(self) -> bool:
        return self.lateral.startswith("turning")


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from vector a to b in degrees (+ = left / counter-clockwise)."""
    ang = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    return float(np.degrees(np.arctan2(np.sin(ang), np.cos(ang))))


def estimate_intent(track_xy: np.ndarray, valid: np.ndarray, dt: float,
                    window: int = 20) -> Intent:
    """Classify one agent's intent from its observed trajectory.

    ``track_xy`` (T, 2) world positions; ``valid`` (T,) bool; ``dt`` seconds/step. Uses the last
    ``window`` valid steps (~2 s at 10 Hz).
    """
    idx = np.where(valid)[0]
    if idx.size < 3:
        return Intent(False, 0.0, 0.0, "unknown", "unknown", "unknown")
    idx = idx[-window:]
    pos = track_xy[idx]
    vel = np.diff(pos, axis=0) / dt                      # (n-1, 2)
    speed = np.linalg.norm(vel, axis=1)
    cur_speed = float(speed[-3:].mean())                 # smoothed current speed
    heading_vec = vel[-3:].mean(axis=0)
    heading = float(np.arctan2(heading_vec[1], heading_vec[0]))

    # Longitudinal: average acceleration across the window.
    accel = float((speed[-1] - speed[0]) / (len(speed) * dt))
    if cur_speed < STOP_SPEED:
        longitudinal = "stopped"
    elif accel < BRAKE_ACCEL:
        longitudinal = "braking"
    elif accel > ACCEL_ACCEL:
        longitudinal = "accelerating"
    else:
        longitudinal = "cruising"

    # Lateral: a turn is a sustained heading change; a lane change is a lateral *shift* while the
    # heading stays roughly straight (this separates a real lane change from a gentle curve).
    net_turn = _signed_angle(vel[0], vel[-1]) if cur_speed > STOP_SPEED else 0.0
    mean_dir = heading_vec / (np.linalg.norm(heading_vec) + 1e-6)
    perp = np.array([-mean_dir[1], mean_dir[0]])
    lateral_drift = float((pos[-1] - pos[0]) @ perp)     # signed lateral displacement
    if cur_speed < MIN_MANEUVER_SPEED:
        lateral = "straight"                             # too slow to reliably call a maneuver
    elif abs(net_turn) > TURN_DEG:
        lateral = "turning left" if net_turn > 0 else "turning right"
    elif abs(lateral_drift) > LANE_CHANGE_LAT and abs(net_turn) < LANE_CHANGE_MAX_TURN:
        lateral = "changing lanes left" if lateral_drift > 0 else "changing lanes right"
    else:
        lateral = "straight"

    label = longitudinal if lateral == "straight" else f"{longitudinal}, {lateral}"
    return Intent(True, cur_speed, heading, longitudinal, lateral, label)


def scene_intents(tracks: np.ndarray, valid: np.ndarray, anchor: int, dt: float,
                  window: int = 20) -> List[Intent]:
    """Estimate intent for every agent up to (and including) step ``anchor``. Returns list len A."""
    A = tracks.shape[0]
    return [estimate_intent(tracks[a, : anchor + 1], valid[a, : anchor + 1], dt, window)
            for a in range(A)]
