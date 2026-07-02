"""Parse AV2 scenarios into model features.

Reads the scenario parquet and HD-map json through the av2 API and produces an agent-centric
Sample dict (focal agent at the origin, heading along +x). The synthetic generator emits the
same format, so everything downstream is agnostic to the data source.

av2 imports are deferred so this module also works without the package installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import FeatureConfig

# Canonical object-type ordering. Indices are used by the model's type embedding and the
# dashboard's type-coded rendering. "ego" is a synthetic class we assign to the AV (the
# data-collection vehicle, track_id == "AV") regardless of its raw ObjectType; "unknown" is
# the fallback for any raw type not in this list.
OBJECT_TYPE_NAMES = [
    "vehicle", "pedestrian", "motorcyclist", "cyclist", "bus",
    "static", "background", "construction", "riderless_bicycle", "unknown", "ego",
]
NAME_TO_ID = {name: i for i, name in enumerate(OBJECT_TYPE_NAMES)}
UNKNOWN_TYPE_ID = NAME_TO_ID["unknown"]
EGO_TYPE_ID = NAME_TO_ID["ego"]

# A Sample is a dict of NumPy arrays / scalars. Keys & shapes (with cfg = FeatureConfig):
#   hist          (A, T_obs, Fa)  float32   per-agent history, agent-centric
#   hist_mask     (A,)            bool       agent slot occupied (has >=1 observed step)
#   object_types  (A,)            int64      semantic class per agent (see OBJECT_TYPE_NAMES)
#   is_ego        (A,)            bool        True for the AV (data-collection vehicle)
#   cur_pos       (A, 2)          float32    each agent's position at the anchor step (agent frame)
#   lanes         (L, P, Fl)      float32    lane centerlines, agent-centric
#   lane_mask     (L,)            bool        lane slot occupied
#   future        (T_fut, 2)      float32    focal future positions, agent-centric
#   future_mask   (T_fut,)        bool        focal future step observed
#   origin        (2,)            float32    focal world position at last observed step
#   rotation      (2, 2)          float32    world->agent rotation R (agent = R @ (world-origin))
#   scenario_id   str
Sample = Dict[str, object]


# --------------------------------------------------------------------------------------
# Geometry helpers (agent-centric framing)
# --------------------------------------------------------------------------------------
def rotation_world_to_agent(heading: float) -> np.ndarray:
    """Rotation matrix R that maps a *world* vector into the agent frame.

    The agent frame's +x axis points along ``heading``. ``agent = R @ (world - origin)``
    and the inverse is ``world = origin + R.T @ agent``.
    """
    c, s = np.cos(heading), np.sin(heading)
    # Rotating the world by -heading aligns the heading direction with +x:
    #   R @ (cos h, sin h) = (1, 0)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def to_agent_frame(points_xy: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply ``agent = R @ (world - origin)`` to an array of points (..., 2)."""
    return (points_xy - origin) @ R.T


def to_world_frame(points_xy: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_agent_frame`: ``world = origin + R.T @ agent``."""
    return points_xy @ R + origin


def resample_polyline(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to ``num_points`` equally spaced (by arc length) points.

    Returns an array of shape ``(num_points, 2)``. Robust to duplicate/degenerate inputs.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] == 1:
        pts = np.repeat(pts, 2, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:  # degenerate (all points coincide) -> just tile
        return np.repeat(pts[:1], num_points, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, num_points)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def polyline_to_features(centerline_agent: np.ndarray) -> np.ndarray:
    """Build per-point lane features [x, y, dir_x, dir_y] from an agent-frame centerline.

    The tangent (dir) encodes lane orientation, which is what a planner cares about.
    """
    xy = centerline_agent
    tangents = np.zeros_like(xy)
    tangents[:-1] = np.diff(xy, axis=0)
    tangents[-1] = tangents[-2] if len(xy) > 1 else 0.0
    norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    tangents = tangents / norm
    return np.concatenate([xy, tangents], axis=1).astype(np.float32)  # (P, 4)


# --------------------------------------------------------------------------------------
# Raw AV2 track extraction
# --------------------------------------------------------------------------------------
def _object_type_id(track) -> int:
    """Map an av2 ``ObjectType`` to our canonical class id (falls back to 'unknown')."""
    raw = getattr(track.object_type, "value", track.object_type)
    return NAME_TO_ID.get(str(raw).lower(), UNKNOWN_TYPE_ID)


def _raw_tracks_from_scenario(scenario, cfg: FeatureConfig):
    """Return dense per-track arrays plus semantics and the focal track index.

    Output arrays are world-frame, NaN-filled where a track has no state at a timestep:
        positions  (N, T_total, 2)
        headings   (N, T_total)
        velocities (N, T_total, 2)
        valid      (N, T_total)  bool
        type_ids   (N,)          int   canonical object-type id
        is_av      (N,)          bool  True for the AV (track_id == "AV")
        focal_idx  int
    """
    T = cfg.num_total_steps
    tracks = scenario.tracks
    n = len(tracks)
    positions = np.full((n, T, 2), np.nan, dtype=np.float32)
    headings = np.full((n, T), np.nan, dtype=np.float32)
    velocities = np.full((n, T, 2), np.nan, dtype=np.float32)
    valid = np.zeros((n, T), dtype=bool)
    type_ids = np.full((n,), UNKNOWN_TYPE_ID, dtype=np.int64)
    is_av = np.zeros((n,), dtype=bool)

    focal_idx = 0
    for i, track in enumerate(tracks):
        if track.track_id == scenario.focal_track_id:
            focal_idx = i
        type_ids[i] = _object_type_id(track)
        is_av[i] = track.track_id == "AV"
        for st in track.object_states:
            t = st.timestep
            if 0 <= t < T:
                positions[i, t] = st.position
                headings[i, t] = st.heading
                velocities[i, t] = st.velocity
                valid[i, t] = True
    return positions, headings, velocities, valid, type_ids, is_av, focal_idx


def _select_agents(positions, valid, focal_idx, last_obs: int, max_agents: int,
                   is_av=None) -> List[int]:
    """Order agents: focal first, then the ego (AV), then nearest neighbours.

    The ego is force-included when present so the downstream risk module always has the AV's
    path to assess conflicts against (it would otherwise be dropped when far from the focal).
    """
    focal_pos = positions[focal_idx, last_obs]
    order = [focal_idx]
    # Force-include the ego/AV right after the focal, if it exists and is observed.
    if is_av is not None:
        for i in np.where(is_av)[0]:
            if i != focal_idx and valid[i, last_obs] and i not in order:
                order.append(int(i))
                break
    candidates = []
    for i in range(positions.shape[0]):
        if i in order or not valid[i, last_obs]:
            continue
        d = float(np.linalg.norm(positions[i, last_obs] - focal_pos))
        candidates.append((d, i))
    candidates.sort(key=lambda x: x[0])
    order += [i for _, i in candidates[: max_agents - len(order)]]
    return order


# --------------------------------------------------------------------------------------
# Public loaders (require av2)
# --------------------------------------------------------------------------------------
def load_scenario(parquet_path: Path):
    """Load an AV2 scenario from its parquet file. Requires the ``av2`` package."""
    try:
        from av2.datasets.motion_forecasting import scenario_serialization
    except ImportError as e:  # pragma: no cover - exercised only with real data
        raise ImportError(
            "The 'av2' package is required to read real scenarios. "
            "Install it with `pip install av2`, or use the synthetic generator."
        ) from e
    return scenario_serialization.load_argoverse_scenario_parquet(Path(parquet_path))


def load_static_map(map_json_path: Path):
    """Load an AV2 HD map from its ``log_map_archive_*.json``. Requires ``av2``."""
    try:
        from av2.map.map_api import ArgoverseStaticMap
    except ImportError as e:  # pragma: no cover
        raise ImportError("The 'av2' package is required to read HD maps.") from e
    return ArgoverseStaticMap.from_json(Path(map_json_path))


def extract_lane_features(
    static_map, origin: np.ndarray, R: np.ndarray, cfg: FeatureConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract up to ``cfg.max_lanes`` nearby lane centerlines in the agent frame.

    Returns ``(lanes, lane_mask)`` with shapes ``(L, P, Fl)`` and ``(L,)``.
    """
    L, P, Fl = cfg.max_lanes, cfg.lane_num_points, cfg.lane_feature_dim
    lanes = np.zeros((L, P, Fl), dtype=np.float32)
    lane_mask = np.zeros((L,), dtype=bool)

    # Query lanes within the map radius of the focal agent's world position.
    try:
        nearby = static_map.get_nearby_lane_segments(origin.astype(np.float64), cfg.map_radius_m)
    except Exception:
        nearby = list(getattr(static_map, "vector_lane_segments", {}).values())

    # Sort by distance of the centerline midpoint to the focal agent (closest first).
    scored = []
    for ls in nearby:
        try:
            centerline = static_map.get_lane_segment_centerline(ls.id)[:, :2]
        except Exception:
            continue
        if centerline.shape[0] < 2:
            continue
        mid = centerline[len(centerline) // 2]
        scored.append((float(np.linalg.norm(mid - origin)), centerline))
    scored.sort(key=lambda x: x[0])

    for slot, (_, centerline) in enumerate(scored[:L]):
        cl_agent = to_agent_frame(centerline.astype(np.float32), origin, R)
        cl_agent = resample_polyline(cl_agent, P)
        lanes[slot] = polyline_to_features(cl_agent)
        lane_mask[slot] = True

    return lanes, lane_mask


def extract_render_map(static_map, origin: np.ndarray, radius: float = 70.0) -> Dict[str, list]:
    """Extract rich HD-map geometry (for *rendering*, not the model) near ``origin``.

    Returns world-frame polylines/polygons: filled drivable areas, lane boundaries (the actual
    road edges, not just centerlines), and pedestrian crossings. Everything within ``radius``
    of the focal agent. Defensive against av2 version differences.
    """
    out: Dict[str, list] = {"drivable": [], "lane_boundaries": [], "centerlines": [],
                            "crosswalks": []}

    def near(xy: np.ndarray) -> bool:
        return bool(np.min(np.linalg.norm(xy - origin, axis=1)) <= radius)

    for da in getattr(static_map, "vector_drivable_areas", {}).values():
        try:
            xy = np.asarray(da.xyz)[:, :2]
            if near(xy):
                out["drivable"].append(xy.astype(np.float32))
        except Exception:
            pass

    for ls in getattr(static_map, "vector_lane_segments", {}).values():
        for bnd in (getattr(ls, "left_lane_boundary", None), getattr(ls, "right_lane_boundary", None)):
            try:
                xy = np.asarray(bnd.xyz)[:, :2]
                if near(xy):
                    out["lane_boundaries"].append(xy.astype(np.float32))
            except Exception:
                pass
        try:  # lane centerline (rendered as a dashed lane divider)
            cl = np.asarray(static_map.get_lane_segment_centerline(ls.id))[:, :2]
            if near(cl):
                out["centerlines"].append(cl.astype(np.float32))
        except Exception:
            pass

    for pc in getattr(static_map, "vector_pedestrian_crossings", {}).values():
        try:
            e1 = np.asarray(pc.edge1.xyz)[:, :2]
            e2 = np.asarray(pc.edge2.xyz)[:, :2]
            if near(e1) or near(e2):
                # Close the two edges into a polygon (e1 -> e2 reversed).
                poly = np.concatenate([e1, e2[::-1]], axis=0)
                out["crosswalks"].append(poly.astype(np.float32))
        except Exception:
            pass

    return out


def sample_from_raw(positions, headings, velocities, valid, type_ids, is_av, focal_idx,
                    static_map, cfg: FeatureConfig, scenario_id: str = "") -> Sample:
    """Build an agent-centric Sample for an *arbitrary* focal agent from already-parsed tracks.

    Factored out so the same scene can be re-framed around any agent (used to predict the intent
    of surrounding vehicles, not just the dataset's designated focal).
    """
    last_obs = cfg.num_history_steps - 1
    if not valid[focal_idx, last_obs]:
        obs_valid = np.where(valid[focal_idx, : cfg.num_history_steps])[0]
        if len(obs_valid) == 0:
            raise ValueError("Focal agent has no observed states")
        last_obs = int(obs_valid[-1])

    origin = positions[focal_idx, last_obs].astype(np.float32)
    R = rotation_world_to_agent(float(headings[focal_idx, last_obs]))
    order = _select_agents(positions, valid, focal_idx, last_obs, cfg.max_agents, is_av=is_av)
    sample = _assemble_sample(positions, headings, velocities, valid, order, origin, R, cfg,
                              type_ids=type_ids, is_av=is_av)
    if static_map is not None:
        sample["lanes"], sample["lane_mask"] = extract_lane_features(static_map, origin, R, cfg)
    sample["scenario_id"] = scenario_id
    return sample


def scenario_to_sample(parquet_path: Path, map_json_path: Path, cfg: FeatureConfig) -> Sample:
    """End-to-end: parquet + map JSON -> agent-centric :data:`Sample`."""
    scenario = load_scenario(parquet_path)
    (positions, headings, velocities, valid,
     type_ids, is_av, focal_idx) = _raw_tracks_from_scenario(scenario, cfg)
    static_map = load_static_map(map_json_path)
    return sample_from_raw(positions, headings, velocities, valid, type_ids, is_av, focal_idx,
                           static_map, cfg, scenario.scenario_id)


def assemble_playback(positions, valid, type_ids, is_av, order, cfg) -> Dict[str, np.ndarray]:
    """Pack selected agents' FULL (110-step) world trajectories for animated playback.

    Unlike :func:`_assemble_sample` (which keeps only the 50-step observed window + the focal's
    future, in the agent frame), this keeps every agent's complete world-frame path so the
    dashboard can animate the scene timestep-by-timestep.
    """
    A, T = cfg.max_agents, cfg.num_total_steps
    tracks = np.full((A, T, 2), np.nan, dtype=np.float32)
    valid_out = np.zeros((A, T), dtype=bool)
    object_types = np.full((A,), UNKNOWN_TYPE_ID, dtype=np.int64)
    is_ego = np.zeros((A,), dtype=bool)
    for slot, idx in enumerate(order):
        v = valid[idx]
        tracks[slot, v] = positions[idx, v]
        valid_out[slot] = v
        if bool(is_av[idx]):
            object_types[slot] = EGO_TYPE_ID
            is_ego[slot] = True
        else:
            object_types[slot] = int(type_ids[idx])
    return {"tracks_world": tracks, "valid": valid_out,
            "object_types": object_types, "is_ego": is_ego}


def raw_world_scene(parquet_path: Path, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    """Parse a real scenario into full world-frame agent trajectories for playback."""
    scenario = load_scenario(parquet_path)
    (positions, headings, velocities, valid,
     type_ids, is_av, focal_idx) = _raw_tracks_from_scenario(scenario, cfg)
    last_obs = cfg.num_history_steps - 1
    if not valid[focal_idx, last_obs]:
        obs_valid = np.where(valid[focal_idx, : cfg.num_history_steps])[0]
        last_obs = int(obs_valid[-1]) if len(obs_valid) else last_obs
    order = _select_agents(positions, valid, focal_idx, last_obs, cfg.max_agents, is_av=is_av)
    return assemble_playback(positions, valid, type_ids, is_av, order, cfg)


def _assemble_sample(positions, headings, velocities, valid, order, origin, R, cfg,
                     type_ids=None, is_av=None) -> Sample:
    """Pack selected agents into fixed-size agent-frame tensors. Shared by AV2 & synthetic."""
    A, T_obs, Fa = cfg.max_agents, cfg.num_history_steps, cfg.agent_feature_dim
    T_fut = cfg.num_future_steps

    hist = np.zeros((A, T_obs, Fa), dtype=np.float32)
    hist_mask = np.zeros((A,), dtype=bool)
    object_types = np.full((A,), UNKNOWN_TYPE_ID, dtype=np.int64)
    is_ego = np.zeros((A,), dtype=bool)
    cur_pos = np.zeros((A, 2), dtype=np.float32)
    future = np.zeros((T_fut, 2), dtype=np.float32)
    future_mask = np.zeros((T_fut,), dtype=bool)

    for slot, idx in enumerate(order):
        # ----- history (observed window) -----
        last_valid_t = None
        for t in range(T_obs):
            if not valid[idx, t]:
                continue
            xy = to_agent_frame(positions[idx, t], origin, R)
            v = velocities[idx, t] @ R.T               # rotate velocity (no translation)
            h = float(headings[idx, t]) - np.arctan2(R[0, 1], R[0, 0])  # heading - focal heading
            hist[slot, t] = [xy[0], xy[1], v[0], v[1], np.sin(h), np.cos(h), 1.0]
            last_valid_t = t
        hist_mask[slot] = bool(valid[idx, :T_obs].any())
        if last_valid_t is not None:
            cur_pos[slot] = hist[slot, last_valid_t, :2]
        # ----- semantics -----
        if is_av is not None and bool(is_av[idx]):
            object_types[slot] = EGO_TYPE_ID
            is_ego[slot] = True
        elif type_ids is not None:
            object_types[slot] = int(type_ids[idx])

    # ----- focal future (slot 0) -----
    focal = order[0]
    for j in range(T_fut):
        t = cfg.num_history_steps + j
        if t < cfg.num_total_steps and valid[focal, t]:
            future[j] = to_agent_frame(positions[focal, t], origin, R)
            future_mask[j] = True

    return {
        "hist": hist,
        "hist_mask": hist_mask,
        "object_types": object_types,
        "is_ego": is_ego,
        "cur_pos": cur_pos,
        "lanes": np.zeros((cfg.max_lanes, cfg.lane_num_points, cfg.lane_feature_dim), np.float32),
        "lane_mask": np.zeros((cfg.max_lanes,), dtype=bool),
        "future": future,
        "future_mask": future_mask,
        "origin": origin.astype(np.float32),
        "rotation": R.astype(np.float32),
        "scenario_id": "",
    }
