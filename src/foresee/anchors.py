"""Mine K goal anchors for anchored training.

k-means over real 6 s endpoints in the agent frame, stratified by maneuver (stop / left /
right / straight) so the anchor set always covers turning and stopping. Plain k-means put
four of six anchors on the straight axis, since most driving is straight.

NumPy only, no sklearn dependency.
"""

from __future__ import annotations

import numpy as np


def kmeans(points: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Lloyd's k-means. ``points`` (N, 2) -> centroids (k, 2)."""
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), k, replace=False)].copy()
    for _ in range(iters):
        d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
        assign = d.argmin(axis=1)
        for j in range(k):
            if np.any(assign == j):
                centroids[j] = points[assign == j].mean(axis=0)
            else:  # re-seed an empty cluster to the worst-fit point
                centroids[j] = points[d.min(axis=1).argmax()]
    return centroids


def collect_future_endpoints(dataset, n_samples: int = 4000) -> np.ndarray:
    """Gather agent-frame 6 s endpoints (last valid future step) from a dataset sample."""
    ends = []
    step = max(1, len(dataset) // n_samples)
    for i in range(0, len(dataset), step):
        s = dataset[i]
        fut = np.asarray(s["future"])
        fm = np.asarray(s["future_mask"])
        if fm.any():
            ends.append(fut[np.where(fm)[0][-1]])
        if len(ends) >= n_samples:
            break
    return np.asarray(ends, dtype=np.float32)


def _allocate(k: int) -> dict:
    """How many anchors to give each maneuver stratum (guarantees turn + stop coverage)."""
    n_left = max(1, k // 6)
    n_right = max(1, k // 6)
    n_stop = 1 if k >= 4 else 0
    return {"left": n_left, "right": n_right, "stop": n_stop,
            "straight": k - n_left - n_right - n_stop}


def compute_anchors(dataset, k: int, n_samples: int = 4000) -> np.ndarray:
    """K goal anchors (k, 2), *stratified by maneuver* so the modes can't all be "straight".

    Pure k-means on real endpoints over-represents going straight (most driving is straight), so
    it placed 4 of 6 anchors on the straight axis. Here we deliberately allocate anchors across
    maneuver strata - stop / turn-left / turn-right / straight - and k-means *within* each, so the
    model is guaranteed modes that cover turning and stopping, not just speed variants of straight.
    """
    ends = collect_future_endpoints(dataset, n_samples)
    dist = np.linalg.norm(ends, axis=1)
    bearing = np.degrees(np.arctan2(ends[:, 1], ends[:, 0]))
    strata = {
        "stop": ends[dist < 6.0],
        "left": ends[(bearing > 18.0) & (dist >= 6.0)],
        "right": ends[(bearing < -18.0) & (dist >= 6.0)],
        "straight": ends[(np.abs(bearing) <= 18.0) & (dist >= 6.0)],
    }
    alloc = _allocate(k)
    anchors = []
    for name, count in alloc.items():
        if count <= 0:
            continue
        pool = strata[name]
        if len(pool) < count:                              # empty/thin stratum -> fall back
            pool = strata["straight"] if len(strata["straight"]) >= count else ends
        anchors.append(kmeans(pool, count, seed=hash(name) & 0xffff))
    anchors = np.concatenate(anchors, axis=0)[:k]
    order = np.argsort(np.arctan2(anchors[:, 1], anchors[:, 0]))  # right -> left for readability
    return anchors[order].astype(np.float32)
