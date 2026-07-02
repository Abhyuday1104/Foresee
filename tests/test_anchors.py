"""Unit tests for goal-anchor mining (k-means + maneuver stratification)."""

import numpy as np

from foresee.anchors import compute_anchors, kmeans


def test_kmeans_recovers_blob_centres():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]])
    pts = np.concatenate([c + rng.normal(scale=0.5, size=(200, 2)) for c in centres])
    got = kmeans(pts.astype(np.float32), 3, iters=50)
    # Each true centre should have a centroid within 1 m.
    for c in centres:
        assert np.min(np.linalg.norm(got - c, axis=1)) < 1.0


class _FakeDataset:
    """Duck-typed dataset yielding endpoints across all maneuver strata."""

    def __init__(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        ends = []
        for _ in range(n):
            r = rng.random()
            if r < 0.10:                                  # stop
                e = rng.normal(scale=1.0, size=2) + [2, 0]
            elif r < 0.25:                                # left turn
                e = [25, 22] + rng.normal(scale=2.0, size=2)
            elif r < 0.40:                                # right turn
                e = [25, -22] + rng.normal(scale=2.0, size=2)
            else:                                         # straight, varied speeds
                e = [rng.uniform(15, 80), rng.normal(scale=1.5)]
            ends.append(e)
        self._ends = np.asarray(ends, dtype=np.float32)

    def __len__(self):
        return len(self._ends)

    def __getitem__(self, i):
        future = np.zeros((60, 2), dtype=np.float32)
        future[-1] = self._ends[i]
        return {"future": future, "future_mask": np.ones(60, dtype=bool)}


def test_stratified_anchors_cover_all_maneuvers():
    anchors = compute_anchors(_FakeDataset(), k=6, n_samples=600)
    assert anchors.shape == (6, 2)
    bearing = np.degrees(np.arctan2(anchors[:, 1], anchors[:, 0]))
    dist = np.linalg.norm(anchors, axis=1)
    assert (bearing > 18).any(), "no left-turn anchor"
    assert (bearing < -18).any(), "no right-turn anchor"
    assert (dist < 6).any(), "no stop anchor"
    assert ((np.abs(bearing) <= 18) & (dist >= 6)).sum() >= 2, "too few straight anchors"
