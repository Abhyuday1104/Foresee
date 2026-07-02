"""Unit tests for the AV2 forecasting metrics on hand-constructed cases."""

import torch

from foresee.metrics import forecasting_metrics


def _case():
    """One sample, two modes: mode 1 is exact, mode 0 is offset by 1 m everywhere."""
    B, K, T = 1, 2, 4
    future = torch.zeros(B, T, 2)
    future[0, :, 0] = torch.arange(T, dtype=torch.float32)
    mask = torch.ones(B, T, dtype=torch.bool)
    traj = torch.zeros(B, K, T, 2)
    traj[0, 1] = future[0]
    traj[0, 0] = future[0] + torch.tensor([0.0, 1.0])
    logits = torch.zeros(B, K)  # uniform probabilities: 0.5 / 0.5
    return {"trajectories": traj, "logits": logits}, future, mask


def test_min_metrics_pick_the_exact_mode():
    pred, future, mask = _case()
    m = forecasting_metrics(pred, future, mask, miss_threshold_m=2.0)
    assert m["minADE"].item() == 0.0
    assert m["minFDE"].item() == 0.0
    assert m["MR"].item() == 0.0


def test_brier_adds_probability_penalty():
    pred, future, mask = _case()
    m = forecasting_metrics(pred, future, mask)
    # Best-endpoint mode has probability 0.5 -> brier-minFDE = 0 + (1 - 0.5)^2.
    assert abs(m["brier_minFDE"].item() - 0.25) < 1e-6


def test_miss_rate_thresholding():
    pred, future, mask = _case()
    pred["trajectories"] += 10.0  # push every mode beyond the 2 m threshold
    m = forecasting_metrics(pred, future, mask, miss_threshold_m=2.0)
    assert m["MR"].item() == 1.0


def test_fde_uses_last_valid_step():
    pred, future, mask = _case()
    mask[0, -1] = False                       # last step unobserved
    pred["trajectories"][0, 1, -1] += 100.0   # error on the unobserved step only
    m = forecasting_metrics(pred, future, mask)
    assert m["minFDE"].item() == 0.0
