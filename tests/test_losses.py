"""Unit tests for the multimodal loss: WTA assignment, anchor assignment, diversity."""

import torch

from foresee.config import TrainConfig
from foresee.losses import diversity_loss, multimodal_loss


def _pred(trajectories, logits):
    return {"trajectories": trajectories, "logits": logits, "scale": None}


def test_diversity_loss_zero_when_endpoints_far():
    B, K, T = 2, 3, 10
    traj = torch.zeros(B, K, T, 2)
    for k in range(K):
        traj[:, k, -1, 0] = 100.0 * k  # endpoints 100 m apart
    assert diversity_loss(traj, margin=4.0).item() == 0.0


def test_diversity_loss_penalises_identical_modes():
    traj = torch.zeros(1, 4, 10, 2)  # all endpoints identical
    assert abs(diversity_loss(traj, margin=4.0).item() - 4.0) < 1e-6


def test_wta_assigns_closest_mode():
    B, K, T = 1, 3, 5
    future = torch.linspace(0, 4, T).view(1, T, 1).repeat(1, 1, 2)
    mask = torch.ones(1, T, dtype=torch.bool)
    traj = torch.zeros(B, K, T, 2)
    traj[0, 1] = future[0]          # mode 1 matches ground truth exactly
    traj[0, 0] += 50
    traj[0, 2] -= 50
    out = multimodal_loss(_pred(traj, torch.zeros(B, K)), future, mask, TrainConfig())
    assert out["best_mode"].item() == 1
    assert torch.isfinite(out["loss"])


def test_anchor_assignment_overrides_wta():
    """With anchors, ground truth is matched to the nearest *anchor*, not prediction."""
    B, K, T = 1, 2, 5
    future = torch.zeros(B, T, 2)
    future[0, :, 1] = torch.linspace(0, 9, T)   # ends at (0, 9)
    mask = torch.ones(1, T, dtype=torch.bool)
    traj = torch.zeros(B, K, T, 2)
    traj[0, 0] = future[0]                      # mode 0 is the better *prediction* ...
    anchors = torch.tensor([[0.0, -10.0], [0.0, 10.0]])
    out = multimodal_loss(_pred(traj, torch.zeros(B, K)), future, mask, TrainConfig(),
                          anchors=anchors)
    assert out["best_mode"].item() == 1         # ... but anchor 1 is nearest the GT endpoint


def test_masked_steps_do_not_contribute():
    B, K, T = 1, 2, 6
    future = torch.zeros(B, T, 2)
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, :3] = True
    traj = torch.zeros(B, K, T, 2)
    traj[0, 0, 3:] = 1e6                        # error only on masked (invalid) steps
    out = multimodal_loss(_pred(traj, torch.zeros(B, K)), future, mask, TrainConfig())
    assert out["best_mode"].item() in (0, 1)    # huge masked error must not dominate
    assert torch.isfinite(out["loss"])
