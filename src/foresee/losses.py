"""Losses for the multimodal head.

multimodal_loss = regression on the assigned mode (Laplace NLL when the model predicts a
scale, smooth-L1 otherwise) plus cross-entropy on which mode was assigned. Assignment is
winner-takes-all by default, or nearest-goal-anchor when anchors are passed; the anchor
assignment is what fixed mode collapse. diversity_loss is an endpoint-repulsion penalty
kept from an earlier attempt that didn't work well (see DESIGN_REVIEW.md).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import TrainConfig


def _per_mode_ade(trajectories: torch.Tensor, future: torch.Tensor,
                  future_mask: torch.Tensor) -> torch.Tensor:
    """Average displacement error of each mode over valid future steps. Returns ``(B, K)``."""
    # trajectories (B,K,T,2); future (B,T,2); future_mask (B,T)
    err = trajectories - future.unsqueeze(1)              # (B, K, T, 2)
    dist = torch.linalg.norm(err, dim=-1)                 # (B, K, T)
    m = future_mask.unsqueeze(1).float()                  # (B, 1, T)
    denom = m.sum(dim=-1).clamp(min=1.0)                  # (B, 1)
    return (dist * m).sum(dim=-1) / denom                 # (B, K)


def diversity_loss(trajectories: torch.Tensor, margin: float) -> torch.Tensor:
    """Hinge repulsion on mode endpoints - the cure for mode collapse.

    Winner-takes-all only trains the single closest mode, so the others drift toward the dominant
    "go straight" prior and the K modes become near-duplicates (measured: ~1.4 distinct maneuvers
    out of 6). This penalises every pair of modes whose endpoints are closer than ``margin``
    metres, forcing the modes apart so they cover *distinct* maneuvers (left / straight / right).
    The WTA term still pins the best mode to the ground truth, so accuracy is preserved while
    coverage improves.
    """
    ends = trajectories[:, :, -1, :]                      # (B, K, 2) endpoints
    d = torch.cdist(ends, ends)                           # (B, K, K) pairwise endpoint distance
    K = ends.shape[1]
    iu = torch.triu_indices(K, K, offset=1, device=ends.device)
    pair_d = d[:, iu[0], iu[1]]                           # (B, num_pairs)
    return torch.relu(margin - pair_d).mean()


def _nearest_anchor(future: torch.Tensor, future_mask: torch.Tensor,
                    anchors: torch.Tensor) -> torch.Tensor:
    """Assign each sample to the anchor closest to its ground-truth 6 s endpoint. Returns (B,)."""
    T = future.shape[1]
    ramp = torch.arange(T, device=future.device)
    last = torch.where(future_mask, ramp, torch.full_like(ramp, -1)).max(dim=1).values.clamp(min=0)
    gt_end = future[torch.arange(future.shape[0]), last]      # (B, 2)
    d = torch.cdist(gt_end.unsqueeze(1), anchors.unsqueeze(0).expand(future.shape[0], -1, -1))
    return d.squeeze(1).argmin(dim=1)                         # (B,)


def multimodal_loss(
    pred: Dict[str, torch.Tensor],
    future: torch.Tensor,
    future_mask: torch.Tensor,
    cfg: TrainConfig,
    anchors: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """WTA (or anchor-assignment) regression + classification (+ diversity) loss.

    If ``anchors`` (K, 2) are given, ground truth is matched to the nearest goal anchor rather
    than the nearest prediction - this is what forces the modes to specialise and cures collapse.
    """
    trajectories = pred["trajectories"]       # (B, K, T, 2)
    logits = pred["logits"]                   # (B, K)
    scale: Optional[torch.Tensor] = pred.get("scale")
    B, K, T, _ = trajectories.shape

    # ----- 1. Mode assignment: nearest anchor (anchored) or winner-takes-all -----
    if anchors is not None:
        best_mode = _nearest_anchor(future, future_mask, anchors)   # (B,) fixed, diverse
    else:
        ade = _per_mode_ade(trajectories, future, future_mask)      # (B, K)
        best_mode = ade.argmin(dim=1)                              # (B,)
    idx = best_mode.view(B, 1, 1, 1).expand(B, 1, T, 2)
    best_traj = trajectories.gather(1, idx).squeeze(1)       # (B, T, 2)

    m = future_mask.float()                                  # (B, T)
    denom = m.sum().clamp(min=1.0)

    # ----- 2. Regression term on the winning mode -----
    if scale is not None:
        best_scale = scale.gather(1, idx).squeeze(1)         # (B, T, 2)
        # Laplace NLL: |y-μ|/b + log(2b), summed over (x,y), masked over valid steps.
        nll = (best_traj - future).abs() / best_scale + torch.log(2.0 * best_scale)
        reg_loss = (nll.sum(dim=-1) * m).sum() / denom
    else:
        huber = F.smooth_l1_loss(best_traj, future, reduction="none").sum(dim=-1)  # (B, T)
        reg_loss = (huber * m).sum() / denom

    # ----- 3. Classification term: predict which mode won -----
    cls_loss = F.cross_entropy(logits, best_mode)

    # ----- 4. Diversity regularizer: keep the K modes distinct -----
    div_loss = (diversity_loss(trajectories, cfg.diversity_margin)
                if cfg.diversity_weight > 0 else trajectories.new_zeros(()))

    total = reg_loss + cfg.cls_loss_weight * cls_loss + cfg.diversity_weight * div_loss
    return {
        "loss": total,
        "reg_loss": reg_loss.detach(),
        "cls_loss": cls_loss.detach(),
        "div_loss": div_loss.detach(),
        "best_mode": best_mode.detach(),
    }
