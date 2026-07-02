"""AV2 forecasting metrics (minADE, minFDE, miss rate, brier-minFDE), computed per sample so
an accumulator can average them over a split.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _last_valid_index(future_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last valid future step per sample. Returns ``(B,)`` long."""
    T = future_mask.shape[1]
    ramp = torch.arange(T, device=future_mask.device).unsqueeze(0)   # (1, T)
    masked = torch.where(future_mask, ramp, torch.full_like(ramp, -1))
    return masked.max(dim=1).values.clamp(min=0)                     # (B,)


@torch.no_grad()
def forecasting_metrics(
    pred: Dict[str, torch.Tensor],
    future: torch.Tensor,
    future_mask: torch.Tensor,
    miss_threshold_m: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample minADE/minFDE/MR/brier-minFDE. Each value is a ``(B,)`` tensor."""
    trajectories = pred["trajectories"]          # (B, K, T, 2)
    logits = pred["logits"]                      # (B, K)
    B, K, T, _ = trajectories.shape
    probs = F.softmax(logits, dim=-1)            # (B, K)

    err = torch.linalg.norm(trajectories - future.unsqueeze(1), dim=-1)  # (B, K, T)
    m = future_mask.unsqueeze(1).float()                                 # (B, 1, T)

    # --- minADE: average displacement over valid steps, min over modes ---
    denom = m.sum(dim=-1).clamp(min=1.0)                                 # (B, 1)
    ade = (err * m).sum(dim=-1) / denom                                  # (B, K)
    min_ade, _ = ade.min(dim=1)                                         # (B,)

    # --- minFDE: endpoint error at each sample's last valid step ---
    last = _last_valid_index(future_mask)                               # (B,)
    fde = err[torch.arange(B), :, last]                                # (B, K)
    min_fde, best_fde_mode = fde.min(dim=1)                            # (B,)

    # --- Miss rate ---
    miss = (min_fde > miss_threshold_m).float()                        # (B,)

    # --- brier-minFDE: add (1 - prob of the endpoint-closest mode)^2 ---
    prob_best = probs[torch.arange(B), best_fde_mode]                  # (B,)
    brier_min_fde = min_fde + (1.0 - prob_best) ** 2                   # (B,)

    return {
        "minADE": min_ade,
        "minFDE": min_fde,
        "MR": miss,
        "brier_minFDE": brier_min_fde,
    }


class MetricAccumulator:
    """Accumulate per-sample metric tensors and report the dataset-level mean."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._count = 0

    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        n = next(iter(metrics.values())).shape[0]
        self._count += n
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v.sum().item())

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            return {k: float("nan") for k in self._sums}
        return {k: s / self._count for k, s in self._sums.items()}
