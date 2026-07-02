"""Goal-anchored variant of the lane transformer.

Each mode query is conditioned on a fixed k-means goal anchor, and the loss assigns ground
truth to the nearest anchor instead of the nearest prediction, which stops the modes from
collapsing onto one maneuver. See anchors.py for how the anchors are mined.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import FeatureConfig, ModelConfig
from .lane_transformer import LaneAnchoredTransformer

POS_SCALE = 30.0  # normalise anchor coordinates (metres) before embedding


class AnchoredTransformer(LaneAnchoredTransformer):
    def __init__(self, feature_cfg: FeatureConfig, model_cfg: ModelConfig) -> None:
        super().__init__(feature_cfg, model_cfg)
        # Goal anchors (set after k-means via `set_anchors`); part of the state_dict so they
        # travel with the checkpoint and inference uses the exact same anchors.
        self.register_buffer("anchors", torch.zeros(self.K, 2))
        self.anchor_proj = nn.Linear(2, model_cfg.hidden_dim)

    def set_anchors(self, anchors) -> None:
        self.anchors.copy_(torch.as_tensor(anchors, dtype=self.anchors.dtype,
                                            device=self.anchors.device))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory, pad = self._encode_tokens(batch)
        B = memory.shape[0]

        # Each query = a learned query + its goal-anchor embedding -> distinct, non-collapsing modes.
        anchor_emb = self.anchor_proj(self.anchors / POS_SCALE)        # (K, H)
        queries = (self.mode_queries + anchor_emb).unsqueeze(0).expand(B, -1, -1)  # (B, K, H)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=pad)       # (B, K, H)

        deltas = self.traj_head(decoded).reshape(B, self.K, self.T_fut, 2)
        trajectories = torch.cumsum(deltas, dim=2)
        logits = self.cls_head(decoded).squeeze(-1)
        if self.scale_head is not None:
            scale = torch.nn.functional.softplus(
                self.scale_head(decoded).reshape(B, self.K, self.T_fut, 2)).clamp(min=1e-2)
        else:
            scale = None
        return {"trajectories": trajectories, "logits": logits, "scale": scale}
