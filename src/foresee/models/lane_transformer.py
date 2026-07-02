"""Transformer decoder over agent and lane tokens (the winner-takes-all baseline).

A per-agent LSTM plus a type embedding encodes each history, a small PointNet encodes each
lane centerline, and K learned queries cross-attend over all tokens to decode trajectories,
Laplace scales and confidence logits.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import FeatureConfig, ModelConfig


class LaneEncoder(nn.Module):
    """PointNet-style encoder: per-point MLP + masked max-pool over each lane polyline."""

    def __init__(self, lane_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(lane_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, lanes: torch.Tensor) -> torch.Tensor:
        # lanes: (B, L, P, Fl) -> (B, L, H)
        B, L, P, Fl = lanes.shape
        x = self.point_mlp(lanes.reshape(B * L, P, Fl))   # (B*L, P, H)
        x = x.max(dim=1).values                           # symmetric pool over points
        return x.reshape(B, L, -1)


class LaneAnchoredTransformer(nn.Module):
    def __init__(self, feature_cfg: FeatureConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.fcfg = feature_cfg
        self.mcfg = model_cfg
        H = model_cfg.hidden_dim
        self.K = model_cfg.num_modes
        self.T_fut = feature_cfg.num_future_steps

        # --- Input normalisation (identical to the LSTM baseline) ---
        agent_scale = torch.tensor([30., 30., 15., 15., 1., 1., 1.])[:feature_cfg.agent_feature_dim]
        lane_scale = torch.tensor([30., 30., 1., 1.])[:feature_cfg.lane_feature_dim]
        self.register_buffer("agent_scale", agent_scale)
        self.register_buffer("lane_scale", lane_scale)
        # Post-hoc confidence calibration (temperature scaling); set after training. 1.0 = none.
        self.register_buffer("temperature", torch.tensor(1.0))

        # --- Encoders ---
        self.type_embed = nn.Embedding(feature_cfg.num_object_types, model_cfg.type_embed_dim)
        self.agent_lstm = nn.LSTM(
            input_size=feature_cfg.agent_feature_dim, hidden_size=H,
            num_layers=model_cfg.lstm_layers, batch_first=True,
            dropout=model_cfg.dropout if model_cfg.lstm_layers > 1 else 0.0,
        )
        self.agent_proj = nn.Sequential(
            nn.Linear(H + model_cfg.type_embed_dim, H), nn.LayerNorm(H), nn.ReLU(inplace=True),
        )
        self.lane_encoder = LaneEncoder(feature_cfg.lane_feature_dim, H)

        # Learned segment embeddings so the decoder can tell agents from lanes.
        self.seg_embed = nn.Parameter(torch.randn(2, H) * 0.02)  # [0]=agent, [1]=lane

        # --- Transformer decoder with K learnable mode queries ---
        self.mode_queries = nn.Parameter(torch.randn(self.K, H) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=H, nhead=model_cfg.attn_heads, dim_feedforward=2 * H,
            dropout=model_cfg.dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=model_cfg.decoder_layers)

        # --- Probabilistic heads (same contract as the LSTM model) ---
        self.traj_head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(inplace=True), nn.Linear(H, self.T_fut * 2),
        )
        self.cls_head = nn.Linear(H, 1)
        if model_cfg.predict_uncertainty:
            self.scale_head = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(inplace=True), nn.Linear(H, self.T_fut * 2),
            )
        else:
            self.scale_head = None

    def _encode_tokens(self, batch):
        """Encode the scene into agent + lane tokens and a key-padding mask."""
        hist = batch["hist"] / self.agent_scale          # (B, A, T_obs, Fa)
        lanes = batch["lanes"] / self.lane_scale          # (B, L, P, Fl)
        hist_mask = batch["hist_mask"]                    # (B, A) bool
        lane_mask = batch["lane_mask"]                    # (B, L) bool
        object_types = batch["object_types"]             # (B, A) long
        B, A, T_obs, Fa = hist.shape

        _, (h_n, _) = self.agent_lstm(hist.reshape(B * A, T_obs, Fa))
        agent_emb = self.agent_proj(
            torch.cat([h_n[-1].reshape(B, A, -1), self.type_embed(object_types)], dim=-1)
        )                                                 # (B, A, H)
        lane_emb = self.lane_encoder(lanes)               # (B, L, H)

        memory = torch.cat([agent_emb + self.seg_embed[0],
                            lane_emb + self.seg_embed[1]], dim=1)   # (B, A+L, H)
        pad = torch.cat([~hist_mask, ~lane_mask], dim=1)            # (B, A+L) True=ignore
        return memory, pad

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory, pad = self._encode_tokens(batch)
        B = memory.shape[0]

        queries = self.mode_queries.unsqueeze(0).expand(B, -1, -1)   # (B, K, H)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=pad)  # (B, K, H)

        deltas = self.traj_head(decoded).reshape(B, self.K, self.T_fut, 2)
        trajectories = torch.cumsum(deltas, dim=2)                   # (B, K, T_fut, 2)
        logits = self.cls_head(decoded).squeeze(-1)                  # (B, K)
        if self.scale_head is not None:
            scale = torch.nn.functional.softplus(
                self.scale_head(decoded).reshape(B, self.K, self.T_fut, 2)
            ).clamp(min=1e-2)
        else:
            scale = None
        return {"trajectories": trajectories, "logits": logits, "scale": scale}
