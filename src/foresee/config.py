"""Shared configuration.

Every tensor shape and hyperparameter lives here so the data pipeline, models, losses and
dashboard can never disagree about dimensions. tests/smoke_test.py asserts the full
forward/backward contract against these values.

AV2 timing at 10 Hz: 110 steps total = 5 s observed (50) + 6 s to predict (60).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------------------
# Feature engineering / data contract
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    # --- Temporal horizon ---
    num_total_steps: int = 110
    num_history_steps: int = 50      # T_obs
    num_future_steps: int = 60       # T_fut  (the prediction horizon)
    sample_rate_hz: float = 10.0

    # --- Scene capacity (padded to fixed size for batching) ---
    max_agents: int = 32             # focal is always index 0; rest are nearest neighbours
    max_lanes: int = 64              # nearest lane centerlines around the focal agent
    lane_num_points: int = 20        # resampled points per lane centerline

    # --- Per-step agent feature vector (agent-centric frame) ---
    # [ x, y, vx, vy, sin(heading), cos(heading), valid ]
    agent_feature_dim: int = 7

    # Number of semantic object classes (vehicle, pedestrian, cyclist, bus, ego, ...).
    # See features.OBJECT_TYPE_NAMES for the canonical ordering. Used for a per-agent
    # type embedding (model) and type-coded rendering (dashboard).
    num_object_types: int = 11

    # --- Per-point lane feature vector (agent-centric frame) ---
    # [ x, y, dir_x, dir_y ]  (dir = unit tangent along the centerline)
    lane_feature_dim: int = 4

    # --- Map query radius (metres) around the focal agent's last observed position ---
    map_radius_m: float = 50.0

    @property
    def observed_slice(self) -> slice:
        return slice(0, self.num_history_steps)

    @property
    def future_slice(self) -> slice:
        return slice(self.num_history_steps, self.num_total_steps)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    lstm_layers: int = 2
    attn_heads: int = 4
    decoder_layers: int = 2          # transformer-decoder depth (lane_transformer arch)
    dropout: float = 0.1
    type_embed_dim: int = 16         # per-agent object-type embedding width
    num_modes: int = 6               # K - number of predicted trajectory hypotheses
    # If True the regression head also predicts a per-point Laplace scale b>0, enabling a
    # proper negative-log-likelihood loss (uncertainty-aware). If False, a Huber/L1 loss
    # is used and the scale is ignored.
    predict_uncertainty: bool = True


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    cls_loss_weight: float = 1.0     # weight on the mode-classification (cross-entropy) term
    # Mode-diversity regularizer: hinge-penalize pairs of modes whose endpoints are closer than
    # `diversity_margin` metres, forcing the K modes to cover *distinct* maneuvers instead of
    # collapsing to near-duplicates of "go straight". 0 disables it.
    diversity_weight: float = 0.6
    diversity_margin: float = 4.0
    miss_threshold_m: float = 2.0    # FDE threshold for the Miss-Rate metric
    num_workers: int = 4
    seed: int = 42
    # Synthetic dataset sizes (used when no real data is found).
    synthetic_train_size: int = 2048
    synthetic_val_size: int = 256


# --------------------------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Model architecture: "anchored" (goal-anchored, default) or "lane_transformer" (WTA baseline).
    arch: str = "anchored"

    # Root of the downloaded AV2 motion-forecasting dataset (contains train/ val/ test/).
    # If unset or non-existent, the pipeline falls back to the synthetic generator ONLY when
    # `require_real_data` is False.
    data_root: str | None = field(
        default_factory=lambda: os.environ.get("FORESEE_DATA_ROOT")
    )

    # When True, the data pipeline must use the real Argoverse 2 dataset and will raise a
    # clear error rather than silently falling back to the synthetic generator. Set via the
    # FORESEE_REQUIRE_REAL=1 environment variable or the --require-real-data CLI flag.
    require_real_data: bool = field(
        default_factory=lambda: os.environ.get("FORESEE_REQUIRE_REAL", "0") == "1"
    )

    def has_real_data(self) -> bool:
        return bool(self.data_root) and os.path.isdir(self.data_root)
