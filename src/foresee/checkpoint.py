"""Checkpoint (de)serialisation shared by training, evaluation and the dashboard."""

from __future__ import annotations

import dataclasses
from typing import Tuple

import torch

from .config import FeatureConfig, ModelConfig
from .models import build_model


def save_checkpoint(path, model, arch: str, feature_cfg: FeatureConfig,
                    model_cfg: ModelConfig, **extra) -> None:
    """Persist a model with everything needed to rebuild it (arch + configs)."""
    payload = {
        "arch": arch,
        "model_state": model.state_dict(),
        "model_cfg": dataclasses.asdict(model_cfg),
        "feature_cfg": dataclasses.asdict(feature_cfg),
    }
    payload.update(extra)
    torch.save(payload, path)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> Tuple[object, FeatureConfig]:
    """Rebuild the saved model (any arch) and its FeatureConfig from a checkpoint.

    weights_only=True keeps this a tensor/primitive deserialization rather than arbitrary
    pickle, which matters because the dashboard loads checkpoints by path.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    feature_cfg = FeatureConfig(**ckpt["feature_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    arch = ckpt.get("arch")
    if arch is None:
        raise ValueError(f"{path} predates the 'arch' field and cannot be loaded.")
    model = build_model(arch, feature_cfg, model_cfg).to(device)
    # strict=False so checkpoints predating newer buffers (e.g. the calibration `temperature`)
    # still load; any missing buffer keeps its registered default.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, feature_cfg
