"""Neural network models and an arch factory."""

from ..config import FeatureConfig, ModelConfig
from .anchored_transformer import AnchoredTransformer
from .lane_transformer import LaneAnchoredTransformer

__all__ = ["LaneAnchoredTransformer", "AnchoredTransformer", "build_model", "ARCHS"]

# Registry of selectable architectures. The chosen name is stored in each checkpoint so
# evaluation and the dashboard rebuild the matching model automatically.
ARCHS = {
    "lane_transformer": LaneAnchoredTransformer,   # winner-takes-all baseline
    "anchored": AnchoredTransformer,               # goal-anchored (default)
}


def build_model(arch: str, feature_cfg: FeatureConfig, model_cfg: ModelConfig):
    """Construct a model by architecture name."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Options: {list(ARCHS)}")
    return ARCHS[arch](feature_cfg, model_cfg)
