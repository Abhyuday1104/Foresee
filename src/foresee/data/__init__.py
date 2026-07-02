"""Data pipeline: AV2 parsing, map fusion, synthetic fallback, and the PyTorch Dataset."""

from .dataset import MotionForecastingDataset, build_datasets, collate_samples

__all__ = ["MotionForecastingDataset", "build_datasets", "collate_samples"]
