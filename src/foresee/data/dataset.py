"""PyTorch Dataset over AV2, with a synthetic fallback and an on-disk npz cache.

build_datasets() returns real-data datasets when FORESEE_DATA_ROOT points at a download and
synthetic ones otherwise, so every entry point runs without the 100 GB dataset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config, FeatureConfig
from . import features as F
from . import synthetic

log = logging.getLogger(__name__)

# Keys that are fixed-size tensors (stacked by the collate); everything else is a list.
_TENSOR_KEYS = ("hist", "hist_mask", "object_types", "is_ego", "cur_pos",
                "lanes", "lane_mask", "future", "future_mask", "origin", "rotation")


def _sample_to_tensors(sample: F.Sample) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k in _TENSOR_KEYS:
        arr = np.asarray(sample[k])
        out[k] = torch.from_numpy(arr.copy())
    out["scenario_id"] = sample["scenario_id"]
    return out


def _feature_signature(fc) -> str:
    """Short signature of the feature contract; cache is invalidated when it changes.

    Bump the version tag whenever the feature extraction logic changes (not just its shapes)
 - e.g. v2 added forced ego inclusion in agent selection.
    """
    return (f"v2-a{fc.max_agents}-l{fc.max_lanes}-p{fc.lane_num_points}-"
            f"h{fc.num_history_steps}-f{fc.num_future_steps}-r{int(fc.map_radius_m)}")


def _load_or_build_cached(parquet: Path, map_json: Path, cfg: FeatureConfig,
                          cache_dir: Optional[Path]) -> F.Sample:
    """Parse a scenario, caching the resulting Sample as .npz for fast subsequent epochs.

    Writes go to a temp file and are moved into place with os.replace, so a killed process
    (or workers racing on epoch 1) can't leave a truncated npz behind. A cache file that
    fails to load is deleted and rebuilt instead of poisoning every later run.
    """
    if cache_dir is None:
        return F.scenario_to_sample(parquet, map_json, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{parquet.parent.name}.npz"
    if cache_file.is_file():
        try:
            data = np.load(cache_file, allow_pickle=False)
            sample = {k: data[k] for k in data.files if k != "scenario_id"}
            sample["scenario_id"] = str(data["scenario_id"])
            return sample
        except Exception:
            log.warning("corrupt cache file %s, rebuilding", cache_file)
            cache_file.unlink(missing_ok=True)
    sample = F.scenario_to_sample(parquet, map_json, cfg)
    to_save = {k: np.asarray(v) for k, v in sample.items() if k != "scenario_id"}
    to_save["scenario_id"] = np.array(sample["scenario_id"])
    tmp = cache_file.parent / f"{cache_file.name}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **to_save)
    os.replace(tmp, cache_file)
    return sample


def _scan_scenarios(split_dir: Path) -> List[Tuple[Path, Path]]:
    """Find (parquet, map_json) pairs under an AV2 split directory.

    AV2 layout: ``<split>/<scenario_id>/scenario_<id>.parquet`` +
    ``log_map_archive_<id>.json``. One listing per scenario directory; a couple of seconds
    for the subsets used here. At full-dataset scale (200k dirs) this should become a
    manifest built once, at the cost of staleness handling when scenarios are added.
    """
    pairs: List[Tuple[Path, Path]] = []
    if not split_dir.is_dir():
        return pairs
    for scen_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        parquet = map_json = None
        for f in scen_dir.iterdir():
            if f.name.startswith("scenario_") and f.suffix == ".parquet":
                parquet = f
            elif f.name.startswith("log_map_archive_") and f.suffix == ".json":
                map_json = f
        if parquet is not None and map_json is not None:
            pairs.append((parquet, map_json))
    return pairs


class MotionForecastingDataset(Dataset):
    """Unified dataset for real AV2 scenarios or deterministic synthetic ones.

    Exactly one of ``scenario_paths`` (real) or ``synthetic_size`` (synthetic) is provided.
    """

    def __init__(
        self,
        feature_cfg: FeatureConfig,
        scenario_paths: Optional[List[Tuple[Path, Path]]] = None,
        synthetic_size: Optional[int] = None,
        synthetic_offset: int = 0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.cfg = feature_cfg
        self.scenario_paths = scenario_paths
        self.synthetic_size = synthetic_size
        self.synthetic_offset = synthetic_offset
        self.cache_dir = cache_dir
        if (scenario_paths is None) == (synthetic_size is None):
            raise ValueError("Provide exactly one of scenario_paths or synthetic_size.")
        self.is_synthetic = scenario_paths is None

    def __len__(self) -> int:
        return self.synthetic_size if self.is_synthetic else len(self.scenario_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.is_synthetic:
            sample = synthetic.generate_sample(self.synthetic_offset + idx, self.cfg)
        else:
            parquet, map_json = self.scenario_paths[idx]
            sample = _load_or_build_cached(parquet, map_json, self.cfg, self.cache_dir)
        return _sample_to_tensors(sample)


def collate_samples(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack fixed-size tensors along a new batch dim; keep ``scenario_id`` as a list."""
    out: Dict[str, object] = {}
    for k in _TENSOR_KEYS:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["scenario_id"] = [b["scenario_id"] for b in batch]
    return out


class RealDataUnavailableError(RuntimeError):
    """Raised when real AV2 data is required (strict mode) but cannot be found."""


def _require_real_data_message(cfg: Config) -> str:
    return (
        "Real Argoverse 2 data was required (require_real_data=True) but was not found.\n"
        f"  data_root = {cfg.data_root!r}\n"
        "Expected layout: <data_root>/train/<scenario_id>/scenario_*.parquet + "
        "log_map_archive_*.json (and likewise for val/).\n"
        "Fix this by:\n"
        "  1. Downloading the dataset:  bash scripts/download_av2.sh\n"
        "  2. Pointing the pipeline at it:  export FORESEE_DATA_ROOT=$HOME/data/datasets/motion-forecasting\n"
        "  3. Installing the parser:  pip install av2\n"
        "Or drop the strict flag (omit --require-real-data / FORESEE_REQUIRE_REAL) to use "
        "the synthetic generator for development."
    )


def build_datasets(cfg: Config) -> Tuple[MotionForecastingDataset, MotionForecastingDataset]:
    """Return ``(train_ds, val_ds)``.

    Uses the real AV2 dataset when available. In strict mode (``cfg.require_real_data``) it
    raises :class:`RealDataUnavailableError` instead of falling back to synthetic data.
    """
    if cfg.has_real_data():
        root = Path(cfg.data_root)
        train_pairs = _scan_scenarios(root / "train")
        val_pairs = _scan_scenarios(root / "val")
        if train_pairs and val_pairs:
            print(f"[data] Real AV2 data: {len(train_pairs)} train / {len(val_pairs)} val scenarios.")
            sig = _feature_signature(cfg.feature)
            cache_root = root / ".foresee_cache"
            return (
                MotionForecastingDataset(cfg.feature, scenario_paths=train_pairs,
                                         cache_dir=cache_root / f"train-{sig}"),
                MotionForecastingDataset(cfg.feature, scenario_paths=val_pairs,
                                         cache_dir=cache_root / f"val-{sig}"),
            )
        if cfg.require_real_data:
            raise RealDataUnavailableError(_require_real_data_message(cfg))
        print(f"[data] FORESEE_DATA_ROOT={cfg.data_root} found but no scenarios; using synthetic.")
    elif cfg.require_real_data:
        raise RealDataUnavailableError(_require_real_data_message(cfg))

    print("[data] Using synthetic dataset (no AV2 download required).")
    train_ds = MotionForecastingDataset(
        cfg.feature, synthetic_size=cfg.train.synthetic_train_size, synthetic_offset=0
    )
    val_ds = MotionForecastingDataset(
        cfg.feature,
        synthetic_size=cfg.train.synthetic_val_size,
        synthetic_offset=10_000_000,  # disjoint indices => disjoint val scenarios
    )
    return train_ds, val_ds
