"""Measure mode quality for a checkpoint: distinct maneuvers, top-1 selection accuracy,
deployed vs oracle ADE, and confidence reliability. Appends a row to
assets/insights/mode_quality.csv so runs accumulate into a comparison table.

    python analysis/evaluate_modes.py --checkpoint runs_anchored/<ts>/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foresee.checkpoint import load_model_from_checkpoint  # noqa: E402
from foresee.config import Config  # noqa: E402
from foresee.data import collate_samples  # noqa: E402
from foresee.data.dataset import (  # noqa: E402
    MotionForecastingDataset,
    _feature_signature,
    _scan_scenarios,
)
from foresee.risk import maneuver_label  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "insights"


def _val_dataset(cfg: Config) -> MotionForecastingDataset:
    if cfg.has_real_data():
        pairs = _scan_scenarios(Path(cfg.data_root) / "val")
        if pairs:
            cache = Path(cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(cfg.feature)}"
            return MotionForecastingDataset(cfg.feature, scenario_paths=pairs, cache_dir=cache)
    print("[warn] no real data found - falling back to synthetic (numbers not comparable)")
    return MotionForecastingDataset(cfg.feature, synthetic_size=512, synthetic_offset=10_000_000)


@torch.no_grad()
def evaluate(model, loader, device: str, limit: int):
    """Return a dict of mode-quality metrics plus raw arrays for the reliability table."""
    temp = float(getattr(model, "temperature", torch.tensor(1.0)))
    K = model.K
    N = top1 = 0
    distinct, ade_top, ade_oracle = [], [], []
    per_mode = [Counter() for _ in range(K)]
    calib_p, calib_hit = [], []

    for batch in loader:
        if N >= limit:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model(batch)
        tr = pred["trajectories"]                                     # (B, K, T, 2)
        probs = torch.softmax(pred["logits"] / temp, dim=-1)          # calibrated
        fut, fm = batch["future"], batch["future_mask"].float()
        err = torch.linalg.norm(tr - fut.unsqueeze(1), dim=-1)
        ade = (err * fm.unsqueeze(1)).sum(-1) / fm.sum(-1, keepdim=True).clamp(min=1)  # (B, K)
        best, top = ade.argmin(1), probs.argmax(1)

        for b in range(tr.shape[0]):
            N += 1
            top1 += int(best[b] == top[b])
            ade_top.append(float(ade[b, top[b]]))
            ade_oracle.append(float(ade[b, best[b]]))
            mans = [maneuver_label(tr[b, k].cpu().numpy()) for k in range(K)]
            distinct.append(len(set(mans)))
            for k in range(K):
                per_mode[k][mans[k]] += 1
                calib_p.append(float(probs[b, k]))
                calib_hit.append(int(k == best[b]))

    return {
        "n": N, "temperature": temp,
        "distinct_maneuvers": float(np.mean(distinct)),
        "top1_accuracy": top1 / max(N, 1),
        "deployed_ade": float(np.mean(ade_top)),
        "oracle_min_ade": float(np.mean(ade_oracle)),
        "per_mode": per_mode,
        "calib": (np.array(calib_p), np.array(calib_hit)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit", type=int, default=400, help="Val scenarios to evaluate.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, feature_cfg = load_model_from_checkpoint(args.checkpoint, args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = ckpt.get("arch", "?")
    cfg = Config()
    cfg.feature = feature_cfg
    loader = DataLoader(_val_dataset(cfg), batch_size=32, collate_fn=collate_samples)

    m = evaluate(model, loader, args.device, args.limit)
    K = model.K

    print(f"\n=== mode quality: {args.checkpoint}  (arch={arch}, {m['n']} val scenarios) ===")
    print(f"  distinct maneuvers / {K} modes : {m['distinct_maneuvers']:.2f}   (1.0 = full collapse)")
    print(f"  top-1 selection accuracy      : {100*m['top1_accuracy']:.0f}%    (random = {100/K:.0f}%)")
    print(f"  deployed (top-mode) ADE       : {m['deployed_ade']:.2f} m")
    print(f"  oracle (best-of-{K}) minADE     : {m['oracle_min_ade']:.2f} m")
    print(f"  calibration temperature       : {m['temperature']:.2f}")

    print("\n  per-mode dominant maneuver (does each mode own a distinct behaviour?):")
    for k in range(K):
        man, cnt = m["per_mode"][k].most_common(1)[0]
        print(f"    mode {k}: {man:<22} {100*cnt/m['n']:.0f}%")

    p, hit = m["calib"]
    print("\n  reliability (stated confidence -> how often that mode was actually best):")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)]:
        sel = (p >= lo) & (p < hi)
        if sel.sum():
            print(f"    {lo:.1f}-{hi:.1f}: empirical {hit[sel].mean():.2f}   (n={sel.sum()})")

    OUT.mkdir(parents=True, exist_ok=True)
    row = {"checkpoint": args.checkpoint, "arch": arch, "n": m["n"],
           "distinct_maneuvers": round(m["distinct_maneuvers"], 2),
           "top1_accuracy": round(m["top1_accuracy"], 3),
           "deployed_ade_m": round(m["deployed_ade"], 2),
           "oracle_min_ade_m": round(m["oracle_min_ade"], 2),
           "temperature": round(m["temperature"], 2)}
    path = OUT / "mode_quality.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"\n[out] appended to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
