"""Run the conflict score across a split and write a CSV ranked by risk.

    python -m foresee.audit --checkpoint runs/<ts>/best.pt --limit 300
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .checkpoint import load_model_from_checkpoint
from .config import Config
from .data.dataset import MotionForecastingDataset, _feature_signature, _scan_scenarios
from .models import build_model
from .playback import build_playback
from .risk import assess


def _val_dataset(cfg: Config) -> MotionForecastingDataset:
    if cfg.has_real_data():
        pairs = _scan_scenarios(Path(cfg.data_root) / "val")
        if pairs:
            cache = Path(cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(cfg.feature)}"
            return MotionForecastingDataset(cfg.feature, scenario_paths=pairs, cache_dir=cache)
    return MotionForecastingDataset(cfg.feature, synthetic_size=512, synthetic_offset=10_000_000)


def run_audit(dataset, model, limit: int, device: str = "cpu"):
    """Return a list of per-scenario risk rows, sorted by descending risk."""
    n = min(limit, len(dataset))
    rows = []
    for idx in range(n):
        pb = build_playback(dataset, idx, model, device=device)
        rep = assess(pb)
        c, it = rep["conflict"], rep["intent"]
        rows.append({
            "scenario_id": pb["scenario_id"],
            "risk": round(c["risk"], 4),
            "risk_level": c["level"],
            "ttc_s": "" if c["ttc_s"] is None else round(c["ttc_s"], 2),
            "closest_m": round(c["closest_m"], 2),
            "threat": c["threat_type"],
            "intent_level": it["level"],
            "intent": it["message"],
        })
        if (idx + 1) % 50 == 0:
            print(f"  audited {idx+1}/{n} ...")
    rows.sort(key=lambda r: (-r["risk"], r["closest_m"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank scenarios by predicted conflict risk.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=300, help="Number of val scenarios to scan.")
    parser.add_argument("--top", type=int, default=15, help="How many to print.")
    parser.add_argument("--out", type=str, default="audit.csv")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = Config()
    if args.checkpoint and Path(args.checkpoint).is_file():
        model, feature_cfg = load_model_from_checkpoint(args.checkpoint, args.device)
        cfg.feature = feature_cfg
    else:
        print("[audit] No checkpoint - using an untrained model (results not meaningful).")
        model = build_model(cfg.arch, cfg.feature, cfg.model).to(args.device).eval()

    dataset = _val_dataset(cfg)
    print(f"[audit] scanning {min(args.limit, len(dataset))} scenarios on {args.device} ...")
    rows = run_audit(dataset, model, args.limit, args.device)

    fields = ["rank", "scenario_id", "risk", "risk_level", "ttc_s", "closest_m",
              "threat", "intent_level", "intent"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **r})

    n_high = sum(r["risk_level"] == "HIGH" for r in rows)
    n_med = sum(r["risk_level"] == "MEDIUM" for r in rows)
    print(f"\n[audit] {len(rows)} scenarios | HIGH={n_high} MEDIUM={n_med} -> {args.out}")
    print(f"\nTop {min(args.top, len(rows))} riskiest scenarios:")
    print(f"  {'#':>2}  {'risk':>5} {'lvl':>6} {'ttc':>5} {'near':>5}  {'threat':<12} intent")
    for i, r in enumerate(rows[:args.top], 1):
        print(f"  {i:>2}  {r['risk']:>5.2f} {r['risk_level']:>6} "
              f"{str(r['ttc_s']):>5} {r['closest_m']:>5.1f}  {r['threat']:<12} {r['intent']}")


if __name__ == "__main__":
    main()
