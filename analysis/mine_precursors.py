"""Scan Argoverse scenarios for near-miss precursors of the NHTSA crash types and compare
the mined distribution against the real crash mix. Writes a figure and CSV to
assets/insights/.

    python analysis/mine_precursors.py --limit 600
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foresee.config import Config  # noqa: E402
from foresee.data.dataset import (  # noqa: E402
    MotionForecastingDataset,
    _feature_signature,
    _scan_scenarios,
)
from foresee.playback import raw_scene  # noqa: E402
from foresee.precursors import NHTSA_MIX, find_precursors  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "insights"
ORDER = ["Closing on lead / fixed object", "Lane departure / swerve",
         "Turn across path", "Lane change / merge"]


def _dataset(cfg: Config) -> MotionForecastingDataset:
    if cfg.has_real_data():
        pairs = _scan_scenarios(Path(cfg.data_root) / "val")
        if pairs:
            cache = Path(cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(cfg.feature)}"
            return MotionForecastingDataset(cfg.feature, scenario_paths=pairs, cache_dir=cache)
    return MotionForecastingDataset(cfg.feature, synthetic_size=512, synthetic_offset=10_000_000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--min-severity", type=float, default=0.3,
                    help="Only count precursors at least this severe as a 'near-miss'.")
    args = ap.parse_args()

    cfg = Config()
    ds = _dataset(cfg)
    n = min(args.limit, len(ds))
    print(f"[mine] scanning {n} scenarios ({'real AV2' if cfg.has_real_data() else 'synthetic'}) ...")

    worst_cat = Counter()        # dominant precursor per scenario (>= min_severity)
    all_cat = Counter()          # every precursor event (>= min_severity)
    flagged = examples = 0
    rows = []
    for idx in range(n):
        scene = raw_scene(ds, idx)
        events = [e for e in find_precursors(scene) if e.severity >= args.min_severity]
        for e in events:
            all_cat[e.category] += 1
        if events:
            flagged += 1
            top = max(events, key=lambda e: e.severity)
            worst_cat[top.category] += 1
            if examples < 12:
                rows.append((scene["scenario_id"][:8], top.category, round(top.severity, 2),
                             top.ttc_s, top.description))
                examples += 1
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{n} ...")

    total_worst = sum(worst_cat.values()) or 1
    print(f"\n[mine] {flagged}/{n} scenarios contain a crash precursor (severity >= {args.min_severity})")
    print(f"\n{'category':<32} {'Argoverse precursors':>20} {'real Tesla crashes':>20}")
    for c in ORDER:
        share = worst_cat.get(c, 0) / total_worst
        print(f"{c:<32} {worst_cat.get(c,0):>6} ({share*100:>3.0f}%){'':>7} {NHTSA_MIX[c]*100:>14.0f}%")

    print("\nExample precursors found:")
    for sid, cat, sev, ttc, desc in rows[:8]:
        print(f"  [{sid}] {cat} (sev {sev}{', TTC '+str(ttc)+'s' if ttc else ''}) - {desc}")

    # --- Figure: Argoverse precursor mix vs real Tesla crash mix ---
    OUT.mkdir(parents=True, exist_ok=True)
    av = [worst_cat.get(c, 0) / total_worst for c in ORDER]
    re = [NHTSA_MIX[c] for c in ORDER]
    y = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.barh(y + 0.2, av, height=0.38, color="#1f9bff", label="Argoverse precursors (mined)")
    ax.barh(y - 0.2, re, height=0.38, color="#e6194B", label="Real Tesla crashes (NHTSA)")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace(" / ", "/\n") for c in ORDER], color="#c8ccd4", fontsize=9)
    ax.set_xlabel("share", color="#8893a5")
    ax.set_title("Mined near-miss precursors vs. real Tesla crash composition", color="white")
    ax.tick_params(colors="#8893a5")
    for s in ax.spines.values():
        s.set_color("#333")
    leg = ax.legend(fontsize=9)
    for t in leg.get_texts():
        t.set_color("black")
    fig.tight_layout()
    fig.savefig(OUT / "07_precursors_vs_crashes.png", dpi=110, facecolor="#0e1117")
    print(f"\n[out] figure -> {OUT/'07_precursors_vs_crashes.png'}")

    with open(OUT / "precursor_mix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "argoverse_precursor_share", "nhtsa_crash_share"])
        for c in ORDER:
            w.writerow([c, round(worst_cat.get(c, 0) / total_worst, 3), NHTSA_MIX[c]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
