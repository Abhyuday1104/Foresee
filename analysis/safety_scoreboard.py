"""Score the conflict detector against ground truth.

Labels real near-miss events (a converging approach to within ~4 m between two agents'
actual futures), then sweeps both the production risk score and the predicted
closest-approach distance to get recall / false-alarm curves. Also counts how many events
involve non-focal agents the detector cannot see at all.

    python analysis/safety_scoreboard.py --checkpoint runs_anchored2/<ts>/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foresee.checkpoint import load_model_from_checkpoint  # noqa: E402
from foresee.config import Config  # noqa: E402
from foresee.data.dataset import (  # noqa: E402
    MotionForecastingDataset,
    _feature_signature,
    _scan_scenarios,
)
from foresee.playback import build_playback, raw_scene  # noqa: E402
from foresee.risk import APPROACH_MARGIN_M, assess_conflict  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "insights"


# Surrogate near-miss definition for ground truth. 4 m centre-to-centre between two vehicles
# is roughly 1.5 m of physical clearance - genuinely close. Unlike the *detector* (which needs a
# sustained-steps gate to reject jittery predicted modes), a real momentary close pass counts:
# ground truth does not jitter.
NEAR_MISS_M = 4.0


def actual_near_miss(track_a, valid_a, track_b, valid_b, type_a, type_b,
                     start: int, horizon: int, near_m: float = NEAR_MISS_M) -> bool:
    """Did agents a and b *actually* converge to a close approach in the future window?

    Event = the pair starts clearly separated (near_m + approach margin) and their actual
    separation then drops below ``near_m`` - i.e. a genuine converging near-miss, not steady
    car-following at a constant gap.
    """
    sl = slice(start, start + horizon)
    both = valid_a[sl] & valid_b[sl]
    if both.sum() < 3:
        return False
    d = np.linalg.norm(track_a[sl] - track_b[sl], axis=1)
    d = np.where(both, d, np.inf)
    first = int(np.where(both)[0][0])
    if d[first] < near_m + APPROACH_MARGIN_M:         # already close: following, not an event
        return False
    return bool((d < near_m).any())


def label_scenario(scene) -> dict:
    """Ground-truth labels for one scenario: focal-vs-ego event + any-agent-vs-ego event."""
    tracks, valid = scene["tracks_world"], scene["valid"]
    types, is_ego = scene["object_types"], scene["is_ego"]
    start, horizon = scene["obs_len"], tracks.shape[1] - scene["obs_len"]
    if not is_ego.any():
        return {"has_ego": False, "focal_event": False, "other_event": False}
    ego = int(np.where(is_ego)[0][0])

    focal_event = actual_near_miss(tracks[0], valid[0], tracks[ego], valid[ego],
                                   int(types[0]), int(types[ego]), start, horizon)
    other_event = False
    for a in range(tracks.shape[0]):
        if a in (0, ego) or not valid[a].any():
            continue
        if actual_near_miss(tracks[a], valid[a], tracks[ego], valid[ego],
                            int(types[a]), int(types[ego]), start, horizon):
            other_event = True
            break
    return {"has_ego": True, "focal_event": focal_event, "other_event": other_event}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config()
    pairs = _scan_scenarios(Path(cfg.data_root) / "val")
    cache = Path(cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(cfg.feature)}"
    ds = MotionForecastingDataset(cfg.feature, scenario_paths=pairs, cache_dir=cache)
    model, _ = load_model_from_checkpoint(args.checkpoint, args.device)
    n = min(args.limit, len(ds))
    print(f"[scoreboard] labelling ground truth + scoring risk on {n} val scenarios ...")

    rows = []
    for idx in range(n):
        try:
            scene = raw_scene(ds, idx)
            lab = label_scenario(scene)
            if not lab["has_ego"]:
                continue
            pb = build_playback(ds, idx, model, device=args.device)
            c = assess_conflict(pb)
            rows.append({"idx": idx, "scenario_id": scene["scenario_id"],
                         "gt_focal_event": int(lab["focal_event"]),
                         "gt_other_event": int(lab["other_event"]),
                         "risk": round(float(c["risk"]), 4),
                         "pred_closest_m": round(float(c["closest_m"]), 2)})
        except Exception as e:
            print(f"  [skip] idx {idx}: {type(e).__name__}: {e}")
        if (idx + 1) % 150 == 0:
            print(f"  {idx+1}/{n} ...")

    events = [r for r in rows if r["gt_focal_event"]]
    benign = [r for r in rows if not r["gt_focal_event"]]
    other_only = [r for r in rows if r["gt_other_event"] and not r["gt_focal_event"]]
    total_events = len(events) + len(other_only)
    print(f"\n[ground truth] {len(rows)} scenarios with an ego vehicle")
    print(f"  focal-vs-ego near-miss events : {len(events)}")
    print(f"  non-focal-vs-ego events        : {len(other_only)}  "
          f"({100*len(other_only)/max(total_events,1):.0f}% of all events are invisible to a "
          "focal-only detector)")

    # ----- sweep 1: the production risk score as the decision variable -----
    print(f"\n[production risk score]\n{'thr':>5} {'recall':>7} {'false-alarm':>12} {'precision':>10}")
    sweep = []
    for thr in np.arange(0.0, 1.001, 0.05):
        tp = sum(r["risk"] >= thr for r in events)
        fp = sum(r["risk"] >= thr for r in benign)
        recall = tp / max(len(events), 1)
        far = fp / max(len(benign), 1)
        prec = tp / max(tp + fp, 1)
        sweep.append((float(thr), recall, far, prec))
        print(f"{thr:>5.2f} {recall:>7.2f} {far:>12.3f} {prec:>10.2f}")

    # ----- sweep 2: predicted closest-approach distance as the decision variable -----
    # The hard conflict gate was hand-tuned; this shows the recall/false-alarm frontier the
    # *same predictions* support when the decision threshold is fitted to ground truth instead.
    print(f"\n[predicted closest approach]\n{'< d m':>6} {'recall':>7} {'false-alarm':>12} {'precision':>10}")
    dsweep = []
    for dm in np.arange(1.0, 15.01, 1.0):
        tp = sum(r["pred_closest_m"] < dm for r in events)
        fp = sum(r["pred_closest_m"] < dm for r in benign)
        recall = tp / max(len(events), 1)
        far = fp / max(len(benign), 1)
        prec = tp / max(tp + fp, 1)
        dsweep.append((float(dm), recall, far, prec))
        print(f"{dm:>6.1f} {recall:>7.2f} {far:>12.3f} {prec:>10.2f}")

    # ----- artefacts -----
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "safety_scoreboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot([s[2] for s in dsweep], [s[1] for s in dsweep], color="#1f9bff", lw=2,
            marker="o", ms=3.5, label="predicted closest approach (fitted)")
    ax.plot([s[2] for s in sweep], [s[1] for s in sweep], color="#f58231", lw=2,
            marker="s", ms=3.5, label="production risk score (hand-tuned)")
    ax.plot([0, 1], [0, 1], color="#444", lw=1, ls="--", label="chance")
    ax.set_xlabel("false-alarm rate (benign scenarios flagged)", color="#c8ccd4")
    ax.set_ylabel("recall (real near-misses caught)", color="#c8ccd4")
    ax.set_title(f"Near-miss detection vs ground truth ({len(events)} events / {len(rows)} scenes)",
                 color="white", fontsize=11)
    leg = ax.legend(fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color("black")
    ax.tick_params(colors="#8893a5")
    for s in ax.spines.values():
        s.set_color("#333")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "08_safety_roc.png", dpi=115, facecolor="#0e1117")
    print(f"\n[out] {OUT/'safety_scoreboard.csv'} and 08_safety_roc.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
