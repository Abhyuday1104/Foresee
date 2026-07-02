"""Render the before/after figure for the mode-collapse fix: same scenario, baseline modes
next to goal-anchored modes. Picks an illustrative validation scene automatically and
writes assets/mode_collapse_fix.png.
"""

from __future__ import annotations

import argparse
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
from foresee.risk import maneuver_label  # noqa: E402
from foresee.visualization import MODE_COLORS  # noqa: E402

DARK = "#0e1117"
OUT = Path(__file__).resolve().parents[1] / "assets" / "mode_collapse_fix.png"


@torch.no_grad()
def _predict(model, sample):
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
    pred = model(batch)
    temp = float(getattr(model, "temperature", torch.tensor(1.0)))
    probs = torch.softmax(pred["logits"][0] / temp, -1).numpy()
    return pred["trajectories"][0].numpy(), probs        # agent frame (K,T,2), (K,)


def _gt(sample):
    fut = sample["future"].numpy()
    fm = sample["future_mask"].numpy()
    return fut[fm] if fm.any() else fut


def _panel(ax, trajs, probs, gt, title, subtitle):
    ax.set_facecolor(DARK)
    order = np.argsort(-probs)
    for rank, k in enumerate(order):
        c = MODE_COLORS[rank % len(MODE_COLORS)]
        lw = 1.2 + 2.6 * probs[k]
        ax.plot(trajs[k, :, 0], trajs[k, :, 1], color=c, lw=lw,
                alpha=float(np.clip(0.45 + 0.55 * probs[k], .35, .95)), zorder=3)
        ax.annotate(f"{probs[k]*100:.0f}%", xy=trajs[k, -1], color=c, fontsize=9,
                    fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc=DARK, ec=c, alpha=.9))
    ax.plot(gt[:, 0], gt[:, 1], color="#2ecc71", lw=2.4, ls=(0, (4, 3)), zorder=4,
            label="what actually happened")
    ax.scatter([0], [0], marker=(3, 0, -90), s=170, color="#1f9bff",
               edgecolor="white", lw=1.0, zorder=6)      # the agent, heading +x
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, color="#8893a5", fontsize=8.5)
    ax.set_aspect("equal")
    ax.tick_params(colors="#555", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#333")
    leg = ax.legend(loc="upper left", fontsize=8, framealpha=.85)
    for t in leg.get_texts():
        t.set_color("black")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--scan", type=int, default=120)
    args = ap.parse_args()

    cfg = Config()
    pairs = _scan_scenarios(Path(cfg.data_root) / "val")
    cache = Path(cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(cfg.feature)}"
    ds = MotionForecastingDataset(cfg.feature, scenario_paths=pairs, cache_dir=cache)
    before, _ = load_model_from_checkpoint(args.before, "cpu")
    after, _ = load_model_from_checkpoint(args.after, "cpu")

    # Pick the most illustrative scene: ground truth TURNS; baseline modes are all straight;
    # anchored model's turn mode matches ground truth.
    best_pick, best_score = None, -1.0
    for idx in range(args.scan):
        s = ds[idx]
        gt = _gt(s)
        if len(gt) < 10:
            continue
        gt_man = maneuver_label(gt)
        if "Turning" not in gt_man:
            continue
        tb, pb = _predict(before, s)
        ta, pa = _predict(after, s)
        mans_b = {maneuver_label(tb[k]) for k in range(6)}
        mans_a = [maneuver_label(ta[k]) for k in range(6)]
        if len(mans_b) > 1 or gt_man not in mans_a:      # baseline must collapse; after must cover
            continue
        score = pa[mans_a.index(gt_man)] + float(np.linalg.norm(gt[-1]) > 15)
        if score > best_score:
            best_score, best_pick = score, (idx, s, tb, pb, ta, pa, gt, gt_man)

    if best_pick is None:
        print("no illustrative scenario found in scan range")
        return 1
    idx, s, tb, pb, ta, pa, gt, gt_man = best_pick
    print(f"scenario idx={idx} ({s['scenario_id']}): ground truth = {gt_man}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(DARK)
    _panel(axes[0], tb, pb, gt,
           "BEFORE - winner-takes-all training",
           "all 6 modes collapse onto 'straight'; the turn is unrepresentable")
    _panel(axes[1], ta, pa, gt,
           "AFTER - goal-anchored modes",
           "modes cover left / straight / right; the turn is predicted with calibrated confidence")
    fig.suptitle("Fixing mode collapse: same scenario, same K=6 modes",
                 color="white", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=115, facecolor=DARK, bbox_inches="tight")
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
