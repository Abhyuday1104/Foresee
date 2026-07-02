"""Matplotlib rendering for the dashboard and exported figures: HD-map layers, typed agents,
predicted modes, intent badges and conflict markers. No Streamlit imports here so it can
run headless.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # safe default; Streamlit/IPython override their own backend first
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .data.features import OBJECT_TYPE_NAMES  # noqa: E402

# Most-likely mode first (warm), trailing modes cooler - perceptually distinct.
MODE_COLORS: List[str] = ["#e6194B", "#f58231", "#ffe119", "#3cb44b", "#4363d8", "#911eb4",
                          "#42d4f4", "#f032e6"]

# Per-object-type rendering: (display label, colour, marker, marker size).
TYPE_STYLE: Dict[str, tuple] = {
    "vehicle":           ("Vehicle", "#c8ccd4", "s", 70),
    "bus":               ("Bus", "#1abc9c", "P", 130),
    "pedestrian":        ("Pedestrian", "#ff8c1a", "o", 45),
    "cyclist":           ("Cyclist", "#17d4f0", "^", 70),
    "motorcyclist":      ("Motorcyclist", "#e84393", "v", 70),
    "riderless_bicycle": ("Riderless bike", "#9b59b6", "X", 55),
    "ego":               ("Ego (AV)", "#ffd700", "*", 240),
}
_DEFAULT_STYLE = ("Other", "#6b7280", ".", 40)  # static / background / construction / unknown


def _style_for(type_id: int) -> tuple:
    name = OBJECT_TYPE_NAMES[type_id] if 0 <= type_id < len(OBJECT_TYPE_NAMES) else "unknown"
    return TYPE_STYLE.get(name, _DEFAULT_STYLE)


def _draw_map(ax, playback):
    """Draw rich HD-map geometry when available (drivable areas, lane edges, crosswalks),
    else fall back to lane centerlines (synthetic data)."""
    rm = playback.get("render_map")
    if rm:
        for poly in rm["drivable"]:                       # filled road surface
            ax.fill(poly[:, 0], poly[:, 1], facecolor="#16181e", edgecolor="#23262f",
                    lw=0.8, zorder=0)
        for cl in rm.get("centerlines", []):              # lane dividers (dashed)
            ax.plot(cl[:, 0], cl[:, 1], color="#4b5160", lw=0.7, ls=(0, (6, 6)),
                    alpha=0.55, zorder=1)
        for bnd in rm["lane_boundaries"]:                 # road edges (solid, brighter)
            ax.plot(bnd[:, 0], bnd[:, 1], color="#5b6273", lw=1.1, alpha=0.9, zorder=1,
                    solid_capstyle="round")
        for cw in rm["crosswalks"]:                       # pedestrian crossings
            ax.fill(cw[:, 0], cw[:, 1], facecolor="#2b3647", edgecolor="#94a3b8",
                    lw=0.8, alpha=0.55, zorder=1, hatch="||||")
    else:
        lanes = playback["lanes"]
        for li in range(lanes.shape[0]):
            if np.isnan(lanes[li]).any():
                continue
            ax.plot(lanes[li, :, 0], lanes[li, :, 1], color="#3a3f4b", lw=1.5, zorder=1)


_INTENT_COLORS = {
    "braking": "#ff6b6b", "stopped": "#ff6b6b", "accelerating": "#51cf66",
    "turning left": "#ffd43b", "turning right": "#ffd43b",
    "changing lanes left": "#22d3ee", "changing lanes right": "#22d3ee",
}


def _intent_tag(it):
    """Short label + colour for a *dynamic* maneuver only.

    Deliberately excludes "stopped" and "cruising": a parked or steadily-moving car is not worth a
    badge and labelling all of them buries the map. Only braking / accelerating / turning /
    lane-changing - the maneuvers that actually signal a developing situation - get a tag.
    """
    if it is None or not it.valid:
        return None
    short = {"braking": "braking", "accelerating": "accel.",
             "turning left": "turn L", "turning right": "turn R",
             "changing lanes left": "lane-chg L", "changing lanes right": "lane-chg R"}
    bits, color = [], "#a1a1aa"
    if it.longitudinal in ("braking", "accelerating"):
        bits.append(short[it.longitudinal])
        color = _INTENT_COLORS[it.longitudinal]
    if it.lateral in short:
        bits.append(short[it.lateral])
        color = _INTENT_COLORS[it.lateral]
    return (" · ".join(bits), color) if bits else None


def render_frame(playback: Dict[str, object], frame: int, top_k: int,
                 bounds=None, report=None, *, show_map: bool = True,
                 show_others: bool = True, show_extrapolation: bool = True,
                 show_ground_truth: bool = True, focus: bool = False,
                 intents=None, precursors=None, show_intent: bool = False) -> plt.Figure:
    """Render a single animation frame of a scenario at timestep ``frame`` (0..total_len-1).

    Observed motion accumulates up to the observation cutoff; at the cutoff the model's
    predicted futures appear and are progressively revealed while the focal agent's *actual*
    position advances along the ground truth - so the viewer sees which mode reality follows.

    If a ``report`` (from :func:`foresee.risk.assess`) is supplied and it found a conflict, the
    predicted conflict location is marked with a warning once playback reaches that moment.
    """
    tracks = playback["tracks_world"]      # (A, T, 2)
    valid = playback["valid"]              # (A, T)
    types = playback["object_types"]       # (A,)
    is_ego = playback["is_ego"]            # (A,)
    obs_len = playback["obs_len"]
    rate = playback["sample_rate_hz"]
    cutoff = obs_len - 1                    # last observed step => prediction time (t = 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    # figure patch too, otherwise saving without an explicit facecolor leaves a white border
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    if show_map:
        _draw_map(ax, playback)

    conf = (report or {}).get("conflict") if report else None
    is_threat = bool(conf and conf.get("level") in ("HIGH", "MEDIUM"))
    focus_active = bool(focus and is_threat)
    conflicting = {i for i, m in enumerate((conf or {}).get("per_mode", []))
                   if m.get("conflicts")} if conf else set()

    present_types = set()
    ego_idx = None
    intent_drawn = 0
    for a in range(tracks.shape[0]):
        if bool(is_ego[a]):
            ego_idx = a
        is_focal, ego = (a == 0), bool(is_ego[a])
        if not show_others and not is_focal and not ego:
            continue
        vis = valid[a, : frame + 1]
        if vis.sum() < 1:
            continue
        tid = int(types[a])
        _, color, marker, msize = _style_for(OBJECT_TYPE_NAMES.index("ego") if ego else tid)
        present_types.add("ego" if ego else OBJECT_TYPE_NAMES[tid])
        # In focus mode, fade everyone who isn't the focal or the AV.
        dim = 0.18 if (focus_active and not is_focal and not ego) else 1.0
        xy = tracks[a, : frame + 1][vis]
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.0, alpha=0.45 * dim, zorder=2)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker=marker, s=msize,
                   edgecolor="#0e1117", linewidths=0.6, zorder=5, alpha=0.9 * dim)
        # Per-agent intent badge (only for *notable, dynamic* intents; capped to avoid clutter).
        if show_intent and intents is not None and a < len(intents) and dim > 0.5 and intent_drawn < 8:
            tag = _intent_tag(intents[a])
            if tag:
                ax.annotate(tag[0], xy=(xy[-1, 0], xy[-1, 1]), xytext=(6, -10),
                            textcoords="offset points", color=tag[1], fontsize=7.5,
                            fontweight="bold", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.12", fc="#0e1117",
                                      ec=tag[1], alpha=0.85))
                intent_drawn += 1
        if is_focal:  # focal / target agent - the one we predict (the potential threat)
            ring = "#ff3b30" if is_threat else "#1f9bff"
            ax.plot(xy[:, 0], xy[:, 1], color="#1f9bff", lw=2.5, zorder=6)
            ax.scatter(xy[-1, 0], xy[-1, 1], facecolor="none", edgecolor=ring,
                       s=300, linewidths=2.4, zorder=7)
        elif ego:  # the AV / ego vehicle - a small persistent tag so the pair is clear
            ax.annotate("AV", xy=(xy[-1, 0], xy[-1, 1]), xytext=(7, 7),
                        textcoords="offset points", color="#ffd700", fontsize=9,
                        fontweight="bold", zorder=12)

    # --- Focal ground-truth future (for verification): faint green dashed ---
    if frame >= cutoff and show_ground_truth:
        gt = tracks[0, cutoff:][valid[0, cutoff:]]
        if gt.shape[0] >= 2:
            ax.plot(gt[:, 0], gt[:, 1], color="#2ecc71", lw=2.0, ls=(0, (4, 3)),
                    alpha=0.75, zorder=6)

    # --- The AV's own path ahead (gold dashed), extended over the full horizon ---
    if frame >= cutoff:
        ego_future = playback.get("ego_future_full")
        if ego_future is None and ego_idx is not None:
            ego_future = tracks[ego_idx, cutoff:][valid[ego_idx, cutoff:]]
        if ego_future is not None and len(ego_future) >= 2:
            ego_future = np.asarray(ego_future)
            ax.plot(ego_future[:, 0], ego_future[:, 1], color="#ffd700", lw=2.0, ls="--",
                    alpha=0.85, zorder=6)

    # --- Predictions: solid over the model horizon, dotted for the extrapolated tail ---
    mode_handles = []
    if frame >= cutoff:
        preds = playback["pred_world"]                 # (K, Tf+E, 2)
        probs = playback["probabilities"]
        start = playback["focal_xy"]
        n_model = int(playback.get("n_model_steps", preds.shape[1]))
        reveal = int(np.clip(frame - cutoff, 0, n_model))
        k = min(top_k, preds.shape[0])
        for i in range(k):
            color = MODE_COLORS[i % len(MODE_COLORS)]
            # In focus mode, draw only the mode(s) that cause the conflict; fade the rest hard.
            faded = focus_active and i not in conflicting
            dim = 0.15 if faded else 1.0
            lw = 1.3 + 2.5 * float(probs[i])
            alpha = float(np.clip(0.45 + 0.55 * probs[i], 0.35, 0.95)) * dim
            shown = np.vstack([start, preds[i, :reveal]]) if reveal > 0 else start[None]
            ax.plot(shown[:, 0], shown[:, 1], color=color, lw=lw, alpha=alpha, zorder=8)
            # Extrapolated tail (beyond the 6 s of data) as a dotted continuation.
            if show_extrapolation and preds.shape[1] > n_model and not faded:
                tail = preds[i, n_model - 1:]
                ax.plot(tail[:, 0], tail[:, 1], color=color, lw=1.2, ls=":", alpha=0.5, zorder=7)
            # Small numbered chip at the endpoint, keyed to the side panel.
            end = preds[i, -1] if show_extrapolation else preds[i, n_model - 1]
            ax.scatter(end[0], end[1], s=130, color=color, edgecolor="#0e1117",
                       linewidths=0.8, zorder=9, alpha=dim)
            ax.annotate(str(i + 1), xy=(end[0], end[1]), color="#0e1117", fontsize=8,
                        fontweight="bold", ha="center", va="center", zorder=10, alpha=dim)
            mode_handles.append(Line2D([0], [0], color=color, lw=2.5,
                                       label=f"{i+1}. {probs[i]*100:.0f}%"))

        # --- Predicted conflict: compact marker + connector to the AV ---
        if conf and conf.get("conflict_point") is not None and conf.get("ttc_s") is not None:
            cx, cy = conf["conflict_point"]                 # threat's predicted position
            conflict_step = int(round(conf["ttc_s"] * rate)) - 1
            reached = (frame - cutoff) >= conflict_step
            col = "#ff3b30" if reached else "#ff9500"
            ego_pt = conf.get("ego_point")
            if ego_pt is not None:
                ax.plot([cx, ego_pt[0]], [cy, ego_pt[1]], color=col, lw=1.8, ls=":", zorder=11)
                # Mark where the AV is at that instant so the conflict pair is unambiguous.
                ax.scatter([ego_pt[0]], [ego_pt[1]], marker="*", s=300, color="#ffd700",
                           edgecolor=col, linewidths=1.4, zorder=11)
                ax.annotate("AV", xy=ego_pt, xytext=(6, -12), textcoords="offset points",
                            color="#ffd700", fontsize=9, fontweight="bold", zorder=12)
            ax.scatter([cx], [cy], marker="X", s=300, color=col,
                       edgecolor="white", linewidths=1.3, zorder=11)
            ax.annotate(f"{'CONFLICT' if reached else 'risk'}: {conf['threat_type']} vs AV, "
                        f"{conf['closest_m']:.1f} m in {conf['ttc_s']:.1f} s",
                        xy=(cx, cy), xytext=(10, 10), textcoords="offset points", color=col,
                        fontsize=9, fontweight="bold", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#0e1117", ec=col))

    # --- Observed precursor highlight: ring the threat agent + link it to the AV ---
    if precursors:
        for p in precursors[:3]:
            if p.threat is None:
                continue
            tv, pv = valid[p.threat, : frame + 1], valid[p.protagonist, : frame + 1]
            if tv.sum() < 1 or pv.sum() < 1:
                continue
            tp = tracks[p.threat, : frame + 1][tv][-1]
            pp = tracks[p.protagonist, : frame + 1][pv][-1]
            ax.scatter(tp[0], tp[1], facecolor="none", edgecolor="#ff9500", s=380,
                       linewidths=2.2, zorder=10)
            ax.plot([pp[0], tp[0]], [pp[1], tp[1]], color="#ff9500", lw=1.2, ls=":",
                    alpha=0.7, zorder=10)

    # --- Phase banner (past vs future, in seconds relative to "now") ---
    t_rel = (frame - cutoff) / rate
    n_model = int(playback.get("n_model_steps", playback["pred_world"].shape[1]))
    extra_s = (playback["pred_world"].shape[1] - n_model) / rate
    if frame < cutoff:
        phase = f"OBSERVING   t = {t_rel:+.1f}s   (watching {obs_len/rate:.0f}s of history)"
    else:
        phase = (f"FORECASTING   t = {t_rel:+.1f}s   "
                 f"(model {n_model/rate:.0f}s + {extra_s:.0f}s extrapolated, dotted)")
    ax.set_title(phase, color="white", fontsize=13)

    # --- Legends ---
    if frame >= cutoff:
        ring = "#ff3b30" if is_threat else "#1f9bff"
        ref_handles = [
            Line2D([0], [0], color=ring, lw=2.5,
                   label=("Threat (predicted agent)" if is_threat else "Target (predicted agent)")),
            Line2D([0], [0], color="#2ecc71", lw=2, ls="--", label="What actually happened"),
            Line2D([0], [0], color="#ffd700", lw=2, ls="--", label="AV path"),
        ]
        legend_handles = ref_handles + mode_handles
        leg1 = ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
                         framealpha=0.9, title="Paths  (numbers = mode %)")
        leg1.get_title().set_color("black")
        ax.add_artist(leg1)
    type_handles = []
    for name in OBJECT_TYPE_NAMES:
        if name not in present_types:
            continue
        label, color, marker, _ = TYPE_STYLE.get(name, _DEFAULT_STYLE)
        type_handles.append(Line2D([0], [0], marker=marker, color="none",
                                   markerfacecolor=color, markeredgecolor="#0e1117",
                                   markersize=10, label=label))
    if type_handles:
        leg2 = ax.legend(handles=type_handles, loc="upper right", fontsize=9,
                         framealpha=0.9, title="Road users")
        leg2.get_title().set_color("black")

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    fig.tight_layout()
    return fig


