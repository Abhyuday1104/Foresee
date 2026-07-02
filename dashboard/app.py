"""Streamlit dashboard.

    streamlit run dashboard/app.py

Conflict verdict, frame-by-frame playback, per-agent intent, detected precursors and a
batch audit. Falls back to the bundled demo data and checkpoint when FORESEE_DATA_ROOT
is not set.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foresee import playback  # noqa: E402
from foresee.agent_intent import predict_agent_intents  # noqa: E402
from foresee.checkpoint import load_model_from_checkpoint  # noqa: E402
from foresee.config import Config  # noqa: E402
from foresee.data.dataset import MotionForecastingDataset  # noqa: E402
from foresee.data.features import OBJECT_TYPE_NAMES  # noqa: E402
from foresee.intent import scene_intents  # noqa: E402
from foresee.models import build_model  # noqa: E402
from foresee.precursors import find_precursors  # noqa: E402
from foresee.risk import assess, describe_modes  # noqa: E402
from foresee.visualization import MODE_COLORS, render_frame  # noqa: E402

AUDIT_LIMIT = 60

# ----------------------------- shadcn-ish styling -----------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
}
[data-testid="stAppViewContainer"], .stApp { background: #09090b; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px; }
/* Cards = Streamlit bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0c0c0f; border: 1px solid #27272a !important;
    border-radius: 12px; padding: 2px 6px;
}
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 650; }
hr { border-color: #27272a; }
/* Buttons */
.stButton > button {
    background: #18181b; color: #fafafa; border: 1px solid #27272a;
    border-radius: 8px; font-weight: 500; transition: all .12s ease;
}
.stButton > button:hover { background: #27272a; border-color: #3f3f46; }
/* Sidebar */
[data-testid="stSidebar"] { background: #0c0c0f; border-right: 1px solid #27272a; }
/* Badges */
.badge { display:inline-block; padding:2px 10px; border-radius:9999px; font-size:.72rem;
         font-weight:600; letter-spacing:.01em; border:1px solid transparent; }
.badge-destructive { background:rgba(239,68,68,.14); color:#fca5a5; border-color:rgba(239,68,68,.4); }
.badge-warning     { background:rgba(245,158,11,.14); color:#fcd34d; border-color:rgba(245,158,11,.4); }
.badge-success     { background:rgba(34,197,94,.14);  color:#86efac; border-color:rgba(34,197,94,.4); }
.badge-secondary   { background:#27272a; color:#d4d4d8; border-color:#3f3f46; }
.badge-outline     { background:transparent; color:#a1a1aa; border-color:#3f3f46; }
/* Stat cards */
.stat { background:#0c0c0f; border:1px solid #27272a; border-radius:12px; padding:14px 16px; }
.stat-label { color:#a1a1aa; font-size:.72rem; font-weight:600; text-transform:uppercase;
              letter-spacing:.05em; }
.stat-value { font-size:1.5rem; font-weight:700; margin-top:2px; }
.stat-sub { color:#71717a; font-size:.74rem; margin-top:1px; }
.muted { color:#a1a1aa; }
.row-item { padding:7px 0; border-bottom:1px solid #1f1f23; font-size:.86rem; }
.dot { display:inline-block; width:8px; height:8px; border-radius:9999px; margin-right:7px; }
</style>
"""


def badge(text: str, variant: str = "secondary") -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'


def stat_card(label: str, value: str, sub: str = "", color: str = "#fafafa") -> None:
    st.markdown(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)


_LEVEL_BADGE = {"HIGH": "destructive", "MEDIUM": "warning", "LOW": "success", "NO_EGO": "secondary"}
_LEVEL_COLOR = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#4ade80", "NO_EGO": "#a1a1aa"}


# ----------------------------- cached resources -----------------------------
_DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"


@st.cache_resource
def get_config() -> Config:
    cfg = Config()
    # fall back to the bundled demo scenarios when no dataset is configured
    if not cfg.has_real_data() and (_DEMO_DIR / "data" / "val").is_dir():
        cfg.data_root = str(_DEMO_DIR / "data")
    return cfg


@st.cache_resource
def get_dataset(_cfg: Config) -> MotionForecastingDataset:
    if _cfg.has_real_data():
        from foresee.data.dataset import _feature_signature, _scan_scenarios
        pairs = _scan_scenarios(Path(_cfg.data_root) / "val")
        if pairs:
            cache = Path(_cfg.data_root) / ".foresee_cache" / f"val-{_feature_signature(_cfg.feature)}"
            return MotionForecastingDataset(_cfg.feature, scenario_paths=pairs, cache_dir=cache)
    return MotionForecastingDataset(_cfg.feature, synthetic_size=512, synthetic_offset=10_000_000)


def discover_checkpoints() -> list[str]:
    """Known checkpoint locations only - the picker is a fixed list, not a free-text path,
    so dashboard users can't point the loader at arbitrary files on the host."""
    found = sorted(Path(".").glob("runs*/**/best.pt"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    demo_ckpt = _DEMO_DIR / "checkpoint.pt"
    if demo_ckpt.is_file():
        found.append(demo_ckpt)
    return [str(p) for p in found]


@st.cache_resource
def get_model(_cfg: Config, checkpoint: str):
    if checkpoint and Path(checkpoint).is_file():
        return load_model_from_checkpoint(checkpoint, device="cpu")[0]
    torch.manual_seed(0)
    return build_model(_cfg.arch, _cfg.feature, _cfg.model).eval()


@st.cache_resource
def get_playback(_dataset, _model, idx: int, checkpoint: str):
    return playback.build_playback(_dataset, idx, _model, device="cpu")


@st.cache_resource
def get_scene_analysis(_dataset, idx: int):
    """Observed-kinematics intent + precursors (no model needed)."""
    scene = playback.raw_scene(_dataset, idx)
    intents = scene_intents(scene["tracks_world"], scene["valid"], scene["obs_len"] - 1,
                            1.0 / scene["sample_rate_hz"])
    return scene, intents, find_precursors(scene)


@st.cache_resource
def get_agent_intents(_dataset, _model, idx: int, checkpoint: str):
    """Model-predicted intent of the nearest surrounding vehicles (one forecast per agent)."""
    return predict_agent_intents(_dataset, idx, _model, n_agents=5, device="cpu")


@st.cache_resource
def get_audit(_dataset, _model, checkpoint: str, limit: int):
    rows = []
    for idx in range(min(limit, len(_dataset))):
        pb = playback.build_playback(_dataset, idx, _model, device="cpu")
        c = assess(pb)["conflict"]
        rows.append({"idx": idx, "scenario": pb["scenario_id"][:8], "risk": round(c["risk"], 2),
                     "level": c["level"], "ttc_s": c["ttc_s"], "closest_m": round(c["closest_m"], 1),
                     "threat": c["threat_type"]})
    rows.sort(key=lambda r: (-r["risk"], r["closest_m"]))
    return rows


def _agent_name(i, scene):
    if bool(scene["is_ego"][i]):
        return "AV (ego)"
    if i == 0:
        return "Target"
    return OBJECT_TYPE_NAMES[int(scene["object_types"][i])].capitalize()


# ----------------------------- app -----------------------------
def main() -> None:
    st.set_page_config(page_title="Foresee", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    cfg = get_config()
    dataset = get_dataset(cfg)

    # ---- sidebar ----
    st.sidebar.markdown("### Controls")
    options = discover_checkpoints()
    checkpoint = st.sidebar.selectbox("Checkpoint", options) if options else ""
    st.sidebar.markdown(
        badge("checkpoint loaded", "success") if (checkpoint and Path(checkpoint).is_file())
        else badge("untrained model", "warning"), unsafe_allow_html=True)
    if str(_DEMO_DIR / "data") == str(cfg.data_root):
        src = "Demo bundle (18 real AV2 scenarios)"
    elif cfg.has_real_data():
        src = "Real Argoverse 2"
    else:
        src = "Synthetic"
    st.sidebar.caption(f"Data source · {src}")
    model = get_model(cfg, checkpoint)
    audit = get_audit(dataset, model, checkpoint, AUDIT_LIMIT)

    if "scenario_idx" not in st.session_state:
        st.session_state.scenario_idx = audit[0]["idx"] if audit else 0
    flagged = [r for r in audit if r["level"] in ("HIGH", "MEDIUM")][:20]
    st.sidebar.markdown("#### Scenario")
    if flagged:
        labels = {f"#{i+1}  ·  risk {r['risk']:.2f}  ·  {r['level']}  ({r['threat']})": r["idx"]
                  for i, r in enumerate(flagged)}
        choice = st.sidebar.selectbox("Jump to a flagged scenario", [" - "] + list(labels))
        if choice != " - " and labels[choice] != st.session_state.scenario_idx:
            st.session_state.scenario_idx = labels[choice]
            st.session_state.frame, st.session_state.playing = 0, False
            st.rerun()
    idx = st.sidebar.slider("Scenario index", 0, len(dataset) - 1, key="scenario_idx")
    top_k = st.sidebar.slider("Predictions shown", 1, cfg.model.num_modes, cfg.model.num_modes)
    speed = st.sidebar.select_slider("Playback speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)

    st.sidebar.markdown("#### View")
    focus = st.sidebar.checkbox("Focus on the conflict", value=False)
    show_intent = st.sidebar.checkbox("Show agent intent", value=True)
    show_map = st.sidebar.checkbox("HD map detail", value=True)
    show_others = st.sidebar.checkbox("Other road users", value=True)
    show_extrap = st.sidebar.checkbox("Extrapolation (+4s)", value=True)
    show_gt = st.sidebar.checkbox("Ground truth", value=True)

    if st.session_state.get("_shown_idx") != idx:
        st.session_state._shown_idx, st.session_state.frame, st.session_state.playing = idx, 0, False

    pb = get_playback(dataset, model, idx, checkpoint)
    report = assess(pb)
    scene, intents, precursors = get_scene_analysis(dataset, idx)
    agent_intents = get_agent_intents(dataset, model, idx, checkpoint)
    c, it = report["conflict"], report["intent"]
    bounds = playback.view_bounds(pb)
    T, cutoff = pb["total_len"], pb["obs_len"] - 1

    # ---- header ----
    st.markdown("## Foresee")
    st.markdown('<span class="muted">Conflict warning and safety audit for autonomous driving - '
                'who is the AV in danger from, and why.</span>', unsafe_allow_html=True)
    with st.expander("About - what am I looking at?"):
        st.markdown(
            "The problem. A self-driving fleet logs millions of miles; the dangerous fraction "
            "is what safety teams must find and can't watch by hand. Foresee predicts where every "
            "road user is going and turns that into a decision.\n\n"
            "On this screen, for one real driving scenario:\n"
            "- Verdict + stats - the headline conflict risk between the predicted *target* "
            "vehicle and the self-driving car (AV): how likely, how soon, how close, and the "
            "reason (e.g. *closing on the AV*, *lane change into the AV*).\n"
            "- Map - a bird's-eye view of the HD map and every road user. The blue ring is "
            "the agent we predict; AV is the self-driving car; green dashed is what the "
            "agent *actually* did (a built-in accuracy check). Coloured paths are the model's "
            "possible futures; the dotted tails extend a few seconds further so the AV can stay "
            "ready for how the situation develops.\n"
            "- Scene intent - what each nearby vehicle is doing *right now*, read from its "
            "motion (braking, turning, changing lanes). This is observed, not predicted.\n"
            "- Observed precursors - moments whose geometry matches how real Tesla driver-"
            "assist crashes actually happen (e.g. the AV closing on a stopped lead without "
            "braking), detected from motion alone.\n"
            "- Batch audit - the same risk score run across many scenarios so the riskiest "
            "surface first.\n\n"
            "Honest limits. The model's confidence scores are being calibrated and its "
            "predicted modes are still close to one another; the conflict score is a screening "
            "signal, not a probability of collision. See `DESIGN_REVIEW.md` and `INSIGHTS.md`.")
    st.write("")

    # ---- verdict card ----
    with st.container(border=True):
        if c["level"] == "NO_EGO":
            st.markdown(f"{badge('NO EGO', 'secondary')} &nbsp; "
                        "<span class='muted'>No AV tracked - conflict risk not applicable.</span>",
                        unsafe_allow_html=True)
        else:
            reason = f" {c['threat_type']} <b>{c['reason']}</b>" if c.get("reason") else \
                     f" threat: <b>{c['threat_type']}</b>"
            ttc = f" · conflict in <b>{c['ttc_s']:.1f}s</b>" if c["ttc_s"] is not None else ""
            st.markdown(
                f"{badge('CONFLICT RISK ' + c['level'], _LEVEL_BADGE[c['level']])} &nbsp; "
                f"<span style='font-size:1.02rem'>{reason}{ttc} · "
                f"closest <b>{c['closest_m']:.1f} m</b></span><br>"
                f"<span class='muted'>-> {c['action']}</span>", unsafe_allow_html=True)

    # ---- stat row ----
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        stat_card("Conflict risk", f"{c['risk']*100:.0f}%" if c["level"] != "NO_EGO" else " - ",
                  c["level"], _LEVEL_COLOR[c["level"]])
    with s2:
        stat_card("Time to conflict", f"{c['ttc_s']:.1f}s" if c["ttc_s"] else " - ", "predicted")
    with s3:
        stat_card("Closest approach",
                  f"{c['closest_m']:.1f} m" if c["level"] != "NO_EGO" else " - ", "AV vs threat")
    with s4:
        stat_card("Intent ambiguity", it["level"], it["message"],
                  {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#4ade80"}[it["level"]])
    st.write("")

    # ---- transport ----
    b1, b2, b3, b4, _ = st.columns([1.1, 1.1, 1.1, 1.5, 3.2])
    if b1.button("Restart", width="stretch"):
        st.session_state.frame, st.session_state.playing = 0, False
    if b2.button("Play", width="stretch"):
        st.session_state.playing = True
    if b3.button("Pause", width="stretch"):
        st.session_state.playing = False
    if b4.button("Jump to now", width="stretch"):
        st.session_state.frame, st.session_state.playing = cutoff, False
    frame = st.slider("Timestep", 0, T - 1, st.session_state.get("frame", 0),
                      label_visibility="collapsed")
    st.session_state.frame = frame

    # ---- map + side panels ----
    col_map, col_side = st.columns([3, 1.25])
    with col_map:
        with st.container(border=True):
            fig = render_frame(pb, frame, top_k, bounds, report=report, show_map=show_map,
                               show_others=show_others, show_extrapolation=show_extrap,
                               show_ground_truth=show_gt, focus=focus,
                               intents=intents, precursors=precursors, show_intent=show_intent)
            # st.pyplot's own savefig leaves a white figure border; save it ourselves instead
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, facecolor="#0e1117", bbox_inches="tight")
            plt.close(fig)
            st.image(buf.getvalue(), width="stretch")
    with col_side:
        with st.container(border=True):
            st.markdown("Observed precursors")
            sev_p = [p for p in precursors if p.severity >= 0.3]
            if sev_p:
                for p in sorted(sev_p, key=lambda x: -x.severity)[:4]:
                    var = "destructive" if p.severity > 0.6 else "warning"
                    st.markdown(f"{badge(p.category.split(' / ')[0], var)} "
                                f"<span class='muted' style='font-size:.8rem'>sev {p.severity:.2f}</span><br>"
                                f"<span style='font-size:.84rem'>{p.description}</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("<span class='muted' style='font-size:.85rem'>No severe precursor "
                            "in this scene.</span>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("Scene intent")
            shown = 0
            for i, intent in enumerate(intents):
                dynamic = (intent.valid and (intent.longitudinal in ("braking", "accelerating")
                           or intent.lateral not in ("straight", "unknown")))
                if not dynamic or shown >= 6:
                    continue
                col = "#f87171" if intent.is_braking() else "#22d3ee" if intent.is_lane_changing() \
                    else "#ffd43b" if intent.is_turning() else "#a1a1aa"
                st.markdown(f"<span class='dot' style='background:{col}'></span>"
                            f"<b>{_agent_name(i, scene)}</b> · "
                            f"<span class='muted' style='font-size:.82rem'>{intent.label}</span>",
                            unsafe_allow_html=True)
                shown += 1
            if shown == 0:
                st.markdown("<span class='muted' style='font-size:.85rem'>All agents cruising "
                            "straight.</span>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("Predicted intent - nearby vehicles")
            st.caption("Forecast (the model re-run per vehicle), not just observed motion.")
            if agent_intents:
                _icol = {"Turning left": "#ffd43b", "Turning right": "#ffd43b",
                         "Slowing / stopping": "#f87171"}
                for a in agent_intents:
                    col = _icol.get(a.maneuver, "#a1a1aa")
                    st.markdown(f"<span class='dot' style='background:{col}'></span>"
                                f"<b>{a.type_name.capitalize()}</b> · "
                                f"<span class='muted' style='font-size:.82rem'>"
                                f"{a.maneuver.lower()} ({a.prob*100:.0f}%)</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("<span class='muted' style='font-size:.85rem'>No nearby vehicles.</span>",
                            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("Predicted modes (target agent)")
            probs, descs = pb["probabilities"], describe_modes(pb, report)
            for i in range(min(top_k, len(probs))):
                col = MODE_COLORS[i % len(MODE_COLORS)]
                st.markdown(f"<span class='dot' style='background:{col}'></span>"
                            f"<b>{probs[i]*100:.0f}%</b> "
                            f"<span class='muted' style='font-size:.8rem'>{descs[i]}</span>",
                            unsafe_allow_html=True)

    # ---- explainer + audit ----
    with st.expander("How to read the map / why predict beyond the sensor horizon"):
        st.markdown(
            "- Blue ring = the agent we predict (red when it's a *threat*). Gold star/AV = "
            "the self-driving car. Green dashed = what actually happened (sanity check).\n"
            "- Intent badges read each vehicle's current maneuver from its motion (braking, "
            "turning, lane-changing) - that's how a precursor like *'AV not braking on a stopped "
            "lead'* is detected.\n"
            "- Paths are extended a few seconds past the model horizon (dotted) so the AV can "
            "stay ready for how the situation evolves, not just the next instant.")
    with st.container(border=True):
        st.markdown(f"Batch safety audit &nbsp; "
                    f"{badge(str(sum(r['level']=='HIGH' for r in audit)) + ' HIGH', 'destructive')} "
                    f"{badge(str(sum(r['level']=='MEDIUM' for r in audit)) + ' MED', 'warning')} "
                    f"<span class='muted'>of first {len(audit)} scenarios</span>",
                    unsafe_allow_html=True)
        st.dataframe([{k: r[k] for k in ("idx", "scenario", "risk", "level", "ttc_s",
                                         "closest_m", "threat")} for r in audit[:20]],
                     width="stretch", hide_index=True)

    st.caption(f"Scenario `{pb['scenario_id']}` · {pb['obs_len']/pb['sample_rate_hz']:.0f}s observed "
               f"-> {pb['n_model_steps']/pb['sample_rate_hz']:.0f}s predicted "
               f"+ {(pb['pred_world'].shape[1]-pb['n_model_steps'])/pb['sample_rate_hz']:.0f}s extrapolated")

    # ---- autoplay ----
    if st.session_state.get("playing"):
        if frame < T - 1:
            st.session_state.frame = frame + 1
            time.sleep(0.12 / speed)
            st.rerun()
        else:
            st.session_state.playing = False


if __name__ == "__main__":
    main()
