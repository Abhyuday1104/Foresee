"""End-to-end contract check on synthetic data (no download needed): dataset -> model ->
loss -> metrics -> inference -> checkpoint -> risk.

    python -m tests.smoke_test
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foresee.checkpoint import load_model_from_checkpoint  # noqa: E402
from foresee.config import Config  # noqa: E402
from foresee.data import build_datasets, collate_samples  # noqa: E402
from foresee.inference import run_inference  # noqa: E402
from foresee.losses import multimodal_loss  # noqa: E402
from foresee.metrics import forecasting_metrics  # noqa: E402
from foresee.models import build_model  # noqa: E402


def main() -> int:
    cfg = Config()
    B = 4
    fc, mc = cfg.feature, cfg.model
    print(f"shapes: A={fc.max_agents} T_obs={fc.num_history_steps} T_fut={fc.num_future_steps} "
          f"L={fc.max_lanes} P={fc.lane_num_points} K={mc.num_modes} H={mc.hidden_dim}")

    # ---- 1. Dataset + collate ----
    train_ds, val_ds = build_datasets(cfg)
    loader = DataLoader(train_ds, batch_size=B, collate_fn=collate_samples, num_workers=0)
    batch = next(iter(loader))
    assert batch["hist"].shape == (B, fc.max_agents, fc.num_history_steps, fc.agent_feature_dim)
    assert batch["lanes"].shape == (B, fc.max_lanes, fc.lane_num_points, fc.lane_feature_dim)
    assert batch["future"].shape == (B, fc.num_future_steps, 2)
    assert batch["rotation"].shape == (B, 2, 2)
    assert batch["object_types"].shape == (B, fc.max_agents)
    assert batch["object_types"].dtype == torch.int64
    assert (batch["object_types"] >= 0).all() and (batch["object_types"] < fc.num_object_types).all()
    assert batch["is_ego"].shape == (B, fc.max_agents)
    assert batch["cur_pos"].shape == (B, fc.max_agents, 2)
    print("[ok] dataset/collate shapes (incl. object types/ego/cur_pos)")

    # ---- 2. Forward ----
    model = build_model("anchored", fc, mc)
    pred = model(batch)
    assert pred["trajectories"].shape == (B, mc.num_modes, fc.num_future_steps, 2)
    assert pred["logits"].shape == (B, mc.num_modes)
    if mc.predict_uncertainty:
        assert pred["scale"].shape == (B, mc.num_modes, fc.num_future_steps, 2)
        assert (pred["scale"] > 0).all(), "Laplace scale must be strictly positive"
    print("[ok] forward shapes")

    # ---- 3. Loss + backward ----
    out = multimodal_loss(pred, batch["future"], batch["future_mask"], cfg.train)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"]), "loss must be finite scalar"
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "gradients must be finite"
    print(f"[ok] loss={out['loss'].item():.3f} reg={out['reg_loss'].item():.3f} "
          f"cls={out['cls_loss'].item():.3f} + backward")

    # ---- 4. Metrics ----
    metrics = forecasting_metrics(pred, batch["future"], batch["future_mask"],
                                  cfg.train.miss_threshold_m)
    for key in ("minADE", "minFDE", "MR", "brier_minFDE"):
        assert metrics[key].shape == (B,), key
        assert torch.isfinite(metrics[key]).all(), key
    assert ((metrics["MR"] == 0) | (metrics["MR"] == 1)).all(), "MR must be binary"
    print(f"[ok] metrics: minADE={metrics['minADE'].mean():.3f} "
          f"minFDE={metrics['minFDE'].mean():.3f} MR={metrics['MR'].mean():.3f}")

    # ---- 5. Inference (world-frame conversion for the dashboard) ----
    scene = run_inference(model, val_ds[0], device="cpu")
    assert scene["pred_trajectories"].shape == (mc.num_modes, fc.num_future_steps, 2)
    assert scene["probabilities"].shape == (mc.num_modes,)
    assert scene["object_types"].shape == (fc.max_agents,)
    assert scene["is_ego"].shape == (fc.max_agents,)
    assert abs(float(scene["probabilities"].sum()) - 1.0) < 1e-4, "probs must sum to 1"
    # Probabilities must be sorted descending (most-likely mode first).
    assert (scene["probabilities"][:-1] >= scene["probabilities"][1:] - 1e-6).all()
    print("[ok] inference + world conversion")

    # ---- 6. Checkpoint round-trip (with arch) ----
    from foresee.checkpoint import save_checkpoint
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        save_checkpoint(path, model, "anchored", fc, mc)
        reloaded, _ = load_model_from_checkpoint(str(path), device="cpu")
        model.eval()
        with torch.no_grad():
            p_eval = model(batch)
            p2 = reloaded(batch)
        assert torch.allclose(p2["trajectories"], p_eval["trajectories"], atol=1e-5)
    print("[ok] checkpoint round-trip")

    # ---- 7. Every registered architecture forwards + backprops ----
    from foresee.models import ARCHS
    for arch in ARCHS:
        m = build_model(arch, fc, mc)
        out = multimodal_loss(m(batch), batch["future"], batch["future_mask"], cfg.train)
        assert torch.isfinite(out["loss"]), arch
        out["loss"].backward()
        assert any(p.grad is not None for p in m.parameters()), arch
    print(f"[ok] archs forward/backward: {list(ARCHS)}")

    # ---- 8. Risk module (conflict + intent) on a playback scene ----
    from foresee.playback import build_playback
    from foresee.risk import assess
    pb = build_playback(train_ds, 0, model, device="cpu")
    rep = assess(pb)
    c, it = rep["conflict"], rep["intent"]
    assert c["level"] in ("HIGH", "MEDIUM", "LOW", "NO_EGO")
    assert 0.0 <= c["risk"] <= 1.0 + 1e-6, "risk must be a probability"
    assert it["level"] in ("HIGH", "MEDIUM", "LOW")
    assert 0.0 <= it["entropy_norm"] <= 1.0 + 1e-6
    assert len(it["maneuvers"]) == mc.num_modes
    print(f"[ok] risk: conflict={c['level']}({c['risk']:.2f}) intent={it['level']} \"{it['message']}\"")

    # ---- 9. Intent estimation + precursor mining (observed-kinematics, no model) ----
    from foresee.intent import scene_intents
    from foresee.playback import raw_scene
    from foresee.precursors import find_precursors
    scene = raw_scene(train_ds, 0)
    intents = scene_intents(scene["tracks_world"], scene["valid"], scene["obs_len"] - 1,
                            1.0 / scene["sample_rate_hz"])
    assert len(intents) == fc.max_agents
    valid_intents = [it for it in intents if it.valid]
    assert valid_intents, "expected at least one agent with an estimated intent"
    assert all(it.longitudinal in ("accelerating", "cruising", "braking", "stopped", "unknown")
               for it in intents)
    precs = find_precursors(scene)  # may be empty for a given synthetic scene; must not error
    for p in precs:
        assert 0.0 <= p.severity <= 1.0 and p.category
    print(f"[ok] intent ({len(valid_intents)} agents) + precursors ({len(precs)} found)")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
