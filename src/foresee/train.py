"""Training loop.

    python -m foresee.train --require-real-data --device cuda --arch anchored

Writes checkpoints and a metrics.csv under runs/<timestamp>/, then fits a calibration
temperature on validation and stores it in the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .anchors import compute_anchors
from .checkpoint import save_checkpoint
from .config import Config, ModelConfig, TrainConfig
from .data import build_datasets, collate_samples
from .losses import multimodal_loss
from .metrics import MetricAccumulator, forecasting_metrics
from .models import ARCHS, build_model


def _move(batch: Dict[str, object], device: str) -> Dict[str, object]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.model = dataclasses.replace(cfg.model, num_modes=args.num_modes,
                                    hidden_dim=args.hidden_dim,
                                    predict_uncertainty=not args.no_uncertainty)
    cfg.arch = args.arch
    cfg.train = dataclasses.replace(cfg.train, epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, num_workers=args.num_workers,
                                    diversity_weight=args.diversity_weight)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.require_real_data:
        cfg.require_real_data = True
    return cfg


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, anchors=None) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "reg_loss": 0.0, "cls_loss": 0.0, "div_loss": 0.0}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        pred = model(batch)
        out = multimodal_loss(pred, batch["future"], batch["future_mask"], cfg.train,
                              anchors=anchors)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        bs = batch["future"].shape[0]
        n += bs
        for k in running:
            running[k] += float(out[k].item()) * bs
    if scheduler is not None:
        scheduler.step()
    return {k: v / max(n, 1) for k, v in running.items()}


def calibrate_temperature(model, loader, device) -> float:
    """Fit a single temperature on val so the mode confidences become calibrated probabilities.

    Target = the mode that is actually closest to ground truth (argmin ADE). We minimise
    cross-entropy of softmax(logits / T) against that target over the val set - standard
    temperature scaling. Returns T; the caller stores it in the model's ``temperature`` buffer.
    """
    model.eval()
    logits_all, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            pred = model(batch)
            tr, fut, fm = pred["trajectories"], batch["future"], batch["future_mask"].float()
            err = torch.linalg.norm(tr - fut.unsqueeze(1), dim=-1)
            ade = (err * fm.unsqueeze(1)).sum(-1) / fm.sum(-1, keepdim=True).clamp(min=1)
            logits_all.append(pred["logits"].cpu())
            targets.append(ade.argmin(1).cpu())
    logits_all = torch.cat(logits_all)
    targets = torch.cat(targets)
    log_t = torch.zeros(1, requires_grad=True)            # optimise log T (T = exp(log_t) > 0)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_all / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


@torch.no_grad()
def validate(model, loader, cfg, device) -> Dict[str, float]:
    model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        batch = _move(batch, device)
        pred = model(batch)
        metrics = forecasting_metrics(pred, batch["future"], batch["future_mask"],
                                      cfg.train.miss_threshold_m)
        acc.update(metrics)
    return acc.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Foresee multimodal forecaster.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--num-modes", type=int, default=ModelConfig.num_modes)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig.hidden_dim)
    parser.add_argument("--diversity-weight", type=float, default=TrainConfig.diversity_weight,
                        help="Weight on the mode-diversity regularizer (0 disables it).")
    parser.add_argument("--arch", type=str, default="lane_transformer", choices=list(ARCHS),
                        help="Model architecture (lane_transformer is more accurate).")
    parser.add_argument("--no-uncertainty", action="store_true",
                        help="Disable the Laplace-scale head (use Huber regression instead).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override FORESEE_DATA_ROOT (path to the AV2 split).")
    parser.add_argument("--require-real-data", action="store_true",
                        help="Use ONLY the real AV2 dataset; error out instead of falling "
                             "back to the synthetic generator.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    cfg = build_config(args)
    torch.manual_seed(cfg.train.seed)

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, collate_fn=collate_samples,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, collate_fn=collate_samples)

    model = build_model(cfg.arch, cfg.feature, cfg.model).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] arch='{cfg.arch}' with {n_params/1e6:.2f}M parameters on {args.device}")

    # Goal anchors for the anchored architecture (k-means over real future endpoints).
    loss_anchors = None
    if cfg.arch == "anchored":
        anchors = compute_anchors(train_ds, cfg.model.num_modes)
        model.set_anchors(anchors)
        loss_anchors = model.anchors
        print("[anchors] goal endpoints (agent frame):")
        for j, a in enumerate(anchors):
            print(f"    mode {j}: ({a[0]:+6.1f}, {a[1]:+6.1f}) m")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "minADE", "minFDE", "MR", "brier_minFDE"])

    best_brier = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, args.device,
                             anchors=loss_anchors)
        val = validate(model, val_loader, cfg, args.device)
        print(f"[epoch {epoch:3d}/{cfg.train.epochs}] "
              f"loss={tr['loss']:.3f} | minADE={val['minADE']:.3f} "
              f"minFDE={val['minFDE']:.3f} MR={val['MR']:.3f} "
              f"brier-minFDE={val['brier_minFDE']:.3f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr['loss']:.4f}", f"{val['minADE']:.4f}",
                                    f"{val['minFDE']:.4f}", f"{val['MR']:.4f}",
                                    f"{val['brier_minFDE']:.4f}"])

        if val["brier_minFDE"] < best_brier:
            best_brier = val["brier_minFDE"]
            save_checkpoint(run_dir / "best.pt", model, cfg.arch, cfg.feature, cfg.model,
                            val_metrics=val, epoch=epoch)
            print(f"    -> new best (brier-minFDE={best_brier:.3f}) saved to {run_dir/'best.pt'}")

    # --- Post-hoc confidence calibration on the best checkpoint ---
    best_path = run_dir / "best.pt"
    if best_path.exists():
        from .checkpoint import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(str(best_path), args.device)
        temp = calibrate_temperature(model, val_loader, args.device)
        model.temperature = torch.tensor(float(temp), device=args.device)
        save_checkpoint(best_path, model, cfg.arch, cfg.feature, cfg.model,
                        val_metrics={}, epoch=0, temperature=float(temp))
        print(f"[calibrate] fitted confidence temperature T={temp:.2f} (saved into best.pt)")

    print(f"[done] Best brier-minFDE={best_brier:.3f}. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
