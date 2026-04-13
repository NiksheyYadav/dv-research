"""
Training script for GPDSTGCN on PEMS04.

Usage
-----
    python train.py

All hyper-parameters are in config.py.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import Config
from utils.data_utils import load_pems04
from utils.graph_utils import build_graph
from utils.metrics import compute_all, format_metrics
from models.gpdstgcn import GPDSTGCN


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── One epoch ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)                       # (B, T, N, C)
        y = y.to(device).squeeze(-1)           # (B, pred_len, N)

        optimiser.zero_grad(set_to_none=True)
        pred = model(x)                        # (B, pred_len, N)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze(-1)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        preds.append(pred.cpu())
        trues.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    return total_loss / len(loader), preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    set_seed(cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dict   = load_pems04(cfg)
    train_loader = data_dict["train_loader"]
    val_loader   = data_dict["val_loader"]
    test_loader  = data_dict["test_loader"]
    scaler       = data_dict["scaler"]

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = build_graph(cfg.adj_path, cfg.num_nodes)
    L_tilde = graph["L_tilde"].to(device)

    # Optional: load sensor coordinates for grid partition
    coords = None
    coords_path = "data/pems04_coords.npy"
    if os.path.exists(coords_path):
        coords = np.load(coords_path)           # (N, 2)
        print(f"[train] Loaded sensor coordinates from {coords_path}")
    else:
        print("[train] No coordinates found – using flat partition")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPDSTGCN(cfg, L_tilde, coords).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimiser, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_gamma)
    criterion = nn.L1Loss()                    # MAE loss

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_ckpt = os.path.join(cfg.save_dir, "gpdstgcn_best.pt")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print("\n" + "=" * 60)
    print("  Starting training")
    print("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, criterion,
                                 device, cfg.grad_clip)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        if epoch % cfg.log_every == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{cfg.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.2e}  time={elapsed:.1f}s")

        # ── Early stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimiser.state_dict(),
                "val_loss":    best_val_loss,
                "config":      cfg.__dict__,
            }, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test evaluation (best checkpoint)")
    print("=" * 60)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, preds, trues = eval_epoch(model, test_loader, criterion, device)

    # Inverse-transform flow dimension
    preds_real = scaler.inverse_transform(preds)
    trues_real = scaler.inverse_transform(trues)

    metrics = compute_all(preds_real, trues_real, horizons=cfg.horizons)
    print(format_metrics(metrics))

    # Save training history
    np.save(os.path.join(cfg.save_dir, "history.npy"), history)
    print(f"\n  Best checkpoint : {best_ckpt}")
    print(f"  Best val loss  : {best_val_loss:.4f}")


if __name__ == "__main__":
    main()