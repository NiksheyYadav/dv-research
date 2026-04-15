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


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def print_config(cfg, n_params, device, data_dict, graph):
    """Print a pretty box with all hyperparameters and setup info."""
    W = 64
    sep = "─" * W
    print()
    print(f"╔{'═' * W}╗")
    print(f"║{'GPDSTGCN  –  Training Configuration':^{W}}║")
    print(f"╠{'═' * W}╣")

    def row(key, val):
        line = f"  {key:<26s} │ {val}"
        print(f"║{line:<{W}}║")

    # Device & Model
    row("Device", device)
    row("Model parameters", f"{n_params:,}")
    print(f"║{sep:^{W}}║")

    # Data
    print(f"║{'  DATA':<{W}}║")
    row("Dataset", cfg.data_path)
    row("Adjacency", cfg.adj_path)
    row("Num nodes", cfg.num_nodes)
    row("Input channels", cfg.in_channels)
    row("Seq len  (input)", f"{cfg.seq_len}  ({cfg.seq_len * 5} min)")
    row("Pred len (output)", f"{cfg.pred_len}  ({cfg.pred_len * 5} min)")
    row("Eval horizons", cfg.horizons)
    row("Train / Val / Test", f"{len(data_dict['train_loader'].dataset)} / "
                              f"{len(data_dict['val_loader'].dataset)} / "
                              f"{len(data_dict['test_loader'].dataset)}")
    row("Batch size", cfg.batch_size)
    n_edges = graph['num_edges']
    density = graph['density']
    row("Graph edges / density", f"{n_edges} / {density:.4f}")
    print(f"║{sep:^{W}}║")

    # Architecture
    print(f"║{'  ARCHITECTURE':<{W}}║")
    row("Hidden dim", cfg.hidden_dim)
    row("ST blocks", cfg.num_layers)
    row("ChebNet order K", cfg.cheb_k)
    row("Attention heads", cfg.num_heads)
    row("GRU layers", cfg.gru_layers)
    row("Dropout", cfg.dropout)
    row("Grid size", f"{cfg.grid_size}×{cfg.grid_size} = {cfg.grid_size**2} cells")
    row("Expand ratio", cfg.expand_ratio)
    print(f"║{sep:^{W}}║")

    # Training
    print(f"║{'  TRAINING':<{W}}║")
    row("Epochs", cfg.epochs)
    row("Learning rate", cfg.lr)
    row("Weight decay", cfg.weight_decay)
    row("LR decay step", cfg.lr_decay_step)
    row("LR decay gamma", cfg.lr_decay_gamma)
    row("Gradient clip", cfg.grad_clip)
    row("Early-stop patience", cfg.patience)
    row("Seed", cfg.seed)
    print(f"╚{'═' * W}╝")
    print()


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
    pbar = tqdm(loader, desc="  ► Train", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}", colour="green")
    for x, y in pbar:
        x = x.to(device)                       # (B, T, N, C)
        y = y.to(device).squeeze(-1)           # (B, pred_len, N)

        optimiser.zero_grad(set_to_none=True)
        pred = model(x)                        # (B, pred_len, N)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="  ► Eval "):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    pbar = tqdm(loader, desc=desc, leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}", colour="cyan")
    for x, y in pbar:
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
    
    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPDSTGCN(cfg, L_tilde, coords).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Print config ──────────────────────────────────────────────────────────
    print_config(cfg, n_params, device, data_dict, graph)

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

    print("🚀 Starting training...\n")

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs",
                     bar_format="{l_bar}{bar:40}{r_bar}", colour="yellow")

    for epoch in epoch_bar:
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, criterion,
                                 device, cfg.grad_clip)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device,
                                    desc="  ► Val  ")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Update epoch bar with live metrics
        star = "★" if val_loss < best_val_loss else " "
        epoch_bar.set_postfix_str(
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"best={best_val_loss:.4f}  lr={lr:.1e}  {elapsed:.0f}s {star}"
        )

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
                tqdm.write(f"\n  ⏹ Early stopping at epoch {epoch} "
                           f"(no improvement for {cfg.patience} epochs)")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print()
    print(f"╔{'═' * 50}╗")
    print(f"║{'TEST  EVALUATION  (best checkpoint)':^50}║")
    print(f"╚{'═' * 50}╝")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, preds, trues = eval_epoch(model, test_loader, criterion, device,
                                  desc="  ► Test ")

    # Inverse-transform flow dimension
    preds_real = scaler.inverse_transform(preds)
    trues_real = scaler.inverse_transform(trues)

    metrics = compute_all(preds_real, trues_real, horizons=cfg.horizons)
    print(format_metrics(metrics))

    # Save training history
    np.save(os.path.join(cfg.save_dir, "history.npy"), history)
    print(f"\n  ✅ Best checkpoint : {best_ckpt}")
    print(f"  ✅ Best val loss   : {best_val_loss:.4f}")


if __name__ == "__main__":
    main()