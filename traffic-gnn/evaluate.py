"""
Evaluation & Plotting for GPDSTGCN.

Usage
-----
    python evaluate.py                          # uses best checkpoint
    python evaluate.py --ckpt checkpoints/gpdstgcn_best.pt

Produces 3 plots:
  1. Training / validation loss curves
  2. Horizon bar chart (MAE / RMSE / MAPE at 15, 30, 60 min)
  3. Prediction vs Actual (time-series overlay for selected nodes)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import Config
from utils.data_utils import load_pems04
from utils.graph_utils import build_graph
from utils.metrics import compute_all, format_metrics
from models.gpdstgcn import GPDSTGCN


# ── Plot 1: Training curves ──────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: str):
    """Plot train/val loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2196F3",
            markersize=3, linewidth=1.5)
    ax.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#FF5722",
            markersize=3, linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MAE)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(history["train_loss"]))

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Loss curves saved to {save_path}")


# ── Plot 2: Horizon bar chart ────────────────────────────────────────────────

def plot_horizon_bars(metrics: dict, save_path: str):
    """Bar chart comparing MAE / RMSE / MAPE across horizons."""
    horizons = [k for k in metrics if k != "overall"]
    labels = {3: "15 min", 6: "30 min", 12: "60 min"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    metric_names = ["MAE", "RMSE", "MAPE"]

    for ax, metric, color in zip(axes, metric_names, colors):
        x = np.arange(len(horizons))
        vals = [metrics[h][metric] for h in horizons]
        bars = ax.bar(x, vals, color=color, alpha=0.85, width=0.5, edgecolor="white")

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(h, str(h)) for h in horizons], fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("Metrics by Prediction Horizon", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Horizon bars saved to {save_path}")


# ── Plot 3: Prediction vs Actual ─────────────────────────────────────────────

def plot_pred_vs_actual(preds: np.ndarray, trues: np.ndarray, save_path: str,
                        nodes=(0, 50, 150), n_samples: int = 200):
    """
    Time-series overlay of predicted vs actual for selected sensor nodes.

    preds, trues : (B, pred_len, N)
    """
    fig, axes = plt.subplots(len(nodes), 1, figsize=(14, 4 * len(nodes)), sharex=True)
    if len(nodes) == 1:
        axes = [axes]

    colors_pred = ["#E91E63", "#9C27B0", "#00BCD4"]
    color_true  = "#333333"

    for ax, node, c in zip(axes, nodes, colors_pred):
        # Use first prediction step for time series
        pred_ts = preds[:n_samples, 0, node]
        true_ts = trues[:n_samples, 0, node]
        t = np.arange(len(pred_ts))

        ax.plot(t, true_ts, "-", color=color_true, linewidth=1.5, label="Actual", alpha=0.8)
        ax.plot(t, pred_ts, "--", color=c, linewidth=1.5, label="Predicted", alpha=0.9)
        ax.fill_between(t, true_ts, pred_ts, alpha=0.15, color=c)

        ax.set_ylabel(f"Node {node}\nFlow", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index", fontsize=12)
    fig.suptitle("Prediction vs Actual (1-step ahead)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Pred vs Actual saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpdstgcn_best.pt")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dict = load_pems04(cfg)
    test_loader = data_dict["test_loader"]
    scaler = data_dict["scaler"]

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = build_graph(cfg.adj_path, cfg.num_nodes)
    L_tilde = graph["L_tilde"].to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPDSTGCN(cfg, L_tilde).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            pred = model(x)
            preds.append(pred.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    preds_real = scaler.inverse_transform(preds).numpy()
    trues_real = scaler.inverse_transform(trues).numpy()

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_all(preds_real, trues_real, horizons=cfg.horizons)
    print(format_metrics(metrics))

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Plot 1: Loss curves
    history_path = os.path.join(cfg.save_dir, "history.npy")
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plot_loss_curves(history, os.path.join(cfg.results_dir, "loss_curves.png"))
    else:
        print("  ⚠ No history.npy found – skipping loss curve plot")

    # Plot 2: Horizon bars
    plot_horizon_bars(metrics, os.path.join(cfg.results_dir, "horizon_bars.png"))

    # Plot 3: Prediction vs Actual
    plot_pred_vs_actual(preds_real, trues_real,
                        os.path.join(cfg.results_dir, "pred_vs_actual.png"))

    print(f"\n  ✓ All results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    main()
