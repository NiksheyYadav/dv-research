"""
Visualization suite for GPDSTGCN traffic prediction.

4 plots:
  1. SpaceTime Cube       — 3D surface: sensor × time × flow
  2. Congestion Heatmap   — rush-hour zones highlighted
  3. Error Heatmap        — where model over/under-predicts
  4. Feature Importance   — gradient-based: flow vs occupancy vs speed

Usage:
    python visualize.py
    python visualize.py --ckpt checkpoints/gpdstgcn_best.pt
"""

import os, argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm

from config import Config
from utils.data_utils import load_pems04
from utils.graph_utils import build_graph
from models.gpdstgcn import GPDSTGCN


# ── 1. SpaceTime Cube ────────────────────────────────────────────────────────

def plot_spacetime_cube(data, save_path, n_nodes=50, n_steps=100):
    """3D surface: sensor-id × timestep × flow."""
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    flow = data[:n_steps, :n_nodes, 0]
    T, N = flow.shape
    X, Y = np.meshgrid(np.arange(N), np.arange(T))

    surf = ax.plot_surface(X, Y, flow, cmap="viridis", edgecolor="none",
                           alpha=0.85, rstride=2, cstride=1)
    ax.set_xlabel("Sensor ID", fontsize=11, labelpad=10)
    ax.set_ylabel("Time Step", fontsize=11, labelpad=10)
    ax.set_zlabel("Traffic Flow", fontsize=11, labelpad=10)
    ax.set_title("SpaceTime Cube — Traffic Flow", fontsize=14, fontweight="bold")
    fig.colorbar(surf, shrink=0.5, aspect=15, label="Flow")
    ax.view_init(elev=25, azim=135)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ SpaceTime Cube → {save_path}")


# ── 2. Congestion Heatmap ────────────────────────────────────────────────────

def plot_congestion_heatmap(data, save_path, n_nodes=80):
    """Heatmap with rush-hour zones annotated."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # One full day = 288 steps (5-min intervals)
    day = data[:288, :n_nodes, 0]
    im = ax.imshow(day.T, aspect="auto", cmap="YlOrRd", interpolation="bilinear")

    # Rush hour bands (7-9 AM, 5-7 PM)
    for start, end, label in [(84, 108, "AM Rush"), (204, 228, "PM Rush")]:
        ax.axvspan(start, end, alpha=0.15, color="blue", linewidth=0)
        ax.text((start + end) / 2, -3, label, ha="center", fontsize=9,
                fontweight="bold", color="#1565C0")

    ax.set_xlabel("Time of Day (5-min steps)", fontsize=12)
    ax.set_ylabel("Sensor ID", fontsize=12)
    ax.set_title("Congestion Heatmap — Rush Hour Zones", fontsize=14,
                 fontweight="bold")

    # Time labels
    ticks = np.arange(0, 289, 36)
    labels = [f"{(t*5)//60:02d}:{(t*5)%60:02d}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)

    fig.colorbar(im, ax=ax, label="Flow", shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Congestion Heatmap → {save_path}")


# ── 3. Error Heatmap ─────────────────────────────────────────────────────────

def plot_error_heatmap(preds, trues, save_path, n_nodes=80, n_steps=200):
    """Signed error heatmap: red=over-predict, blue=under-predict."""
    err = (preds[:n_steps, 0, :n_nodes] - trues[:n_steps, 0, :n_nodes])

    fig, ax = plt.subplots(figsize=(16, 6))
    vmax = np.percentile(np.abs(err), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(err.T, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="bilinear")

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Sensor ID", fontsize=12)
    ax.set_title("Prediction Error Heatmap (red=over, blue=under)",
                 fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Error (pred − true)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Error Heatmap → {save_path}")


# ── 4. Feature Importance (gradient-based) ────────────────────────────────────

def plot_feature_importance(model, sample_x, device, save_path):
    """Gradient-based feature importance: |dL/dX| aggregated per feature."""
    model.eval()
    x = torch.FloatTensor(sample_x).unsqueeze(0).to(device)
    x.requires_grad_(True)

    pred = model(x)
    target = pred.sum()
    target.backward()

    grad = x.grad.abs().squeeze(0).cpu().numpy()   # (T, N, C)
    importance = grad.mean(axis=(0, 1))             # (C,)

    feature_names = ["Flow", "Occupancy", "Speed"]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(feature_names[:len(importance)], importance,
                  color=colors[:len(importance)], alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, importance):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean |Gradient|", fontsize=12)
    ax.set_title("Feature Importance (Gradient-Based)", fontsize=14,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Feature Importance → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/gpdstgcn_best.pt")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Load raw data for visualizations
    raw = np.load(cfg.data_path)["data"].astype(np.float32)
    print(f"[viz] Raw data: {raw.shape}")

    # Plot 1 & 2 — only need raw data
    plot_spacetime_cube(raw, os.path.join(cfg.results_dir, "spacetime_cube.png"))
    plot_congestion_heatmap(raw, os.path.join(cfg.results_dir, "congestion_heatmap.png"))

    # Load model for plots 3 & 4
    if not os.path.exists(args.ckpt):
        print(f"\n  ⚠ Checkpoint not found ({args.ckpt})")
        print("    Run train.py first. Skipping error heatmap & feature importance.")
        return

    data = load_pems04(cfg)
    graph = build_graph(cfg.adj_path, cfg.num_nodes)
    L = graph["L_tilde"].to(device)

    model = GPDSTGCN(cfg, L).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Inference on test set
    preds, trues = [], []
    with torch.no_grad():
        for x, y in data["test_loader"]:
            preds.append(model(x.to(device)).cpu())
            trues.append(y.to(device).squeeze(-1).cpu())
    preds = data["scaler"].inverse_transform(torch.cat(preds)).numpy()
    trues = data["scaler"].inverse_transform(torch.cat(trues)).numpy()

    # Plot 3
    plot_error_heatmap(preds, trues,
                       os.path.join(cfg.results_dir, "error_heatmap.png"))

    # Plot 4 — grab a sample
    sample_x = next(iter(data["test_loader"]))[0][0].numpy()
    plot_feature_importance(model, sample_x, device,
                            os.path.join(cfg.results_dir, "feature_importance.png"))

    print(f"\n✓ All visualizations saved to {cfg.results_dir}/")


if __name__ == "__main__":
    main()
