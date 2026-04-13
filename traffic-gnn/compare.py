"""
Compare all models on PEMS04 — produces Table 3 for the paper.

Usage:  python compare.py

Output:
  results/comparison_table.csv
  results/comparison_bars.png
"""

import os, time, numpy as np, torch, torch.nn as nn, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from utils.data_utils import load_pems04
from utils.graph_utils import build_graph
from utils.metrics import compute_all, format_metrics
from models.gpdstgcn import GPDSTGCN
from models.baselines import STGCN, DCRNN, HistoricalAverage, ARIMABaseline


def train_nn_model(model, train_loader, val_loader, cfg, device):
    """Quick training loop for a neural baseline."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_decay_step, cfg.lr_decay_gamma)
    crit = nn.L1Loss()
    best_val, best_state = float("inf"), None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).squeeze(-1)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).squeeze(-1)
                val_loss += crit(model(x), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

        if epoch % cfg.log_every == 0:
            print(f"    epoch {epoch:3d}  val={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_nn(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device).squeeze(-1)
        preds.append(model(x).cpu())
        trues.append(y.cpu())
    preds = scaler.inverse_transform(torch.cat(preds)).numpy()
    trues = scaler.inverse_transform(torch.cat(trues)).numpy()
    return preds, trues


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.results_dir, exist_ok=True)

    data = load_pems04(cfg)
    graph = build_graph(cfg.adj_path, cfg.num_nodes)
    L = graph["L_tilde"].to(device)

    all_results = {}

    # --- 1. GPDSTGCN (ours) ---
    print("\n══ GPDSTGCN (ours) ══")
    t0 = time.time()
    model = GPDSTGCN(cfg, L).to(device)
    model = train_nn_model(model, data["train_loader"], data["val_loader"], cfg, device)
    preds, trues = evaluate_nn(model, data["test_loader"], data["scaler"], device)
    all_results["GPDSTGCN"] = compute_all(preds, trues, cfg.horizons)
    print(f"  time: {time.time()-t0:.0f}s")
    print(format_metrics(all_results["GPDSTGCN"]))

    # --- 2. STGCN ---
    print("\n══ STGCN ══")
    t0 = time.time()
    model = STGCN(cfg, L).to(device)
    model = train_nn_model(model, data["train_loader"], data["val_loader"], cfg, device)
    preds, trues = evaluate_nn(model, data["test_loader"], data["scaler"], device)
    all_results["STGCN"] = compute_all(preds, trues, cfg.horizons)
    print(f"  time: {time.time()-t0:.0f}s")
    print(format_metrics(all_results["STGCN"]))

    # --- 3. DCRNN ---
    print("\n══ DCRNN ══")
    t0 = time.time()
    model = DCRNN(cfg, L).to(device)
    model = train_nn_model(model, data["train_loader"], data["val_loader"], cfg, device)
    preds, trues = evaluate_nn(model, data["test_loader"], data["scaler"], device)
    all_results["DCRNN"] = compute_all(preds, trues, cfg.horizons)
    print(f"  time: {time.time()-t0:.0f}s")
    print(format_metrics(all_results["DCRNN"]))

    # --- 4. Historical Average ---
    print("\n══ HistoricalAverage ══")
    raw = np.load(cfg.data_path)["data"].astype(np.float32)
    n_train = int(raw.shape[0] * cfg.train_ratio)
    ha = HistoricalAverage().fit(raw[:n_train])
    n_test_start = int(raw.shape[0] * (cfg.train_ratio + cfg.val_ratio))
    test_raw = raw[n_test_start:]
    ha_preds, ha_trues = [], []
    for i in range(len(test_raw) - cfg.seq_len - cfg.pred_len + 1):
        slot = (n_test_start + i + cfg.seq_len) % 288
        p = ha.predict(None, start_slot=slot, pred_len=cfg.pred_len)
        t = test_raw[i + cfg.seq_len : i + cfg.seq_len + cfg.pred_len, :, 0]
        ha_preds.append(p[None])
        ha_trues.append(t[None])
    ha_preds = np.concatenate(ha_preds)
    ha_trues = np.concatenate(ha_trues)
    all_results["HistAvg"] = compute_all(ha_preds, ha_trues, cfg.horizons)
    print(format_metrics(all_results["HistAvg"]))

    # --- Build comparison table ---
    rows = []
    for name, metrics in all_results.items():
        for h in cfg.horizons:
            m = metrics[h]
            rows.append({"Model": name, "Horizon": f"{h*5}min",
                          "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"]})
        ov = metrics["overall"]
        rows.append({"Model": name, "Horizon": "Overall",
                      "MAE": ov["MAE"], "RMSE": ov["RMSE"], "MAPE": ov["MAPE"]})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.results_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Table saved: {csv_path}")
    print(df.to_string(index=False))

    # --- Bar chart ---
    models = list(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        vals = [all_results[m]["overall"][metric] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)], alpha=0.85,
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Model Comparison (Overall)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.results_dir, "comparison_bars.png"), dpi=200)
    plt.close(fig)
    print(f"✓ Bar chart saved: {cfg.results_dir}/comparison_bars.png")


if __name__ == "__main__":
    main()
