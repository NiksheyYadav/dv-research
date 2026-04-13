"""
Evaluation metrics: MAE · RMSE · MAPE at multiple horizons.

All functions expect numpy arrays or torch tensors with shape (B, pred_len, N).
"""

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── Individual metrics ───────────────────────────────────────────────────────

def mae(pred, true):
    """Mean Absolute Error."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    """Root Mean Squared Error."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    return np.sqrt(np.mean((pred - true) ** 2))


def mape(pred, true, eps=1e-5):
    """Mean Absolute Percentage Error (%)."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    mask = np.abs(true) > eps
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100.0


# ── Compute at multiple horizons ─────────────────────────────────────────────

def compute_all(pred, true, horizons=None):
    """
    Parameters
    ----------
    pred     : (B, pred_len, N) – predicted traffic flow
    true     : (B, pred_len, N) – ground truth
    horizons : list of int, e.g. [3, 6, 12]  →  15 / 30 / 60 min

    Returns
    -------
    dict  {horizon_step: {"MAE": ..., "RMSE": ..., "MAPE": ...}, "overall": {...}}
    """
    pred, true = _to_numpy(pred), _to_numpy(true)

    if horizons is None:
        horizons = [3, 6, 12]

    results = {}

    for h in horizons:
        p = pred[:, :h, :]
        t = true[:, :h, :]
        results[h] = {
            "MAE":  mae(p, t),
            "RMSE": rmse(p, t),
            "MAPE": mape(p, t),
        }

    # Overall (all horizons)
    results["overall"] = {
        "MAE":  mae(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
    }
    return results


# ── Pretty print ─────────────────────────────────────────────────────────────

def format_metrics(results: dict) -> str:
    """Return a formatted table string."""
    lines = [
        "",
        "  ┌────────────┬──────────┬──────────┬──────────┐",
        "  │  Horizon   │   MAE    │   RMSE   │  MAPE(%) │",
        "  ├────────────┼──────────┼──────────┼──────────┤",
    ]

    horizon_labels = {3: "15 min", 6: "30 min", 12: "60 min"}

    for h, m in results.items():
        if h == "overall":
            continue
        label = horizon_labels.get(h, f"h={h}")
        lines.append(
            f"  │ {label:>10s} │ {m['MAE']:8.4f} │ {m['RMSE']:8.4f} │ {m['MAPE']:8.2f} │"
        )

    if "overall" in results:
        m = results["overall"]
        lines.append("  ├────────────┼──────────┼──────────┼──────────┤")
        lines.append(
            f"  │ {'Overall':>10s} │ {m['MAE']:8.4f} │ {m['RMSE']:8.4f} │ {m['MAPE']:8.2f} │"
        )

    lines.append("  └────────────┴──────────┴──────────┴──────────┘")
    return "\n".join(lines)
