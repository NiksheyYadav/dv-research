"""
Data loading and preprocessing for PEMS04.

─ Z-score normalisation (per-feature, per-node)
─ Sliding-window dataset  (seq_len → pred_len)
─ Train / Val / Test DataLoader creation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Z-Score Scaler ────────────────────────────────────────────────────────────

class ZScoreScaler:
    """Per-feature Z-score normalisation: x' = (x - μ) / σ."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std  = np.where(std < 1e-8, 1.0, std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        """Inverse-transform; only restores flow feature (index 0)."""
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean[..., 0], dtype=x.dtype, device=x.device)
            std  = torch.tensor(self.std[..., 0],  dtype=x.dtype, device=x.device)
        else:
            mean = self.mean[..., 0]
            std  = self.std[..., 0]
        return x * std + mean


# ── Sliding-Window Dataset ────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """
    Generates (x, y) pairs from a contiguous traffic tensor.

    x : (seq_len,  N, C)   – past observations
    y : (pred_len, N, 1)   – future flow values (feature index 0)
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        Parameters
        ----------
        data     : (T_total, N, C)
        seq_len  : int – input window length
        pred_len : int – prediction horizon
        """
        self.data     = data
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_samples = data.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                        # (T, N, C)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, 0:1]  # (P, N, 1)
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def load_pems04(cfg) -> dict:
    """
    Load PEMS04 dataset, apply Z-score normalisation, create DataLoaders.

    Expects  cfg.data_path   → 'data/pems04.npz'  (key 'data': (T, N, C))
    Expects  cfg.seq_len, cfg.pred_len, cfg.batch_size, cfg.num_workers

    Returns
    -------
    dict with keys: train_loader, val_loader, test_loader, scaler
    """
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"PEMS04 data not found at '{cfg.data_path}'.\n"
            f"Run:  python data/download_pems04.py\n"
        )

    raw = np.load(cfg.data_path)
    data = raw["data"].astype(np.float32)      # (T, N, C)   e.g. (16992, 307, 3)
    T_total, N, C = data.shape
    print(f"[data] Loaded PEMS04: shape={data.shape}  "
          f"({T_total} timesteps × {N} nodes × {C} features)")

    # ── Train / Val / Test split ──────────────────────────────────────────────
    n_train = int(T_total * cfg.train_ratio)
    n_val   = int(T_total * cfg.val_ratio)

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val:]

    print(f"[data] Split: train={train_data.shape[0]}  "
          f"val={val_data.shape[0]}  test={test_data.shape[0]}")

    # ── Z-score normalisation (fit on train only) ─────────────────────────────
    mean = train_data.mean(axis=0, keepdims=True)   # (1, N, C)
    std  = train_data.std(axis=0, keepdims=True)

    scaler = ZScoreScaler(mean, std)
    train_data = scaler.transform(train_data)
    val_data   = scaler.transform(val_data)
    test_data  = scaler.transform(test_data)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = TrafficDataset(train_data, cfg.seq_len, cfg.pred_len)
    val_ds   = TrafficDataset(val_data,   cfg.seq_len, cfg.pred_len)
    test_ds  = TrafficDataset(test_data,  cfg.seq_len, cfg.pred_len)

    loader_kw = dict(
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"[data] Batches: train={len(train_loader)}  "
          f"val={len(val_loader)}  test={len(test_loader)}")

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "scaler":       scaler,
    }
