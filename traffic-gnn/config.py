"""
Centralised hyper-parameters for GPDSTGCN on PEMS04.

All values can be overridden with command-line args:
    python train.py --epochs 200 --lr 0.0005

Defaults match the paper's PEMS04 configuration.
"""

import argparse


class Config:
    """All hyper-parameters in one place."""

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path:   str = "data/pems04.npz"
    adj_path:    str = "data/pems04.csv"
    num_nodes:   int = 307
    in_channels: int = 3            # flow, occupancy, speed

    # ── Sliding window ────────────────────────────────────────────────────────
    seq_len:     int = 12           # input  = 12 × 5 min = 60 min
    pred_len:    int = 12           # output = 12 × 5 min = 60 min

    # Horizons to evaluate: 3→15 min, 6→30 min, 12→60 min
    horizons: list = None           # set in __init__

    # ── Model architecture ────────────────────────────────────────────────────
    hidden_dim:   int   = 64
    num_layers:   int   = 2         # number of ST blocks
    cheb_k:       int   = 3         # Chebyshev polynomial order K
    num_heads:    int   = 4         # temporal attention heads
    gru_layers:   int   = 2
    dropout:      float = 0.1

    # ── Grid partition ────────────────────────────────────────────────────────
    grid_size:    int   = 4         # M = 4×4 = 16 cells
    expand_ratio: float = 0.15     # boundary expansion ratio

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:         int   = 100
    batch_size:     int   = 32
    lr:             float = 0.001
    weight_decay:   float = 1e-4
    lr_decay_step:  int   = 20
    lr_decay_gamma: float = 0.5
    grad_clip:      float = 5.0
    patience:       int   = 15      # early stopping patience

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:       int = 42
    log_every:  int = 5
    save_dir:   str = "checkpoints"
    results_dir: str = "results"
    num_workers: int = 4

    # ── Train / val / test split ratios ───────────────────────────────────────
    train_ratio: float = 0.6
    val_ratio:   float = 0.2
    # test_ratio = 1 - train_ratio - val_ratio

    def __init__(self, **overrides):
        self.horizons = [3, 6, 12]   # 15 / 30 / 60 min
        # Apply any keyword overrides
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    # ── CLI override helper ───────────────────────────────────────────────────
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="GPDSTGCN config")
        for k, v in vars(cls()).items():
            if isinstance(v, bool):
                parser.add_argument(f"--{k}", type=lambda x: x.lower() == "true", default=v)
            elif isinstance(v, (int, float, str)):
                parser.add_argument(f"--{k}", type=type(v), default=v)
        args, _ = parser.parse_known_args()
        return cls(**{k: v for k, v in vars(args).items() if v is not None})

    def __repr__(self):
        items = "\n".join(f"  {k:20s} = {v}" for k, v in vars(self).items())
        return f"Config(\n{items}\n)"
