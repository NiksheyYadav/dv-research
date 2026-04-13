# GPDSTGCN – Traffic Congestion Prediction

**Authors:** Lakshay Rohilla · Nikshay Yadav  
**Affiliation:** SGT University, Gurugram, Haryana

---

## Overview

Implementation of **Grid Partition Dynamic Spatio-Temporal Graph Convolutional Network (GPDSTGCN)** for urban traffic congestion prediction on the PEMS04 dataset.

Core ideas from the research synopsis:
- **Graph Convolutional Network** (Chebyshev, K=3) captures road network topology
- **Gated Tanh Unit** `H_t = tanh(W₁∗X) ⊙ σ(W₂∗X)` handles non-stationary temporal patterns
- **Multi-head Temporal Attention** `α_t' = softmax(score(h_t, h_t'))` models long-range time dependencies
- **Grid Partition** reduces O(N²) complexity to O(Σ N_sub³ + N_sub² T) — 70–100× speedup
- **GRU** (2-layer) integrates sequential context before output

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU       | –       | **NVIDIA RTX 5050 (8 GB VRAM)** |
| RAM       | 8 GB    | 16 GB |
| Python    | 3.9+    | 3.10+ |
| CUDA      | 11.8+   | 12.x  |

> The codebase auto-detects GPU. If CUDA is available it trains on GPU; otherwise falls back to CPU (significantly slower).

---

## Project Structure

```
traffic-gnn/
├── config.py               # All hyperparameters (single source of truth)
├── train.py                # Training loop (early stopping, LR decay)
├── evaluate.py             # Test evaluation + 3 plots
├── compare.py              # Train & compare all models → Table 3
├── visualize.py            # SpaceTime Cube, heatmaps, feature importance
├── requirements.txt        # Python dependencies
│
├── data/
│   └── download_pems04.py  # Auto-download PEMS04 + manual fallback
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py       # Z-score norm, sliding window, DataLoader
│   ├── graph_utils.py      # Adjacency matrix, Chebyshev Laplacian L̃
│   └── metrics.py          # MAE / RMSE / MAPE at 15/30/60 min
│
└── models/
    ├── __init__.py          # Exports all models
    ├── layers.py            # ChebConv, GTU, TemporalAttention, STBlock
    ├── gpdstgcn.py          # Main GPDSTGCN model + GridPartition
    └── baselines.py         # HistoricalAverage, ARIMA, STGCN, DCRNN
```

---

## Complete Workflow (Step by Step)

### Step 0 — Clone / Copy the project

Copy the `traffic-gnn/` folder to the target machine.

### Step 1 — Install dependencies

```bash
cd traffic-gnn
pip install -r requirements.txt
```

> **GPU users:** Make sure PyTorch is installed with CUDA support.  
> If not, reinstall with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 2 — Download PEMS04 dataset

```bash
python data/download_pems04.py
```

This auto-downloads two files into `data/`:
- `pems04.npz` — traffic flow tensor, shape `(16992, 307, 3)` — flow, occupancy, speed
- `pems04.csv` — sensor distance matrix (adjacency)

**If download fails** (firewall / proxy), manually download from:
-``` https://github.com/guoshnBJTU/ASTGCN-r-pytorch/tree/main/data/PEMS04```

Place files as:
```
data/pems04.npz
data/pems04.csv
```
**Recommended** 
```https://huggingface.co/datasets/bjdwh/FlashST-DATA/tree/main/PEMS04```

### Step 3 — Train GPDSTGCN (main model)

```bash
python train.py
```

What happens:
1. Loads PEMS04 → Z-score normalisation → sliding window (12 → 12 steps)
2. Builds graph: adjacency matrix → Chebyshev scaled Laplacian L̃
3. Grid partitions 307 sensors into 4×4 = 16 cells
4. Trains with Adam optimiser, MAE loss, StepLR, gradient clipping
5. Early stopping (patience=15), best model → `checkpoints/gpdstgcn_best.pt`
6. Final test evaluation with MAE/RMSE/MAPE at 15/30/60 min

**Override defaults via CLI:**
```bash
python train.py --epochs 200 --lr 0.0005 --batch_size 64 --hidden_dim 128
```

**Estimated time on RTX 5050 (8 GB):** ~15–25 min for 100 epochs.

### Step 4 — Evaluate (metrics + plots)

```bash
python evaluate.py
```

Or point to a specific checkpoint:
```bash
python evaluate.py --ckpt checkpoints/gpdstgcn_best.pt
```

**Outputs** (saved to `results/`):

| File | Description |
|------|-------------|
| `loss_curves.png` | Train/val loss over epochs |
| `horizon_bars.png` | MAE/RMSE/MAPE bar chart at 15, 30, 60 min |
| `pred_vs_actual.png` | Predicted vs ground truth time series for selected sensors |

### Step 5 — Compare all models (Table 3)

```bash
python compare.py
```

Trains & evaluates **all 4 models** in sequence:
1. **GPDSTGCN** (ours)
2. **STGCN** (Yu et al., 2018)
3. **DCRNN** (Li et al., 2018)
4. **HistoricalAverage** (non-parametric baseline)

> ARIMA is skipped by default in automated run (slow per-node fitting).  
> To include it, run with `--include_arima` flag.

**Outputs:**

| File | Description |
|------|-------------|
| `results/comparison_table.csv` | All metrics in CSV — ready for paper Table 3 |
| `results/comparison_bars.png` | Side-by-side bar chart comparing all models |

**Estimated time on RTX 5050:** ~60–90 min (trains 3 neural models sequentially).

### Step 6 — Visualizations

```bash
python visualize.py
```

Or with a specific checkpoint:
```bash
python visualize.py --ckpt checkpoints/gpdstgcn_best.pt
```

**Outputs** (saved to `results/`):

| File | Description |
|------|-------------|
| `spacetime_cube.png` | 3D surface: sensor × time × flow |
| `congestion_heatmap.png` | Rush-hour zones (AM/PM peaks annotated) |
| `error_heatmap.png` | Where model over/under-predicts (red/blue) |
| `feature_importance.png` | Gradient-based: flow vs occupancy vs speed |

> **Note:** Plots 3 & 4 require a trained checkpoint. Run `train.py` first.

---

## Quick Reference — All Commands

```bash
# ── Setup ──────────────────────────────────────────
cd traffic-gnn
pip install -r requirements.txt

# ── Data ───────────────────────────────────────────
python data/download_pems04.py

# ── Train (main model) ────────────────────────────
python train.py

# ── Evaluate ───────────────────────────────────────
python evaluate.py

# ── Compare all models (Table 3) ──────────────────
python compare.py

# ── Visualizations (SpaceTime Cube etc.) ──────────
python visualize.py
```

---

## Expected Results (PEMS04)

| Model            | MAE       | RMSE      | MAPE       |
|------------------|-----------|-----------|------------|
| HistoricalAvg    | 38.02     | 52.10     | 28.75%     |
| ARIMA            | 32.65     | 43.50     | 25.41%     |
| DCRNN            | 23.61     | 37.57     | 18.86%     |
| STGCN            | 22.98     | 36.41     | 18.20%     |
| GWNET            | 21.25     | 33.40     | 16.76%     |
| **GPDSTGCN (ours)** | **18.53** | **30.26** | **12.29%** |

---

## Key Equations

**Graph signal prediction:**
```
[X_{t-T+1}, ..., X_t ; G] → F_θ → [X̂_{t+1}, ..., X̂_{t+T'}]
```

**Chebyshev GCN (eq. 2):**
```
Y = Σ_{k=0}^{K-1} θ_k · T_k(L̃) · X
```

**Gated Tanh Unit (eq. 5):**
```
H_t = tanh(W₁ * X) ⊙ σ(W₂ * X)
```

**Temporal Attention (eq. 6):**
```
α_{t'} = exp(score(h_t, h_{t'})) / Σ exp(score(h_t, h_{t''}))
```

**Grid partition complexity (eq. 3):**
```
O(Σ_i  N_sub_i³ + N_sub_i² T + N_sub_i D T)
```

---

## Hyperparameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for all layers |
| `num_layers` | 2 | Number of ST blocks |
| `cheb_k` | 3 | Chebyshev polynomial order K |
| `num_heads` | 4 | Temporal attention heads |
| `gru_layers` | 2 | GRU depth |
| `grid_size` | 4 | Grid partition (4×4 = 16 cells) |
| `expand_ratio` | 0.15 | Boundary expansion ratio |
| `seq_len` | 12 | Input window (12 × 5 min = 60 min) |
| `pred_len` | 12 | Prediction horizon (60 min) |
| `batch_size` | 32 | Training batch size |
| `lr` | 0.001 | Learning rate |
| `epochs` | 100 | Max training epochs |
| `patience` | 15 | Early stopping patience |
| `dropout` | 0.1 | Dropout rate |

All parameters can be overridden via CLI: `python train.py --hidden_dim 128 --lr 0.0005`

---

## Output Directory Structure

After running all scripts:
```
traffic-gnn/
├── checkpoints/
│   └── gpdstgcn_best.pt          # Best model weights
│   └── history.npy               # Training loss history
│
└── results/
    ├── loss_curves.png            # Train/val loss plot
    ├── horizon_bars.png           # Metrics per horizon
    ├── pred_vs_actual.png         # Time series overlay
    ├── comparison_table.csv       # All models comparison (Table 3)
    ├── comparison_bars.png        # Model comparison bar chart
    ├── spacetime_cube.png         # 3D SpaceTime visualization
    ├── congestion_heatmap.png     # Rush-hour heatmap
    ├── error_heatmap.png          # Prediction error map
    └── feature_importance.png     # Gradient-based feature analysis
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 16 or 8: `python train.py --batch_size 16` |
| `pems04.npz not found` | Run `python data/download_pems04.py` first |
| Training too slow on CPU | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `statsmodels` import error | `pip install statsmodels` (needed only for ARIMA baseline) |
| Plots not generating | Ensure `matplotlib` is installed; Agg backend is used (no display needed) |