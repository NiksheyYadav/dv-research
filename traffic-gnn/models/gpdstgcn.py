"""
GPDSTGCN – Grid Partition Dynamic Spatio-Temporal Graph Convolutional Network

Architecture (from synopsis)
─────────────────────────────
  Input (B, T, N, C)
      │
  Input projection  →  Linear(C → H)
      │
  Grid Partition    →  divide N sensors into M cells, expand boundaries
      │
  STBlock × num_layers  (ChebConv + GTU + Temporal Attention)  per cell
      │
  Hierarchical aggregation (scatter back to full N)
      │
  GRU (2-layer)  →  take last hidden state
      │
  Output MLP     →  (B, N, pred_len)
      │
  (B, pred_len, N)

Complexity reduction:
  Global GCN: O(N²)
  GPDSTGCN:   O( Σ_i  (N_i³ + N_i² T + N_i D T) )
  Typical speedup: 70–100× for N=3834
"""

import numpy as np
import torch
import torch.nn as nn

from .layers import STBlock


# ── Grid Partition ────────────────────────────────────────────────────────────

class GridPartition:
    """
    Partition N sensors into M = grid_size² cells using their 2-D coordinates.
    Each cell is optionally expanded to include neighbouring sensors at the
    boundary (controlled by expand_ratio).
    """

    def __init__(
        self,
        coords:       np.ndarray,   # (N, 2)  – normalised [0,1] coordinates
        grid_size:    int   = 4,
        expand_ratio: float = 0.15,
    ):
        self.N            = coords.shape[0]
        self.grid_size    = grid_size
        self.expand_ratio = expand_ratio
        self.cells        = self._partition(coords)

    def _partition(self, coords: np.ndarray) -> list:
        """Return list of dicts {core, expanded} – both 1-D int arrays."""
        g = self.grid_size
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        xedges = np.linspace(x_min, x_max + 1e-8, g + 1)
        yedges = np.linspace(y_min, y_max + 1e-8, g + 1)

        cells = []
        for i in range(g):
            for j in range(g):
                # Core nodes
                mask_core = (
                    (coords[:, 0] >= xedges[i])   & (coords[:, 0] < xedges[i + 1]) &
                    (coords[:, 1] >= yedges[j])   & (coords[:, 1] < yedges[j + 1])
                )
                core = np.where(mask_core)[0]
                if len(core) == 0:
                    continue

                # Expanded nodes (boundary overlap)
                dx = (xedges[i + 1] - xedges[i]) * self.expand_ratio
                dy = (yedges[j + 1] - yedges[j]) * self.expand_ratio
                mask_exp = (
                    (coords[:, 0] >= xedges[i]   - dx) & (coords[:, 0] < xedges[i + 1] + dx) &
                    (coords[:, 1] >= yedges[j]   - dy) & (coords[:, 1] < yedges[j + 1] + dy)
                )
                expanded = np.where(mask_exp)[0]
                cells.append({"core": core, "expanded": expanded})

        return cells

    # Convenience ─────────────────────────────────────────────────────────────
    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def summary(self):
        sizes = [len(c["expanded"]) for c in self.cells]
        print(f"[partition] {self.num_cells} cells | "
              f"nodes/cell: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")


# ── Dummy flat partition (no coordinates available) ───────────────────────────

class FlatPartition:
    """
    Fallback: split N nodes into chunks of ~chunk_size.
    Used when sensor coordinates are unavailable.
    """

    def __init__(self, N: int, chunk_size: int = 20):
        self.cells = []
        for start in range(0, N, chunk_size):
            idx = np.arange(start, min(start + chunk_size, N))
            self.cells.append({"core": idx, "expanded": idx})

    @property
    def num_cells(self):
        return len(self.cells)


# ── Main Model ────────────────────────────────────────────────────────────────

class GPDSTGCN(nn.Module):
    """
    Grid-Partition Dynamic Spatio-Temporal Graph Convolutional Network.
    """

    def __init__(self, config, L_tilde: torch.Tensor, coords: np.ndarray = None):
        """
        Parameters
        ----------
        config   : Config object (see config.py)
        L_tilde  : (N, N)  normalised scaled Laplacian (from graph_utils)
        coords   : (N, 2)  sensor coordinates (optional; uses flat partition if None)
        """
        super().__init__()

        N  = config.num_nodes
        C  = config.in_channels
        H  = config.hidden_dim
        T  = config.seq_len
        P  = config.pred_len

        # Register Laplacian as buffer (moves to device automatically)
        self.register_buffer("L_tilde", L_tilde)

        # ── Grid Partition ───────────────────────────────────────────────────
        if coords is not None:
            self.partition = GridPartition(
                coords, config.grid_size, config.expand_ratio
            )
            self.partition.summary()
        else:
            chunk = max(1, N // (config.grid_size ** 2))
            self.partition = FlatPartition(N, chunk_size=chunk)
            print(f"[model] No coords – flat partition into {self.partition.num_cells} cells")

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(C, H),
            nn.LayerNorm(H),
        )

        # ── ST Blocks ────────────────────────────────────────────────────────
        # Build with progressively larger dilation for multi-scale receptive field
        self.st_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            dil = 2 ** i
            self.st_blocks.append(
                STBlock(
                    in_channels  = H,
                    hidden_dim   = H,
                    K            = config.cheb_k,
                    num_heads    = config.num_heads,
                    dropout      = config.dropout,
                    gtu_kernel   = 3,
                    gtu_dilation = dil,
                )
            )

        # ── GRU for global sequential context ────────────────────────────────
        self.gru = nn.GRU(
            input_size  = H,
            hidden_size = H,
            num_layers  = config.gru_layers,
            batch_first = True,
            dropout     = config.dropout if config.gru_layers > 1 else 0.0,
        )

        # ── Output projection ─────────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, P),               # → pred_len steps
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, N, C)

        Returns
        -------
        out : (B, pred_len, N)
        """
        B, T, N, C = x.shape
        device = x.device

        # ── Input projection ─────────────────────────────────────────────────
        h = self.input_proj(x)             # (B, T, N, H)

        # ── Grid-partitioned ST Blocks ────────────────────────────────────────
        # We apply the ST blocks to the FULL graph at once here.
        # (True GPDSTGCN would process each cell separately and aggregate;
        #  that optimisation is enabled when cells have < ~100 nodes each.
        #  For 307 nodes total it is fast enough without partitioning the forward pass.)
        #
        # To stay faithful to the paper's complexity analysis, we process each
        # cell's sub-graph with its own local Laplacian sub-matrix.

        h = self._partitioned_st(h)        # (B, T, N, H)

        # ── GRU over time ─────────────────────────────────────────────────────
        # Merge N into batch dimension for GRU
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)   # (B*N, T, H)
        h, _ = self.gru(h)
        h = h[:, -1, :]                    # last timestep: (B*N, H)
        h = h.reshape(B, N, -1)            # (B, N, H)

        # ── Output projection ─────────────────────────────────────────────────
        out = self.output_proj(h)          # (B, N, pred_len)
        return out.permute(0, 2, 1)        # (B, pred_len, N)

    # ── Partitioned ST processing ─────────────────────────────────────────────

    def _partitioned_st(self, h: torch.Tensor) -> torch.Tensor:
        """
        Process each grid cell through ST blocks using its local sub-Laplacian.
        Core nodes' features are updated; expanded boundary nodes are used as
        context but their updates are discarded.

        h : (B, T, N, H)  →  (B, T, N, H)
        """
        B, T, N, H_dim = h.shape
        out = torch.zeros_like(h)           # accumulate core outputs
        counts = torch.zeros(N, device=h.device)

        for cell in self.partition.cells:
            core_idx = torch.LongTensor(cell["core"]).to(h.device)
            exp_idx  = torch.LongTensor(cell["expanded"]).to(h.device)

            # Extract sub-graph features
            h_sub = h[:, :, exp_idx, :]     # (B, T, n_exp, H)

            # Extract sub-Laplacian for expanded region
            L_sub = self.L_tilde[exp_idx][:, exp_idx]   # (n_exp, n_exp)

            # Apply ST blocks
            for block in self.st_blocks:
                h_sub = block(h_sub, L_sub)

            # Map core-only nodes back (core positions within expanded)
            core_in_exp = self._core_positions(cell["core"], cell["expanded"])
            out[:, :, core_idx, :] += h_sub[:, :, core_in_exp, :]
            counts[core_idx] += 1.0

        # Average over cells that contributed to each node
        counts = counts.clamp(min=1.0).view(1, 1, N, 1)
        return out / counts

    @staticmethod
    def _core_positions(core: np.ndarray, expanded: np.ndarray) -> torch.LongTensor:
        """Return indices of core nodes within the expanded array."""
        exp_set = {v: i for i, v in enumerate(expanded)}
        return torch.LongTensor([exp_set[c] for c in core])