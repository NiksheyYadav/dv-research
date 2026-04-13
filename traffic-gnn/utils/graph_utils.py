"""
Graph construction utilities.

─ Load adjacency from PEMS04 distance CSV
─ Build normalised weighted adjacency A
─ Compute Chebyshev scaled Laplacian  L̃ = 2L/λ_max − I
"""

import os
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


# ── Distance-based adjacency ─────────────────────────────────────────────────

def _load_adj_from_csv(csv_path: str, num_nodes: int, sigma2: float = 0.1,
                       epsilon: float = 0.5) -> np.ndarray:
    """
    Build a weighted adjacency matrix from a PEMS-style distance CSV.

    CSV columns: from, to, distance
    Weights: w_ij = exp( -d²_ij / σ² )  if w_ij ≥ ε else 0
    """
    import pandas as pd

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if not os.path.exists(csv_path):
        print(f"[graph] CSV not found at {csv_path} – using identity adjacency")
        return np.eye(num_nodes, dtype=np.float32)

    df = pd.read_csv(csv_path)
    # Some PEMS CSVs have 2 or 3 columns
    if df.shape[1] == 3:
        src, dst, dist = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values
    elif df.shape[1] == 2:
        src, dst = df.iloc[:, 0].values, df.iloc[:, 1].values
        dist = np.ones(len(src), dtype=np.float32)
    else:
        raise ValueError(f"Unexpected CSV format with {df.shape[1]} columns")

    for s, d, w in zip(src, dst, dist):
        s, d = int(s), int(d)
        if s >= num_nodes or d >= num_nodes:
            continue
        weight = np.exp(-w * w / sigma2)
        if weight >= epsilon:
            A[s, d] = weight
            A[d, s] = weight

    # Self-loops
    np.fill_diagonal(A, 1.0)
    return A


# ── Laplacian computation ─────────────────────────────────────────────────────

def _compute_laplacian(A: np.ndarray) -> np.ndarray:
    """Symmetric normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-12)))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.astype(np.float32)


def _scaled_laplacian(L: np.ndarray) -> np.ndarray:
    """Chebyshev scaled Laplacian:  L̃ = 2L / λ_max  −  I."""
    N = L.shape[0]
    try:
        L_sparse = coo_matrix(L)
        lambda_max = eigsh(L_sparse, k=1, which="LM", return_eigenvectors=False)[0]
    except Exception:
        lambda_max = 2.0  # fallback

    L_tilde = (2.0 / lambda_max) * L - np.eye(N, dtype=np.float32)
    return L_tilde


# ── Public API ────────────────────────────────────────────────────────────────

def build_graph(adj_path: str, num_nodes: int) -> dict:
    """
    Build graph from PEMS distance CSV.

    Returns
    -------
    dict with keys:
        A       : (N, N) np.ndarray  – weighted adjacency
        L       : (N, N) np.ndarray  – normalised Laplacian
        L_tilde : (N, N) torch.Tensor – scaled Chebyshev Laplacian
    """
    A = _load_adj_from_csv(adj_path, num_nodes)
    L = _compute_laplacian(A)
    L_tilde = _scaled_laplacian(L)

    print(f"[graph] Adjacency: {num_nodes} nodes, "
          f"{(A > 0).sum() - num_nodes} edges, "
          f"density={((A > 0).sum() - num_nodes) / (num_nodes * (num_nodes - 1)):.4f}")

    return {
        "A":       A,
        "L":       L,
        "L_tilde": torch.FloatTensor(L_tilde),
    }
