"""
Core neural-network layers for GPDSTGCN.

─ ChebConv       – Chebyshev Graph Convolutional layer  (eq. 2 in paper)
─ GTU            – Gated Tanh Unit for temporal conv    (eq. 5)
─ TemporalAttn   – Multi-head self-attention over time  (eq. 6)
─ STBlock        – One full Spatio-Temporal block
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Chebyshev Graph Convolution ──────────────────────────────────────────────

class ChebConv(nn.Module):
    """
    Y = Σ_{k=0}^{K-1}  θ_k · T_k(L̃) · X

    Inputs
    ------
    x      : (B, N, in_channels)
    L_tilde: (N, N)  – pre-computed scaled normalised Laplacian

    Output : (B, N, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.K = K
        # θ : (K, in_channels, out_channels)
        self.theta = nn.Parameter(
            torch.empty(K, in_channels, out_channels)
        )
        nn.init.xavier_uniform_(self.theta)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # T_0(L̃)·X = X
        Tx_prev2 = x                                          # (B, N, C)
        out = torch.einsum("bnc,co->bno", Tx_prev2, self.theta[0])

        if self.K == 1:
            return out + self.bias

        # T_1(L̃)·X = L̃·X
        L = L_tilde.unsqueeze(0).expand(B, -1, -1)           # (B, N, N)
        Tx_prev1 = torch.bmm(L, x)
        out = out + torch.einsum("bnc,co->bno", Tx_prev1, self.theta[1])

        for k in range(2, self.K):
            Tx_curr = 2.0 * torch.bmm(L, Tx_prev1) - Tx_prev2
            out = out + torch.einsum("bnc,co->bno", Tx_curr, self.theta[k])
            Tx_prev2, Tx_prev1 = Tx_prev1, Tx_curr

        return out + self.bias


# ── Gated Tanh Unit ──────────────────────────────────────────────────────────

class GTU(nn.Module):
    """
    H_t = tanh(W1 * X)  ⊙  σ(W2 * X)

    Applies 1-D dilated causal convolution over the time axis.
    Input  : (B, C_in, N, T)
    Output : (B, C_out, N, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        conv_kwargs = dict(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (1, kernel_size),
            padding      = (0, pad),
            dilation     = (1, dilation),
        )
        self.W_tanh = nn.Conv2d(**conv_kwargs)
        self.W_gate = nn.Conv2d(**conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_tanh(x)) * torch.sigmoid(self.W_gate(x))


# ── Temporal Self-Attention ──────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Multi-head dot-product attention over the time dimension.

    α_{t'} = softmax( score(h_t, h_{t'}) )   ∀ t' ∈ [t-T, t-1]

    Input  : (B*N, T, d_model)
    Output : (B*N, T, d_model)
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.Wq  = nn.Linear(d_model, d_model, bias=False)
        self.Wk  = nn.Linear(d_model, d_model, bias=False)
        self.Wv  = nn.Linear(d_model, d_model, bias=False)
        self.Wo  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BN, T, D = x.shape
        H, dk = self.num_heads, self.d_k

        q = self.Wq(x).view(BN, T, H, dk).transpose(1, 2)   # (BN, H, T, dk)
        k = self.Wk(x).view(BN, T, H, dk).transpose(1, 2)
        v = self.Wv(x).view(BN, T, H, dk).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (BN, H, T, T)
        attn   = self.drop(F.softmax(scores, dim=-1))
        out    = torch.matmul(attn, v)                                  # (BN, H, T, dk)

        out = out.transpose(1, 2).contiguous().view(BN, T, D)
        return self.Wo(out)


# ── Spatio-Temporal Block ────────────────────────────────────────────────────

class STBlock(nn.Module):
    """
    One complete ST block:
        GTU  →  ChebConv  →  Temporal Attention  →  LayerNorm
    with a residual connection from input to output.

    Input / Output : (B, T, N, hidden_dim)
    """

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        K:            int   = 3,
        num_heads:    int   = 4,
        dropout:      float = 0.1,
        gtu_kernel:   int   = 3,
        gtu_dilation: int   = 1,
    ):
        super().__init__()

        self.gtu = GTU(in_channels, hidden_dim,
                       kernel_size=gtu_kernel, dilation=gtu_dilation)
        self.gcn = ChebConv(hidden_dim, hidden_dim, K=K)
        self.t_attn = TemporalAttention(hidden_dim, num_heads, dropout)

        self.norm_gcn  = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # Project residual if channel dims differ
        self.residual_proj = (
            nn.Linear(in_channels, hidden_dim, bias=False)
            if in_channels != hidden_dim else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, N, C)
        L_tilde : (N, N)
        """
        B, T, N, C = x.shape
        residual = self.residual_proj(x)   # (B, T, N, H)

        # ── 1. Gated temporal convolution ────────────────────────────────────
        # reshape for Conv2d: (B, C, N, T)
        xt = x.permute(0, 3, 2, 1)
        xt = self.gtu(xt)                   # (B, H, N, T)
        xt = xt.permute(0, 3, 2, 1)         # (B, T, N, H)

        # ── 2. Chebyshev graph convolution ───────────────────────────────────
        H = xt.shape[-1]
        xs = xt.reshape(B * T, N, H)
        xs = self.gcn(xs, L_tilde)          # (B*T, N, H)
        xs = xs.reshape(B, T, N, H)
        xs = self.norm_gcn(xs + residual)

        # ── 3. Temporal multi-head attention ─────────────────────────────────
        # swap to (B*N, T, H) for attention over time
        xa = xs.permute(0, 2, 1, 3).reshape(B * N, T, H)
        xa = self.t_attn(xa)
        xa = xa.reshape(B, N, T, H).permute(0, 2, 1, 3)   # (B, T, N, H)
        xa = self.norm_attn(xa + xs)

        # ── 4. Feed-forward + residual ───────────────────────────────────────
        out = self.norm_ffn(self.ffn(xa) + xa)
        return self.dropout(out)