"""
Baseline models for comparison (paper Table 3).

- HistoricalAverage  : same time-of-day mean
- ARIMABaseline      : per-node ARIMA(2, 0, 1)
- STGCN              : sandwich Temporal->Spatial->Temporal block
- DCRNN              : bidirectional diffusion GRU encoder-decoder
"""

import numpy as np
import torch
import torch.nn as nn
from .layers import ChebConv


# 1. Historical Average
class HistoricalAverage:
    def __init__(self, period=288):
        self.period = period
        self.slot_means = None

    def fit(self, train_data):
        T, N, C = train_data.shape
        slots = np.zeros((self.period, N), dtype=np.float64)
        counts = np.zeros(self.period, dtype=np.float64)
        for t in range(T):
            s = t % self.period
            slots[s] += train_data[t, :, 0]
            counts[s] += 1.0
        self.slot_means = (slots / np.maximum(counts[:, None], 1.0)).astype(np.float32)
        return self

    def predict(self, x, start_slot=0, pred_len=12):
        N = self.slot_means.shape[1]
        preds = np.zeros((pred_len, N), dtype=np.float32)
        for p in range(pred_len):
            preds[p] = self.slot_means[(start_slot + p) % self.period]
        return preds


# 2. ARIMA Baseline
class ARIMABaseline:
    def __init__(self, order=(2, 0, 1), max_nodes=None):
        self.order = order
        self.max_nodes = max_nodes
        self.models_ = {}

    def fit(self, train_data):
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        T, N, C = train_data.shape
        n_fit = min(N, self.max_nodes) if self.max_nodes else N
        print(f"  Fitting ARIMA{self.order} on {n_fit} nodes ...")
        for node in range(n_fit):
            series = train_data[:, node, 0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=self.order)
                    self.models_[node] = model.fit()
            except Exception:
                self.models_[node] = None
        return self

    def predict(self, x, start_slot=0, pred_len=12):
        seq_len, N, C = x.shape
        preds = np.zeros((pred_len, N), dtype=np.float32)
        for node in range(N):
            model = self.models_.get(node)
            if model is not None:
                try:
                    preds[:, node] = model.forecast(steps=pred_len)
                    continue
                except Exception:
                    pass
            preds[:, node] = x[-1, node, 0]
        return preds


# 3. STGCN helper modules
class _TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2 * out_ch, (1, kernel_size),
                              padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        h = self.conv(x)
        h1, h2 = h.chunk(2, dim=1)
        return h1 * torch.sigmoid(h2)


class _STConvBlock(nn.Module):
    def __init__(self, in_ch, hid, out_ch, K=3):
        super().__init__()
        self.t1 = _TemporalConv(in_ch, hid)
        self.gcn = ChebConv(hid, hid, K=K)
        self.t2 = _TemporalConv(hid, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.residual = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x, L_tilde):
        res = self.residual(x)
        h = self.t1(x)
        B, H, N, T = h.shape
        h = h.permute(0, 3, 2, 1).reshape(B * T, N, H)
        h = self.gcn(h, L_tilde)
        h = h.reshape(B, T, N, H).permute(0, 3, 2, 1)
        h = self.t2(h)
        out = (h + res).permute(0, 2, 3, 1)
        return self.norm(out).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    """STGCN (Yu et al., 2018). Two ST-Conv blocks + output MLP."""
    def __init__(self, config, L_tilde):
        super().__init__()
        C, H, P = config.in_channels, config.hidden_dim, config.pred_len
        self.register_buffer("L_tilde", L_tilde)
        self.block1 = _STConvBlock(C, H, H, K=config.cheb_k)
        self.block2 = _STConvBlock(H, H, H, K=config.cheb_k)
        self.output_proj = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Linear(H, P))

    def forward(self, x):
        B, T, N, C = x.shape
        h = x.permute(0, 3, 2, 1)
        h = self.block1(h, self.L_tilde)
        h = self.block2(h, self.L_tilde)
        h = h.permute(0, 2, 3, 1)[:, :, -1, :]
        return self.output_proj(h).permute(0, 2, 1)


# 4. DCRNN helper modules
class _DiffusionConv(nn.Module):
    def __init__(self, in_ch, out_ch, K=2):
        super().__init__()
        self.K = K
        self.theta = nn.Linear(in_ch * 2 * K, out_ch)

    def forward(self, x, supports):
        B, N, C = x.shape
        parts = []
        for P in supports:
            Pk = x
            parts.append(Pk)
            for _ in range(1, self.K):
                Pk = torch.bmm(P.unsqueeze(0).expand(B, -1, -1), Pk)
                parts.append(Pk)
        return self.theta(torch.cat(parts, dim=-1))


class _DCGRUCell(nn.Module):
    def __init__(self, in_ch, hid, K=2):
        super().__init__()
        self.hidden_dim = hid
        self.gate_conv = _DiffusionConv(in_ch + hid, 2 * hid, K)
        self.cand_conv = _DiffusionConv(in_ch + hid, hid, K)

    def forward(self, x, h, supports):
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(combined, supports))
        r, z = gates.chunk(2, dim=-1)
        cand = torch.tanh(self.cand_conv(torch.cat([x, r * h], -1), supports))
        return z * h + (1 - z) * cand


class DCRNN(nn.Module):
    """DCRNN (Li et al., 2018). 2-layer DCGRU encoder + output MLP."""
    def __init__(self, config, L_tilde):
        super().__init__()
        C, H, N, P = (config.in_channels, config.hidden_dim,
                       config.num_nodes, config.pred_len)
        self.hidden_dim = H
        self.register_buffer("L_tilde", L_tilde)
        A = torch.clamp(torch.eye(N) - L_tilde, 0.0)
        P_f = A / A.sum(1, keepdim=True).clamp(min=1e-8)
        P_b = A.T / A.T.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("P_f", P_f)
        self.register_buffer("P_b", P_b)
        self.enc1 = _DCGRUCell(C, H)
        self.enc2 = _DCGRUCell(H, H)
        self.output_proj = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(H, P))

    def forward(self, x):
        B, T, N, C = x.shape
        supports = [self.P_f, self.P_b]
        h1 = torch.zeros(B, N, self.hidden_dim, device=x.device)
        h2 = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for t in range(T):
            h1 = self.enc1(x[:, t], h1, supports)
            h2 = self.enc2(h1, h2, supports)
        return self.output_proj(h2).permute(0, 2, 1)
