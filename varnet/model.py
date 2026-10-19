"""VARnet (Paz 2024, Section 3, Figure 8), reimplemented in PyTorch.

Phase 1: Conv1d 3->16->24->4, discrete wavelet transform (bior2.2) into
approximation and detail (4 x N/2 each), repeat_interleave back to N,
concatenate with the original 3 channels (11 channels), Conv1d 11->32->16->4.

Phase 2: learned finite-embedding Fourier transform (FEFT, Section 3.2) maps
each of the 4 variable-length channels to a fixed `samples` length;
Conv1d 4->24->12->1; fully connected samples -> samples/2 -> samples/4 -> 4.

ASSUMPTIONS (not specified in the paper): kernel size 5 with same padding,
FEFT output taken as the magnitude of the complex projection.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# bior2.2 decomposition filters (PyWavelets: pywt.Wavelet("bior2.2"))
BIOR22_LO = [0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798212,
             0.3535533905932738, -0.1767766952966369]
BIOR22_HI = [0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, 0.0, 0.0]


def conv_stack(channels: list[int], k: int = 5) -> nn.Sequential:
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.ReLU()]
    return nn.Sequential(*layers)


class DWT1d(nn.Module):
    """Single-level discrete wavelet transform applied per channel."""

    def __init__(self, channels: int):
        super().__init__()
        lo = torch.tensor(BIOR22_LO[::-1]).view(1, 1, -1).repeat(channels, 1, 1)
        hi = torch.tensor(BIOR22_HI[::-1]).view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.c = channels

    def forward(self, x):
        pad = self.lo.shape[-1] - 2
        xp = F.pad(x, (pad // 2, pad - pad // 2), mode="reflect")
        approx = F.conv1d(xp, self.lo, stride=2, groups=self.c)
        detail = F.conv1d(xp, self.hi, stride=2, groups=self.c)
        n = x.shape[-1] // 2
        return approx[..., :n], detail[..., :n]


class FEFT(nn.Module):
    """Finite-embedding Fourier transform: a DFT evaluated at `samples`
    learned frequency positions u_k = alpha_k (N - 1), alpha_k init k/samples.
    F(k) = 1/sqrt(N) * sum_j a_j exp(-2 pi i u_k j / N)."""

    def __init__(self, samples: int = 700):
        super().__init__()
        self.alpha = nn.Parameter(torch.arange(samples, dtype=torch.float32) / samples)

    def forward(self, a):                       # a: (B, C, N)
        n = a.shape[-1]
        u = self.alpha * (n - 1)                # (S,)
        j = torch.arange(n, device=a.device, dtype=a.dtype)
        ang = -2 * math.pi * torch.outer(u, j) / n          # (S, N)
        re = torch.einsum("bcn,sn->bcs", a, torch.cos(ang))
        im = torch.einsum("bcn,sn->bcs", a, torch.sin(ang))
        return torch.sqrt(re ** 2 + im ** 2 + 1e-12) / math.sqrt(n)


class VARnet(nn.Module):
    def __init__(self, samples: int = 700, n_classes: int = 4):
        super().__init__()
        self.pre = conv_stack([3, 16, 24, 4])
        self.dwt = DWT1d(4)
        self.post = conv_stack([11, 32, 16, 4])
        self.feft = FEFT(samples)
        self.freq = conv_stack([4, 24, 12, 1])
        self.head = nn.Sequential(
            nn.Linear(samples, samples // 2), nn.ReLU(),
            nn.Linear(samples // 2, samples // 4), nn.ReLU(),
            nn.Linear(samples // 4, n_classes),
        )

    def forward(self, x):                       # x: (B, 3, N), N even
        h = self.pre(x)
        approx, detail = self.dwt(h)
        approx = approx.repeat_interleave(2, dim=-1)
        detail = detail.repeat_interleave(2, dim=-1)
        h = torch.cat([x, approx, detail], dim=1)            # (B, 11, N)
        h = self.post(h)                                     # (B, 4, N)
        h = self.feft(h)                                     # (B, 4, S)
        h = self.freq(h).squeeze(1)                          # (B, S)
        return self.head(h)                                  # logits (B, 4)
