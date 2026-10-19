"""Preprocessing (Paz 2024, Section 2.2) and batching."""
from __future__ import annotations

import numpy as np
import torch

from .generator import CLASSES, TOTAL_DAYS, make_curve


def iqr(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25) or 1e-12


def preprocess(t: np.ndarray, flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Return a (3, T) array: [brightness, uncertainty, time].
    Eq. 2-4: flux -> arcsinh((flux - median) / IQR)
    Eq. 5:   err  -> arcsinh(err / IQR(flux))
    time     -> (t - t0) / 4000
    Real NEOWISE magnitudes must first be converted with Eq. 1:
    flux = 309.25 * 10**(-w1mpro / 2.5)."""
    order = np.argsort(t)
    t, flux, err = t[order], flux[order], err[order]
    scale = iqr(flux)
    f = np.arcsinh((flux - np.median(flux)) / scale)
    e = np.arcsinh(err / scale)
    tt = (t - t[0]) / TOTAL_DAYS
    return np.stack([f, e, tt]).astype(np.float32)


def collate(series: list[np.ndarray]) -> torch.Tensor:
    """Zero-pad a list of (3, T_i) arrays to (B, 3, max T)."""
    n = max(s.shape[1] for s in series)
    n += n % 2                                  # even length for the DWT
    out = np.zeros((len(series), 3, n), np.float32)
    for i, s in enumerate(series):
        out[i, :, : s.shape[1]] = s
    return torch.from_numpy(out)


def synthetic_batch(batch_size: int, rng: np.random.Generator):
    """Balanced batch generated on demand (pseudo-online learning, Sec 2.3)."""
    labels = rng.integers(0, len(CLASSES), batch_size)
    series = [preprocess(*make_curve(int(y), rng)) for y in labels]
    return collate(series), torch.from_numpy(labels.astype(np.int64))


def fixed_set(n: int, seed: int):
    """A reproducible held-out synthetic set, generated in chunks."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for start in range(0, n, 256):
        x, y = synthetic_batch(min(256, n - start), rng)
        xs.append(x); ys.append(y)
    return xs, ys
