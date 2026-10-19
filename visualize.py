"""Explanatory figures for the VARnet replication (run after train.py).

    python visualize.py                 # uses results/varnet_best.pt
    python visualize.py --results path  # another results folder

Saves to <results>/figures/:
  1_preprocessing.png      raw flux -> median/IQR -> arcsinh, one curve per class
  2_wavelet.png            original vs bior2.2 approximation and detail
  3_learned_frequencies.png FEFT frequency positions: initial vs learned
  4_f1_by_snr.png          macro F1 by signal-to-noise bucket
  5_misclassified.png      examples the model gets wrong
  6_confidence.png         prediction confidence, correct vs wrong
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from varnet.data import collate, preprocess
from varnet.generator import CLASSES, make_curve
from varnet.model import DWT1d, VARnet

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
p.add_argument("--n", type=int, default=4000, help="fresh synthetic curves for the analyses")
args = p.parse_args()
res = Path(args.results); fig_dir = res / "figures"; fig_dir.mkdir(exist_ok=True)
COL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

state = torch.load(res / "varnet_best.pt", map_location="cpu")
samples = state["feft.alpha"].numel()
model = VARnet(samples=samples); model.load_state_dict(state); model.eval()
rng = np.random.default_rng(123)

# ------------------------------------------------ 1. preprocessing steps ---
fig, ax = plt.subplots(4, 3, figsize=(13, 9))
for c in range(4):
    t, f, e = make_curve(c, rng)
    q75, q25 = np.percentile(f, [75, 25]); iqr = q75 - q25
    steps = [f, (f - np.median(f)) / iqr, np.arcsinh((f - np.median(f)) / iqr)]
    titles = ["raw flux (Jy)", "(flux - median) / IQR", "arcsinh(...)  -> model input"]
    for k in range(3):
        ax[c, k].plot(t, steps[k], ".", ms=2, color=COL[c])
        ax[c, k].set_title(f"{CLASSES[c]}: {titles[k]}", fontsize=9)
fig.supxlabel("days"); fig.tight_layout(); fig.savefig(fig_dir / "1_preprocessing.png", dpi=130); plt.close(fig)

# ------------------------------------------------------- 2. wavelet view ---
dwt = DWT1d(1)
fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for c in range(4):
    x = preprocess(*make_curve(c, rng))[0]
    x = x[: len(x) - len(x) % 2]
    a, d = dwt(torch.tensor(x).view(1, 1, -1))
    idx = np.arange(len(x))
    ax[c].plot(idx, x, color="grey", lw=0.6, label="input")
    ax[c].plot(idx[::2], a.view(-1).numpy() / np.sqrt(2), color=COL[c], lw=1.2, label="approximation (low pass)")
    ax[c].plot(idx[::2], d.view(-1).numpy(), color="black", lw=0.6, alpha=0.6, label="detail (high pass)")
    ax[c].set_title(CLASSES[c], fontsize=10)
ax[0].legend(fontsize=8, loc="upper right"); ax[-1].set_xlabel("observation index")
fig.suptitle("bior2.2 wavelet decomposition (single level)")
fig.tight_layout(); fig.savefig(fig_dir / "2_wavelet.png", dpi=130); plt.close(fig)

# ----------------------------------------- 3. learned FEFT frequencies ---
alpha = state["feft.alpha"].numpy(); init = np.arange(samples) / samples
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(init, label="initial k / samples"); ax[0].plot(alpha, label="learned")
ax[0].set_xlabel("output index k"); ax[0].set_ylabel("alpha_k (fraction of N-1)"); ax[0].legend()
ax[0].set_title("Finite-embedding Fourier transform: frequency positions")
ax[1].hist(alpha - init, bins=60, color="#4c72b0"); ax[1].set_title("Shift from initial position")
fig.tight_layout(); fig.savefig(fig_dir / "3_learned_frequencies.png", dpi=130); plt.close(fig)

# --------------------------------------- predictions on fresh curves -------
labels = rng.integers(0, 4, args.n)
raw, snr = [], []
for y in labels:
    t, f, e = make_curve(int(y), rng)
    raw.append((t, f, e)); snr.append(np.std(f) / e[0])
probs = []
with torch.no_grad():
    for s in range(0, args.n, 256):
        x = collate([preprocess(*r) for r in raw[s:s + 256]])
        probs.append(torch.softmax(model(x), 1).numpy())
probs = np.concatenate(probs); pred = probs.argmax(1); conf = probs.max(1)
snr = np.asarray(snr)

# ---------------------------------------------------- 4. F1 by S/N bucket ---
edges = np.quantile(snr[labels > 0], np.linspace(0, 1, 6))
mids, f1s = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (labels == 0) | ((snr >= lo) & (snr <= hi))
    f1s.append(f1_score(labels[m], pred[m], average="macro"))
    mids.append(f"{lo:.0f}-{hi:.0f}")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(mids, f1s, color="#4c72b0"); ax.set_ylim(0, 1.12)
ax.set_xlabel("variable-curve spread / reported error (quintiles)"); ax.set_ylabel("macro F1")
ax.set_title("Performance by signal strength")
for i, v in enumerate(f1s): ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
fig.tight_layout(); fig.savefig(fig_dir / "4_f1_by_snr.png", dpi=130); plt.close(fig)

# ------------------------------------------------- 5. misclassified curves ---
wrong = np.where(pred != labels)[0][:8]
if len(wrong):
    fig, ax = plt.subplots(2, 4, figsize=(15, 6)); ax = ax.ravel()
    for a, i in zip(ax, wrong):
        t, f, e = raw[i]
        a.plot(t, f, ".", ms=2, color=COL[labels[i]])
        a.set_title(f"true {CLASSES[labels[i]]} / pred {CLASSES[pred[i]]} ({conf[i]:.2f})", fontsize=9)
    for a in ax[len(wrong):]: a.axis("off")
    fig.supxlabel("days"); fig.supylabel("flux (Jy)")
    fig.tight_layout(); fig.savefig(fig_dir / "5_misclassified.png", dpi=130); plt.close(fig)

# ----------------------------------------------------- 6. confidence ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(conf[pred == labels], bins=40, alpha=0.7, label="correct", color="#55a868")
ax.hist(conf[pred != labels], bins=40, alpha=0.7, label="wrong", color="#c44e52")
ax.set_xlabel("softmax confidence"); ax.set_ylabel("curves"); ax.set_yscale("log"); ax.legend()
ax.set_title("Prediction confidence (paper ranks candidates by this)")
fig.tight_layout(); fig.savefig(fig_dir / "6_confidence.png", dpi=130); plt.close(fig)

print(f"accuracy on {args.n} fresh curves: {(pred == labels).mean():.3f}; figures in {fig_dir}/")
