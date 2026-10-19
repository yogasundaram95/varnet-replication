"""Train and evaluate the VARnet replication on synthetic light curves.

Paper settings (Paz 2024, Section 4.1): AdamW, lr = 8.5e-5,
weight decay = 10**-5.5, 53 epochs = 1.06M unique curves (20k per epoch),
macro precision / recall / F1, 4-class and real-vs-bogus confusion matrices.

Default here is a laptop-scale run; pass --paper-scale for the full setting.

    python train.py                 # ~100k curves, laptop scale
    python train.py --paper-scale   # 53 epochs x 20,000 curves (GPU advised)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from varnet.data import fixed_set, synthetic_batch
from varnet.generator import CLASSES, make_curve
from varnet.model import VARnet

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--per-epoch", type=int, default=4000, help="new synthetic curves per epoch")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--samples", type=int, default=700, help="FEFT output length (paper: 700)")
p.add_argument("--lr", type=float, default=8.5e-5)
p.add_argument("--wd", type=float, default=10 ** -5.5)
p.add_argument("--val", type=int, default=2000)
p.add_argument("--test", type=int, default=4000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--paper-scale", action="store_true")
p.add_argument("--out", default="results")
args = p.parse_args()
if args.paper_scale:
    args.epochs, args.per_epoch = 53, 20000

out = Path(args.out); out.mkdir(exist_ok=True)
torch.manual_seed(args.seed)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device={device}  epochs={args.epochs}  curves/epoch={args.per_epoch}")


def evaluate(model, xs, ys):
    model.eval(); preds, loss = [], 0.0
    with torch.no_grad():
        for x, y in zip(xs, ys):
            logits = model(x.to(device))
            loss += F.cross_entropy(logits, y.to(device), reduction="sum").item()
            preds.append(logits.argmax(1).cpu())
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(preds).numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return loss / len(y_true), pr, rc, f1, y_true, y_pred


# ------------------------------------------------------------------ data ---
print("building fixed validation and test sets ...")
val_x, val_y = fixed_set(args.val, seed=10_001)
test_x, test_y = fixed_set(args.test, seed=20_002)

# --------------------------------------------------------------- training ---
model = VARnet(samples=args.samples).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
rng = np.random.default_rng(args.seed)
history, best_f1 = [], -1.0
t0 = time.time()
for epoch in range(1, args.epochs + 1):
    model.train(); run = 0.0
    for _ in range(args.per_epoch // args.batch):
        x, y = synthetic_batch(args.batch, rng)          # never-repeating data
        loss = F.cross_entropy(model(x.to(device)), y.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item()
    vloss, vp, vr, vf1, *_ = evaluate(model, val_x, val_y)
    history.append(dict(epoch=epoch, train_loss=run / (args.per_epoch // args.batch),
                        val_loss=vloss, val_precision=vp, val_recall=vr, val_f1=vf1))
    print(f"epoch {epoch:3d}  train {history[-1]['train_loss']:.3f}  "
          f"val loss {vloss:.3f}  macro F1 {vf1:.3f}  ({time.time() - t0:.0f}s)")
    if vf1 > best_f1:
        best_f1 = vf1
        torch.save(model.state_dict(), out / "varnet_best.pt")

# ------------------------------------------------------------- evaluation ---
model.load_state_dict(torch.load(out / "varnet_best.pt", map_location=device))
tloss, tp, tr, tf1, y_true, y_pred = evaluate(model, test_x, test_y)
acc = float((y_true == y_pred).mean())
cm4 = confusion_matrix(y_true, y_pred, labels=range(4))
rb_true, rb_pred = (y_true > 0).astype(int), (y_pred > 0).astype(int)
rbp, rbr, rbf, _ = precision_recall_fscore_support(rb_true, rb_pred, average="macro", zero_division=0)
cm2 = confusion_matrix(rb_true, rb_pred, labels=[0, 1])

# inference speed per source
model.eval(); x = test_x[0].to(device)
with torch.no_grad():
    def sync():
        if device == "cuda": torch.cuda.synchronize()
        if device == "mps": torch.mps.synchronize()   # GPU work is asynchronous
    for _ in range(3): model(x)
    sync(); s = time.time()
    for _ in range(10): model(x)
    sync()
per_source_us = (time.time() - s) / (10 * x.shape[0]) * 1e6

results = dict(
    setting=vars(args), device=device,
    curves_seen=args.epochs * (args.per_epoch // args.batch) * args.batch,
    test_set="synthetic, held out", test_size=int(len(y_true)),
    four_class=dict(precision=tp, recall=tr, f1=tf1, accuracy=acc, confusion=cm4.tolist()),
    real_bogus=dict(precision=rbp, recall=rbr, f1=rbf,
                    accuracy=float((rb_true == rb_pred).mean()), confusion=cm2.tolist()),
    inference_us_per_source=per_source_us,
    paper_reference=dict(note="Paper metrics are on 231 REAL NEOWISE sources, not synthetic data",
                         four_class=dict(precision=0.918, recall=0.910, f1=0.914, accuracy=0.922),
                         real_bogus=dict(precision=0.973, recall=0.975, f1=0.974, accuracy=0.974)),
)
(out / "metrics.json").write_text(json.dumps(results, indent=2))
(out / "history.json").write_text(json.dumps(history, indent=2))
print(json.dumps({k: results[k] for k in ("four_class", "real_bogus", "inference_us_per_source")}, indent=2))

# ---------------------------------------------------------------- figures ---
def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

plot_cm(cm4, [c.title() for c in CLASSES], "4-class (synthetic test set)", out / "confusion_4class.png")
plot_cm(cm2, ["Null", "Variable"], "Real vs. bogus (synthetic test set)", out / "confusion_real_bogus.png")

ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
ax[0].set_title("Cross-entropy loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, [h["val_f1"] for h in history]); ax[1].set_title("Validation macro F1"); ax[1].set_xlabel("epoch")
fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

rng = np.random.default_rng(7)
fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
for c in range(4):
    for k in range(3):
        t, f, e = make_curve(c, rng)
        axes[c, k].errorbar(t, f, yerr=e, fmt=".", ms=2, lw=0.3)
        axes[c, k].set_title(CLASSES[c], fontsize=9)
fig.supxlabel("days"); fig.supylabel("W1 flux (Jy)")
fig.tight_layout(); fig.savefig(out / "synthetic_examples.png", dpi=130); plt.close(fig)
print(f"saved results to {out}/")
