"""Evaluate a trained VARnet checkpoint on labeled, real NEOWISE W1 light curves.

Input CSV columns: source_id,ra_deg,dec_deg,label. Labels must be one of
null, transient, pulsator, transit. Observations are queried from IRSA's
NEOWISE-R Single-exposure Source Table; no synthetic observations are used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from varnet.data import collate, preprocess
from varnet.generator import CLASSES
from varnet.model import VARnet

ZERO_POINT_JY = 309.54
CATALOG = "neowiser_p1bs_psd"


def finite_column(table, name):
    return np.asarray(table[name], dtype=float)


def fetch_curve(source, radius_arcsec: float, min_points: int):
    coord = SkyCoord(float(source.ra_deg) * u.deg, float(source.dec_deg) * u.deg)
    table = Irsa.query_region(
        coord,
        catalog=CATALOG,
        spatial="Cone",
        radius=radius_arcsec * u.arcsec,
    )
    if len(table) == 0:
        return None, "no_irsa_rows"

    required = {"mjd", "ra", "dec", "w1mpro", "w1sigmpro"}
    missing = required - set(table.colnames)
    if missing:
        raise RuntimeError(f"IRSA response is missing required columns: {sorted(missing)}")

    ra, dec = finite_column(table, "ra"), finite_column(table, "dec")
    observed = SkyCoord(ra * u.deg, dec * u.deg)
    separation = coord.separation(observed).arcsec
    keep = np.isfinite(separation) & (separation <= radius_arcsec)

    mags = finite_column(table, "w1mpro")
    mag_err = finite_column(table, "w1sigmpro")
    mjd = finite_column(table, "mjd")
    keep &= np.isfinite(mags) & np.isfinite(mag_err) & np.isfinite(mjd)
    keep &= (mag_err > 0) & (mags > 0)

    # Retain only clean W1 detections. In cc_flags, character 0 is W1.
    if "cc_flags" in table.colnames:
        flags = np.asarray(table["cc_flags"]).astype(str)
        keep &= np.array([len(flag) > 0 and flag[0] == "0" for flag in flags])

    # qual_frame=0 marks a frame that should not be used for science.
    if "qual_frame" in table.colnames:
        quality = finite_column(table, "qual_frame")
        keep &= np.isfinite(quality) & (quality > 0)

    if int(keep.sum()) < min_points:
        return None, f"too_few_clean_points:{int(keep.sum())}"

    mjd, mags, mag_err = mjd[keep], mags[keep], mag_err[keep]
    flux = ZERO_POINT_JY * np.power(10.0, -mags / 2.5)
    # Propagate magnitude uncertainty through F = F0 * 10**(-m/2.5).
    flux_err = (math.log(10.0) / 2.5) * flux * mag_err
    return preprocess(mjd, flux, flux_err), f"ok:{len(mjd)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="CSV with source_id, ra_deg, dec_deg, label")
    parser.add_argument("--checkpoint", default="results/varnet_best.pt")
    parser.add_argument("--out", default="results/real")
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sources = pd.read_csv(args.catalog)
    required = {"source_id", "ra_deg", "dec_deg", "label"}
    missing = required - set(sources.columns)
    if missing:
        raise SystemExit(f"Catalog is missing columns: {sorted(missing)}")
    unknown = sorted(set(sources["label"].dropna()) - set(CLASSES))
    if unknown:
        raise SystemExit(f"Unknown labels {unknown}; choose from {CLASSES}")
    if sources["source_id"].duplicated().any():
        raise SystemExit("source_id values must be unique")
    if sources[list(required)].isna().any().any():
        raise SystemExit("Catalog contains missing IDs, coordinates, or labels")

    records, curves, labels = [], [], []
    for row in sources.itertuples(index=False):
        try:
            curve, status = fetch_curve(row, args.radius_arcsec, args.min_points)
        except Exception as exc:
            curve, status = None, f"query_error:{type(exc).__name__}:{exc}"
        records.append({
            "source_id": row.source_id,
            "label": row.label,
            "status": status,
            "n_observations": int(status.split(":")[-1]) if status.startswith("ok:") else 0,
        })
        if curve is not None:
            curves.append(curve)
            labels.append(CLASSES.index(row.label))

    usable_indices = [i for i, r in enumerate(records) if r["status"].startswith("ok:")]
    if not curves:
        raise SystemExit("No usable light curves were retrieved; see per-source statuses.")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = VARnet().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(curves), args.batch_size):
            batch_curves = curves[start:start + args.batch_size]
            batch = collate(batch_curves).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probs = np.concatenate(probabilities)
    y_true = np.asarray(labels, dtype=int)
    y_pred = probs.argmax(axis=1)

    for idx, p in zip(usable_indices, probs):
        records[idx].update({
            "prediction": CLASSES[int(p.argmax())],
            "confidence": float(p.max()),
            **{f"p_{name}": float(p[j]) for j, name in enumerate(CLASSES)},
        })

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    metrics = {
        "data_source": "NASA/IPAC IRSA NEOWISE-R Single-exposure Source Table",
        "catalog": CATALOG,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "input_sources": len(sources),
        "usable_sources": len(curves),
        "excluded_sources": len(sources) - len(curves),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_order": CLASSES,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i, name in enumerate(CLASSES)
        },
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(CLASSES))
        ).tolist(),
        "quality_policy": {
            "cone_radius_arcsec": args.radius_arcsec,
            "minimum_points": args.min_points,
            "clean_w1_cc_flags_only": True,
            "positive_qual_frame_only_when_column_present": True,
        },
        "note": "Real-data result on the supplied labeled catalog; not directly comparable to the paper unless the exact source list and selection are reproduced.",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
