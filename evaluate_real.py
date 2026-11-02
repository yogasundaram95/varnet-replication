"""Evaluate a trained VARnet checkpoint on labeled, real NEOWISE W1 light curves.

Input CSV columns: source_id,ra_deg,dec_deg,label. Labels must be one of
null, transient, pulsator, transit. Observations are queried from IRSA's
NEOWISE-R Single-exposure Source Table; no synthetic observations are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import IrsaClass
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
    for attempt in range(3):
        try:
            table = IrsaClass().query_region(
                coord,
                catalog=CATALOG,
                spatial="Cone",
                radius=radius_arcsec * u.arcsec,
                columns="mjd,ra,dec,w1mpro,w1sigmpro,cc_flags,qual_frame",
                async_job=True,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
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


def calibration_metrics(probabilities, y_true):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    bins = np.minimum(np.digitize(confidence, np.linspace(0, 1, 11)) - 1, 9)
    ece = sum(
        (bins == i).mean() * abs(correct[bins == i].mean() - confidence[bins == i].mean())
        for i in range(10) if np.any(bins == i)
    )
    targets = np.eye(len(CLASSES))[y_true]
    return float(ece), float(np.square(probabilities - targets).sum(axis=1).mean())


def bootstrap_intervals(y_true, y_pred, n_bootstrap: int, seed: int = 2026):
    support = np.bincount(y_true, minlength=len(CLASSES))
    if n_bootstrap < 1 or support.min() < 10:
        return None
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == label) for label in range(len(CLASSES))]
    accuracies, macro_f1s = [], []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in by_class
        ])
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true[sampled], y_pred[sampled], labels=np.arange(len(CLASSES)), zero_division=0
        )
        accuracies.append(float((y_true[sampled] == y_pred[sampled]).mean()))
        macro_f1s.append(float(f1.mean()))
    return {
        "method": "stratified source bootstrap",
        "replicates": n_bootstrap,
        "accuracy_95_percent": np.percentile(accuracies, [2.5, 97.5]).tolist(),
        "macro_f1_95_percent": np.percentile(macro_f1s, [2.5, 97.5]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="CSV with source_id, ra_deg, dec_deg, label")
    parser.add_argument("--checkpoint", default="results/varnet_best.pt")
    parser.add_argument("--out", default="results/real")
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1, help="concurrent asynchronous IRSA queries; use 1 to limit archive load")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    # Keep the literal label "null"; pandas otherwise interprets it as NaN.
    sources = pd.read_csv(args.catalog, keep_default_na=False)
    required = {"source_id", "ra_deg", "dec_deg", "label"}
    missing = required - set(sources.columns)
    if missing:
        raise SystemExit(f"Catalog is missing columns: {sorted(missing)}")
    sources["source_id"] = sources["source_id"].astype(str).str.strip()
    sources["label"] = sources["label"].astype(str).str.strip()
    if sources["source_id"].duplicated().any():
        raise SystemExit("source_id values must be unique after trimming whitespace")
    unknown = sorted(set(sources["label"]) - set(CLASSES))
    if unknown:
        raise SystemExit(f"Unknown labels {unknown}; choose from {CLASSES}")
    if sources["source_id"].duplicated().any():
        raise SystemExit("source_id values must be unique")
    if sources[list(required)].isna().any().any():
        raise SystemExit("Catalog contains missing IDs, coordinates, or labels")

    sources["ra_deg"] = pd.to_numeric(sources["ra_deg"], errors="coerce")
    sources["dec_deg"] = pd.to_numeric(sources["dec_deg"], errors="coerce")
    if (sources["source_id"].eq("").any() or sources["ra_deg"].isna().any()
            or sources["dec_deg"].isna().any() or sources["label"].eq("").any()):
        raise SystemExit("Catalog contains blank IDs, invalid coordinates, or blank labels")
    if (not sources["ra_deg"].between(0, 360).all()
            or not sources["dec_deg"].between(-90, 90).all()):
        raise SystemExit("Catalog coordinates are outside valid RA/Dec ranges")

    records = []
    for row in sources.itertuples(index=False):
        record = {
            "source_id": row.source_id,
            "label": row.label,
            "status": "pending",
            "n_observations": 0,
        }
        for metadata in ("catalog_type", "nova_year", "catalog"):
            if hasattr(row, metadata):
                record[metadata] = getattr(row, metadata)
        records.append(record)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "retrieval_status.csv"
    pd.DataFrame(records).to_csv(status_path, index=False)
    curves_by_index = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, row in enumerate(sources.itertuples(index=False)):
            futures[pool.submit(fetch_curve, row, args.radius_arcsec, args.min_points)] = index
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            row = sources.iloc[index]
            try:
                curve, status = future.result()
            except Exception as exc:
                curve, status = None, f"query_error:{type(exc).__name__}:{exc}"
            records[index]["status"] = status
            records[index]["n_observations"] = int(status.split(":")[-1]) if status.startswith("ok:") else 0
            if curve is not None:
                curves_by_index[index] = curve
            pd.DataFrame(records).to_csv(status_path, index=False)
            print(f"[{completed}/{len(sources)}] {row['source_id']}: {status}", flush=True)

    usable_indices = sorted(curves_by_index)
    curves = [curves_by_index[i] for i in usable_indices]
    labels = [CLASSES.index(sources.iloc[i]["label"]) for i in usable_indices]
    if not curves:
        pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
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
    supported = support > 0
    ece, brier = calibration_metrics(probs, y_true)
    input_support = sources["label"].value_counts().to_dict()
    usable_support = {name: int(support[i]) for i, name in enumerate(CLASSES)}
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "NASA/IPAC IRSA NEOWISE-R Single-exposure Source Table",
        "catalog": CATALOG,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "input_catalog_sha256": hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest(),
        "device": device,
        "input_sources": len(sources),
        "usable_sources": len(curves),
        "excluded_sources": len(sources) - len(curves),
        "input_class_support": {name: int(input_support.get(name, 0)) for name in CLASSES},
        "usable_class_support": usable_support,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision[supported].mean()),
        "macro_recall": float(recall[supported].mean()),
        "macro_f1": float(f1[supported].mean()),
        "macro_averaging_scope": "classes with at least one usable labeled source",
        "confidence_calibration": {"expected_calibration_error_10_bins": ece, "multiclass_brier_score": brier},
        "bootstrap_confidence_intervals": bootstrap_intervals(y_true, y_pred, args.bootstrap),
        "baselines": {
            "uniform_random_accuracy": 1.0 / len(CLASSES),
            "majority_class_accuracy": max(usable_support.values()) / len(curves),
        },
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

    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
