"""Build a reproducible 4-class labeled source catalog from NASA HEASARC GCVS.

The output is a source list for external evaluation, not a collection of model
predictions. It selects confirmed GCVS classes and holds out the four-source
pilot identities so the expanded evaluation is independent of that pilot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PULSATOR_TYPES = {
    "RRAB", "RRC", "RR", "RRD", "RR(B)", "DCEP", "DCEPS", "CEP",
    "M", "SR", "SRA", "SRB", "SRC", "SRD", "DSCT", "DSCTC",
    "BCEP", "SXPHE", "CWB", "CWA", "RVA", "RVB", "RVTAU", "BLBOO",
    "ACYG",
}
PILOT_IDENTITIES = {"vzboo", "cvboo", "v2860ori", "hd165459"}
CLASSES = ("null", "transient", "pulsator", "transit")


def load_catalog(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path, keep_default_na=False)

    from astroquery.heasarc import Heasarc

    query = (
        "SELECT name,ra,dec,variability_type,nova_year,pm_ra,pm_dec,"
        "coordinates_epoch,coordinates_epoch_flag,ref_astrometry_flag,comment_flag "
        "FROM gcvs"
    )
    return Heasarc.query_tap(query).to_table().to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-csv", help="cached HEASARC GCVS extract; fetches live if omitted")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="data/real_sources_20_per_class.csv")
    args = parser.parse_args()
    if args.per_class < 2:
        raise SystemExit("Use at least two sources per class.")

    data = load_catalog(args.catalog_csv)
    required = {
        "name", "ra", "dec", "variability_type", "nova_year", "pm_ra",
        "pm_dec", "coordinates_epoch", "coordinates_epoch_flag",
        "ref_astrometry_flag", "comment_flag",
    }
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"HEASARC GCVS extract is missing fields: {sorted(missing)}")

    data["source_id"] = data["name"].astype(str).str.strip()
    data["catalog_type"] = data["variability_type"].astype(str).str.strip().str.upper()
    for column in ("ra", "dec", "nova_year", "pm_ra", "pm_dec", "coordinates_epoch"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["identity_key"] = data["source_id"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    # Coordinates are J2000. Keep low-proper-motion entries so a 3 arcsec IRSA
    # cone remains appropriate at the NEOWISE epoch without large corrections.
    sound_position = (
        data["ra"].between(0, 360)
        & data["dec"].between(-90, 90)
        & data["coordinates_epoch"].eq(2000)
        & data["coordinates_epoch_flag"].astype(str).str.strip().eq("")
        & data["ref_astrometry_flag"].astype(str).str.strip().eq("")
        & data["pm_ra"].notna()
        & data["pm_dec"].notna()
        & data["pm_ra"].abs().lt(0.02)
        & data["pm_dec"].abs().lt(0.02)
        & data["comment_flag"].astype(str).str.strip().ne("Y")
        & data["source_id"].ne("")
        & ~data["identity_key"].isin(PILOT_IDENTITIES)
    )
    data = data.loc[sound_position].drop_duplicates("identity_key").copy()
    certain_type = ~data["catalog_type"].str.endswith(":")

    masks = {
        "null": data["catalog_type"].eq("CST"),
        "transient": data["nova_year"].ge(2014),
        "pulsator": data["catalog_type"].isin(PULSATOR_TYPES) & certain_type,
        "transit": data["catalog_type"].str.match(r"^(EA|EB|EW)(/.*)?$") & certain_type,
    }

    rng = np.random.default_rng(args.seed)
    pieces = []
    for label in CLASSES:
        candidates = data.loc[masks[label]].copy()
        if len(candidates) < args.per_class:
            raise SystemExit(
                f"Only {len(candidates)} eligible {label} sources; need {args.per_class}."
            )
        sampled = candidates.sample(n=args.per_class, random_state=int(rng.integers(0, 2**32 - 1)))
        sampled["label"] = label
        pieces.append(sampled)

    result = pd.concat(pieces, ignore_index=True)
    result = result.rename(columns={
        "ra": "ra_deg",
        "dec": "dec_deg",
        "pm_ra": "pm_ra_arcsec_per_year",
        "pm_dec": "pm_dec_arcsec_per_year",
    })
    result["catalog"] = "HEASARC GCVS"
    columns = [
        "source_id", "ra_deg", "dec_deg", "label", "catalog", "catalog_type",
        "nova_year", "pm_ra_arcsec_per_year", "pm_dec_arcsec_per_year",
        "coordinates_epoch",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result[columns].sort_values(["label", "source_id"]).to_csv(args.out, index=False)
    print(f"Wrote {len(result)} sources to {args.out}")
    print(result.groupby("label").size().to_string())


if __name__ == "__main__":
    main()
