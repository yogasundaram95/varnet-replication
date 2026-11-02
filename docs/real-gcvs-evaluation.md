# Expanded GCVS evaluation set

This fixed source list contains 80 independently catalog-labeled targets, with
20 per VARnet output class. It excludes the four objects in the initial pilot.
It was sampled deterministically (seed 2026) from NASA HEASARC's current
General Catalog of Variable Stars table. The [manifest](../data/real_sources_20_per_class.csv)
records each star name, coordinates, catalog type, and the labels used.

## Class mapping and selection

- `null`: GCVS variability type `CST` (cataloged as constant).
- `transient`: GCVS `nova_year >= 2014`, within the NEOWISE Reactivation period.
- `pulsator`: established pulsating types (RR Lyrae, Cepheid, Mira, semiregular,
  Delta Scuti, and listed related types).
- `transit`: GCVS eclipsing types `EA`, `EB`, and `EW`, including subtypes.

The builder keeps finite J2000 positions, entries with unambiguous astrometric
flags, and proper motions below 0.02 arcsec/year in both coordinates so the
catalog positions remain compatible with the evaluator's 3 arcsec IRSA cone.
It excludes uncertain variability-type suffixes for pulsators/eclipsing
binaries and excludes all four initial-pilot identities. The fixed manifest is
checked in, so rerunning the evaluator does not resample the holdout.

Build a fresh source list from HEASARC with:

```bash
python scripts/build_real_catalog.py --per-class 20 --seed 2026 --out data/real_sources_20_per_class.csv
```

Then evaluate:

```bash
python evaluate_real.py --catalog data/real_sources_20_per_class.csv --out results/real-gcvs-20-per-class --workers 1
```

Each query is asynchronous, sequential by default, retried after transient
failures, and writes a retrieval-status CSV as it completes. The evaluator
reports class support, per-class metrics, calibration, random/majority
baselines, and stratified bootstrap intervals when at least ten sources per
class have usable curves.

## Status

The catalog and evaluation path are prepared. Per-source requests timed out during the full run, and a single uploaded-table query for all 80 positions did not return within 15 minutes.
The completed real-data result remains the four-source pilot in
[`results/real-pilot/`](../results/real-pilot/); no score is claimed for the
80-source list until the archive queries finish.
