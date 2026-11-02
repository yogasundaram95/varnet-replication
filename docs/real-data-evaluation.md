# Real NEOWISE evaluation

The training and published checkpoint remain trained on synthetic curves. This
workflow evaluates that checkpoint against **real W1 single-exposure
observations** fetched from NASA/IPAC IRSA.

## 1. Prepare labeled sources

Create a CSV with the four columns shown in
[`real_catalog_template.csv`](../data/real_catalog_template.csv):

- `source_id`: a unique catalog identifier
- `ra_deg`, `dec_deg`: ICRS coordinates in decimal degrees
- `label`: one of `null`, `transient`, `pulsator`, `transit`

The paper's validation set was curated from the General Catalog of Variable
Stars (48 eclipsing binaries and 49 pulsators), Koji Mukai's Galactic novae
list (34 transient sources whose events occurred during NEOWISE Reactivation),
and 100 manually inspected nonvariable sources. The paper does not publish the
complete 231-row coordinate/label manifest in its article, so this repo does
not invent those identities. Use source lists with traceable catalog references
and record any manual label review in a companion notes file. Eclipsing binaries
map to `transit`; the paper's class name is morphological, not a statement
that these are exoplanet transits.

Primary references:

- Paz (2024), [A Sub-millisecond Fourier and Wavelet-based Model to Extract
  Variable Candidates from the NEOWISE Single-exposure Database](https://doi.org/10.3847/1538-3881/ad7fe6)
- [General Catalog of Variable Stars at HEASARC](https://heasarc.gsfc.nasa.gov/W3Browse/all/gcvs.html)
- [Koji Mukai's Galactic novae list](https://asd.gsfc.nasa.gov/Koji.Mukai/novae/novae.html)
- [NEOWISE Single-exposure Source Database column documentation](https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/sec2_1a.html)

## 2. Install and run

Install the repository requirements, then run:

```bash
python evaluate_real.py --catalog data/my_labeled_sources.csv
```

The script fetches a 3 arcsec cone around each coordinate from the IRSA
`neowiser_p1bs_psd` table, keeps finite W1 detections with positive magnitude
uncertainty, rejects nonzero W1 artifact flags and (when present) frames with
`qual_frame <= 0`, and requires at least 32 usable observations. It converts
W1 magnitude and magnitude uncertainty to Jy and Jy uncertainty, then applies
the repository's paper-based preprocessing. Outputs are written to
`results/real/predictions.csv` and `results/real/metrics.json`.

Network access to IRSA is required. Queries are performed one source at a time;
a 231-source catalog can take several minutes. IRSA TAP is queried asynchronously with retries for transient archive failures. Per-source failures and
insufficient light curves are retained in the predictions file as statuses.

## Interpretation

This is an external real-observation evaluation of the existing synthetic
checkpoint. It does not make the training data real, retrain the model, or
reproduce the paper's score by itself. The result is meaningful only to the
extent that the supplied labels are independently grounded and the sample
selection is documented. Scores should report the number of usable objects per
class alongside the metrics.


## Four-source pilot

Run the checked-in pilot with:

```bash
python evaluate_real.py --catalog data/real_pilot_sources.csv --out results/real-pilot
```

The recorded pilot fetched all four curves (351, 438, 69, and 885 clean observations). The model got 2/4 correct: it classified CV Boo and V2860 Ori correctly, but predicted null for pulsator VZ Boo and transit for null example HD 165459. Accuracy was 0.50 and macro F1 was 0.417. With only one source per class, these values are a smoke evaluation, not a useful performance estimate. See [provenance](real-pilot-provenance.md), [predictions](../results/real-pilot/predictions.csv), and [metrics](../results/real-pilot/metrics.json).
