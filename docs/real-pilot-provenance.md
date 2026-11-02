# Real-source pilot provenance

This is a deliberately tiny, four-object smoke evaluation using actual named
objects from the VARnet paper and a catalog-confirmed pulsator. It is not a
statistically representative validation set and should not be used to claim
model performance.

| source_id | label | provenance for identity/class | J2000 coordinates |
|---|---|---|---|
| VZ Boo | pulsator | SIMBAD identifies it as an RR Lyrae variable; coordinates taken from SIMBAD ICRS J2000 entry. | 14:53:15.8992, +25:57:19.495 |
| CV Boo | transit | Paz (2024) uses CV Boo as a transit/eclipsing-binary example; the published eclipsing-binary study gives J2000 coordinates. | 15:26:19.54, +36:58:53 |
| V2860 Ori | transient | Paz (2024) uses V2860 Ori as a nova example; NASA/GSFC's Galactic novae list identifies it as Nova Orionis 2019 and gives its coordinates. | 06:09:57.453, +12:12:25.15 |
| HD 165459 | null | Paz (2024) labels HD 165459 as a null example in its sample light-curve figure; coordinates from SIMBAD/Aladin's indexed entry. | 18:02:30.7413, +58:37:38.150 |

Sources:
- Paz (2024), [paper](https://doi.org/10.3847/1538-3881/ad7fe6), including Figure 4 examples and the validation-catalog discussion.
- [SIMBAD: VZ Boo](https://simbad.u-strasbg.fr/simbad/sim-id?Ident=VZ+Boo)
- [CV Boo eclipsing-binary study](https://doi.org/10.1088/0004-6256/136/5/2158)
- [NASA/GSFC Galactic novae list](https://asd.gsfc.nasa.gov/Koji.Mukai/novae/novae.html)
- [SIMBAD: HD 165459](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=HD+165459)

The coordinate columns are the decimal-degree conversion of the cited J2000
positions. The evaluator queries real NEOWISE-R W1 single-exposure records at
these positions. It may exclude sources that fail minimum coverage or quality
filters; report the usable per-class support along with any metrics.
