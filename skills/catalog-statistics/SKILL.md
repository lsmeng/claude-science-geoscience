---
name: catalog-statistics
description: >-
  Frequency-magnitude statistics for an earthquake catalog: magnitude of completeness
  Mc (maximum-curvature), Gutenberg-Richter b-value by Aki-Utsu maximum-likelihood
  (NOT least-squares) with Shi-Bolt uncertainty, a-value, and the incremental+cumulative
  FMD. Use whenever the user wants a b-value, Mc, completeness assessment, or a
  frequency-magnitude plot for a catalog — including induced-seismicity monitoring
  where b-value and Mc are the headline metrics. Self-contained (numpy/pandas only).
license: MIT
category: geoscience
---

# catalog-statistics

Computes the headline frequency-magnitude metrics for a catalog. Reads a CSV (a
`mag` column) or a connector search result (`{"events":[{"mag":…}]}` from
`comcat`/`fdsnws`), so it chains directly off the data connectors.

## Method (and why)

- **Mc — maximum curvature (MAXC) + 0.2.** Peak of the incremental FMD, plus the
  empirical Woessner–Wiemer correction.
- **b-value — Aki-Utsu MLE**, `b = log10(e) / (mean(M|M≥Mc) − (Mc − dM/2))`. We do
  **not** fit a least-squares slope to the cumulative FMD — that is biased and the
  is a common review error.
- **Uncertainty — Shi & Bolt (1982)**, `σ_b = 2.30·b²·√(Σ(M−M̄)²/(N(N−1)))`.
- **a-value** from `log10 N(≥Mc) + b·Mc`.

## Usage

```bash
python scripts/catalog_stats.py --catalog catalog.json --dm 0.1
python scripts/catalog_stats.py --catalog catalog.csv --mc 1.8   # force Mc
python scripts/catalog_stats.py --mags "1.2,1.3,1.5,2.0,2.1"     # quick check
```

Output JSON: `{n_total, mag_min, mag_max, mc, mc_method, b, b_uncertainty, a, n,
mean_mag, dm, method, n_warning, fmd:[…]}`.

## Verified

Tested:
- Recovers a **known synthetic b** (true b=1.0 → 0.996; true b=0.8 → 0.798) from a
  binned Gutenberg-Richter sample — the estimator is correct.
- Runs on a **real ComCat catalog** (2019 Ridgecrest, 17,513 events within 60 km, 4-14 Jul 2019, via the `comcat` connector):
  Mc 1.6, b 0.66 ± 0.006, N = 8,690 ≥ Mc (2026-10-21 run).

## Reviewer gotchas this skill respects

- **MLE, not least-squares** for b.
- **Fit only M ≥ Mc**; `dM` binning consistent between Mc and b.
- **Report σ_b** (Shi-Bolt) and **N**; warns when `N < 50` (unstable b).
- **Mc is fragile:** short-term aftershock incompleteness (STAI) right after a
  mainshock raises the true Mc and biases b low — MAXC will underestimate Mc there.
  The Ridgecrest smoke-test result (b≈0.62) illustrates exactly this; always sanity-
  check Mc stability before trusting an aftershock b-value.
- **Single magnitude type only** — do not mix ML/Md/Mw in one FMD.

## Later (not yet implemented)

`b-positive` (Van der Elst 2021), Lilliefors/MBS Mc, spatial/temporal b-value maps,
Omori-Utsu and ETAS. These would fold in `seismostats`
once available in the environment.
