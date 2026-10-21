# catalog-statistics/scripts — additions

Extensions to the base `catalog-statistics` skill. The original
`catalog_stats.py` (MAXC Mc + Aki-Utsu b) is unchanged; these files add Mc method
cross-checks and a STAI-robust b-value estimator. All pure numpy.

## `mc_methods.py` — three Mc estimators + a GR goodness-of-fit test

A single Mc is fragile, so this module computes three independent estimators and
reports their spread (agreement within one magnitude bin is the sanity check):

| function | method | reference |
|---|---|---|
| `mc_maxc(mags, dm, correction)` | **MAXC** — maximum curvature, peak of the incremental FMD (+0.2 by default) | Wiemer & Wyss 2000; Woessner & Wiemer 2005 |
| `mc_mbs(mags, dm)` | **MBS** — b-value stability; first cutoff where b flattens to within Shi-Bolt δb | Cao & Gao 2002; Woessner & Wiemer 2005 |
| `mc_gft(mags, dm)` | **GFT** — goodness-of-fit; lowest cutoff where a GR model reproduces ≥90/95% of the cumulative FMD | Wiemer & Wyss 2000 |
| `lilliefors_exponential(values)` | Lilliefors-style KS test that magnitudes above Mc are exponential (the GR hypothesis) | Lilliefors 1969 |
| `estimate_mc_all(mags, dm)` | runs all three, reports `spread_bins` and `agree_within_bin` | — |

```bash
python scripts/mc_methods.py --catalog cat.json --dm 0.1
python scripts/mc_methods.py --mags "1.0,1.1,1.2,..." --dm 0.1
```

Notes / gotchas:
- Every internal b-value is Aki-Utsu **MLE** (house rule), never least-squares.
- **MBS runs high on very large catalogs**: with millions of events δb shrinks so
  the stability criterion is never satisfied until an artificially high cutoff. It
  is intended for ordinary-sized catalogs.
- On a real aftershock catalog with strong **short-term aftershock incompleteness
  (STAI)** the three methods legitimately disagree (the FMD is not a single clean
  GR line) — that disagreement is itself the diagnostic.
- `lilliefors_exponential` compares the KS distance to **asymptotic** Lilliefors
  critical values; magnitudes rounded to `dm` are discretized, which inflates the
  KS statistic, so treat rejection on binned data with care. Approximate check.

## `b_positive.py` — the b-positive estimator (Van der Elst 2021)

Ordinary b is biased **low** whenever Mc drifts in time (classic STAI right after a
mainshock). b-positive fixes this by using only the **positive differences** between
successive (time-ordered) magnitudes, which stay exponential with the true slope b
even under a moving completeness threshold.

    b+ = log10(e) / ( mean(δ | δ ≥ δMc) − (δMc − Δm/2) ) ,   δ_i = m_{i+1} − m_i

```bash
python scripts/b_positive.py --mags "1.2,0.8,1.5,..." --dm 0.1   # mags IN TIME ORDER
python scripts/b_positive.py --catalog cat.json --dm 0.1         # sorted by 'time' here
```

Returns `{b, b_uncertainty (Shi-Bolt), n_pos, n_total, dmc, dm, mean_diff, method}`.
Feed magnitudes in **time order** (the CLI sorts a connector result by its UTC
`time` field automatically); do not sort by magnitude.

## Validation — `../tests_extra.py`

Kept separate from the skill's own `smoke_test.py`. Verified results (seeded, so
reproducible):
- b-positive recovers true b=1.0 → 0.995 and b=0.8 → 0.797 on clean GR catalogs.
- Under a **time-varying Mc**, ordinary b = 0.82 (biased low) while b-positive =
  0.996 — recovering the true value.
- MAXC / MBS / GFT all return the completeness bin (raw 1.1 for Mc_true=1.0),
  **spread 0.0 bins**, agreeing within a bin.
- Real 2019 Ridgecrest (15,297 events via `comcat-mcp`): ordinary b = 0.62±0.005
  (STAI low-bias) vs **b-positive = 0.95±0.01** — the correction in action.
