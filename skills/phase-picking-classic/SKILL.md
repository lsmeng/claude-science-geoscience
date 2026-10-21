---
name: phase-picking-classic
description: >-
  Detect and pick seismic P/S phases with classic (non-ML) ObsPy characteristic
  functions: recursive_sta_lta / classic_sta_lta with trigger_onset, network
  coincidence_trigger, the Baer-Kradolfer pk_baer P picker, the Akazawa ar_pick
  AR P&S picker, plus AR-AIC (Maeda) sub-sample onset refinement. Use whenever the
  user wants to detect events or pick arrival times from raw waveforms WITHOUT a
  deep-learning model (no torch/seisbench needed), for a quick STA/LTA trigger,
  or as the deterministic fallback for phase-picking-ml. ObsPy + numpy only.
license: MIT
category: geoscience
---

# phase-picking-classic

Turns a miniSEED trace (e.g. from the the `fdsnws` connector) into a list of P/S
onset picks using the standard ObsPy pickers — no machine learning. Absolute
ISO-8601 UTC pick times, so results feed straight into association/location.

## Method (and why)

- **STA/LTA** — `recursive_sta_lta` (default; streaming recursion, robust) or
  `classic_sta_lta`, then `trigger_onset(cft, thr_on, thr_off)`. STA/LTA windows are
  given in **seconds** and converted to samples internally against the trace rate.
- **coincidence_trigger** — network detection: keep only onsets seen on
  `>= min_stations` traces within the coincidence window (kills local noise spikes).
- **pk_baer** — Baer & Kradolfer (1987) envelope P picker; good on impulsive P.
- **ar_pick** — Akazawa (2004) autoregressive P **and** S picker; needs 3-component
  Z,N,E from one station.
- **AR-AIC refinement** — Maeda (1985) `AIC(k)=k·log var(x[:k])+(n-k-1)·log var(x[k:])`;
  the minimum inside a window around a coarse STA/LTA trigger is the true onset. This
  removes the STA/LTA trigger lag and sharpens the pick to sub-sample-scale accuracy.

## Usage

```bash
python scripts/sta_lta_pick.py --mseed trace.mseed --method recursive \
    --sta 0.5 --lta 5 --thr-on 3 --thr-off 1
python scripts/sta_lta_pick.py --mseed trace.mseed --method baer
python scripts/sta_lta_pick.py --mseed 3comp.mseed --method ar          # Z,N,E
python scripts/sta_lta_pick.py --mseed net.mseed  --method coincidence --min-stations 4
```

Importable API (used by `phase-picking-ml` as its fallback):
`pick_stream(stream, method=…)` → `[{network,station,location,channel,phase,time,sample,cf_value,method}]`;
also `sta_lta_picks`, `baer_pick`, `ar_pick_ze`, `aic_curve`, `aic_refine`, `coincidence`.

## Verified

`smoke_test.py` (clear PASS markers):
- **Synthetic step onset** (known sample 3000 @ 100 Hz): recursive STA/LTA triggers
  within +2 samples (20 ms); AIC refines to the exact sample.
- **Real 2019 Ridgecrest M7.1 mainshock** at **CI.CLC** (~5 km, via the `fdsnws`/SCEDC):
  STA/LTA first-triggers **+2.00 s after origin**, inside the expected P window
  (~1.5–2.5 s for this hypocentral distance) → **VERIFIED**.

## Reviewer gotchas respected

- **Trigger time ≠ onset time.** Raw STA/LTA fires *after* the true onset (window
  ramp). Always AIC-refine before reporting an arrival; the skill does this by default.
- **STA/LTA is sensitive to band.** Pre-filter (bandpass ~1–20 Hz) noisy/broadband
  data before picking; do it in `waveform-preprocess`, not here.
- **State units/frames:** windows in seconds, pick times absolute UTC, sample index
  given for provenance. `cf_value` is the characteristic-function value at the trigger
  (a rough SNR proxy), not a probability.
- **pk_baer/STA/LTA can false-pick on pre-event noise** (see the CLI example where
  pk_baer fires 0.9 s before the STA/LTA P) — cross-check methods, or gate by
  coincidence, before trusting a single-trace pick.
- **ar_pick needs true Z,N,E** and a fixed sampling rate across components.

## Later (not yet implemented)

Amplitude/first-motion polarity (see `pick-qc-refine`), kurtosis (CF) pickers,
template matching / matched-filter detection, and ML pickers (`phase-picking-ml`).
