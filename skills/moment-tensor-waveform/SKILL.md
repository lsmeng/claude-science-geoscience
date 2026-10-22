---
name: moment-tensor-waveform
description: >-
  Regional moment-tensor (focal mechanism) inversion from broadband waveforms, using Syngine
  Green's functions. Use whenever the user wants a focal mechanism, moment tensor, Mw or
  nodal planes of a regional earthquake (about Mw 4.5 to 7, 50 to 400 km), or wants to
  check a catalog mechanism against waveforms. Downloads the data, builds synthetics for the
  six elementary moment tensors with the EarthScope Syngine service, aligns each station
  (the 1-D Earth is a few seconds off), solves the linear problem, and reports Mw, both
  nodal planes, CLVD fraction, variance reduction per station and, if given, the Kagan angle
  to a reference mechanism.
license: MIT
category: geoscience
---

# moment-tensor-waveform

`scripts/mt_invert.py` does the whole chain for one event. Needs ObsPy, NumPy and
matplotlib (for the figure); network access to an FDSN data center (default SCEDC) and to
`service.earthscope.org` (Syngine).

```bash
python scripts/mt_invert.py --lat 35.76 --lon -117.575 --depth 6.95 \
    --time 2019-07-05T11:07:53.040 --net CI --minr 60 --maxr 250 --nsta 12 \
    --band 0.03 0.07 --ref 312,74,201 --out ./mt_out
```

Output: `mt_result.json` (Mw, planes `[strike, dip, rake]` x 2, CLVD fraction, variance
reduction, per-station VR and time shift, stations used, stations dropped and why,
Kagan angle) and `mt_fit.png` (observed black, synthetic red, beachball).

## Method (and why)

1. **Stations.** Azimuth-balanced round-robin over 45 degree sectors, closest first.
2. **Data.** Response removed to velocity, rotated to ZNE, band-passed (default 0.03 to
   0.07 Hz), resampled to 2 Hz. Stations with signal-to-noise below 3 against the pre-event
   window are skipped.
3. **Green's functions.** Six Syngine requests per station, one per elementary tensor
   (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp), `ak135f_2s`, velocity, same filter. Syngine is linear in
   the tensor, so any mechanism is a linear combination (checked: reconstructing a double
   couple from the six gives correlation 1.000 with Syngine's own).
4. **Inversion.** Deviatoric (trace-free) least squares. The station weights come from the
   synthetics, **not** from the data: weighting by observed amplitude gives a noise-only
   station the largest weight and collapses the solution to nearly zero (this happened
   during development).
5. **Time shifts.** Each station gets its own shift, first from a mechanism-independent
   energy-envelope match, then refined by cross-correlation (up to +/- 12 s). Regional
   shifts of 3 to 7 s are normal because the 1-D model is wrong for the local crust.
6. **Pruning.** After the first solution, stations with variance reduction below 0.3 are
   dropped (keeping at least six) and the inversion is repeated. Dropped stations are
   reported.

## Verified (2019 Ridgecrest aftershocks, ComCat moment tensors as reference)

| Event | Stations | Mw (this / ComCat) | Variance reduction | Nodal planes (this) | Kagan angle |
|---|---|---|---|---|---|
| 2019-07-05 Mw 5.37 (ci38450263) | 8 of 12 | 5.38 / 5.37 | 0.89 | [313,75,-156] [216,67,-17] | 2.9 deg |
| 2019-07-12 Mw 4.9 (ci38548295) | 11 | 4.82 / 4.9 | 0.83 | [54,88,-6] [144,84,-178] | 10.3 deg |

Not usable: 2019-07-06 Mw 5.5 inside the active sequence. Every station was rejected because the
pre-event noise window was full of aftershock coda.

## Limits (say these out loud)

- **Point source in a 1-D Earth.** This is a regional long-period mechanism, not a 3-D
  waveform inversion. Expect 10 degree level agreement, not better, for Mw 4.5 to 5.5.
- **Fixed location and depth.** The depth is an input. Try two or three depths and compare
  the variance reduction; the skill does not search depth itself.
- **Deviatoric, not full.** The CLVD fraction can be large for a poor station set (0.34 with
  all 11 stations including bad ones, 0.15 after pruning in the Mw 5.37 test). Treat a large
  CLVD as a warning, not a result.
- **Events inside active sequences** fail the signal-to-noise screen.
- **Short periods are not modelled.** The synthetics are limited to the database's shortest
  period (2 s for `ak135f_2s`); the default band is much longer.
- Syngine is a public service: the skill makes 6 requests per station with a short pause;
  use `--nsta` modestly.
