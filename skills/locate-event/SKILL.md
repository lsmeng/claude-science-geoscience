---
name: locate-event
description: >-
  Locate a single earthquake from P and S picks with a transparent grid search in a 1-D
  iasp91 Earth, with outlier rejection and a bootstrap error estimate. Use when the user has
  P and S picks (for example from phase-picking-ml) at several stations and wants a
  hypocentre, origin time, residuals and an uncertainty, or wants to check a catalog
  location. In auto mode it downloads waveforms around an approximate origin time, runs
  PhaseNet, and takes the strongest P and S per station. Pure Python, no Fortran locator.
license: MIT
category: geoscience
---

# locate-event

`scripts/locate_event.py`. Needs ObsPy and NumPy (TauP is part of ObsPy), matplotlib for the
figure, and for `--auto` also seisbench and torch (same as `phase-picking-ml`).

```bash
# from a picks file written by phase-picking-ml
python scripts/locate_event.py --picks picks.csv --time 2019-07-12T13:11:38 --dc SCEDC --out ./loc_out

# or let it pick (approximate time and place only choose the stations)
python scripts/locate_event.py --auto --lat 35.64 --lon -117.59 --time 2019-07-12T13:11:38 \
    --net CI --minr 25 --maxr 120 --nsta 15 --out ./loc_out \
    --ref 35.6368,-117.5862,9.48,2019-07-12T13:11:37.990
```

Output: `location.json` (lat, lon, depth, origin time, RMS, picks and stations used, azimuthal
gap, horizontal and depth uncertainty, every residual with a `used` flag, bootstrap cloud,
comparison with the reference) and `location.png` (map with bootstrap cloud, residuals).

## Method

1. iasp91 P and S travel times on a (distance, depth) grid from TauP, bilinear lookup.
2. Coarse grid over latitude, longitude and depth; for every trial point the origin time is
   the weighted mean residual and the misfit is the weighted RMS. Two finer grids follow.
3. Outliers (more than 3 MAD, at least 0.5 s) are removed and the location repeated.
4. Uncertainty: bootstrap over stations (100 resamples).

## Verified (2019-07-12 Mw 4.9 Ridgecrest aftershock, ComCat ci38548295)

15 stations at 27 to 75 km, 29 picks, RMS 0.22 s. Horizontal position 0.99 km from ComCat
(bootstrap 0.6 km), origin time 0.73 s earlier than ComCat, azimuthal gap 113 degrees.
**Depth 21.7 km against ComCat's 9.5 km** (see limits).

## Limits (say these out loud)

- **Depth is poorly resolved when the nearest station is far.** Here the nearest station is
  27 km away, about three focal depths, so depth and origin time trade off. The bootstrap
  gives 3.7 km, which understates the real error (the offset is 12 km): the bootstrap only
  sees pick noise, not the model error.
- **1-D iasp91 is not the local crust.** All picks arrive earlier than iasp91 predicts,
  which pushes the solution deeper and the origin time earlier. A regional 1-D model, or
  NonLinLoc or HYPOINVERSE with station corrections, would be better.
- **Auto mode assumes one event in the window.** Overlapping events need an associator.
- Needs P and S at four or more stations (at least 8 picks).
