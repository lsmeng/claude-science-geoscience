#!/usr/bin/env python3
"""
b_positive.py — the b-positive b-value estimator (Van der Elst, 2021).

An ADDITION to catalog-statistics. b-positive is robust to time-varying magnitude
of completeness (e.g. short-term aftershock incompleteness, STAI, right after a
mainshock) — the situation where the ordinary Aki-Utsu b-value is biased low.

Idea (Van der Elst, JGR 2021, doi:10.1029/2020JB021027): take the catalog in TIME
order and form the differences between successive magnitudes,

    δ_i = m_{i+1} − m_i .

The POSITIVE differences (δ_i above a small threshold δMc) are distributed
exponentially with the SAME slope b as the underlying Gutenberg-Richter
distribution, and — crucially — this holds even when the completeness magnitude
drifts in time, because a difference only requires that BOTH events were detected,
not that any absolute threshold was met. The b-value then follows from the
Aki-Utsu MLE applied to the positive differences:

    b+ = log10(e) / ( mean(δ | δ ≥ δMc) − (δMc − Δm/2) )

with Δm the magnitude bin width (binning correction), δMc the difference threshold
(default one bin, Δm). Shi-Bolt gives the uncertainty.

House rules: MLE (not least-squares); reports σ and N; UTC only for any time input.

Usage:
    # magnitudes already in time order
    python b_positive.py --mags "1.2,0.8,1.5,..." --dm 0.1
    # from a connector catalog result (sorted by 'time' before differencing)
    python b_positive.py --catalog cat.json --dm 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

LOG10E = math.log10(math.e)


def b_positive(mags_time_ordered, dm=0.1, dmc=None):
    """b-positive estimator (Van der Elst 2021).

    Parameters
    ----------
    mags_time_ordered : sequence of magnitudes in TIME order (not sorted by size!).
    dm   : magnitude bin width Δm (default 0.1).
    dmc  : threshold on positive differences δMc (default = dm, i.e. keep δ >= dm).

    Returns dict {b, b_uncertainty, n_pos, n_total, dmc, dm, mean_diff, method}.
    """
    if dmc is None:
        dmc = dm
    m = np.asarray(mags_time_ordered, float)
    m = m[np.isfinite(m)]
    if m.size < 3:
        return {"error": "need >= 3 magnitudes in time order", "n_total": int(m.size)}
    diffs = np.diff(m)                                   # successive differences
    pos = diffs[diffs >= dmc - dm / 2 + 1e-9]            # positive differences >= δMc
    n = int(pos.size)
    if n < 2:
        return {"error": "too few positive differences", "n_pos": n,
                "n_total": int(m.size)}
    mean_d = float(pos.mean())
    denom = mean_d - (dmc - dm / 2)
    if denom <= 0:
        return {"error": "non-positive denominator; check dm/dmc", "n_pos": n,
                "mean_diff": mean_d}
    b = LOG10E / denom
    var = float(np.sum((pos - mean_d) ** 2) / (n * (n - 1)))
    sigma_b = 2.30 * b * b * math.sqrt(var)              # Shi & Bolt (1982)
    return {
        "b": round(b, 4),
        "b_uncertainty": round(sigma_b, 4),
        "n_pos": n,
        "n_total": int(m.size),
        "dmc": dmc,
        "dm": dm,
        "mean_diff": round(mean_d, 4),
        "method": "b-positive (Van der Elst 2021), Aki-Utsu MLE on positive diffs; "
                  "Shi-Bolt sigma",
        "note": "robust to time-varying Mc; feed magnitudes in TIME order",
    }


def _load_time_ordered_mags(args):
    if args.mags:
        # assumed already in time order
        return [float(x) for x in args.mags.split(",") if x.strip()]
    if not args.catalog:
        raise SystemExit("provide --catalog or --mags")
    obj = json.load(open(args.catalog))
    events = obj.get("events", obj if isinstance(obj, list) else [])
    # sort by time (ISO-8601 UTC strings sort lexicographically) so diffs are temporal
    ev = [e for e in events if e.get("mag") is not None and e.get("time")]
    ev.sort(key=lambda e: e["time"])
    return [e["mag"] for e in ev]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", help="JSON connector result (sorted by time here)")
    ap.add_argument("--mags", help="comma-separated magnitudes IN TIME ORDER")
    ap.add_argument("--dm", type=float, default=0.1, help="magnitude bin width Δm")
    ap.add_argument("--dmc", type=float, default=None,
                    help="positive-difference threshold δMc (default = dm)")
    args = ap.parse_args(argv)
    mags = _load_time_ordered_mags(args)
    print(json.dumps(b_positive(mags, dm=args.dm, dmc=args.dmc), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
