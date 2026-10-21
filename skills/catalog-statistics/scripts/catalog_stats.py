#!/usr/bin/env python3
"""
catalog_stats.py — frequency-magnitude statistics for an earthquake catalog.

Self-contained (numpy + pandas only, NO seismostats dependency) implementation of
the headline monitoring metrics:

  - Magnitude of completeness Mc   (maximum-curvature, MAXC, with the usual +0.2)
  - b-value                        (Aki-Utsu maximum-likelihood — NOT least squares)
  - b-value uncertainty            (Shi & Bolt 1982)
  - a-value                        (from the fit)
  - the frequency-magnitude distribution (incremental + cumulative)

Why MLE and not a least-squares slope: fitting a line to the cumulative FMD is
biased and the reviewer skill flags it. Aki-Utsu MLE is the correct estimator:

    b   = log10(e) / ( mean(M | M>=Mc) - (Mc - dM/2) )
    σ_b = 2.30 * b^2 * sqrt( Σ (M - mean)^2 / (N (N-1)) )        (Shi & Bolt)

Input: a catalog with magnitudes. Accepts
  --catalog file.csv   (a column named mag/magnitude/Magnitude)
  --catalog file.json  (a comcat/fdsns connector search result: {"events":[{"mag":..}]})
  --mags "1.2,3.4,..." (comma list, for quick checks)

Usage:
  python catalog_stats.py --catalog cat.json --dm 0.1
  python catalog_stats.py --catalog cat.csv --mc 1.8      # force Mc instead of MAXC
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

LOG10E = math.log10(math.e)  # 0.4342944819...


def frequency_magnitude_distribution(mags, dm=0.1):
    """Return (bin_centers, incremental_counts, cumulative_counts) for the FMD."""
    mags = np.asarray(mags, float)
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return np.array([]), np.array([]), np.array([])
    # bin edges aligned to dm
    lo = math.floor(mags.min() / dm) * dm
    hi = math.ceil(mags.max() / dm) * dm + dm
    edges = np.arange(lo - dm / 2, hi + dm, dm)
    centers = np.round(edges[:-1] + dm / 2, 4)
    incremental, _ = np.histogram(mags, bins=edges)
    cumulative = np.cumsum(incremental[::-1])[::-1]  # N(>=M)
    return centers, incremental, cumulative


def mc_maxc(mags, dm=0.1, correction=0.2):
    """Magnitude of completeness by maximum curvature (peak of the incremental FMD).

    correction: the empirical +0.2 (Woessner & Wiemer 2005) added to the raw MAXC.
    """
    centers, incremental, _ = frequency_magnitude_distribution(mags, dm)
    if centers.size == 0:
        return None
    mc_raw = float(centers[int(np.argmax(incremental))])
    return round(mc_raw + correction, 4)


def b_value_aki_utsu(mags, mc, dm=0.1):
    """Aki-Utsu MLE b-value (+ Shi-Bolt σ, a-value) on events with M >= Mc.

    Returns a dict with b, b_uncertainty, a, n, mean_mag, mc, dm.
    """
    mags = np.asarray(mags, float)
    m = mags[np.isfinite(mags) & (mags >= mc - dm / 2 + 1e-9)]
    n = int(m.size)
    if n < 2:
        return {"error": "too few events >= Mc", "n": n, "mc": mc}
    mean_m = float(m.mean())
    denom = mean_m - (mc - dm / 2)
    if denom <= 0:
        return {"error": "non-positive (mean - (Mc - dM/2)); check Mc/dM", "n": n,
                "mean_mag": mean_m, "mc": mc}
    b = LOG10E / denom
    # Shi & Bolt (1982) uncertainty
    var = float(np.sum((m - mean_m) ** 2) / (n * (n - 1)))
    sigma_b = 2.30 * b * b * math.sqrt(var)
    # a-value: log10 N(>=Mc) + b*Mc
    a = math.log10(n) + b * mc
    return {
        "b": round(b, 4),
        "b_uncertainty": round(sigma_b, 4),
        "a": round(a, 4),
        "n": n,
        "mean_mag": round(mean_m, 4),
        "mc": mc,
        "dm": dm,
        "method": "Aki-Utsu MLE; Shi-Bolt sigma",
        "n_warning": None if n >= 50 else f"N={n} < 50: b-value unstable (reviewer WARN)",
    }


def analyze(mags, dm=0.1, mc=None):
    mags = np.asarray(mags, float)
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return {"error": "no finite magnitudes"}
    if mc is None:
        mc = mc_maxc(mags, dm)
    res = {"n_total": int(mags.size), "mag_min": round(float(mags.min()), 2),
           "mag_max": round(float(mags.max()), 2),
           "mc": mc, "mc_method": "MAXC+0.2" if mc == mc_maxc(mags, dm) else "user-supplied"}
    res.update(b_value_aki_utsu(mags, mc, dm))
    centers, inc, cum = frequency_magnitude_distribution(mags, dm)
    res["fmd"] = [{"mag": float(c), "incremental": int(i), "cumulative": int(cu)}
                  for c, i, cu in zip(centers, inc, cum)]
    return res


def _load_mags(args):
    if args.mags:
        return [float(x) for x in args.mags.split(",") if x.strip()]
    if not args.catalog:
        raise SystemExit("provide --catalog or --mags")
    if args.catalog.endswith(".json"):
        obj = json.load(open(args.catalog))
        events = obj.get("events", obj if isinstance(obj, list) else [])
        return [e["mag"] for e in events if e.get("mag") is not None]
    # CSV
    df = pd.read_csv(args.catalog)
    for col in ("mag", "magnitude", "Magnitude", "Mag", "ml", "Mw"):
        if col in df.columns:
            return df[col].dropna().tolist()
    raise SystemExit(f"no magnitude column in {args.catalog} (have {list(df.columns)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", help="CSV (mag column) or JSON (connector search result)")
    ap.add_argument("--mags", help="comma-separated magnitudes (quick check)")
    ap.add_argument("--dm", type=float, default=0.1, help="magnitude bin width (default 0.1)")
    ap.add_argument("--mc", type=float, default=None, help="force Mc (default: MAXC+0.2)")
    ap.add_argument("--no-fmd", action="store_true", help="omit the FMD table from output")
    args = ap.parse_args(argv)

    mags = _load_mags(args)
    res = analyze(mags, dm=args.dm, mc=args.mc)
    if args.no_fmd:
        res.pop("fmd", None)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
