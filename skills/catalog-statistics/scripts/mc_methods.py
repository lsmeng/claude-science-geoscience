#!/usr/bin/env python3
"""
mc_methods.py — magnitude-of-completeness (Mc) estimators for an earthquake catalog.

An ADDITION to catalog-statistics (the base skill ships MAXC+0.2 in
catalog_stats.py). This module gathers the three standard Mc estimators so a
result can be cross-checked — a single Mc is fragile, and agreement between
independent methods is the reviewer-agent's sanity check.

Methods (all pure numpy):

  1. MAXC  — maximum curvature (Wiemer & Wyss 2000). Mc = magnitude bin with the
             most events (peak of the incremental FMD). Fast but biased low; the
             usual empirical +0.2 correction (Woessner & Wiemer 2005) is offered.

  2. MBS   — b-value stability (Cao & Gao 2002; Woessner & Wiemer 2005). Sweep the
             cutoff Mco upward; b(Mco) is unstable while the sample is incomplete
             and flattens once Mc is reached. Mc = lowest Mco where the b-value
             averaged over the next 0.5 mag agrees with b(Mco) to within its
             Shi-Bolt uncertainty δb.

  3. GFT   — goodness-of-fit test (Wiemer & Wyss 2000). For each Mco fit a GR model
             (Aki-Utsu MLE) and measure the % of the observed cumulative FMD the
             model reproduces, R = 100·(1 − Σ|obs−pred| / Σobs). Mc = lowest Mco
             with R ≥ 90% (95% first if attainable). Falls back to MAXC if neither
             confidence level is reached.

Also provides `lilliefors_exponential` — a Lilliefors-style goodness-of-fit test of
the exponential (Gutenberg-Richter) hypothesis for magnitudes above a candidate Mc.
The KS distance is compared to the asymptotic Lilliefors critical values for the
exponential with an estimated rate (Lilliefors 1969); treat p as approximate.

House rules: MLE not least-squares for every b-value used internally; single
magnitude type only; magnitudes are assumed already binned to `dm`.

Usage:
    python mc_methods.py --catalog cat.json --dm 0.1
    python mc_methods.py --mags "1.0,1.1,1.2,..." --dm 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

LOG10E = math.log10(math.e)  # 0.4342944819...


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _clean(mags):
    m = np.asarray(mags, float)
    return m[np.isfinite(m)]


def fmd(mags, dm=0.1):
    """Incremental + cumulative frequency-magnitude distribution.

    Returns (centers, incremental, cumulative) with bin centers aligned to dm.
    cumulative[k] = N(M >= centers[k]).
    """
    m = _clean(mags)
    if m.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo = math.floor(m.min() / dm) * dm
    hi = math.ceil(m.max() / dm) * dm + dm
    edges = np.arange(lo - dm / 2, hi + dm, dm)
    centers = np.round(edges[:-1] + dm / 2, 4)
    inc, _ = np.histogram(m, bins=edges)
    cum = np.cumsum(inc[::-1])[::-1]
    return centers, inc, cum


def b_aki_utsu(mags, mc, dm=0.1):
    """Aki-Utsu MLE b-value on M >= Mc, with Shi-Bolt sigma. Returns (b, sigma_b, N)."""
    m = _clean(mags)
    m = m[m >= mc - dm / 2 + 1e-9]
    n = int(m.size)
    if n < 2:
        return math.nan, math.nan, n
    mean_m = float(m.mean())
    denom = mean_m - (mc - dm / 2)
    if denom <= 0:
        return math.nan, math.nan, n
    b = LOG10E / denom
    var = float(np.sum((m - mean_m) ** 2) / (n * (n - 1)))
    sigma_b = 2.30 * b * b * math.sqrt(var)
    return b, sigma_b, n


# --------------------------------------------------------------------------- #
# 1. maximum curvature
# --------------------------------------------------------------------------- #
def mc_maxc(mags, dm=0.1, correction=0.2):
    """Mc by maximum curvature. `correction` is the empirical add-on (0.2 default).

    Returns dict {mc, mc_raw, method}.
    """
    centers, inc, _ = fmd(mags, dm)
    if centers.size == 0:
        return {"mc": None, "mc_raw": None, "method": "MAXC"}
    mc_raw = float(centers[int(np.argmax(inc))])
    return {"mc": round(mc_raw + correction, 4), "mc_raw": round(mc_raw, 4),
            "method": f"MAXC+{correction}" if correction else "MAXC"}


# --------------------------------------------------------------------------- #
# 2. b-value stability (MBS)
# --------------------------------------------------------------------------- #
def mc_mbs(mags, dm=0.1, avg_window=0.5):
    """Mc by b-value stability (Cao & Gao 2002 / Woessner & Wiemer 2005).

    For each candidate cutoff Mco, compare b(Mco) with the average of b over the
    next `avg_window` magnitude units; Mc is the first Mco where the two agree to
    within the Shi-Bolt uncertainty δb(Mco).

    Returns dict {mc, method, b_at_mc, sweep:[{mco,b,b_avg,db}]}.
    """
    centers, inc, _ = fmd(mags, dm)
    if centers.size == 0:
        return {"mc": None, "method": "MBS"}
    k = max(1, int(round(avg_window / dm)))
    bs, sig = [], []
    for mco in centers:
        b, s, n = b_aki_utsu(mags, float(mco), dm)
        bs.append(b)
        sig.append(s)
    bs = np.array(bs)
    sig = np.array(sig)

    sweep = []
    mc = None
    for i, mco in enumerate(centers):
        if not np.isfinite(bs[i]) or i + k >= len(centers):
            continue
        window = bs[i:i + k + 1]
        if not np.all(np.isfinite(window)):
            continue
        b_avg = float(np.mean(window))
        db = float(sig[i])
        sweep.append({"mco": round(float(mco), 4), "b": round(float(bs[i]), 4),
                      "b_avg": round(b_avg, 4), "db": round(db, 4)})
        if mc is None and np.isfinite(db) and abs(b_avg - bs[i]) <= db:
            mc = float(mco)
    return {"mc": round(mc, 4) if mc is not None else None,
            "method": "MBS (b-value stability)",
            "b_at_mc": round(float(bs[list(centers).index(mc)]), 4) if mc is not None else None,
            "sweep": sweep}


# --------------------------------------------------------------------------- #
# 3. goodness-of-fit test (GFT)
# --------------------------------------------------------------------------- #
def mc_gft(mags, dm=0.1, levels=(95.0, 90.0)):
    """Mc by the goodness-of-fit test (Wiemer & Wyss 2000).

    For each candidate Mco: fit GR (Aki-Utsu b, a), predict the cumulative FMD, and
    compute R = 100·(1 − Σ|obs−pred|/Σobs) over bins >= Mco. Mc is the lowest Mco
    reaching the first confidence level in `levels` (95% then 90%). If neither is
    reached, fall back to MAXC and flag it.

    Returns dict {mc, confidence, method, R_at_mc, fallback}.
    """
    centers, inc, cum = fmd(mags, dm)
    if centers.size == 0:
        return {"mc": None, "method": "GFT"}
    R = np.full(centers.size, np.nan)
    for i, mco in enumerate(centers):
        b, _, n = b_aki_utsu(mags, float(mco), dm)
        if not np.isfinite(b) or n < 2:
            continue
        a = math.log10(n) + b * float(mco)          # so pred(Mco) = n
        sub = centers[i:]
        obs = cum[i:]
        pred = 10.0 ** (a - b * sub)
        denom = obs.sum()
        if denom <= 0:
            continue
        R[i] = 100.0 * (1.0 - np.abs(obs - pred).sum() / denom)

    for lvl in levels:
        idx = np.where(R >= lvl)[0]
        if idx.size:
            i = int(idx[0])
            return {"mc": round(float(centers[i]), 4), "confidence": lvl,
                    "R_at_mc": round(float(R[i]), 2),
                    "method": "GFT (goodness-of-fit)", "fallback": False}
    # neither confidence reached → fall back to MAXC
    fb = mc_maxc(mags, dm)
    return {"mc": fb["mc"], "confidence": None,
            "R_at_mc": round(float(np.nanmax(R)), 2) if np.any(np.isfinite(R)) else None,
            "method": "GFT (goodness-of-fit) — fell back to MAXC", "fallback": True}


# --------------------------------------------------------------------------- #
# Lilliefors exponentiality test (GR hypothesis check above Mc)
# --------------------------------------------------------------------------- #
def lilliefors_exponential(values):
    """Lilliefors-style KS test of exponentiality with an estimated rate.

    `values` should be (M - Mc) for events above a candidate Mc; under GR these are
    exponential with rate λ = b·ln(10). We estimate λ = 1/mean(values), form the KS
    distance between the empirical CDF and the fitted exponential CDF, and compare to
    the asymptotic Lilliefors critical values for the exponential (Lilliefors 1969).

    Returns dict {D, n, lambda, crit_05, crit_01, reject_05, note}. p is NOT computed
    exactly — the critical values are asymptotic; treat as an approximate check.
    """
    x = _clean(values)
    x = x[x >= 0]
    n = int(x.size)
    if n < 4:
        return {"D": None, "n": n, "note": "too few values"}
    lam = 1.0 / float(x.mean())
    xs = np.sort(x)
    cdf = 1.0 - np.exp(-lam * xs)                 # fitted exponential CDF
    ecdf_hi = (np.arange(1, n + 1)) / n
    ecdf_lo = (np.arange(0, n)) / n
    D = float(max(np.max(ecdf_hi - cdf), np.max(cdf - ecdf_lo)))
    # asymptotic Lilliefors critical values for the exponential (large n):
    #   D_crit(alpha) ≈ c(alpha) / sqrt(n)
    crit_05 = 1.094 / math.sqrt(n)
    crit_01 = 1.308 / math.sqrt(n)
    return {"D": round(D, 4), "n": n, "lambda": round(lam, 4),
            "crit_05": round(crit_05, 4), "crit_01": round(crit_01, 4),
            "reject_05": bool(D > crit_05),
            "note": "asymptotic Lilliefors critical values; approximate"}


# --------------------------------------------------------------------------- #
# convenience: all methods at once
# --------------------------------------------------------------------------- #
def estimate_mc_all(mags, dm=0.1):
    """Run MAXC, MBS and GFT and report agreement (spread in bins)."""
    maxc = mc_maxc(mags, dm)
    mbs = mc_mbs(mags, dm)
    gft = mc_gft(mags, dm)
    # compare RAW estimates (MAXC without the +0.2 offset) so the comparison is fair
    vals = [maxc["mc_raw"], mbs["mc"], gft["mc"]]
    vals = [v for v in vals if v is not None]
    spread_bins = round((max(vals) - min(vals)) / dm, 2) if len(vals) >= 2 else None
    return {
        "dm": dm,
        "maxc": maxc,
        "mbs": {k: v for k, v in mbs.items() if k != "sweep"},
        "gft": gft,
        "raw_estimates": {"maxc_raw": maxc["mc_raw"], "mbs": mbs["mc"], "gft": gft["mc"]},
        "spread_bins": spread_bins,
        "agree_within_bin": (spread_bins is not None and spread_bins <= 1.0),
    }


def _load_mags(args):
    if args.mags:
        return [float(x) for x in args.mags.split(",") if x.strip()]
    if not args.catalog:
        raise SystemExit("provide --catalog or --mags")
    if args.catalog.endswith(".json"):
        obj = json.load(open(args.catalog))
        events = obj.get("events", obj if isinstance(obj, list) else [])
        return [e["mag"] for e in events if e.get("mag") is not None]
    import pandas as pd
    df = pd.read_csv(args.catalog)
    for col in ("mag", "magnitude", "Magnitude", "Mag", "ml", "Mw"):
        if col in df.columns:
            return df[col].dropna().tolist()
    raise SystemExit(f"no magnitude column in {args.catalog}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", help="CSV (mag column) or JSON connector result")
    ap.add_argument("--mags", help="comma-separated magnitudes (quick check)")
    ap.add_argument("--dm", type=float, default=0.1, help="magnitude bin width (default 0.1)")
    args = ap.parse_args(argv)
    mags = _load_mags(args)
    print(json.dumps(estimate_mc_all(mags, dm=args.dm), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
