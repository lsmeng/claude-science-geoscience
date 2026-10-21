#!/usr/bin/env python3
"""
sta_lta_pick.py — classic (non-ML) seismic phase detection & picking with ObsPy.

Wraps the standard ObsPy characteristic-function pickers so an agent can turn a
miniSEED trace (e.g. from the fdsnws-mcp connector) into a list of P/S onset picks:

  - recursive_sta_lta / classic_sta_lta   → trigger_onset picks (single trace)
  - coincidence_trigger                    → network-coincidence detections (stream)
  - pk_baer                                (Baer & Kradolfer 1987 P picker)
  - ar_pick                                (Akazawa 2004 AR P & S, needs Z,N,E)
  - AR-AIC refinement (Maeda 1985 AIC)     → sub-sample onset refinement of a coarse pick

Everything is ObsPy + numpy — no ML, no torch. All times are ISO-8601 UTC.

Design:
  - A "pick" is {network, station, location, channel, phase, time (ISO UTC), sample,
    cf_value, method}. Times are absolute (trace starttime + sample/df), never relative.
  - `pick_stream()` is the high-level entry point (used by the phase-picking-ml
    fallback): give it an ObsPy Stream, get picks back.

Usage:
  python sta_lta_pick.py --mseed trace.mseed --method recursive --sta 0.5 --lta 5 \
      --thr-on 3 --thr-off 1
  python sta_lta_pick.py --mseed trace.mseed --method baer
  python sta_lta_pick.py --mseed 3comp.mseed --method ar        # needs Z,N,E
"""

from __future__ import annotations

import argparse
import json
import math
import sys


# ---------------------------------------------------------------------------
# Characteristic-function pickers (single trace)
# ---------------------------------------------------------------------------
def sta_lta_picks(data, df, sta=0.5, lta=5.0, thr_on=3.0, thr_off=1.0,
                  recursive=True):
    """STA/LTA detection on one trace. Returns list of (onset_sample, cf_value).

    sta, lta are WINDOW LENGTHS IN SECONDS (converted to samples here).
    recursive=True uses recursive_sta_lta (streaming form), else classic_sta_lta.
    """
    import numpy as np
    from obspy.signal.trigger import (recursive_sta_lta, classic_sta_lta,
                                       trigger_onset)
    data = np.asarray(data, float)
    nsta, nlta = max(1, int(round(sta * df))), max(2, int(round(lta * df)))
    if data.size < nlta + 2:
        return []
    fn = recursive_sta_lta if recursive else classic_sta_lta
    cft = fn(data, nsta, nlta)
    onsets = trigger_onset(cft, thr_on, thr_off)
    return [(int(on), float(cft[int(on)])) for on, off in onsets]


def baer_pick(data, df, tdownmax=20, tupevent=60, thr1=7.0, thr2=12.0,
              preset_len=100, p_dur=100):
    """Baer & Kradolfer (1987) P picker. Returns (onset_sample, quality_str) or None."""
    import numpy as np
    from obspy.signal.trigger import pk_baer
    data = np.asarray(data, float)
    p_sample, quality = pk_baer(data, df, int(tdownmax), int(tupevent),
                                thr1, thr2, int(preset_len), int(p_dur))
    if p_sample <= 0:
        return None
    return int(p_sample), str(quality)


def ar_pick_ze(z, n, e, df, **kw):
    """Akazawa (2004) AR P & S picker on 3-component data (Z, N, E arrays).

    Returns (p_sample, s_sample) in samples relative to trace start (either may be
    None if not picked). kw overrides the ar_pick tuning constants.
    """
    import numpy as np
    from obspy.signal.trigger import ar_pick
    params = dict(f1=1.0, f2=20.0, lta_p=1.0, sta_p=0.1, lta_s=4.0, sta_s=1.0,
                  m_p=2, m_s=8, l_p=0.1, l_s=0.2, s_pick=True)
    params.update(kw)
    p_t, s_t = ar_pick(np.asarray(z, float), np.asarray(n, float),
                       np.asarray(e, float), df,
                       params["f1"], params["f2"], params["lta_p"], params["sta_p"],
                       params["lta_s"], params["sta_s"], params["m_p"], params["m_s"],
                       params["l_p"], params["l_s"], s_pick=params["s_pick"])
    p_s = int(round(p_t * df)) if p_t and p_t > 0 else None
    s_s = int(round(s_t * df)) if s_t and s_t > 0 else None
    return p_s, s_s


# ---------------------------------------------------------------------------
# AR-AIC sub-sample refinement (Maeda 1985)
# ---------------------------------------------------------------------------
def aic_curve(x):
    """Maeda (1985) AIC function of a 1-D array; onset = global minimum index.

    AIC(k) = k*log(var(x[:k+1])) + (n-k-1)*log(var(x[k+1:]))
    """
    import numpy as np
    x = np.asarray(x, float)
    n = x.size
    aic = np.full(n, np.inf)
    for k in range(1, n - 1):
        v1 = np.var(x[:k + 1])
        v2 = np.var(x[k + 1:])
        if v1 <= 0 or v2 <= 0:
            continue
        aic[k] = k * math.log(v1) + (n - k - 1) * math.log(v2)
    return aic


def aic_refine(data, df, coarse_sample, win_s=1.0):
    """Refine a coarse onset (sample index) with AIC inside ±win_s seconds.

    Returns (refined_sample, refined_time_offset_s). The AIC minimum inside the
    window is the onset; this de-biases the STA/LTA lag and sharpens the pick.
    """
    import numpy as np
    data = np.asarray(data, float)
    half = max(2, int(round(win_s * df)))
    a = max(0, coarse_sample - half)
    b = min(data.size, coarse_sample + half)
    seg = data[a:b]
    if seg.size < 5:
        return coarse_sample, coarse_sample / df
    aic = aic_curve(seg)
    k = int(np.argmin(aic))
    refined = a + k
    return refined, refined / df


# ---------------------------------------------------------------------------
# Stream-level convenience (high-level entry point)
# ---------------------------------------------------------------------------
def _trace_id(tr):
    s = tr.stats
    return s.network, s.station, s.location, s.channel


def pick_stream(stream, method="recursive", sta=0.5, lta=5.0, thr_on=3.0,
                thr_off=1.0, refine=True, refine_win_s=1.0, detrend=True):
    """Pick every trace in an ObsPy Stream. Returns a list of pick dicts.

    method: 'recursive' | 'classic' | 'baer'. Each STA/LTA onset is (optionally)
    AIC-refined. Times are absolute ISO-8601 UTC.
    """
    picks = []
    for tr in stream:
        tr = tr.copy()
        if detrend:
            tr.detrend("demean")
        df = float(tr.stats.sampling_rate)
        net, stn, loc, cha = _trace_id(tr)
        phase = "P" if cha and cha[-1] in ("Z", "z") else "?"
        if method == "baer":
            r = baer_pick(tr.data, df)
            onsets = [(r[0], None)] if r else []
        else:
            onsets = sta_lta_picks(tr.data, df, sta, lta, thr_on, thr_off,
                                   recursive=(method != "classic"))
        for onset_sample, cf in onsets:
            samp = onset_sample
            if refine:
                samp, _ = aic_refine(tr.data, df, onset_sample, refine_win_s)
            t = tr.stats.starttime + samp / df
            picks.append({
                "network": net, "station": stn, "location": loc, "channel": cha,
                "phase": phase,
                "time": str(t),
                "sample": int(samp),
                "cf_value": (round(cf, 3) if cf is not None else None),
                "method": f"{method}+AIC" if refine else method,
            })
    return picks


def coincidence(stream, thr_on=3.0, thr_off=1.0, min_stations=2,
                sta=0.5, lta=5.0):
    """Network coincidence_trigger over a Stream. Returns detection dicts.

    Requires >= min_stations traces triggering within the coincidence window.
    """
    from obspy.signal.trigger import coincidence_trigger
    st = stream.copy()
    st.detrend("demean")
    nsta = max(1, int(round(sta * st[0].stats.sampling_rate)))
    nlta = max(2, int(round(lta * st[0].stats.sampling_rate)))
    trig = coincidence_trigger("recstalta", thr_on, thr_off, st, min_stations,
                               sta=nsta / st[0].stats.sampling_rate,
                               lta=nlta / st[0].stats.sampling_rate)
    out = []
    for d in trig:
        out.append({
            "time": str(d["time"]),
            "coincidence_sum": d.get("coincidence_sum"),
            "stations": d.get("stations"),
            "duration": d.get("duration"),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mseed", required=True, help="miniSEED file (one or more traces)")
    ap.add_argument("--method", default="recursive",
                    choices=["recursive", "classic", "baer", "ar", "coincidence"])
    ap.add_argument("--sta", type=float, default=0.5, help="STA window (s)")
    ap.add_argument("--lta", type=float, default=5.0, help="LTA window (s)")
    ap.add_argument("--thr-on", type=float, default=3.0, help="trigger on threshold")
    ap.add_argument("--thr-off", type=float, default=1.0, help="trigger off threshold")
    ap.add_argument("--min-stations", type=int, default=2, help="coincidence min stations")
    ap.add_argument("--no-refine", action="store_true", help="skip AIC refinement")
    args = ap.parse_args(argv)

    import obspy
    st = obspy.read(args.mseed)

    if args.method == "coincidence":
        res = {"detections": coincidence(st, args.thr_on, args.thr_off,
                                         args.min_stations, args.sta, args.lta)}
    elif args.method == "ar":
        # need Z,N,E from a single station
        comps = {tr.stats.channel[-1]: tr for tr in st}
        if not ({"Z", "N", "E"} <= set(comps)):
            res = {"error": "ar_pick needs Z,N,E components", "have": list(comps)}
        else:
            z, n, e = comps["Z"], comps["N"], comps["E"]
            df = float(z.stats.sampling_rate)
            p_s, s_s = ar_pick_ze(z.data, n.data, e.data, df)
            res = {"picks": [
                {"phase": ph, "sample": smp,
                 "time": (str(z.stats.starttime + smp / df) if smp is not None else None)}
                for ph, smp in (("P", p_s), ("S", s_s))]}
    else:
        picks = pick_stream(st, method=args.method, sta=args.sta, lta=args.lta,
                            thr_on=args.thr_on, thr_off=args.thr_off,
                            refine=not args.no_refine)
        res = {"n_picks": len(picks), "picks": picks}
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
