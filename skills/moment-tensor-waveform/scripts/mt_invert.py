#!/usr/bin/env python3
"""
mt_invert.py - regional moment-tensor inversion with Syngine Green's functions.

Given an event (location, depth, origin time) it
  1. picks broadband stations around the event from an FDSN data center (azimuth balanced),
  2. downloads and corrects the observed 3-component waveforms (response removed,
     band-passed, resampled),
  3. asks the EarthScope Syngine service for the six elementary moment-tensor
     synthetics at each station (1-D Earth model, AxiSEM/Instaseis),
  4. solves the linear least-squares problem for the deviatoric moment tensor, with
     iterative per-station time shifts (the 1-D model is wrong at regional distance),
  5. decomposes it (best double couple, CLVD, Mw) and reports variance reduction,
     both nodal planes and, if a reference mechanism is given, the Kagan angle.

Everything is a point source in a 1-D Earth: the answer is a *regional long-period
mechanism*, not a full 3-D waveform inversion. Use it for events of about Mw 4.5 to
7 at 50 to 400 km distance, in the 20 to 40 s period band.

    python mt_invert.py --lat 35.6368 --lon -117.5862 --depth 9.48 \
        --time 2019-07-12T13:11:37.990 --net CI --minr 50 --maxr 250 --nsta 12 \
        --band 0.03 0.07 --out ./mt_out [--ref 322,87,-173]

Network access: the data center (default SCEDC) and service.earthscope.org.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

SYNGINE = os.environ.get("SYNGINE_BASE", "https://service.earthscope.org/irisws/syngine/1")
BASIS = ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"]
M0_UNIT = 1.0e16  # N*m for each elementary tensor


# --------------------------------------------------------------------------- Syngine
def syngine_gf(src, rec, model, t0, t1, dt=None, tries=3):
    """Six elementary synthetics (velocity, ZNE) at one receiver -> dict comp -> (6, n) array."""
    from obspy import read
    out = {}
    for k, name in enumerate(BASIS):
        mt = [0.0] * 6
        mt[k] = M0_UNIT
        q = dict(model=model, components="ZNE", units="velocity", format="miniseed",
                 sourcelatitude=src["lat"], sourcelongitude=src["lon"],
                 sourcedepthinmeters=int(round(src["depth_km"] * 1000)),
                 origintime=src["time"], starttime=t0, endtime=t1,
                 receiverlatitude=rec["lat"], receiverlongitude=rec["lon"],
                 sourcemomenttensor=",".join(f"{v:g}" for v in mt))
        if dt:
            q["dt"] = dt
        url = f"{SYNGINE}/query?" + urllib.parse.urlencode(q, safe=",:-")
        last = None
        for i in range(tries):
            time.sleep(0.4)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "claude-science-geoscience/0.1 (mt_invert)"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                break
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2 * (i + 1))
        else:
            raise RuntimeError(f"Syngine failed for {rec['code']} {name}: {last}")
        st = read(io.BytesIO(body), format="MSEED")
        for tr in st:
            out.setdefault(tr.stats.channel[-1], []).append(tr)
    return out


# --------------------------------------------------------------------------- stations
def pick_stations(client, net, lat, lon, minr, maxr, nsta, t0, chan="BH?"):
    from obspy.geodetics import gps2dist_azimuth
    from obspy.geodetics import kilometers2degrees as k2d
    inv = client.get_stations(network=net, channel=chan, latitude=lat, longitude=lon,
                              minradius=k2d(minr), maxradius=k2d(maxr),
                              starttime=t0, endtime=t0 + 600, level="station")
    cands = []
    for n in inv:
        for s in n:
            d, az, _ = gps2dist_azimuth(lat, lon, s.latitude, s.longitude)
            cands.append(dict(net=n.code, sta=s.code, lat=s.latitude, lon=s.longitude,
                              dist_km=d / 1000.0, az=az))
    # azimuth balance: round-robin over 45 degree sectors, closest first
    sectors = {}
    for c in sorted(cands, key=lambda c: c["dist_km"]):
        sectors.setdefault(int(c["az"] // 45), []).append(c)
    chosen = []
    while len(chosen) < nsta and any(sectors.values()):
        for k in sorted(sectors):
            if sectors[k] and len(chosen) < nsta:
                chosen.append(sectors[k].pop(0))
    return chosen


def load_observed(client, s, t0, tend, band, fs, chan="BH?"):
    """Download, remove response (velocity), rotate to ZNE, band-pass, resample."""
    from obspy import UTCDateTime
    ta, tb = UTCDateTime(t0) - 120, UTCDateTime(t0) + tend + 120
    st = client.get_waveforms(s["net"], s["sta"], "*", chan, ta, tb)
    inv = client.get_stations(network=s["net"], station=s["sta"], channel=chan,
                              starttime=ta, endtime=tb, level="response")
    st.merge(method=1, fill_value=0)
    st.detrend("linear"); st.taper(0.05)
    f1, f2 = band
    st.remove_response(inventory=inv, output="VEL", pre_filt=(f1 / 2, f1, f2, f2 * 1.5),
                       zero_mean=True, taper=True)
    st.rotate("->ZNE", inventory=inv)
    st.filter("bandpass", freqmin=f1, freqmax=f2, corners=4, zerophase=True)
    st.resample(fs)
    return st


# --------------------------------------------------------------------------- inversion
def _window(tr, t_start, t_end):
    from obspy import UTCDateTime
    return tr.slice(UTCDateTime(t_start), UTCDateTime(t_end)).data.astype(float)


def invert(obs, gfs, fs, t_start, t_end, n_iter=3, max_shift=12.0, deviatoric=True):
    """obs[sta][comp] -> trace ; gfs[sta][comp] -> list of 6 traces (already filtered)."""
    stas = sorted(obs)
    nmax = int(max_shift * fs)

    def smooth(x, n):
        k = np.ones(n) / n
        return np.convolve(x, k, mode="same")

    def envelope_shift(s):
        """Mechanism-independent first guess: align the observed energy envelope with the
        energy of all six elementary synthetics (the 1-D model is a few seconds off)."""
        n_sm = max(3, int(10 * fs))
        g_env = np.sqrt(sum(np.sum([_window(gfs[s][c][k], t_start, t_end) ** 2 for k in range(6)], axis=0)
                            for c in "ZNE"))
        g_env = smooth(g_env, n_sm)
        best, bestc = 0.0, -1e9
        for lag in range(-nmax, nmax + 1):
            sh = lag / fs
            o_env = np.sqrt(sum(_window(obs[s][c], t_start + sh, t_end + sh) ** 2 for c in "ZNE"))
            n = min(len(o_env), len(g_env))
            o = smooth(o_env[:n], n_sm); g = g_env[:n]
            cc = float(np.dot(o - o.mean(), g - g.mean()) /
                       (np.linalg.norm(o - o.mean()) * np.linalg.norm(g - g.mean()) + 1e-30))
            if cc > bestc:
                bestc, best = cc, sh
        return best

    shifts = {s: envelope_shift(s) for s in stas}
    # Station weights from the Green's functions (distance normalisation). Weighting by the
    # observed amplitude would give a noise-only station the largest weight.
    wsta = {}
    for s in stas:
        e = np.sqrt(np.mean([np.mean(_window(gfs[s][c][k], t_start, t_end) ** 2)
                             for c in "ZNE" for k in range(6)]))
        wsta[s] = 1.0 / (e + 1e-30)

    def stack(shifts):
        D, G, idx = [], [], []
        for s in stas:
            sh = shifts[s]
            for comp in "ZNE":
                d = _window(obs[s][comp], t_start + sh, t_end + sh)
                g = np.column_stack([_window(gfs[s][comp][k], t_start, t_end) for k in range(6)])
                n = min(len(d), len(g))
                d, g = d[:n], g[:n]
                w = wsta[s]  # one weight per station (from the synthetics, not the data)
                D.append(d * w); G.append(g * w); idx.append((s, comp, n))
        return D, G, idx

    for it in range(n_iter + 1):
        D, G, idx = stack(shifts)
        d = np.concatenate(D); A = np.vstack(G)
        if deviatoric:  # Mrr + Mtt + Mpp = 0  ->  Mpp = -(Mrr + Mtt)
            T = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [-1, -1, 0, 0, 0],
                          [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], float)
            x5, *_ = np.linalg.lstsq(A @ T, d, rcond=None)
            m = T @ x5
        else:
            m, *_ = np.linalg.lstsq(A, d, rcond=None)
        if it == n_iter:
            break
        # update shifts: cross-correlate observed with the current synthetic (3 comps summed)
        for s in stas:
            best, bestc = 0.0, -1e9
            syn = {c: np.column_stack([_window(gfs[s][c][k], t_start, t_end) for k in range(6)]) @ m
                   for c in "ZNE"}
            lo = int(round(shifts[s] * fs)) - int(3 * fs)
            hi = int(round(shifts[s] * fs)) + int(3 * fs)
            for lag in range(max(lo, -nmax), min(hi, nmax) + 1):
                sh = lag / fs
                cc = 0.0
                for c in "ZNE":
                    o = _window(obs[s][c], t_start + sh, t_end + sh)
                    n = min(len(o), len(syn[c]))
                    cc += float(np.dot(o[:n], syn[c][:n]))
                if cc > bestc:
                    bestc, best = cc, sh
            shifts[s] = best
    # misfit per station and overall (unweighted, in physical units)
    res = {}
    num = den = 0.0
    for s in stas:
        n_s = d_s = 0.0
        for comp in "ZNE":
            o = _window(obs[s][comp], t_start + shifts[s], t_end + shifts[s])
            g = np.column_stack([_window(gfs[s][comp][k], t_start, t_end) for k in range(6)])
            n = min(len(o), len(g)); o, y = o[:n], g[:n] @ m
            n_s += float(np.sum((o - y) ** 2)); d_s += float(np.sum(o ** 2))
        res[s] = dict(vr=round(1 - n_s / d_s, 3), shift_s=round(shifts[s], 2))
        num += n_s; den += d_s
    return m * M0_UNIT, 1 - num / den, res


def decompose(m):
    """m = (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) in N*m -> dict with M0, Mw, planes, CLVD."""
    from obspy.imaging.beachball import MomentTensor, mt2plane, mt2axes, aux_plane
    from obspy.imaging.beachball import mt2plane as _mt2plane
    mt = MomentTensor(*[float(v) for v in m], 1)
    M = np.array([[m[0], m[3], m[4]], [m[3], m[1], m[5]], [m[4], m[5], m[2]]])
    ev = np.sort(np.linalg.eigvalsh(M))
    m0 = float(np.sqrt(np.sum(M * M) / 2.0))
    mw = (2.0 / 3.0) * (np.log10(m0) - 9.1)
    clvd = abs(ev[1]) / max(abs(ev[0]), abs(ev[2]))  # |lambda_mid| / |lambda_max|
    p1 = _mt2plane(mt)
    p2 = aux_plane(p1.strike, p1.dip, p1.rake)
    return dict(m0_Nm=m0, Mw=round(float(mw), 2),
                clvd_fraction=round(float(clvd), 3),
                planes=[[round(p1.strike), round(p1.dip), round(((p1.rake + 180) % 360) - 180)],
                        [round(p2[0]), round(p2[1]), round(((p2[2] + 180) % 360) - 180)]],
                mt=[float(v) for v in m])


def _axes(strike, dip, rake):
    """Right-handed rotation matrix whose columns are the T, B, P axes (north, east, down)."""
    s, d, r = np.radians([strike, dip, rake])
    n = np.array([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)])
    u = np.array([np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
                  -np.sin(r) * np.sin(d)])
    T = (n + u) / np.sqrt(2.0)
    P = (n - u) / np.sqrt(2.0)
    B = np.cross(P, T)
    return np.column_stack([T, B, P])


def kagan(plane_a, plane_b):
    """Kagan angle (degrees) between a plane (strike,dip,rake) and the nearer of a list of planes.

    Double couples have a four-fold symmetry (the two nodal planes are interchangeable), so
    the angle is the smallest rotation over the identity and the three 180-degree rotations.
    """
    Ra = _axes(*plane_a)
    sym = [np.diag(v) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    best = 180.0
    for b in plane_b:
        Rb = _axes(*b)
        for S in sym:
            c = (np.trace(Ra.T @ Rb @ S) - 1.0) / 2.0
            best = min(best, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return best


# --------------------------------------------------------------------------- main
def run(lat, lon, depth_km, time_iso, net="CI", dc="SCEDC", minr=50, maxr=250, nsta=12,
        band=(0.03, 0.07), model="ak135f_2s", fs=2.0, ref=None, out=None, n_iter=3,
        verbose=True, min_snr=3.0, min_station_vr=0.3):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.taup import TauPyModel
    t0 = UTCDateTime(time_iso)
    client = Client(dc)
    tp = TauPyModel("iasp91")
    chosen = pick_stations(client, net, lat, lon, minr, maxr, nsta, t0)
    if verbose:
        print(f"{len(chosen)} stations", file=sys.stderr)
    obs, gfs, used, skipped = {}, {}, [], []
    src = dict(lat=lat, lon=lon, depth_km=depth_km, time=time_iso)
    # window: origin time to S + 60 s of the farthest station, same for all
    tS = max(tp.get_travel_times(depth_km, s["dist_km"] / 111.19, ["s", "S"])[0].time for s in chosen)
    t_start, t_end = t0 + 0.0, t0 + tS + 60.0
    tend_rel = float(tS + 60.0)
    for s in chosen:
        key = f"{s['net']}.{s['sta']}"
        try:
            st = load_observed(client, s, t0, tend_rel, band, fs)
            comp = {c: st.select(channel=f"*{c}")[0] for c in "ZNE"}
            sig = np.sqrt(np.mean(np.concatenate([_window(comp[c], t0, t0 + tend_rel) ** 2 for c in "ZNE"])))
            noi = np.sqrt(np.mean(np.concatenate([_window(comp[c], t0 - 110, t0 - 20) ** 2 for c in "ZNE"])))
            snr = float(sig / (noi + 1e-30))
            if snr < min_snr:
                skipped.append(dict(code=key, reason=f"signal-to-noise {snr:.1f} < {min_snr}"))
                if verbose:
                    print("  skip", key, f"SNR {snr:.1f}", file=sys.stderr)
                continue
            rec = dict(lat=s["lat"], lon=s["lon"], code=key)
            g = syngine_gf(src, rec, model, str(t0 - 120), str(t0 + tend_rel + 120), dt=1.0 / fs)
            f1, f2 = band
            gset = {}
            for c in "ZNE":
                trs = g[c]
                for tr in trs:
                    tr.filter("bandpass", freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                    tr.resample(fs)
                gset[c] = trs
            obs[key], gfs[key] = comp, gset
            used.append(dict(s, code=key, snr=round(snr, 1)))
            if verbose:
                print("  ok", key, round(s["dist_km"]), "km", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            skipped.append(dict(code=key, reason=f"{type(e).__name__}: {str(e)[:100]}"))
            if verbose:
                print("  skip", key, type(e).__name__, str(e)[:80], file=sys.stderr)
    if len(obs) < 4:
        raise RuntimeError(f"only {len(obs)} usable stations; widen --minr/--maxr or the band")
    m, vr, per = invert(obs, gfs, fs, t_start, t_end, n_iter=n_iter)
    # robust second pass: drop stations the first solution cannot explain (bad alignment,
    # 3-D path effects, instrument problems) but keep at least six
    dropped = []
    for _ in range(2):
        bad = sorted([k for k, v in per.items() if v["vr"] < min_station_vr], key=lambda k: per[k]["vr"])
        bad = bad[: max(0, len(obs) - 6)]
        if not bad:
            break
        for k in bad:
            dropped.append(dict(code=k, vr=per[k]["vr"], shift_s=per[k]["shift_s"]))
            obs.pop(k); gfs.pop(k)
        m, vr, per = invert(obs, gfs, fs, t_start, t_end, n_iter=n_iter)
    used = [u for u in used if u["code"] in obs]
    dec = decompose(m)
    result = dict(source=src, band_hz=list(band), model=model, n_stations=len(obs),
                  variance_reduction=round(float(vr), 3), per_station=per, **dec,
                  skipped=skipped, dropped_after_fit=dropped, stations=[{k: (round(v, 3) if isinstance(v, float) else v)
                                              for k, v in s.items()} for s in used])
    if ref:
        a = [float(v) for v in ref]
        result["reference_plane"] = a
        result["kagan_angle_deg"] = round(kagan(a, dec["planes"]), 1)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "mt_result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        try:
            _figure(obs, gfs, m, per, fs, t_start, t_end, result, os.path.join(out, "mt_fit.png"))
        except Exception as e:  # noqa: BLE001
            print("figure failed:", e, file=sys.stderr)
    return result


def _figure(obs, gfs, m, per, fs, t_start, t_end, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from obspy.imaging.beachball import beach
    stas = sorted(obs)
    n = len(stas)
    fig = plt.figure(figsize=(12, 1.0 * n + 2))
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1, 1, 0.9], hspace=0.15, wspace=0.12)
    for i, s in enumerate(stas):
        for j, c in enumerate("ZNE"):
            ax = fig.add_subplot(gs[i, j])
            sh = per[s]["shift_s"]
            o = _window(obs[s][c], t_start + sh, t_end + sh)
            g = np.column_stack([_window(gfs[s][c][k], t_start, t_end) for k in range(6)]) @ (m / M0_UNIT)
            k = min(len(o), len(g)); t = np.arange(k) / fs
            ax.plot(t, o[:k], "k", lw=0.8); ax.plot(t, g[:k], "r", lw=0.8)
            ax.set_yticks([]); ax.set_xticks([] if i < n - 1 else ax.get_xticks())
            if j == 0:
                ax.set_ylabel(s.split(".")[1], rotation=0, ha="right", va="center", fontsize=9)
            if i == 0:
                ax.set_title(c, fontsize=10)
    ax = fig.add_subplot(gs[:, 3]); ax.set_aspect("equal"); ax.axis("off")
    ax.add_collection(beach(list(m / 1e15), width=180, xy=(0, 0), facecolor="0.3"))
    ax.set_xlim(-110, 110); ax.set_ylim(-150, 110)
    p = result["planes"]
    ax.text(0, -110, f"Mw {result['Mw']}  VR {result['variance_reduction']:.2f}\n"
                     f"plane 1 {p[0]}\nplane 2 {p[1]}", ha="center", va="top", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--depth", type=float, required=True, help="km"); ap.add_argument("--time", required=True)
    ap.add_argument("--net", default="CI"); ap.add_argument("--dc", default="SCEDC")
    ap.add_argument("--minr", type=float, default=50); ap.add_argument("--maxr", type=float, default=250)
    ap.add_argument("--nsta", type=int, default=12)
    ap.add_argument("--band", type=float, nargs=2, default=[0.03, 0.07])
    ap.add_argument("--model", default="ak135f_2s")
    ap.add_argument("--ref", help="reference plane 'strike,dip,rake' for the Kagan angle")
    ap.add_argument("--out", default="./mt_out")
    a = ap.parse_args(argv)
    ref = [float(v) for v in a.ref.split(",")] if a.ref else None
    r = run(a.lat, a.lon, a.depth, a.time, a.net, a.dc, a.minr, a.maxr, a.nsta, tuple(a.band),
            a.model, ref=ref, out=a.out)
    print(json.dumps({k: r[k] for k in ("Mw", "planes", "clvd_fraction", "variance_reduction",
                                        "n_stations", "kagan_angle_deg") if k in r}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
