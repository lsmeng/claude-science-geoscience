#!/usr/bin/env python3
"""
locate_event.py - locate a single earthquake from P and S picks (pure Python).

Method: grid search over latitude, longitude and depth in a 1-D iasp91 Earth. For every
trial hypocentre the origin time is the (weighted) mean of the pick residuals, the
misfit is the weighted RMS residual. Outlier picks are removed iteratively (3 x MAD).
Uncertainty comes from a bootstrap over stations. No Fortran locator is needed (use
HYPOINVERSE / NonLinLoc for production catalogs; this is the transparent single-event tool).

Input:
  * a picks CSV with columns network,station,phase,time[,probability]  (the output of the
    phase-picking-ml skill works as is), and station coordinates fetched from an FDSN data
    center (--dc), OR
  * --auto: download 3-component waveforms around an approximate origin time, run PhaseNet
    (needs seisbench) and take the strongest P and S per station. The approximate location
    is only used to choose stations, not as a starting point.

    python locate_event.py --auto --lat 35.64 --lon -117.59 --time 2019-07-12T13:11:38 \
        --net CI --minr 25 --maxr 120 --out ./loc_out [--ref 35.6368,-117.5862,9.48,2019-07-12T13:11:37.990]

Network: the data center (default SCEDC); PhaseNet weights from the SeisBench host in --auto.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

KM_PER_DEG = 111.19


# ----------------------------------------------------------------------------- travel times
class TravelTimes:
    """iasp91 P and S first-arrival times on a (distance, depth) grid, bilinear lookup."""

    def __init__(self, dmax_km=400.0, zmax_km=40.0, dd=2.0, dz=2.0):
        from obspy.taup import TauPyModel
        m = TauPyModel("iasp91")
        self.d = np.arange(0.0, dmax_km + dd, dd)
        self.z = np.arange(0.0, zmax_km + dz, dz)
        self.tab = {}
        for ph, names in (("P", ["p", "P"]), ("S", ["s", "S"])):
            T = np.full((len(self.z), len(self.d)), np.nan)
            for i, z in enumerate(self.z):
                for j, d in enumerate(self.d):
                    deg = max(d / KM_PER_DEG, 0.01)
                    arr = m.get_travel_times(max(z, 0.1), deg, names)
                    if arr:
                        T[i, j] = min(a.time for a in arr)
            self.tab[ph] = T

    def __call__(self, ph, dist_km, depth_km):
        T = self.tab[ph]
        dist_km = np.asarray(dist_km, float); depth_km = np.asarray(depth_km, float)
        fd = np.clip(dist_km / (self.d[1] - self.d[0]), 0, len(self.d) - 1.001)
        fz = np.clip(depth_km / (self.z[1] - self.z[0]), 0, len(self.z) - 1.001)
        i0 = fz.astype(int); j0 = fd.astype(int); a = fz - i0; b = fd - j0
        return ((1 - a) * (1 - b) * T[i0, j0] + a * (1 - b) * T[i0 + 1, j0]
                + (1 - a) * b * T[i0, j0 + 1] + a * b * T[i0 + 1, j0 + 1])


# ----------------------------------------------------------------------------- locating
def _dist_km(lat, lon, slat, slon):
    """Equirectangular distance (km) from trial points (arrays) to stations (arrays)."""
    lat = np.asarray(lat)[..., None]; lon = np.asarray(lon)[..., None]
    dy = (slat - lat) * KM_PER_DEG
    dx = (slon - lon) * KM_PER_DEG * np.cos(np.radians((slat + lat) / 2))
    return np.hypot(dx, dy)


def grid_locate(picks, tt, center, half_deg=0.6, n=61, depths=None, w=None):
    """picks: list of dict(phase, t (s, relative), lat, lon, w). Returns best and the misfit cube."""
    depths = np.arange(1.0, 30.1, 1.0) if depths is None else depths
    lats = center[0] + np.linspace(-half_deg, half_deg, n)
    lons = center[1] + np.linspace(-half_deg, half_deg, n) / np.cos(np.radians(center[0]))
    slat = np.array([p["lat"] for p in picks]); slon = np.array([p["lon"] for p in picks])
    t = np.array([p["t"] for p in picks]); ph = np.array([p["phase"] for p in picks])
    w = np.array([p.get("w", 1.0) for p in picks]) if w is None else w
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    D = _dist_km(LA, LO, slat, slon)  # (n, n, npick)
    best = (np.inf, None)
    cube = np.zeros((len(depths), n, n))
    for k, z in enumerate(depths):
        tc = np.where(ph == "P", tt("P", D, z), tt("S", D, z))
        res = t - tc
        t0 = np.sum(w * res, axis=-1) / np.sum(w)
        r = res - t0[..., None]
        rms = np.sqrt(np.sum(w * r ** 2, axis=-1) / np.sum(w))
        cube[k] = rms
        i, j = np.unravel_index(np.argmin(rms), rms.shape)
        if rms[i, j] < best[0]:
            best = (rms[i, j], (lats[i], lons[j], z, t0[i, j]))
    return best, cube, lats, lons, depths


def refine(picks, tt, start, w=None):
    """Local refinement on a fine grid around the coarse solution."""
    lat, lon, z, _ = start
    out = start
    for half, n, dz in ((0.05, 41, 0.5), (0.01, 41, 0.2)):
        depths = np.arange(max(0.2, z - 4), z + 4.01, dz)
        (rms, sol), _, _, _, _ = grid_locate(picks, tt, (lat, lon), half_deg=half, n=n, depths=depths, w=w)
        lat, lon, z, t0 = sol
        out = (lat, lon, z, t0, rms)
    return out


def locate(picks, tt, center=None, n_boot=100, seed=0, outlier_mad=3.0):
    """Full single-event location with outlier rejection and a station bootstrap."""
    rng = np.random.default_rng(seed)
    picks = [dict(p) for p in picks]
    if center is None:
        first = min(picks, key=lambda p: p["t"])
        center = (first["lat"], first["lon"])
    # coarse grid is wide; fine grids follow
    (rms, sol), _, _, _, _ = grid_locate(picks, tt, center, half_deg=0.8, n=81)
    sol = refine(picks, tt, sol)
    kept = picks
    for _ in range(3):  # iterative outlier removal
        lat, lon, z, t0, _ = sol
        D = _dist_km(np.array([lat]), np.array([lon]),
                     np.array([p["lat"] for p in kept]), np.array([p["lon"] for p in kept]))[0]
        tc = np.array([tt(p["phase"], d, z) for p, d in zip(kept, D)])
        res = np.array([p["t"] for p in kept]) - tc - t0
        mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-3
        keep = np.abs(res - np.median(res)) <= max(outlier_mad * mad, 0.5)
        if keep.all() or keep.sum() < 6:
            break
        kept = [p for p, k in zip(kept, keep) if k]
        sol = refine(kept, tt, sol)
    lat, lon, z, t0, rms = sol
    # residuals for the report
    D = _dist_km(np.array([lat]), np.array([lon]),
                 np.array([p["lat"] for p in picks]), np.array([p["lon"] for p in picks]))[0]
    resid = []
    for p, d in zip(picks, D):
        r = p["t"] - tt(p["phase"], d, z) - t0
        resid.append(dict(station=p.get("station"), phase=p["phase"], dist_km=round(float(d), 1),
                          residual_s=round(float(r), 2), used=any(p is q for q in kept)))
    # bootstrap over stations
    stas = sorted({p["station"] for p in kept})
    boots = []
    for _ in range(n_boot):
        pick_sta = rng.choice(stas, size=len(stas), replace=True)
        bp = [p for s in pick_sta for p in kept if p["station"] == s]
        try:
            _, sb = grid_locate(bp, tt, (lat, lon), half_deg=0.1, n=41,
                                depths=np.arange(max(0.5, z - 8), z + 8.01, 1.0))[0]
            boots.append(sb)
        except Exception:  # noqa: BLE001
            pass
    B = np.array(boots)
    ex = (B[:, 1] - lon) * KM_PER_DEG * np.cos(np.radians(lat))
    ny = (B[:, 0] - lat) * KM_PER_DEG
    cov = np.cov(np.vstack([ex, ny]))
    ev, evec = np.linalg.eigh(cov)
    az_stations = np.degrees(np.arctan2(
        (np.array([p["lon"] for p in kept]) - lon) * np.cos(np.radians(lat)),
        np.array([p["lat"] for p in kept]) - lat)) % 360
    srt = np.sort(np.unique(az_stations.round()))
    gap = float(np.max(np.diff(np.concatenate([srt, [srt[0] + 360]])))) if len(srt) > 1 else 360.0
    return dict(lat=float(lat), lon=float(lon), depth_km=float(z), origin_rel_s=float(t0), rms_s=float(rms),
                n_picks=len(kept), n_picks_total=len(picks), n_stations=len(stas), gap_deg=round(gap),
                sigma_horizontal_km=[round(float(np.sqrt(max(v, 0))), 2) for v in ev[::-1]],
                ellipse_major_azimuth_deg=round(float(np.degrees(np.arctan2(evec[0, 1], evec[1, 1])) % 180)),
                sigma_depth_km=round(float(np.std(B[:, 2])), 2), residuals=resid,
                bootstrap=[[float(b[0]), float(b[1]), float(b[2])] for b in B[:60]])


# ----------------------------------------------------------------------------- data in
def stations_for(client, picks_sta, t0):
    out = {}
    for net, sta in picks_sta:
        try:
            inv = client.get_stations(network=net, station=sta, level="station", starttime=t0, endtime=t0 + 60)
            out[(net, sta)] = (inv[0][0].latitude, inv[0][0].longitude)
        except Exception:  # noqa: BLE001
            pass
    return out


def auto_picks(client, net, lat, lon, time_iso, minr, maxr, nsta, p_thr=0.5, s_thr=0.4, verbose=True):
    """Download waveforms near an approximate origin, run PhaseNet, take the strongest P and S."""
    from obspy import UTCDateTime
    from obspy.geodetics import gps2dist_azimuth, kilometers2degrees as k2d
    import seisbench.models as sbm
    t0 = UTCDateTime(time_iso)
    inv = client.get_stations(network=net, channel="BH?", latitude=lat, longitude=lon,
                              minradius=k2d(minr), maxradius=k2d(maxr), starttime=t0, endtime=t0 + 60,
                              level="station")
    cands = sorted([(gps2dist_azimuth(lat, lon, s.latitude, s.longitude)[0] / 1000, n.code, s)
                    for n in inv for s in n], key=lambda c: c[0])[: nsta * 2]
    model = sbm.PhaseNet.from_pretrained("original")
    picks = []
    for d, ncode, s in cands:
        if len({p["station"] for p in picks}) >= nsta:
            break
        try:
            st = client.get_waveforms(ncode, s.code, "*", "BH?", t0 - 20, t0 + 100)
            if len(st) < 3:
                continue
            st.merge(fill_value=0)
            ann = model.annotate(st)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("  skip", s.code, type(e).__name__, file=sys.stderr)
            continue
        found = {}
        for tr in ann:
            ph = tr.stats.channel[-1]
            if ph not in "PS":
                continue
            tt_ = tr.times("timestamp") - t0.timestamp
            sel = (tt_ > -5) & (tt_ < 60)
            i = int(np.argmax(np.where(sel, tr.data, -1)))
            found[ph] = (float(tt_[i]), float(tr.data[i]))
        if "P" in found and found["P"][1] >= p_thr:
            picks.append(dict(network=ncode, station=s.code, phase="P", t=found["P"][0],
                              w=found["P"][1], lat=s.latitude, lon=s.longitude))
            if "S" in found and found["S"][1] >= s_thr and found["S"][0] > found["P"][0] + 1.0:
                picks.append(dict(network=ncode, station=s.code, phase="S", t=found["S"][0],
                                  w=found["S"][1], lat=s.latitude, lon=s.longitude))
            if verbose:
                print("  picks", s.code, round(d), "km", {k: (round(v[0], 2), round(v[1], 2)) for k, v in found.items()},
                      file=sys.stderr)
    return picks, t0


def read_picks_csv(path, client, t_ref):
    from obspy import UTCDateTime
    rows = list(csv.DictReader(open(path)))
    sta = stations_for(client, {(r["network"], r["station"]) for r in rows}, t_ref)
    picks = []
    for r in rows:
        k = (r["network"], r["station"])
        if k not in sta:
            continue
        picks.append(dict(network=k[0], station=k[1], phase=r["phase"], t=UTCDateTime(r["time"]) - t_ref,
                          w=float(r.get("probability") or 1.0), lat=sta[k][0], lon=sta[k][1]))
    return picks


def _figure(res, picks, ref, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    for p in picks:
        a.plot(p["lon"], p["lat"], "^", color="0.4", ms=7)
    B = np.array(res["bootstrap"])
    a.plot(B[:, 1], B[:, 0], ".", color="tab:blue", alpha=0.35, ms=4, label="bootstrap")
    a.plot(res["lon"], res["lat"], "*", color="tab:blue", ms=16, label="this location")
    if ref:
        a.plot(ref[1], ref[0], "o", mfc="none", mec="tab:red", ms=12, mew=2, label="reference")
    a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.legend(loc="best", fontsize=9)
    a.set_aspect(1 / np.cos(np.radians(res["lat"])))
    b = ax[1]
    r = res["residuals"]
    for i, q in enumerate(r):
        b.plot(q["dist_km"], q["residual_s"], "o" if q["phase"] == "P" else "s",
               color="tab:blue" if q["used"] else "0.7", ms=6)
    b.axhline(0, color="k", lw=0.6); b.set_xlabel("distance (km)"); b.set_ylabel("pick residual (s)")
    b.set_title(f"RMS {res['rms_s']:.2f} s, {res['n_picks']} picks, {res['n_stations']} stations", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", help="picks CSV (network,station,phase,time[,probability])")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--time", help="approximate origin time (UTC); reference time for picks")
    ap.add_argument("--net", default="CI"); ap.add_argument("--dc", default="SCEDC")
    ap.add_argument("--minr", type=float, default=25); ap.add_argument("--maxr", type=float, default=120)
    ap.add_argument("--nsta", type=int, default=15)
    ap.add_argument("--ref", help="reference 'lat,lon,depth_km,origin_time' for comparison")
    ap.add_argument("--out", default="./loc_out")
    a = ap.parse_args(argv)
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client(a.dc)
    if a.auto:
        picks, t_ref = auto_picks(client, a.net, a.lat, a.lon, a.time, a.minr, a.maxr, a.nsta)
    else:
        t_ref = UTCDateTime(a.time)
        picks = read_picks_csv(a.picks, client, t_ref)
    if len(picks) < 8:
        raise SystemExit(f"only {len(picks)} usable picks; need at least 8 (P and S at 4+ stations)")
    tt = TravelTimes()
    res = locate(picks, tt)
    res["origin_time"] = str(t_ref + res["origin_rel_s"])
    if a.ref:
        rl, ro, rz, rt = a.ref.split(",")
        rl, ro, rz = float(rl), float(ro), float(rz)
        dx = (res["lon"] - ro) * KM_PER_DEG * np.cos(np.radians(rl)); dy = (res["lat"] - rl) * KM_PER_DEG
        res["reference"] = dict(lat=rl, lon=ro, depth_km=rz, origin_time=rt,
                                horizontal_offset_km=round(float(np.hypot(dx, dy)), 2),
                                depth_offset_km=round(res["depth_km"] - rz, 2),
                                origin_offset_s=round(float(UTCDateTime(res["origin_time"]) - UTCDateTime(rt)), 2))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "location.json"), "w"), indent=2)
    try:
        _figure(res, picks, [float(v) for v in a.ref.split(",")[:2]] if a.ref else None,
                os.path.join(a.out, "location.png"))
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e, file=sys.stderr)
    brief = {k: res[k] for k in ("lat", "lon", "depth_km", "origin_time", "rms_s", "n_picks", "n_stations",
                                 "gap_deg", "sigma_horizontal_km", "sigma_depth_km") if k in res}
    if "reference" in res:
        brief["reference"] = res["reference"]
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
