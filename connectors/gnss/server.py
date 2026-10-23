#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2"]
# ///
"""
gnss-mcp — Model Context Protocol connector for GNSS daily position time series.

Continuous-GPS/GNSS daily position solutions from the **Nevada Geodetic Laboratory
(UNR / NGL)**, which processes >20,000 stations worldwide into a single consistent
reference frame. Daily east/north/up displacement series are the geodetic backbone
for measuring coseismic offsets, postseismic transients, interseismic strain and
slow-slip — the surface-deformation analogue to the seismic-catalogue connectors.

NGL distributes each station as a plain-text **`.tenv3`** file. Current frame is
**IGS20**; the per-station series live at
    geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/<STA>.tenv3
(the frame token appears twice in the path — `.../IGS20/tenv3/IGS20/...`).
Station metadata is scanned from the master `DataHoldings.txt`.

Design (shared with comcat-mcp / gcmt-mcp / literature-mcp):
- Big payloads are returned **by reference**: the full parsed series is written to a
  temp JSON file and the tool returns a compact statistical summary + a `handle`
  (the file path) the agent can hand to a plotting/analysis step.
- Every result carries the source `url` for provenance.
- Polite etiquette: a descriptive User-Agent and a ~1 req/sec floor between calls.
- `DataHoldings.txt` (~2.7 MB) is cached locally so station lookups are cheap.

Displacement convention: NGL reports position as an integer reference (`_e0/_n0/_u0`,
metres) plus a fractional part (`__east/__north/__up`, metres). This connector sums
them, de-means each component over the series, and reports **mm relative to the
series mean** — the standard "displacement time series" plotted for GNSS.

Run (stdio):  python server.py     (needs `pip install mcp`)
"""

from __future__ import annotations

# Optional bundled packages (see the fdsnws connector): `site/` next to `connectors/`.
import os as _os
import sys as _sys
_site = _os.environ.get("GEOSCI_SITE_PACKAGES") or _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "site")
if _os.path.isdir(_site) and _os.path.abspath(_site) not in _sys.path:
    _sys.path.append(_os.path.abspath(_site))


import json
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://geodesy.unr.edu"
TENV3_URL = BASE + "/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3"
HOLDINGS_URL = BASE + "/NGLStationPages/DataHoldings.txt"
CACHE = Path(os.environ.get("GNSS_MCP_CACHE", tempfile.gettempdir())).resolve()
_UA = {"User-Agent": "claude-science-geoscience/0.1 (gnss-mcp; mailto:meng.caltech@gmail.com)"}
_MIN_INTERVAL = 1.0  # seconds — one request/sec polite floor
_last_call = [0.0]

# .tenv3 column indices (whitespace-split); NGL header:
# site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m)
#      u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) ... _lat _lon _height
C_YYMMMDD, C_DECYR, C_MJD = 1, 2, 3
C_E0, C_EAST, C_N0, C_NORTH, C_U0, C_UP = 7, 8, 9, 10, 11, 12
C_SIGE, C_SIGN, C_SIGU = 14, 15, 16
C_LAT, C_LON, C_HGT = 20, 21, 22

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("gnss")
except ModuleNotFoundError:  # pragma: no cover
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit("Install 'mcp' to run as an MCP server; tools remain importable.")
    mcp = _NoMCP()


def _throttle():
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _fetch_text(url: str) -> str:
    _throttle()
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode("utf-8", "replace")


_HOLD = [0.0, ""]


def _holdings_text() -> str:
    """DataHoldings.txt, cached in memory for 24 h (and on disk when writable)."""
    if _HOLD[1] and (time.time() - _HOLD[0]) < 86400:
        return _HOLD[1]
    text = None
    local = CACHE / "gnssmcp_DataHoldings.txt"
    try:
        if local.exists() and local.stat().st_size > 0 and (time.time() - local.stat().st_mtime) < 86400:
            text = local.read_text()
    except OSError:
        pass
    if text is None:
        text = _fetch_text(HOLDINGS_URL)
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            local.write_text(text)
        except OSError:
            pass
    _HOLD[0], _HOLD[1] = time.time(), text
    return text


def _mjd_to_iso(mjd: float) -> str:
    """Modified Julian Date → ISO calendar date (MJD epoch = 1858-11-17)."""
    d = datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=float(mjd))
    return d.date().isoformat()


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _stats_mm(vals_m):
    """De-mean a metre-series and return {mean_removed_m, ptp_mm, std_mm, ...}."""
    mu = _mean(vals_m)
    mm = [(v - mu) * 1000.0 for v in vals_m]
    n = len(mm)
    var = sum(x * x for x in mm) / n if n else 0.0
    return {
        "n": n,
        "reference_mean_m": round(mu, 5) if mu is not None else None,
        "ptp_mm": round(max(mm) - min(mm), 3) if mm else None,
        "std_mm": round(var ** 0.5, 3),
        "first_mm": round(mm[0], 3) if mm else None,
        "last_mm": round(mm[-1], 3) if mm else None,
    }, mm


@mcp.tool()
def gnss_get_timeseries(station: str, inline: bool = False,
                        start: str = "", end: str = "") -> str:
    """Fetch + parse one NGL station's daily .tenv3 series (IGS20 frame).

    `station` is a 4-char NGL id (case-insensitive, e.g. 'P056', 'MKEA', 'AB06').
    Downloads the .tenv3, parses every epoch, converts east/north/up to **mm
    relative to the series mean**, and writes the full series (dates + e/n/u mm +
    1-sigma mm) to a temp JSON file returned **by reference** as `handle`.

    Returns JSON {station, url, handle, n_epochs, date_start, date_end, span_days,
    east/north/up: {ptp_mm, std_mm, first_mm, last_mm, reference_mean_m}}.
    Hand `handle` to a plotting/analysis step for the full arrays.
    For a time window (e.g. around an earthquake) pass `start` and `end` (YYYY-MM-DD):
    the daily rows in that window (date, east, north, up, sigmas in mm) come back
    inline as `window`, so no file access is needed (the Claude Science sandbox
    cannot read the handle file).
    """
    sta = station.strip().upper()
    url = TENV3_URL.format(sta=sta)
    try:
        text = _fetch_text(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return json.dumps({
                "error": f"no IGS20 .tenv3 for station '{sta}' (404)",
                "hint": "check the id with gnss_station_info; NGL ids are 4 chars.",
                "url": url,
            }, indent=2)
        raise

    dates, decyr, east_m, north_m, up_m = [], [], [], [], []
    sig_e, sig_n, sig_u = [], [], []
    for line in text.splitlines():
        c = line.split()
        if len(c) < 23 or c[0].lower() == "site":
            continue
        try:
            dates.append(_mjd_to_iso(c[C_MJD]))
            decyr.append(float(c[C_DECYR]))
            east_m.append(float(c[C_E0]) + float(c[C_EAST]))
            north_m.append(float(c[C_N0]) + float(c[C_NORTH]))
            up_m.append(float(c[C_U0]) + float(c[C_UP]))
            sig_e.append(float(c[C_SIGE]) * 1000.0)
            sig_n.append(float(c[C_SIGN]) * 1000.0)
            sig_u.append(float(c[C_SIGU]) * 1000.0)
        except (ValueError, IndexError):
            continue

    if not dates:
        return json.dumps({"error": f"parsed 0 epochs for '{sta}'", "url": url}, indent=2)

    e_stat, e_mm = _stats_mm(east_m)
    n_stat, n_mm = _stats_mm(north_m)
    u_stat, u_mm = _stats_mm(up_m)

    full = {
        "station": sta, "frame": "IGS20", "url": url,
        "convention": "east/north/up in mm relative to series mean",
        "dates": dates, "decimal_year": decyr,
        "east_mm": [round(x, 3) for x in e_mm],
        "north_mm": [round(x, 3) for x in n_mm],
        "up_mm": [round(x, 3) for x in u_mm],
        "sig_east_mm": [round(x, 3) for x in sig_e],
        "sig_north_mm": [round(x, 3) for x in sig_n],
        "sig_up_mm": [round(x, 3) for x in sig_u],
    }
    handle, write_error = None, None
    try:
        fd, handle = tempfile.mkstemp(prefix=f"gnssmcp_{sta}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(full, f)
    except OSError as e:  # sandboxed host: cannot write files
        handle, write_error = None, f"{type(e).__name__}: {e}"

    span = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days
    out = {
        "station": sta, "frame": "IGS20", "url": url, "handle": handle,
        "n_epochs": len(dates),
        "date_start": dates[0], "date_end": dates[-1], "span_days": span,
        "east": e_stat, "north": n_stat, "up": u_stat,
        "note": "full dates + e/n/u/sigma arrays (mm) written by reference at `handle`.",
    }
    if write_error:
        out["write_error"] = write_error
    if start or end:
        lo, hi = start or "0000-01-01", end or "9999-12-31"
        out["window"] = {"start": lo, "end": hi, "columns": ["date", "east_mm", "north_mm", "up_mm",
                         "sig_e_mm", "sig_n_mm", "sig_u_mm"], "rows": [
            [d[:10], full["east_mm"][i], full["north_mm"][i], full["up_mm"][i],
             full["sig_east_mm"][i], full["sig_north_mm"][i], full["sig_up_mm"][i]]
            for i, d in enumerate(dates) if lo <= d[:10] <= hi]}
    if inline or handle is None:
        out["data"] = {k: full[k] for k in ("dates", "decimal_year", "east_mm", "north_mm", "up_mm")}
    return json.dumps(out, indent=2)


@mcp.tool()
def gnss_station_info(station: str) -> str:
    """Look up an NGL station's metadata from the master DataHoldings.txt.

    Returns JSON {station, lat, lon, height_m, date_begin, date_end, n_solutions,
    orig_name, url, source}. Use this to confirm a station id / operating window
    before gnss_get_timeseries, or to discover valid ids if one 404s.
    """
    sta = station.strip().upper()
    hit = None
    for line in _holdings_text().splitlines():
        c = line.split()
        if c and c[0].upper() == sta:
            hit = c
            break
    if not hit:
        return json.dumps({"error": f"station '{sta}' not in DataHoldings.txt",
                           "source": HOLDINGS_URL}, indent=2)
    return json.dumps({
        "station": hit[0],
        "lat": float(hit[1]), "lon": float(hit[2]), "height_m": float(hit[3]),
        "date_begin": hit[7], "date_end": hit[8], "date_modified": hit[9],
        "n_solutions": int(hit[10]) if len(hit) > 10 and hit[10].isdigit() else None,
        "orig_name": hit[11] if len(hit) > 11 else None,
        "url": TENV3_URL.format(sta=hit[0]),
        "source": HOLDINGS_URL,
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
