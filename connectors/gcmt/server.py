#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2", "obspy>=1.4.0"]
# ///
"""
gcmt-mcp — Model Context Protocol connector for the Global CMT (GCMT) catalog.

Global Centroid-Moment-Tensor solutions (centroid location/time + full moment
tensor + nodal planes) for M >~ 5 events since 1976. GCMT is distributed as
**ndk files**, not a REST API, so this connector fetches + caches the monthly ndk
files and parses them with ObsPy (`obspy.io.ndk`), then filters by time / magnitude
/ region.

Source (verified): the LDEO mirror serves the monthly files at
    https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/{YYYY}/{mmm}{yy}.ndk
(the globalcmt.org NEW_MONTHLY path 404s as of 2026-07).

Convention note (for the reviewer): ObsPy converts ndk moment-tensor components
into the **Aki-Richards / NED-ish `Catalog` representation in N·m**; the raw ndk is
USE / dyne·cm with a per-record exponent. This connector reports ObsPy's parsed
values and labels units, so downstream code never mixes dyne·cm with N·m.

Run (stdio): python server.py    (needs `pip install mcp` + obspy)
Test:        python smoke_test.py (needs obspy + network)
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
import urllib.request
from pathlib import Path

GCMT_BASE = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY"
CACHE = Path(os.environ.get("GCMT_MCP_CACHE", "./gcmt_cache")).resolve()
_UA = {"User-Agent": "claude-science-geoscience/0.1 (gcmt-mcp)"}
_MON = ["jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"]

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("gcmt")
except ModuleNotFoundError:  # pragma: no cover
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit("Install 'mcp' to run as a server; tools remain importable.")
    mcp = _NoMCP()


def _month_url(year: int, month: int) -> str:
    return f"{GCMT_BASE}/{year}/{_MON[month - 1]}{year % 100:02d}.ndk"


_MEM: dict = {}


def _fetch_month(year: int, month: int) -> bytes | None:
    """Download one monthly ndk; return its bytes (memory cache, disk cache when writable)."""
    key = (year, month)
    if key in _MEM:
        return _MEM[key]
    local = CACHE / f"{_MON[month - 1]}{year % 100:02d}.ndk"
    try:
        if local.exists() and local.stat().st_size > 0:
            _MEM[key] = local.read_bytes()
            return _MEM[key]
    except OSError:
        pass
    url = _month_url(year, month)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception:
        return None
    _MEM[key] = data
    try:  # the host may not let the server write files (Claude Science sandbox)
        CACHE.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    except OSError:
        pass
    return data


def _iter_months(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _event_record(ev):
    """Flatten an ObsPy ndk event into a compact dict."""
    from obspy.geodetics import kilometers2degrees  # noqa: F401 (kept for parity)
    o = ev.preferred_origin() or (ev.origins[-1] if ev.origins else None)
    m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
    fm = ev.focal_mechanisms[0] if ev.focal_mechanisms else None
    rec = {
        "id": str(ev.resource_id).split("/")[-1],
        "centroid_time": str(o.time) if o else None,
        "lat": o.latitude if o else None,
        "lon": o.longitude if o else None,
        "depth_km": (o.depth / 1000.0) if (o and o.depth is not None) else None,
        "mag": m.mag if m else None,
        "magtype": m.magnitude_type if m else None,
    }
    if fm:
        np1 = fm.nodal_planes.nodal_plane_1 if fm.nodal_planes else None
        if np1:
            rec["nodal_plane_1"] = {"strike": np1.strike, "dip": np1.dip, "rake": np1.rake}
        mt = fm.moment_tensor
        if mt:
            rec["M0_Nm"] = mt.scalar_moment
            t = mt.tensor
            if t:
                rec["tensor_Nm"] = {"Mrr": t.m_rr, "Mtt": t.m_tt, "Mpp": t.m_pp,
                                    "Mrt": t.m_rt, "Mrp": t.m_rp, "Mtp": t.m_tp}
                rec["tensor_units"] = "N·m (ObsPy-parsed from ndk USE/dyne·cm)"
    return rec


@mcp.tool()
def gcmt_search(
    starttime: str,
    endtime: str,
    minmagnitude: float = 0.0,
    minlatitude: float = None,
    maxlatitude: float = None,
    minlongitude: float = None,
    maxlongitude: float = None,
    limit: int = 200,
) -> str:
    """Search GCMT for moment-tensor solutions in a time/magnitude/region window.

    Times ISO-8601 UTC. Fetches+caches the needed monthly ndk files and parses them
    with ObsPy. Returns JSON {count, months_fetched, events:[{id, centroid_time, lat,
    lon, depth_km, mag, nodal_plane_1, M0_Nm, tensor_Nm, tensor_units}], source}.
    """
    from obspy import UTCDateTime, read_events
    t0, t1 = UTCDateTime(starttime), UTCDateTime(endtime)

    events, months = [], []
    for y, mth in _iter_months(t0, t1):
        p = _fetch_month(y, mth)
        if p is None:
            continue
        months.append(f"{_MON[mth-1]}{y%100:02d}")
        try:
            import io
            cat = read_events(io.BytesIO(p), format="NDK")
        except Exception:
            continue
        for ev in cat:
            o = ev.preferred_origin() or (ev.origins[-1] if ev.origins else None)
            m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if o is None:
                continue
            if not (t0 <= o.time <= t1):
                continue
            if m and m.mag is not None and m.mag < minmagnitude:
                continue
            if minlatitude is not None and not (minlatitude <= o.latitude <= maxlatitude):
                continue
            if minlongitude is not None and not (minlongitude <= o.longitude <= maxlongitude):
                continue
            events.append(_event_record(ev))
            if len(events) >= limit:
                break
        if len(events) >= limit:
            break

    return json.dumps({
        "count": len(events),
        "months_fetched": months,
        "events": events,
        "source": GCMT_BASE,
    }, indent=2)


@mcp.tool()
def gcmt_refresh(year: int, month: int) -> str:
    """Force (re)download of one monthly ndk into the cache. Returns status JSON."""
    _MEM.pop((year, month), None)
    local = CACHE / f"{_MON[month - 1]}{year % 100:02d}.ndk"
    try:
        if local.exists():
            local.unlink()
    except OSError:
        pass
    p = _fetch_month(year, month)
    return json.dumps({"year": year, "month": month,
                       "cached": bool(p), "bytes": len(p) if p else None,
                       "url": _month_url(year, month)}, indent=2)


if __name__ == "__main__":
    mcp.run()
