#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2", "obspy>=1.4.0"]
# ///
"""
isc-mcp — Model Context Protocol connector for the ISC Bulletin.

The International Seismological Centre (ISC) Bulletin is the definitive long-term
global earthquake catalogue: the ISC re-picks, re-associates and relocates events
months-to-years after they occur, integrating the reports of ~150 agencies. It is
the reference catalogue for global seismicity studies and the source most GCMT /
ISC-EHB / EHB relocations build on — the "gold-standard, reviewed" analogue to the
faster-but-preliminary ComCat/USGS feed.

Implemented as a **thin, dependency-light HTTP client** over the ISC FDSNWS-event
service (`www.isc.ac.uk/fdsnws/event/1/`), which returns standard QuakeML. QuakeML
is parsed with ObsPy (`read_events`) so origins/magnitudes come back fully typed.

Design (shared with comcat-mcp / fdsnws-mcp):
- Times are ISO-8601 UTC only.
- Radial search is expressed in **km** (converted to degrees for the FDSN param).
- Every result carries a `request` (the exact query URL) for provenance.
- Polite etiquette: a descriptive User-Agent and a ~1 req/sec floor between calls.

Caveat surfaced to the agent: the most recent ~2 years of the ISC Bulletin are
"pre-review" (not yet analyst-reviewed); `isc_search` echoes this in `note`.

Run (stdio):  uv run server.py     (dependencies install themselves)
"""

from __future__ import annotations

# Optional bundled packages: if a `site/` folder sits next to `connectors/`, or
# GEOSCI_SITE_PACKAGES points to one, use it. This lets hosts that start the server
# with a fixed Python (e.g. Claude Science) find obspy without extra setup.
import os as _os
import sys as _sys
_site = _os.environ.get("GEOSCI_SITE_PACKAGES") or _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "site")
if _os.path.isdir(_site) and _os.path.abspath(_site) not in _sys.path:
    _sys.path.append(_os.path.abspath(_site))


import json
import time
import urllib.parse
import urllib.request

BASE = "http://www.isc.ac.uk/fdsnws/event/1"
import os
_UA = {"User-Agent": "claude-science-geoscience/0.1 (isc-mcp; contact: %s)" % os.environ.get("ISC_MCP_CONTACT", "unset")}
_MIN_INTERVAL = 1.0  # seconds — one request/sec polite floor
_last_call = [0.0]

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("isc")
except ModuleNotFoundError:  # pragma: no cover - SDK-less fallback so tools stay callable
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit("Install 'mcp' to run as an MCP server; tools remain importable.")
    mcp = _NoMCP()


def _throttle():
    """Enforce the ~1 req/sec polite floor across all outbound calls."""
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _get(params: dict):
    """GET {BASE}/query?{params} → (raw QuakeML text, url). Throttled + polite UA.

    ISC returns HTTP 204 (No Content) when a query matches zero events; we surface
    that as an empty body rather than raising.
    """
    q = {k: v for k, v in params.items() if v is not None}
    url = f"{BASE}/query?" + urllib.parse.urlencode(q)
    _throttle()
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status == 204:
                return "", url
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # ISC uses 404 for "no data" on some paths
        if e.code in (204, 404):
            return "", url
        if e.code == 429:
            raise RuntimeError(
                "ISC rate limit (HTTP 429): too many requests. Wait about a minute "
                "and retry with a narrower query; do not loop.") from e
        raise
    if body.strip() and "<" in body[:200] and "quakeml" not in body[:2000].lower():
        # ISC sometimes answers 200 with an HTML/CGI error page instead of QuakeML.
        raise RuntimeError("ISC returned a non-QuakeML reply (server-side error): "
                           + " ".join(body.split())[:200])
    return body, url


def _catalog_from_quakeml(qml: str):
    """Parse a QuakeML string into an ObsPy Catalog (empty Catalog if blank)."""
    from obspy import read_events
    from obspy.core.event import Catalog
    if not qml.strip():
        return Catalog()
    import io
    return read_events(io.BytesIO(qml.encode("utf-8")), format="QUAKEML")


def _event_record(ev):
    """Flatten an ObsPy event (from ISC QuakeML) into a compact dict."""
    o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
    m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
    desc = ev.event_descriptions[0].text if ev.event_descriptions else None
    return {
        "id": str(ev.resource_id).split("=")[-1].split("/")[-1],
        "time": str(o.time) if o else None,
        "lat": o.latitude if o else None,
        "lon": o.longitude if o else None,
        "depth_km": (o.depth / 1000.0) if (o and o.depth is not None) else None,
        "mag": m.mag if m else None,
        "magtype": m.magnitude_type if m else None,
        "magauthor": (m.creation_info.agency_id if (m and m.creation_info) else None),
        "author": (o.creation_info.agency_id if (o and o.creation_info) else None),
        "region": desc,
    }


@mcp.tool()
def isc_search(
    starttime: str,
    endtime: str,
    minmagnitude: float = None,
    maxmagnitude: float = None,
    latitude: float = None,
    longitude: float = None,
    maxradius_km: float = None,
    minlatitude: float = None,
    maxlatitude: float = None,
    minlongitude: float = None,
    maxlongitude: float = None,
    mindepth_km: float = None,
    maxdepth_km: float = None,
    limit: int = 500,
) -> str:
    """Search the ISC Bulletin and return a compact event list.

    Times ISO-8601 UTC. Region is EITHER a box (min/max lat/lon) OR radial
    (latitude, longitude, maxradius_km — km, converted to degrees for FDSN).
    Depths in km, positive down. Returns JSON {count, events:[{id, time, lat, lon,
    depth_km, mag, magtype, magauthor, author, region}], note, request}.

    Note: the ISC Bulletin lags real time; the most recent ~2 years are pre-review
    (analyst review pending) — reflected in `note`.
    """
    from obspy.geodetics import kilometers2degrees
    kw = dict(starttime=starttime, endtime=endtime,
              minmag=minmagnitude, maxmag=maxmagnitude,
              mindepth=mindepth_km, maxdepth=maxdepth_km,
              limit=limit, format="xml")
    if None not in (latitude, longitude, maxradius_km):
        kw.update(lat=latitude, lon=longitude,
                  maxradius=kilometers2degrees(maxradius_km))
    if None not in (minlatitude, maxlatitude, minlongitude, maxlongitude):
        kw.update(minlat=minlatitude, maxlat=maxlatitude,
                  minlon=minlongitude, maxlon=maxlongitude)
    qml, url = _get(kw)
    cat = _catalog_from_quakeml(qml)
    events = [_event_record(ev) for ev in cat]
    return json.dumps({
        "count": len(events),
        "events": events,
        "note": "ISC Bulletin: reviewed long-term catalogue; most recent ~2 yr are pre-review.",
        "request": url,
    }, indent=2)


@mcp.tool()
def isc_get(eventid: str) -> str:
    """Fetch one ISC Bulletin event by its ISC event id (e.g. '611634051').

    Returns JSON with the flattened preferred origin + magnitude, plus all reported
    origins and magnitudes (multi-agency solutions) so the agent can compare e.g.
    the ISC location against NEIC/GCMT contributions. Returns {event, n_origins,
    n_magnitudes, origins, magnitudes, request}.
    """
    qml, url = _get({"eventid": eventid, "format": "xml",
                     "includeallorigins": "true", "includeallmagnitudes": "true"})
    cat = _catalog_from_quakeml(qml)
    if len(cat) == 0:
        return json.dumps({"error": f"no ISC event '{eventid}'", "request": url}, indent=2)
    ev = cat[0]
    origins = [{
        "time": str(o.time), "lat": o.latitude, "lon": o.longitude,
        "depth_km": (o.depth / 1000.0) if o.depth is not None else None,
        "author": (o.creation_info.agency_id if o.creation_info else None),
    } for o in ev.origins]
    magnitudes = [{
        "mag": m.mag, "magtype": m.magnitude_type,
        "author": (m.creation_info.agency_id if m.creation_info else None),
    } for m in ev.magnitudes]
    return json.dumps({
        "event": _event_record(ev),
        "n_origins": len(origins),
        "n_magnitudes": len(magnitudes),
        "origins": origins,
        "magnitudes": magnitudes,
        "request": url,
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
