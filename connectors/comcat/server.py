#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2"]
# ///
"""
comcat-mcp — Model Context Protocol connector for the USGS ComCat catalog.

The authoritative, deduplicated ANSS Comprehensive Catalog (ComCat): origins,
magnitudes, and per-event "products" (ShakeMap, moment tensor, focal mechanism,
phase data, finite-fault, PAGER). Analogue of Claude Science's biology catalogs.

Implemented as a **thin, dependency-light HTTP client** over the USGS FDSNWS-event
service (`earthquake.usgs.gov/fdsnws/event/1/`) — no `libcomcat` required, only the
Python stdlib. GeoJSON is used because it carries the `products` index that plain
FDSN-event does not.

Design (shared with fdsnws-mcp):
- Times are ISO-8601 UTC only.
- `comcat_count` is a cheap pre-check against the hard **20,000-event cap** so the
  agent can auto-chunk large queries.
- Every result carries a `request` (the exact query URL) for provenance.

Run (stdio):  uv run server.py     (dependencies install themselves)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
_UA = {"User-Agent": "claude-science-geoscience/0.1 (comcat-mcp)"}
CAP = 20000  # USGS hard cap per query
PAGE = 500  # events per request when paging a large search
INLINE_MAX = 200  # bigger results are saved to a file instead of returned inline
DATA_DIR = os.environ.get("COMCAT_MCP_DATADIR", "./comcat_data")

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("comcat")
except ModuleNotFoundError:  # pragma: no cover - SDK-less fallback so tools stay callable
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit("Install 'mcp' to run as an MCP server; tools remain importable.")
    mcp = _NoMCP()


def _fetch(url: str, tries: int = 3) -> str:
    """GET with retries: USGS sometimes drops a long response mid-stream."""
    import http.client
    import time
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", "replace")
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"USGS request failed after {tries} tries: {last}")


def _get(method: str, params: dict, raw: bool = False):
    """GET {BASE}/{method}?{params}; return parsed JSON (or raw text)."""
    q = {k: v for k, v in params.items() if v is not None}
    url = f"{BASE}/{method}?" + urllib.parse.urlencode(q)
    body = _fetch(url)
    return (body, url) if raw else (json.loads(body), url)


def _bbox_radius(kw, minlatitude, maxlatitude, minlongitude, maxlongitude,
                 latitude, longitude, maxradius_km):
    if None not in (minlatitude, maxlatitude, minlongitude, maxlongitude):
        kw.update(minlatitude=minlatitude, maxlatitude=maxlatitude,
                  minlongitude=minlongitude, maxlongitude=maxlongitude)
    if None not in (latitude, longitude, maxradius_km):
        kw.update(latitude=latitude, longitude=longitude,
                  maxradiuskm=maxradius_km)
    return kw


@mcp.tool()
def comcat_count(
    starttime: str,
    endtime: str,
    minmagnitude: float = None,
    maxmagnitude: float = None,
    minlatitude: float = None,
    maxlatitude: float = None,
    minlongitude: float = None,
    maxlongitude: float = None,
    latitude: float = None,
    longitude: float = None,
    maxradius_km: float = None,
    mindepth_km: float = None,
    maxdepth_km: float = None,
) -> str:
    """Return how many ComCat events match a query (cheap pre-check vs the 20k cap).

    Use this before comcat_search on wide windows: if count > 20000, split the time
    range. Returns JSON {count, over_cap, request}.
    """
    kw = dict(starttime=starttime, endtime=endtime,
              minmagnitude=minmagnitude, maxmagnitude=maxmagnitude,
              mindepth=mindepth_km, maxdepth=maxdepth_km, format="text")
    _bbox_radius(kw, minlatitude, maxlatitude, minlongitude, maxlongitude,
                 latitude, longitude, maxradius_km)
    body, url = _get("count", kw, raw=True)
    n = int(body.strip())
    return json.dumps({"count": n, "over_cap": n > CAP, "request": url}, indent=2)


@mcp.tool()
def comcat_search(
    starttime: str,
    endtime: str,
    minmagnitude: float = None,
    maxmagnitude: float = None,
    minlatitude: float = None,
    maxlatitude: float = None,
    minlongitude: float = None,
    maxlongitude: float = None,
    latitude: float = None,
    longitude: float = None,
    maxradius_km: float = None,
    mindepth_km: float = None,
    maxdepth_km: float = None,
    eventtype: str = None,
    orderby: str = "time",
    limit: int = 500,
    save_to: str = None,
) -> str:
    """Search ComCat and return a compact event list.

    Times ISO-8601 UTC. Box (min/max lat/lon) OR radial (latitude, longitude,
    maxradius_km). Depths km, positive down. Large searches are fetched in pages
    (limit up to 20000). Returns JSON {count, events:[{id, time, lat, lon, depth_km,
    mag, magtype, place, types}], request}. `types` lists the products available per
    event (shakemap, moment-tensor, ...) for comcat_get_product.

    To keep the conversation small, a result with more than 200 events is written to
    a JSON file (`save_to`, or COMCAT_MCP_DATADIR) and only the first 20 events are
    returned together with {"file": path, "saved": true}. The file has the same
    {"events": [...]} layout, so analysis skills (e.g. catalog-statistics) read it directly.
    """
    base = dict(starttime=starttime, endtime=endtime,
                minmagnitude=minmagnitude, maxmagnitude=maxmagnitude,
                mindepth=mindepth_km, maxdepth=maxdepth_km,
                eventtype=eventtype, orderby=orderby, format="geojson")
    _bbox_radius(base, minlatitude, maxlatitude, minlongitude, maxlongitude,
                 latitude, longitude, maxradius_km)
    limit = min(limit, CAP)

    events, offset, url = [], 1, None
    while len(events) < limit:
        n = min(PAGE, limit - len(events))
        data, url = _get("query", dict(base, limit=n, offset=offset))
        feats = data.get("features", [])
        for f in feats:
            p = f.get("properties", {})
            coords = (f.get("geometry") or {}).get("coordinates", [None, None, None])
            events.append({
                "id": f.get("id"),
                "time": _epoch_ms_to_iso(p.get("time")),
                "lat": coords[1], "lon": coords[0],
                "depth_km": coords[2],
                "mag": p.get("mag"), "magtype": p.get("magType"),
                "place": p.get("place"),
                "types": [t for t in (p.get("types") or "").split(",") if t],
            })
        if len(feats) < n:
            break
        offset += n

    out = {"count": len(events), "events": events, "request": url,
           "truncated_at_limit": len(events) >= limit}
    if save_to or len(events) > INLINE_MAX:
        path = save_to or os.path.join(
            DATA_DIR, f"comcat_{starttime[:10]}_{endtime[:10]}_{len(events)}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh)
        out = {"count": len(events), "saved": True, "file": os.path.abspath(path),
               "events": events[:20], "note": "first 20 of the saved events shown",
               "request": url, "truncated_at_limit": out["truncated_at_limit"]}
    return json.dumps(out, indent=2)


@mcp.tool()
def comcat_get_event(eventid: str) -> str:
    """Fetch full detail for one ComCat event by id (e.g. 'hv70116556').

    Returns JSON with origin, all magnitudes, and the product index (which products
    exist and their download URLs). Use comcat_get_product to pull a product's files.
    """
    data, url = _get("query", {"eventid": eventid, "format": "geojson"})
    p = data.get("properties", {})
    g = data.get("geometry", {}) or {}
    coords = g.get("coordinates", [None, None, None])
    products = {}
    for ptype, plist in (p.get("products") or {}).items():
        if plist:
            contents = list((plist[0].get("contents") or {}).keys())
            products[ptype] = {"n": len(plist), "contents_sample": contents[:20]}
    out = {
        "id": data.get("id"),
        "time": _epoch_ms_to_iso(p.get("time")),
        "lat": coords[1], "lon": coords[0], "depth_km": coords[2],
        "mag": p.get("mag"), "magtype": p.get("magType"), "place": p.get("place"),
        "products_available": products,
        "request": url,
    }
    return json.dumps(out, indent=2)


@mcp.tool()
def comcat_get_product(eventid: str, producttype: str, content: str = None) -> str:
    """List (or fetch the URL of) a product's contents for an event.

    producttype e.g. 'moment-tensor', 'focal-mechanism', 'shakemap', 'phase-data',
    'origin'. Without `content`, lists available content files + their URLs. With
    `content` (a filename from that list), returns that file's direct URL and, for
    small text/quakeml/json files, its body. Returns JSON.
    """
    data, url = _get("query", {"eventid": eventid, "format": "geojson"})
    plist = (data.get("properties", {}).get("products") or {}).get(producttype)
    if not plist:
        return json.dumps({"error": f"no product '{producttype}' for {eventid}",
                           "request": url}, indent=2)
    contents = plist[0].get("contents") or {}
    if content is None:
        return json.dumps({
            "eventid": eventid, "producttype": producttype,
            "contents": {k: v.get("url") for k, v in contents.items()},
            "request": url,
        }, indent=2)
    item = contents.get(content)
    if not item:
        return json.dumps({"error": f"no content '{content}'",
                           "available": list(contents)[:40]}, indent=2)
    result = {"eventid": eventid, "producttype": producttype,
              "content": content, "url": item.get("url"),
              "bytes": item.get("length"), "type": item.get("contentType")}
    # inline small text-ish payloads
    ctype = (item.get("contentType") or "")
    if item.get("url") and (item.get("length") or 0) < 200000 and \
       any(t in ctype for t in ("text", "xml", "json")):
        try:
            body, _ = _get_url(item["url"])
            result["body"] = body[:200000]
        except Exception as e:  # pragma: no cover
            result["body_error"] = str(e)
    return json.dumps(result, indent=2)


def _get_url(url):
    return _fetch(url), url


def _epoch_ms_to_iso(ms):
    """USGS times are epoch milliseconds UTC. Convert without Date.now-style calls."""
    if ms is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    mcp.run()
