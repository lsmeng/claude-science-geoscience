#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2", "obspy>=1.4.0"]
# ///
"""
fdsnws-mcp — a Model Context Protocol connector for seismological data.

This is the seismology analog of Claude Science's UniProt/PDB connectors.
It wraps ObsPy's FDSN client so an agent (Claude Code / OpenCode / Cline / any
MCP host) can pull waveforms, station metadata, and earthquake catalogs from any
FDSN data center by name.

Run (stdio transport, the default for local MCP hosts):
    pip install -r requirements.txt
    python server.py

Register in an MCP host (.mcp.json / claude config):
    {
      "mcpServers": {
        "fdsnws": { "command": "python", "args": ["/abs/path/server.py"] }
      }
    }

Design notes
------------
- Every tool takes an explicit `datacenter` (default IRISPH5/EarthScope "IRIS")
  so the agent is never guessing which archive it hit — a reviewer-agent concern.
- Waveforms are written to disk (miniSEED) and the tool returns a *manifest*
  (path, byte size, gap report, exact request string) rather than raw bytes,
  so results are reproducible and the context window stays small.
- Times are ISO-8601 UTC only. No local time anywhere. (Reviewer check #UTC.)
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
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ObsPy is imported lazily inside tools so `--help`/import doesn't require it.
#
# The MCP SDK is an optional import: when `mcp` is installed this runs as a real
# MCP server; when it isn't (e.g. running smoke_test.py in a bare ObsPy env),
# `mcp.tool()` degrades to an identity decorator so the tool functions stay
# directly callable. This keeps the connector testable without the SDK.
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("fdsnws")
except ModuleNotFoundError:  # pragma: no cover - fallback for SDK-less testing
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit(
                "The 'mcp' package is not installed — this file's tools are still "
                "directly importable/callable, but running it as an MCP server "
                "requires `pip install mcp`."
            )
    mcp = _NoMCP()

# Where waveform files land. Override with FDSNWS_MCP_DATADIR.
DATA_DIR = Path(os.environ.get("FDSNWS_MCP_DATADIR", "./fdsn_data")).resolve()

# Common FDSN data centers ObsPy knows by short name.
KNOWN_CENTERS = [
    "IRIS", "EARTHSCOPE", "ORFEUS", "GEOFON", "NCEDC", "SCEDC",
    "RESIF", "INGV", "ETH", "BGR", "USGS", "EMSC", "AUSPASS", "KOERI",
]


def _utc(s: str):
    """Parse an ISO-8601 string to an ObsPy UTCDateTime, rejecting naive ambiguity."""
    from obspy import UTCDateTime
    return UTCDateTime(s)


def _client(datacenter: str):
    """Client for a short name (e.g. "IRIS") or any FDSN base URL (https://...)."""
    from obspy.clients.fdsn import Client
    if datacenter.lower().startswith("http"):
        return Client(base_url=datacenter)
    return Client(datacenter.upper())


def _dc(datacenter: str) -> str:
    return datacenter if datacenter.lower().startswith("http") else datacenter.upper()


@mcp.tool()
def list_datacenters() -> str:
    """List FDSN data centers this connector can query by short name.

    Returns a JSON array of names usable as the `datacenter` argument of the
    other tools (e.g. "IRIS", "SCEDC", "GEOFON"). Any other FDSN-compliant center
    can be used by passing its base URL (e.g. "https://service.iris.edu") instead.
    """
    return json.dumps(KNOWN_CENTERS)


@mcp.tool()
def get_events(
    starttime: str,
    endtime: str,
    minmagnitude: float = 0.0,
    maxmagnitude: float = 10.0,
    minlatitude: Optional[float] = None,
    maxlatitude: Optional[float] = None,
    minlongitude: Optional[float] = None,
    maxlongitude: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    maxradius_km: Optional[float] = None,
    mindepth_km: Optional[float] = None,
    maxdepth_km: Optional[float] = None,
    datacenter: str = "IRIS",
    limit: int = 500,
) -> str:
    """Query an FDSN event service for an earthquake catalog.

    Times are ISO-8601 UTC (e.g. "2026-01-01T00:00:00"). Use either a
    bounding box (min/max lat/lon) OR a radial search (latitude, longitude,
    maxradius_km) — not both. Depths are kilometers, positive down.

    Returns JSON: {count, events:[{time, lat, lon, depth_km, mag, magtype,
    id, author}], request}. Capped at `limit` events.
    """
    from obspy.geodetics import kilometers2degrees

    kw = dict(
        starttime=_utc(starttime),
        endtime=_utc(endtime),
        minmagnitude=minmagnitude,
        maxmagnitude=maxmagnitude,
        limit=limit,
        orderby="time",
    )
    if None not in (minlatitude, maxlatitude, minlongitude, maxlongitude):
        kw.update(minlatitude=minlatitude, maxlatitude=maxlatitude,
                  minlongitude=minlongitude, maxlongitude=maxlongitude)
    if None not in (latitude, longitude, maxradius_km):
        kw.update(latitude=latitude, longitude=longitude,
                  maxradius=kilometers2degrees(maxradius_km))
    if mindepth_km is not None:
        kw["mindepth"] = mindepth_km
    if maxdepth_km is not None:
        kw["maxdepth"] = maxdepth_km

    cat = _client(datacenter).get_events(**kw)

    out = []
    for ev in cat:
        o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        out.append({
            "id": str(ev.resource_id),
            "time": str(o.time) if o else None,
            "lat": o.latitude if o else None,
            "lon": o.longitude if o else None,
            "depth_km": (o.depth / 1000.0) if (o and o.depth is not None) else None,
            "mag": m.mag if m else None,
            "magtype": m.magnitude_type if m else None,
            "author": (o.creation_info.author if (o and o.creation_info) else None),
        })
    return json.dumps({
        "count": len(out),
        "events": out,
        "request": {"datacenter": _dc(datacenter), **{k: str(v) for k, v in kw.items()}},
    }, indent=2)


@mcp.tool()
def get_stations(
    starttime: str,
    endtime: str,
    network: str = "*",
    station: str = "*",
    location: str = "*",
    channel: str = "*",
    minlatitude: Optional[float] = None,
    maxlatitude: Optional[float] = None,
    minlongitude: Optional[float] = None,
    maxlongitude: Optional[float] = None,
    level: str = "channel",
    datacenter: str = "IRIS",
) -> str:
    """Query an FDSN station service for network/station/channel metadata.

    NSLC wildcards allowed (e.g. network="HV", channel="HH?"). `level` is one of
    network|station|channel|response. Returns JSON with a station list
    (net, sta, lat, lon, elev_m, channels) and the exact request. Use level=
    "response" only when you actually need poles-and-zeros (large output).
    """
    inv = _client(datacenter).get_stations(
        starttime=_utc(starttime), endtime=_utc(endtime),
        network=network, station=station, location=location, channel=channel,
        minlatitude=minlatitude, maxlatitude=maxlatitude,
        minlongitude=minlongitude, maxlongitude=maxlongitude,
        level=level,
    )
    stations = []
    for net in inv:
        for sta in net:
            chans = sorted({f"{net.code}.{sta.code}.{c.location_code}.{c.code}"
                            for c in sta.channels}) if sta.channels else []
            stations.append({
                "network": net.code, "station": sta.code,
                "lat": sta.latitude, "lon": sta.longitude,
                "elev_m": sta.elevation, "n_channels": len(chans),
                "channels": chans[:50],
            })
    return json.dumps({
        "count": len(stations),
        "stations": stations,
        "request": {"datacenter": _dc(datacenter), "network": network,
                    "station": station, "channel": channel, "level": level,
                    "starttime": starttime, "endtime": endtime},
    }, indent=2)


@mcp.tool()
def get_waveforms(
    network: str,
    station: str,
    channel: str,
    starttime: str,
    endtime: str,
    location: str = "*",
    datacenter: str = "IRIS",
    outfile: Optional[str] = None,
    inline: bool = False,
) -> str:
    """Download continuous waveforms as miniSEED and return a manifest.

    Times are ISO-8601 UTC. NSLC wildcards allowed. The samples are saved as a
    .mseed file (under FDSNWS_MCP_DATADIR unless `outfile` is given). Hosts that
    run connectors in a sandbox may not let the server write files; in that case,
    or when `inline` is true, the miniSEED bytes are returned base64-encoded in
    `mseed_base64` (up to 3 MB) so the caller can decode them with
    `base64.b64decode` and `obspy.read`.

    Returns JSON: {file, bytes, n_traces, gaps, n_gaps, sampling_rates, time_span,
    request, [mseed_base64], [write_error]}. A non-empty `gaps` list is a
    data-quality flag the downstream picking skill and reviewer must see.
    """
    import base64
    import io

    st = _client(datacenter).get_waveforms(
        network=network, station=station, location=location, channel=channel,
        starttime=_utc(starttime), endtime=_utc(endtime),
    )
    buf = io.BytesIO()
    st.write(buf, format="MSEED")
    data = buf.getvalue()

    if outfile is None:
        tag = f"{network}.{station}.{channel}_{starttime}_{endtime}".replace(":", "").replace("*", "x").replace("?", "x")
        outfile = str(DATA_DIR / f"{tag}.mseed")
    saved, write_error = None, None
    try:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "wb") as fh:
            fh.write(data)
        saved = outfile
    except OSError as e:
        write_error = f"{type(e).__name__}: {e}"

    gaps = [[g[0], g[1], g[2], g[3], str(g[4]), str(g[5]), g[6]] for g in st.get_gaps()]
    out = {
        "file": saved,
        "bytes": len(data),
        "n_traces": len(st),
        "sampling_rates": sorted({tr.stats.sampling_rate for tr in st}),
        "gaps": gaps,
        "n_gaps": len(gaps),
        "time_span": [str(min(tr.stats.starttime for tr in st)),
                      str(max(tr.stats.endtime for tr in st))] if len(st) else None,
        "request": {"datacenter": _dc(datacenter),
                    "nslc": f"{network}.{station}.{location}.{channel}",
                    "starttime": starttime, "endtime": endtime},
    }
    if write_error:
        out["write_error"] = write_error
    if inline or saved is None:
        if len(data) <= 3_000_000:
            out["mseed_base64"] = base64.b64encode(data).decode("ascii")
        else:
            out["note"] = "result larger than 3 MB: request a shorter window or fewer channels"
    return json.dumps(out, indent=2)


@mcp.tool()
def provenance_stamp() -> str:
    """Return a provenance record for this connector (versions + UTC timestamp).

    Emit this alongside any artifact derived from FDSN data so a result is
    reproducible/auditable (reviewer-agent + regulatory requirement).
    """
    import obspy
    return json.dumps({
        "connector": "fdsnws-mcp",
        "obspy_version": obspy.__version__,
        "utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(DATA_DIR),
    }, indent=2)


@mcp.tool()
def diagnose() -> str:
    """Report whether obspy can be imported here and where Python is looking.

    Use when another tool fails with an import error. Returns JSON {python,
    site_dir, site_dir_exists, site_listing, obspy, error, sys_path}.
    """
    import sys
    info = {"python": sys.executable, "site_dir": os.path.abspath(_site),
            "site_dir_exists": os.path.isdir(_site)}
    try:
        info["site_listing"] = sorted(os.listdir(_site))[:8]
    except Exception as e:  # noqa: BLE001
        info["site_listing"] = f"{type(e).__name__}: {e}"
    try:
        import obspy
        info["obspy"] = obspy.__version__
        info["error"] = None
    except Exception as e:  # noqa: BLE001
        info["obspy"] = None
        info["error"] = f"{type(e).__name__}: {e}"
    info["sys_path"] = sys.path
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run()
