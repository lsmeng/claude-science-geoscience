#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2", "obspy>=1.4.0"]
# ///
"""
syngine-mcp — Model Context Protocol connector for the EarthScope Syngine service.

Syngine returns synthetic seismograms for a point source (moment tensor or double
couple) in a 1-D Earth model, computed from AxiSEM wavefield databases with Instaseis.
For 1-D Earth models this is equivalent to a normal-mode summation, so it is the
quickest way to get "what would this source look like at this station" anywhere on the
globe, without installing or running a solver.

Docs: https://service.iris.edu/irisws/syngine/1/  (also served at service.earthscope.org)

Design (shared with the other connectors in this repo):
- Times are ISO-8601 UTC. Every result carries the exact request URL (provenance).
- Waveforms are returned inline as base64 miniSEED (up to 3 MB) when the host does not
  let the server write files (Claude Science sandbox), or when `inline` is true.
- Be polite: a ~0.5 s floor between calls and a descriptive User-Agent.

Run (stdio):  uv run server.py     (dependencies install themselves)
"""

from __future__ import annotations

# Optional bundled packages (see the fdsnws connector): `site/` next to `connectors/`.
import os as _os
import sys as _sys
_site = _os.environ.get("GEOSCI_SITE_PACKAGES") or _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "site")
if _os.path.isdir(_site) and _os.path.abspath(_site) not in _sys.path:
    _sys.path.append(_os.path.abspath(_site))

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

BASE = os.environ.get("SYNGINE_BASE", "https://service.earthscope.org/irisws/syngine/1")
_UA = {"User-Agent": "claude-science-geoscience/0.1 (syngine-mcp; contact: %s)"
       % os.environ.get("SYNGINE_MCP_CONTACT", "unset")}
_MIN_INTERVAL = 0.5
_last = [0.0]
INLINE_MAX = 3_000_000
DATA_DIR = Path(os.environ.get("SYNGINE_MCP_DATADIR", "./syngine_data")).resolve()

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("syngine")
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
    dt = time.monotonic() - _last[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last[0] = time.monotonic()


def _get(path: str, params: dict | None = None, raw: bool = False, tries: int = 3):
    q = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{BASE}/{path}" + (("?" + urllib.parse.urlencode(q, safe=",:-")) if q else "")
    last = None
    for i in range(tries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            return (body, url) if raw else (json.loads(body.decode("utf-8", "replace")), url)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 503) and i < tries - 1:
                time.sleep(3 * (i + 1))
                last = RuntimeError(f"HTTP {e.code}: {msg}")
                continue
            raise RuntimeError(f"Syngine HTTP {e.code}: {' '.join(msg.split())}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Syngine request failed: {last}")


@mcp.tool()
def syngine_models() -> str:
    """List the Earth models (wavefield databases) Syngine can use.

    Returns JSON {models: {name: {description, min_period_s, max_period, default_dt}}}.
    Good defaults: `ak135f_2s` (shortest period 2 s, global), `iasp91_2s`, `prem_a_2s`
    (PREM with anisotropy). Longer-period databases (`_5s`, `_10s`) are faster.
    """
    data, url = _get("models")
    out = {k: {"description": v.get("description"), "min_period_s": v.get("min_period"),
               "max_period": v.get("max_period"), "default_dt": v.get("default_dt")}
           for k, v in data.items()}
    return json.dumps({"models": out, "request": url}, indent=2)


@mcp.tool()
def syngine_waveforms(
    sourcelatitude: float,
    sourcelongitude: float,
    sourcedepth_km: float,
    origintime: str,
    network: Optional[str] = None,
    station: Optional[str] = None,
    receiverlatitude: Optional[float] = None,
    receiverlongitude: Optional[float] = None,
    moment_tensor: Optional[str] = None,
    double_couple: Optional[str] = None,
    model: str = "ak135f_2s",
    components: str = "ZNE",
    units: str = "velocity",
    starttime: str = "P-10",
    endtime: str = "P+120",
    dt: Optional[float] = None,
    inline: bool = False,
    outfile: Optional[str] = None,
) -> str:
    """Synthetic 3-component seismograms at one receiver for a point source.

    Source: give EITHER `moment_tensor` = "Mrr,Mtt,Mpp,Mrt,Mrp,Mtp" in N*m (Harvard/GCMT
    convention, r up, theta south, phi east) OR `double_couple` = "strike,dip,rake" or
    "strike,dip,rake,M0" (degrees, M0 in N*m; default M0 = 1e19 N*m, about Mw 6.6).
    Receiver: either (network, station) known to EarthScope, or receiverlatitude and
    receiverlongitude (preferred when you have coordinates; the station codes are then
    only labels). `origintime` ISO-8601 UTC; start/end may be relative to the P arrival
    ("P-10", "P+120") or absolute ISO times. `units` is displacement|velocity|acceleration.

    Returns JSON {file, bytes, n_traces, sampling_rates, time_span, request, source,
    [mseed_base64], [write_error]}. The synthetics are 1-D-Earth point-source waveforms
    without instrument response, filtered by the database's shortest period (e.g. 2 s):
    do not compare them with observations at shorter periods, and expect 3-D structure
    to matter more at regional distances.
    """
    if (moment_tensor is None) == (double_couple is None):
        return json.dumps({"error": "give exactly one of moment_tensor or double_couple"})
    p = dict(model=model, components=components, units=units, format="miniseed",
             sourcelatitude=sourcelatitude, sourcelongitude=sourcelongitude,
             sourcedepthinmeters=int(round(sourcedepth_km * 1000)),
             origintime=origintime, starttime=starttime, endtime=endtime, dt=dt)
    if moment_tensor:
        p["sourcemomenttensor"] = moment_tensor
    else:
        p["sourcedoublecouple"] = double_couple
    if receiverlatitude is not None and receiverlongitude is not None:
        p.update(receiverlatitude=receiverlatitude, receiverlongitude=receiverlongitude)
        if network:
            p["networkcode"] = network
        if station:
            p["stationcode"] = station
    elif network and station:
        p.update(network=network, station=station)
    else:
        return json.dumps({"error": "give (network, station) or receiverlatitude+receiverlongitude"})

    body, url = _get("query", p, raw=True)
    import io
    from obspy import read
    st = read(io.BytesIO(body), format="MSEED")

    saved, write_error = None, None
    path = outfile or str(DATA_DIR / f"syn_{network or 'XX'}_{station or 'XXX'}_{origintime.replace(':', '')}.mseed")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(body)
        saved = path
    except OSError as e:
        write_error = f"{type(e).__name__}: {e}"
    out = {
        "file": saved,
        "bytes": len(body),
        "n_traces": len(st),
        "sampling_rates": sorted({tr.stats.sampling_rate for tr in st}),
        "time_span": [str(min(tr.stats.starttime for tr in st)),
                      str(max(tr.stats.endtime for tr in st))] if len(st) else None,
        "source": {"lat": sourcelatitude, "lon": sourcelongitude, "depth_km": sourcedepth_km,
                   "origintime": origintime, "moment_tensor": moment_tensor,
                   "double_couple": double_couple, "model": model, "units": units},
        "request": url,
    }
    if write_error:
        out["write_error"] = write_error
    if inline or saved is None:
        if len(body) <= INLINE_MAX:
            out["mseed_base64"] = base64.b64encode(body).decode("ascii")
        else:
            out["note"] = "result larger than 3 MB: request a shorter window"
    return json.dumps(out, indent=2)


@mcp.tool()
def provenance_stamp() -> str:
    """Connector version info: Syngine service version, ObsPy version and a UTC timestamp."""
    from datetime import datetime, timezone
    try:
        body, url = _get("version", raw=True)
        version = body.decode("utf-8", "replace").strip()
    except Exception as e:  # noqa: BLE001
        version = f"unavailable ({e})"
    import obspy
    return json.dumps({"connector": "syngine-mcp", "service_version": version,
                       "base": BASE, "obspy_version": obspy.__version__,
                       "utc": datetime.now(timezone.utc).isoformat()}, indent=2)


if __name__ == "__main__":
    mcp.run()
