#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2", "obspy>=1.4.0"]
# ///
"""
taup-velocity-mcp — Model Context Protocol connector for theoretical travel times.

Theoretical seismic-phase travel times, ray parameters and take-off angles for the
standard 1-D global earth models (IASP91, AK135, PREM, …), computed with ObsPy's
`obspy.taup` — the Python port of Crotwell's TauP toolkit. This is the reference an
agent uses to predict when P/S/PKP/etc should arrive at a station, to compute the
take-off angle needed for a focal-mechanism / first-motion analysis, or to sanity-
check a located event's picks.

**No network.** Everything is analytic against bundled velocity models, so this
connector is fully offline and deterministic — the numerical counterpart to the
data-fetching catalogue connectors.

Design (shared with the sibling connectors):
- Distances in degrees (great-circle), depths in km positive-down.
- Every result echoes the `model` used, for provenance/reproducibility.
- Deterministic: identical inputs always yield identical times.

Run (stdio):  python server.py     (needs `pip install mcp` + obspy)
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

# Models shipped with obspy.taup (TauP .npz bundles).
_MODELS = ["iasp91", "ak135", "prem", "1066a", "1066b",
           "herrin", "jb", "pwdk", "sp6"]
_model_cache = {}

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("taup-velocity")
except ModuleNotFoundError:  # pragma: no cover
    class _NoMCP:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def run(self):
            raise SystemExit("Install 'mcp' to run as an MCP server; tools remain importable.")
    mcp = _NoMCP()


def _get_model(name: str):
    """Load (and cache) a TauPyModel by name."""
    from obspy.taup import TauPyModel
    key = name.lower()
    if key not in _model_cache:
        _model_cache[key] = TauPyModel(model=key)
    return _model_cache[key]


def _arrival_dict(a) -> dict:
    return {
        "phase": a.name,
        "time_s": round(a.time, 4),
        "ray_param_s_per_deg": round(a.ray_param_sec_degree, 5),
        "takeoff_angle_deg": round(a.takeoff_angle, 4),
        "incident_angle_deg": round(a.incident_angle, 4),
        "purist_distance_deg": round(a.purist_distance, 4),
    }


@mcp.tool()
def list_models() -> str:
    """List the bundled 1-D earth models available to this connector.

    Returns JSON {models, default}. All are offline TauP model bundles; 'iasp91' and
    'ak135' are the usual global-travel-time references.
    """
    return json.dumps({"models": _MODELS, "default": "iasp91"}, indent=2)


def _travel(m, depth_km, distance_deg, phases):
    """get_travel_times; if nothing comes back and 'P'/'S' were asked for, also try the
    local branch names 'p'/'s' (TauP calls the direct upgoing/turning phase p/s when the
    source is in the crust at short distance). Returns (arrivals, phases_used, note)."""
    arrivals = m.get_travel_times(source_depth_in_km=depth_km,
                                  distance_in_degree=distance_deg, phase_list=phases)
    note = None
    if not arrivals and any(x in ("P", "S") for x in phases):
        extra = [x for x in ("p", "s") if x not in phases]
        phases = phases + extra
        arrivals = m.get_travel_times(source_depth_in_km=depth_km,
                                      distance_in_degree=distance_deg, phase_list=phases)
        note = ("no arrivals for the requested phases; added local phases p,s "
                "(crustal source at short distance)")
    return arrivals, phases, note


@mcp.tool()
def tt_compute(
    distance_deg: float,
    depth_km: float,
    phase_list: str = "p,s,P,S",
    model: str = "iasp91",
) -> str:
    """Theoretical travel times for phases at a given distance & source depth.

    `phase_list` is comma-separated TauP phase names (e.g. 'p,s,P,S', 'P,pP,PcP,PKIKP'; use lower-case p,s for a crustal source within ~1-2 deg).
    `distance_deg` great-circle degrees; `depth_km` positive down. Returns JSON
    {model, distance_deg, depth_km, phase_list, n, arrivals:[{phase, time_s,
    ray_param_s_per_deg, takeoff_angle_deg, incident_angle_deg, purist_distance_deg}]}.
    Arrivals are ordered by arrival time; a phase absent at this geometry is omitted.
    """
    phases = [p.strip() for p in phase_list.split(",") if p.strip()]
    m = _get_model(model)
    arrivals, phases, note = _travel(m, depth_km, distance_deg, phases)
    arrs = sorted((_arrival_dict(a) for a in arrivals), key=lambda d: d["time_s"])
    out = {"model": model, "distance_deg": distance_deg, "depth_km": depth_km,
           "phase_list": phases, "n": len(arrs), "arrivals": arrs}
    if note:
        out["note"] = note
    return json.dumps(out, indent=2)


@mcp.tool()
def takeoff_angle(
    distance_deg: float,
    depth_km: float,
    phase_list: str = "p,s,P,S",
    model: str = "iasp91",
) -> str:
    """Take-off (and incidence) angles for phases — for focal-mechanism / first-motion work.

    Take-off angle is measured from the downward vertical at the source (0° = straight
    down, 90° = horizontal); incidence angle is measured at the receiver. Same params
    as tt_compute. Returns JSON {model, distance_deg, depth_km, arrivals:[{phase,
    takeoff_angle_deg, incident_angle_deg, ray_param_s_per_deg, time_s}]}.
    """
    phases = [p.strip() for p in phase_list.split(",") if p.strip()]
    m = _get_model(model)
    arrivals, phases, note = _travel(m, depth_km, distance_deg, phases)
    out = [{
        "phase": a.name,
        "takeoff_angle_deg": round(a.takeoff_angle, 4),
        "incident_angle_deg": round(a.incident_angle, 4),
        "ray_param_s_per_deg": round(a.ray_param_sec_degree, 5),
        "time_s": round(a.time, 4),
    } for a in arrivals]
    res = {"model": model, "distance_deg": distance_deg, "depth_km": depth_km,
           "arrivals": out}
    if note:
        res["note"] = note
    return json.dumps(res, indent=2)


if __name__ == "__main__":
    mcp.run()
