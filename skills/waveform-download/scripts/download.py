#!/usr/bin/env python3
"""
download.py — bulk FDSN waveform download via ObsPy, two ways:

  method="bulk"          : explicit station list -> Client.get_waveforms_bulk
                           (fast, deterministic; you know exactly what you asked for)
  method="massdownloader": geographic domain + Restrictions -> MassDownloader
                           (discovers stations in a region/time window, saves
                           miniSEED + matching StationXML, resumable)

Both return a MANIFEST (files, byte sizes, per-trace gap report, station list,
the exact request) rather than raw samples, so results are reproducible and the
agent context stays small — the same data-by-reference discipline as the fdsnws
connector.

House rules honoured: times ISO-8601 UTC only; a non-empty gap list is surfaced
as a data-quality flag; every product carries its request parameters.

Usage
-----
  # explicit stations (bulk)
  python download.py --bulk --datacenter IRIS \\
      --net HV --stations UWE,SBL,BYL --channel HHZ \\
      --start 2018-05-04T22:32:00 --end 2018-05-04T22:34:00 --outdir ./wf

  # regional discovery (mass_downloader), circular domain
  python download.py --massdownloader --datacenter IRIS \\
      --lat 19.40 --lon -155.28 --maxradius-km 30 \\
      --net HV --channel-priorities HHZ,BHZ \\
      --start 2018-05-04T22:32:00 --end 2018-05-04T22:34:00 --outdir ./wf
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _utc(s):
    from obspy import UTCDateTime
    return UTCDateTime(s)


def _gap_report(st):
    return [[g[0], g[1], g[2], g[3], str(g[4]), str(g[5]), g[6]] for g in st.get_gaps()]


def download_bulk(datacenter, network, stations, channel, start, end,
                  location="*", outdir="./wf"):
    """Download an explicit station list in one bulk request; write per-station mseed."""
    from obspy.clients.fdsn import Client
    os.makedirs(outdir, exist_ok=True)
    t0, t1 = _utc(start), _utc(end)
    bulk = [(network, s.strip(), location, channel, t0, t1)
            for s in stations if s.strip()]
    cl = Client(datacenter)
    st = cl.get_waveforms_bulk(bulk)

    files = []
    for sta in sorted({tr.stats.station for tr in st}):
        sub = st.select(station=sta)
        fn = os.path.join(outdir, f"{network}.{sta}.{channel}"
                                  f"_{start}_{end}.mseed".replace(":", ""))
        sub.write(fn, format="MSEED")
        files.append({"file": fn, "bytes": os.path.getsize(fn),
                      "station": sta, "n_traces": len(sub)})
    gaps = _gap_report(st)
    return {
        "method": "bulk", "datacenter": datacenter,
        "n_traces": len(st), "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "stations": sorted({tr.stats.station for tr in st}),
        "sampling_rates": sorted({tr.stats.sampling_rate for tr in st}),
        "n_gaps": len(gaps), "gaps": gaps,
        "time_span": [str(min(tr.stats.starttime for tr in st)),
                      str(max(tr.stats.endtime for tr in st))] if len(st) else None,
        "files": files,
        "request": {"network": network, "stations": stations, "channel": channel,
                    "location": location, "start": start, "end": end},
    }


def download_massdownloader(datacenter, lat, lon, maxradius_km, start, end,
                            network="*", channel_priorities=("HH[ZNE]", "BH[ZNE]"),
                            location_priorities=("", "00", "10"),
                            outdir="./wf", minradius_km=0.0,
                            min_interstation_distance_km=0.0):
    """Regional discovery download via MassDownloader (circular domain).

    `min_interstation_distance_km` thins dense arrays (mass_downloader keeps the
    first station and drops any nearer than this to one already chosen) — a simple
    way to bound how many stations a wide domain pulls.
    """
    from obspy.clients.fdsn.mass_downloader import (
        CircularDomain, Restrictions, MassDownloader)
    from obspy.geodetics import kilometers2degrees

    wf_dir = os.path.join(outdir, "waveforms")
    xml_dir = os.path.join(outdir, "stations")
    domain = CircularDomain(latitude=lat, longitude=lon,
                            minradius=kilometers2degrees(minradius_km),
                            maxradius=kilometers2degrees(maxradius_km))
    restr = Restrictions(
        starttime=_utc(start), endtime=_utc(end),
        network=network, reject_channels_with_gaps=False,
        minimum_length=0.0,
        channel_priorities=list(channel_priorities),
        location_priorities=list(location_priorities),
        minimum_interstation_distance_in_m=min_interstation_distance_km * 1000.0,
    )
    mdl = MassDownloader(providers=[datacenter])
    mdl.download(domain, restr, mseed_storage=wf_dir, stationxml_storage=xml_dir)

    mseed_files = sorted(glob.glob(os.path.join(wf_dir, "*")))
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*")))
    manifest_files = [{"file": f, "bytes": os.path.getsize(f)}
                      for f in mseed_files if os.path.isfile(f)]
    return {
        "method": "massdownloader", "datacenter": datacenter,
        "n_mseed_files": len(manifest_files),
        "n_stationxml_files": len([f for f in xml_files if os.path.isfile(f)]),
        "total_bytes": sum(f["bytes"] for f in manifest_files),
        "waveform_dir": wf_dir, "stationxml_dir": xml_dir,
        "files": manifest_files,
        "request": {"lat": lat, "lon": lon, "maxradius_km": maxradius_km,
                    "network": network, "start": start, "end": end,
                    "channel_priorities": list(channel_priorities)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bulk", action="store_true", help="explicit station list")
    mode.add_argument("--massdownloader", action="store_true",
                      help="regional discovery download")
    ap.add_argument("--datacenter", default="IRIS")
    ap.add_argument("--start", required=True, help="ISO-8601 UTC start")
    ap.add_argument("--end", required=True, help="ISO-8601 UTC end")
    ap.add_argument("--outdir", default="./wf")
    # bulk
    ap.add_argument("--net", default="*")
    ap.add_argument("--stations", help="comma list (bulk mode)")
    ap.add_argument("--channel", default="HHZ")
    ap.add_argument("--location", default="*")
    # massdownloader
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--maxradius-km", type=float)
    ap.add_argument("--minradius-km", type=float, default=0.0)
    ap.add_argument("--channel-priorities", default="HH[ZNE],BH[ZNE]")
    ap.add_argument("--min-interstation-km", type=float, default=0.0,
                    help="thin dense arrays: drop stations closer than this (km)")
    args = ap.parse_args(argv)

    if args.bulk:
        if not args.stations:
            raise SystemExit("--stations required for --bulk")
        res = download_bulk(args.datacenter, args.net, args.stations.split(","),
                            args.channel, args.start, args.end, args.location,
                            args.outdir)
    else:
        for req in ("lat", "lon", "maxradius_km"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_','-')} required for --massdownloader")
        res = download_massdownloader(
            args.datacenter, args.lat, args.lon, args.maxradius_km,
            args.start, args.end, network=args.net,
            channel_priorities=tuple(args.channel_priorities.split(",")),
            outdir=args.outdir, minradius_km=args.minradius_km,
            min_interstation_distance_km=args.min_interstation_km)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
