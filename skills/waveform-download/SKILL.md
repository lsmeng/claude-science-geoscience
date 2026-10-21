---
name: waveform-download
description: >-
  Bulk FDSN waveform download via ObsPy — an explicit station list
  (Client.get_waveforms_bulk) or regional discovery over a geographic domain
  (MassDownloader, saving miniSEED + matching StationXML, resumable). Returns a
  manifest with files, sizes, gap report and the exact request. Use whenever the
  user needs to fetch continuous waveforms for an event or region. Pure ObsPy.
license: MIT
category: geoscience
---

# waveform-download

Two complementary download paths, both returning a data-by-reference manifest so
the agent context stays small and every product is reproducible.

## Method (and why)

- **`bulk`** — you name the stations; one `get_waveforms_bulk` request pulls them.
  Fast and deterministic: you know exactly what you asked for. Best when the
  station set is already decided (e.g. from a `get_stations` search).
- **`massdownloader`** — you name a **domain** (circular here: lat/lon + radius) and
  a time window; `MassDownloader` discovers the stations, downloads miniSEED **and
  the matching StationXML** (needed for later response removal), and is **resumable**
  (skips already-downloaded pieces). `min_interstation_distance_km` thins dense
  arrays to bound how many stations a wide domain pulls.
- **Gaps surfaced.** The bulk manifest reports a per-trace gap list — a non-empty
  gap list is a data-quality flag the downstream picking/preprocess skills must see.
- **UTC only**, sizes in bytes, sampling rate in Hz; the exact `request` travels
  with the manifest.

## Usage

```bash
# explicit stations
python scripts/download.py --bulk --datacenter IRIS \
    --net HV --stations UWE,SBL,BYL --channel HHZ \
    --start 2018-05-04T22:32:00 --end 2018-05-04T22:34:00 --outdir ./wf

# regional discovery (circular domain), saves mseed + StationXML
python scripts/download.py --massdownloader --datacenter IRIS \
    --lat 19.40 --lon -155.28 --maxradius-km 30 \
    --net HV --channel-priorities HHZ,BHZ --min-interstation-km 5 \
    --start 2018-05-04T22:32:00 --end 2018-05-04T22:34:00 --outdir ./wf
```

Importable: `download_bulk(...)`, `download_massdownloader(...)`. This pairs with
the the `fdsnws` connector's `get_stations` (pick a station set) and hands its
output to `waveform-preprocess`.

## Verified

Tested by downloading a **tiny real window** (2018 Kilauea, ~2 min, HV network)
both ways and **cleans up after itself**:
- **bulk**: HV UWE/SBL/BYL HHZ -> 3 files, ~107 KB, 100 Hz, 0 gaps; manifest byte
  sizes match the files on disk.
- **massdownloader**: circular domain around the summit, HHZ, 8 km thinning ->
  3 miniSEED + 3 matching StationXML files.

## Reviewer gotchas

- **Keep windows small.** Continuous data is huge; request only what you need and
  chunk long spans. The smoke test deliberately uses 2 minutes.
- **StationXML travels with mass_downloader** but NOT with a bare bulk request —
  fetch the inventory (`get_stations level=response`) before response removal.
- **Data centre matters.** HV/regional data may live at IRIS/EarthScope, SCEDC
  (CI), or NCEDC — state the `datacenter`; a wrong archive returns nothing, not an
  error.
- **Gaps are expected** in real continuous data; the manifest flags them so they
  are handled (not silently zero-filled) downstream.
- **`location`/channel wildcards** can multiply the download; be explicit for tight
  requests.

## Later

Rectangular/global domains, event-based `Restrictions` (origin-time windows,
distance/azimuth), multi-provider fan-out with de-duplication, and automatic
chunking of multi-day requests.
