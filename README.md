# Seismology connectors and skills for Claude Science

Seven data and simulation connectors (MCP servers) and six skills for earthquake data. Early version, built to show what Earth-science support in Claude Science could look like.

## What is in it

| Piece | What it does | Needs |
|---|---|---|
| `connectors/fdsnws` | Events, stations and waveforms from any FDSN data center (IRIS/EarthScope, SCEDC, NCEDC, GFZ/GEOFON, ORFEUS, USGS, ... or any base URL). Waveforms are saved as miniSEED and a manifest is returned. | mcp, obspy |
| `connectors/comcat` | USGS ComCat catalog: count, search (paged, large results saved to a file), one event, event products (moment tensor, ShakeMap, ...). | mcp |
| `connectors/isc` | ISC Bulletin search and per-event multi-agency origins and magnitudes. | mcp, obspy |
| `connectors/syngine` | Synthetic seismograms for any moment tensor or double couple, anywhere on the globe (EarthScope Syngine: AxiSEM in 1-D Earth models, 2 s and longer). The quick "simulator". | mcp, obspy |
| `connectors/gcmt` | Global CMT moment-tensor catalog search. | mcp, obspy |
| `connectors/taup` | iasp91, ak135, prem travel times, ray parameters, take-off angles (TauP). | mcp, obspy |
| `connectors/gnss` | Nevada Geodetic Laboratory GNSS station lookup and daily time series. | mcp |
| `skills/waveform-download` | Bulk or regional (MassDownloader) waveform download with StationXML and a gap report. | obspy |
| `skills/phase-picking-ml` | PhaseNet / EQTransformer P and S picks (SeisBench). Falls back to STA/LTA if torch is absent, and tags every pick with the backend. | obspy, seisbench, torch |
| `skills/phase-picking-classic` | STA/LTA + AIC picker (used as the fallback above). | obspy |
| `skills/moment-tensor-waveform` | Regional moment-tensor inversion from waveforms with Syngine Green's functions; Mw, nodal planes, fit per station, Kagan angle to a reference. | obspy, numpy, matplotlib |
| `skills/locate-event` | Single-event location from P and S picks (iasp91 grid search, outlier rejection, bootstrap), auto mode with PhaseNet. | obspy, numpy (+ seisbench for auto) |
| `skills/catalog-statistics` | Mc, Gutenberg-Richter b-value (Aki-Utsu maximum likelihood, Shi-Bolt error), Mc cross-checks, b-positive. | numpy, pandas |

## Install in Claude Science (tested 2026-10-21, v0.1.54)

Claude Science runs each connector in a sandbox, which shapes the install:

- It starts the connector with its own Python (3.13, `mcp` already installed) and does **not** pass `PYTHONPATH`. `uv run` does not start inside the sandbox.
- It can only see files under system paths such as `/opt` or `/usr/local`. A file under your home or `~/.claude-science` fails with "not visible inside the MCP sandbox" (or grant it with `[sandbox] user_read_paths` in `config.toml`).
- A connector cannot write files anywhere visible. `fdsnws` therefore returns the waveform inline (base64 miniSEED) when it cannot write.

Steps:

1. Put the repo under a system path and install obspy for the app's Python into a folder next to it:

   ```
   D=/opt/homebrew/share/claude-science-geoscience
   mkdir -p $D && rsync -a connectors/ $D/connectors/
   uv pip install --python ~/.claude-science/conda/envs/claude-science-mcp/bin/python --target $D/site obspy
   ```

   `fdsnws` and `isc` add `$D/site` to `sys.path` themselves.

2. In Settings > Connectors > Add connector > Local command, add one entry per connector:

   ```
   name:    comcat   (and fdsnws, isc)
   command: ~/.claude-science/conda/envs/claude-science-mcp/bin/python /opt/homebrew/share/claude-science-geoscience/connectors/comcat/server.py
   ```

   Use the full path to the interpreter. Optional environment variables: `COMCAT_MCP_DATADIR`, `FDSNWS_MCP_DATADIR`, `ISC_MCP_CONTACT` (your email for the ISC User-Agent).

3. Settings > Skills: the folders in `skills/` appear as drafts if copied to the org's `skills/` folder (`~/.claude-science/orgs/<org>/skills/`); open each and press Publish. Alternatively add them from GitHub.

4. Settings > Network > Allowed domains: add `service.earthscope.org` (IRIS/EarthScope data and Syngine), `service.scedc.caltech.edu`, `service.ncedc.org`, `geofon.gfz.de`, `earthquake.usgs.gov`, `www.isc.ac.uk`, `hifis-storage.desy.de` (SeisBench downloads the PhaseNet weights from there), and for the two catalog connectors `www.ldeo.columbia.edu` (GCMT) and `geodesy.unr.edu` (GNSS). **The allowlist applies to the connectors as well as to code the agent runs**: a connector whose host is not listed returns nothing (the `gcmt` connector returned an empty result for July 2019 until the host was allowed; the app then asked for permission in the chat). NIED has no ObsPy short name; pass its FDSN base URL as `datacenter` if it offers open access.

5. After editing a connector file, stop its process (`pkill -f connectors/<name>/server.py`) and toggle the connector off and on in Settings.

Each connector also runs standalone: `uv run connectors/<name>/server.py` (dependencies install themselves).

## Test

```
uv run tests/test_connectors.py
```

Starts each connector over stdio exactly as an MCP host does, lists the tools and makes one real call (2019 Ridgecrest M7.1).

## Status (2026-10-21)

Tested on macOS, and inside Claude Science 0.1.54 (project "Seismology demo", Sonnet 5.5) with the one-prompt Ridgecrest task.

| Item | Result |
|---|---|
| comcat | In Claude Science: found the mainshock (ci38457511, 2019-07-06T03:19:53Z, Mw 7.1), counted 17,513 events within 60 km (4-14 Jul 2019), fetched them all |
| fdsnws | In Claude Science: `get_waveforms` for CI.CLC from SCEDC returned the miniSEED inline; identical to a direct ObsPy fetch (3 traces, 40 Hz, 4,801 samples, SHA-256 equal). `get_events` and `get_stations` also work outside the app |
| isc | Tools list in the app. **Live query not verified:** on 2026-10-21 the ISC server answered every query with an internal error ("unable to open ... query line 165") or HTTP 429. The connector reports both clearly. Re-test later |
| syngine | In Claude Science: synthetic seismograms for the ComCat and the inverted mechanism at CI.JRC2, returned inline. Outside the app: any moment tensor or double couple, any receiver; a synthetic record section for a Mw 7.0 tensor at 33 distances, 10 to 170 degrees |
| gcmt, taup, gnss | Tested outside the app. `gcmt` was used inside the app once the host `www.ldeo.columbia.edu` was allowed (it returned the Ridgecrest Mw 7.03 tensor); `taup` and `gnss` not yet used inside the app |
| moment-tensor-waveform | In the app, on the Mw 4.9 aftershock (11 stations, 0.04 to 0.1 Hz): Mw 4.82 vs ComCat 4.90, nodal planes [54,88,-6] [144,84,-178], Kagan angle 8.9 deg, variance reduction 0.83. On the Mac the Mw 5.37 aftershock gave Kagan 2.9 deg. The Mw 5.5 event inside the active sequence failed the signal-to-noise screen |
| locate-event | In the app, auto mode on the Mw 4.9 aftershock: 15 stations, 29 picks, RMS 0.22 s; 0.99 km from ComCat horizontally, origin time 0.73 s early, depth 21.7 km vs 9.5 km (unresolved: nearest station 27 km, 1-D iasp91) |
| Global waveforms (honest) | Observed IU long-period records against Syngine synthetics for the Mw 7.0 Ridgecrest GCMT tensor were a poor fit (12 stations). P timing is within about 3 s, but amplitudes are 2 to 8 times too large (point source, no source time function) and observed surface waves are 40 s or more late (1-D Earth). Not a result to show as a fit |
| waveform-download | Bulk download of HV.UWE and HV.SBL works (Mac). Not exercised inside the app |
| phase-picking-ml | In the app, a clean example: the Mw 4.9 aftershock of 2019-07-12 (ci38548295) at CI.JRC2, 43 km. PhaseNet (SeisBench 0.12.6, "original" weights) gave P at +7.49 s (p = 0.92) and S at +13.03 s (p = 0.82), within 0.17 s of iasp91; S-P 5.54 s observed vs 5.55 s predicted. A poor example, for contrast: the Mw 7.1 mainshock at CI.CLC (5 km) gave a P at +1.04 s but no S, and picks inside the coda are spurious (S arrives within the P coda and the record is dominated by the mainshock). Pick close-in, large events with care |
| catalog-statistics | In the app: Mc 1.6, b = 0.661 ± 0.006 (N = 8,690), the same as on the Mac. The app's own checks showed b is not stable: 0.51 to 1.12 for Mc 1.0 to 3.4, b-positive gives 0.905, and the other Mc estimators give 3.4 to 3.6. Treat the headline b as a demo number |

Not tested: a clean install on another machine, `waveform-download` and `phase-picking-classic` inside the app, `taup` and `gnss` inside the app, and `isc` against a working ISC server.

## Notes

- `mcp` is pinned below 2.0 because the 2.x SDK renamed `FastMCP`.
- ComCat is fetched in pages of 500 events with retries, because long single responses were cut off mid-transfer.
- Times are ISO-8601 UTC everywhere, and every result carries the exact request so it can be reproduced.

## Contributing

This is a first set for seismology. Connectors and skills for other Earth and environmental science fields (geodesy, geochemistry, hydrology, planetary science, ...) are welcome: open an issue or a pull request. Each connector is one file and each skill is one folder with a `SKILL.md`.

## License

MIT, see `LICENSE`.
