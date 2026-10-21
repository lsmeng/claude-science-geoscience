---
name: phase-picking-ml
description: >-
  Deep-learning seismic phase picking with SeisBench — PhaseNet and EQTransformer
  pretrained models — turning raw 3-component waveforms into P/S picks with
  probabilities. Use whenever the user wants ML/neural phase picking, PhaseNet or
  EQTransformer, or high-recall automatic picks for building a catalog. Portable:
  runs the real model when seisbench+torch are installed, and falls back to the
  classic STA/LTA picker (phase-picking-classic) when they are not, so it always
  returns picks. Prefer this over phase-picking-classic when a trained model is available.
license: MIT
category: geoscience
---

# phase-picking-ml

Runs a SeisBench `WaveformModel` (PhaseNet / EQTransformer) over an ObsPy Stream and
emits picks as a CSV. Feed it 3-component (Z,N,E) data — the ML models need all three;
a lone vertical usually yields no ML pick (the fallback still handles one component).

## Method (and why)

- **PhaseNet** (Zhu & Beroza 2019) / **EQTransformer** (Mousavi 2020) via
  `seisbench.models`, loaded with `.from_pretrained("original")`. `annotate()` gives
  continuous P/S probability traces; `classify()` thresholds them into discrete picks
  with a `peak_time` (the pick) and `peak_value` (probability).
- **Portability guard + fallback.** `pick()` never hard-fails on a missing dependency:
  if `import seisbench` fails it prints
  `requires seisbench+torch (pip install seisbench torch)` and falls back to
  `phase-picking-classic`'s STA/LTA+AIC picker. Every pick carries a `backend` tag
  (`phasenet` | `eqtransformer` | `classic-fallback`) so provenance is explicit.
- **OpenMP coexistence.** torch and ObsPy/numpy can each link an OpenMP runtime; on
  macOS that triggers `OMP: Error #15`. The module sets `KMP_DUPLICATE_LIB_OK=TRUE`
  before importing seisbench so the two coexist.

## Sandbox note (Claude Science)

SeisBench caches model weights in `~/.seisbench`, which a sandbox may block
(`PermissionError` on import). Set `SEISBENCH_CACHE_ROOT` to a writable folder (for example
the workspace) before importing `seisbench`. The weights are downloaded on first use, so the
host serving them (`hifis-storage.desy.de`) must be reachable.

## Usage

```bash
python scripts/phasenet_pick.py --mseed 3comp.mseed --model phasenet --out picks.csv
python scripts/phasenet_pick.py --mseed 3comp.mseed --model eqtransformer \
    --p-thr 0.3 --s-thr 0.3
```

Importable API: `pick(stream, model="phasenet", p_thr=…, s_thr=…) -> (picks, backend)`;
`annotate(stream)` (probability traces), `classify(stream)` (picks),
`picks_to_csv(picks, path)`. CSV columns:
`network,station,location,channel,phase,time,probability,backend`.

## Verified

Tested: **seisbench 0.11.7 + torch 2.12.0 are present in
this env**, so the test runs **real PhaseNet** on 3-component **CI.CLC** for the 2019
Ridgecrest M7.1 mainshock (via the `fdsnws`/SCEDC): PhaseNet returns a **P pick at
+1.06 s after origin, probability 0.786** — inside the expected P window and sharper
than the classic STA/LTA trigger (+2.0 s). It also verifies the **classic fallback**
returns picks + a valid CSV. → **VERIFIED (real PhaseNet).** In a torch-less
environment the same test degrades to SCAFFOLD mode (guard message + fallback only).

## Reviewer gotchas respected

- **3-component input required** for PhaseNet/EQTransformer; document when only Z is
  available (ML recall drops / no picks) and note the fallback took over.
- **Probability is model confidence, not detection truth** — always report the
  `probability` and the `pretrained` weights used; different weights give different
  picks. State the model+weights in provenance.
- **Resampling.** SeisBench resamples internally to the model rate; keep the original
  sampling rate in the manifest so pick times stay reproducible. Times are absolute UTC.
- **Backend is always tagged** so a downstream reviewer/associator can tell a
  0.79-probability PhaseNet pick from a fallback STA/LTA trigger.

## Later (not yet implemented)

GPU batching for continuous data, EQTransformer detection windows + associator handoff,
model ensembling / picker blending, uncertainty from the probability curve width, and
fine-tuned regional weights. These fold in once a GPU box + `seisbench` model zoo is
wired to the workbench.
