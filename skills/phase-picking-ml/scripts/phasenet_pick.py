#!/usr/bin/env python3
"""
phasenet_pick.py — deep-learning phase picking via SeisBench (PhaseNet / EQTransformer).

Portable: runs the real SeisBench model when seisbench+torch are installed, and
degrades gracefully to the classic ObsPy picker when they are NOT — so a caller
gets picks either way and the skill is safe to ship into a torch-less environment.

  1. Tries to import seisbench; if absent, `pick()` prints a clear
     'requires seisbench+torch (pip install seisbench torch)' guard message and
     FALLS BACK to the STA/LTA picker in the sibling `phase-picking-classic` skill.
  2. Exposes the intended API — annotate() (continuous P/S probabilities) and
     classify()/pick() (discrete picks) — driven by a SeisBench WaveformModel.
  3. Every pick is tagged with the backend used
     ('phasenet'|'eqtransformer'|'classic-fallback') so provenance is never lost.

NB: PhaseNet/EQTransformer expect 3-component (Z,N,E) input; a lone Z trace usually
yields no ML picks (the classic fallback still works on one component).

Usage:
  python phasenet_pick.py --mseed 3comp.mseed --model phasenet --out picks.csv
  python phasenet_pick.py --mseed 3comp.mseed --model eqtransformer --p-thr 0.3 --s-thr 0.3
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

# torch (via seisbench) and ObsPy/numpy can both drag in an OpenMP runtime; on macOS
# that trips "OMP: Error #15 ... libomp already initialized". This unblocks the
# duplicate-runtime check so the two coexist. Set before any seisbench/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Optional heavy deps — the whole point of the SCAFFOLD guard.
# ---------------------------------------------------------------------------
GUARD_MSG = ("requires seisbench+torch (pip install seisbench torch) — "
             "falling back to the classic STA/LTA picker (phase-picking-classic)")

try:
    import seisbench.models as sbm  # noqa: F401
    HAVE_SEISBENCH = True
except Exception:
    HAVE_SEISBENCH = False


def _load_classic():
    """Import the phase-picking-classic picker (deterministic fallback)."""
    here = os.path.dirname(os.path.abspath(__file__))
    classic = os.path.abspath(os.path.join(
        here, "..", "..", "phase-picking-classic", "scripts"))
    if classic not in sys.path:
        sys.path.insert(0, classic)
    import sta_lta_pick as slp
    return slp


# ---------------------------------------------------------------------------
# Intended SeisBench API (real path) + classic fallback (this environment)
# ---------------------------------------------------------------------------
_MODEL_NAMES = {"phasenet": "PhaseNet", "eqtransformer": "EQTransformer"}


def _get_model(model="phasenet", pretrained="original"):
    """Return a loaded SeisBench WaveformModel. Raises if seisbench absent."""
    if not HAVE_SEISBENCH:
        raise RuntimeError(GUARD_MSG)
    import seisbench.models as sbm
    cls = getattr(sbm, _MODEL_NAMES[model])
    return cls.from_pretrained(pretrained)


def annotate(stream, model="phasenet", pretrained="original"):
    """Real path: continuous P/S probability traces from a SeisBench model.

    Returns an ObsPy Stream of *_P / *_S probability traces. Requires seisbench+torch.
    """
    mdl = _get_model(model, pretrained)
    return mdl.annotate(stream)


def classify(stream, model="phasenet", pretrained="original",
             p_thr=0.3, s_thr=0.3):
    """Real path: discrete picks from a SeisBench model. Requires seisbench+torch.

    Returns a list of pick dicts (same schema as pick()).
    """
    mdl = _get_model(model, pretrained)
    out = mdl.classify(stream, P_threshold=p_thr, S_threshold=s_thr)
    picks = []
    for p in getattr(out, "picks", out):
        # trace_id is "NET.STA.LOC" (SeisBench drops the component for a pick)
        tid = (getattr(p, "trace_id", "") or "").split(".")
        net = tid[0] if len(tid) > 0 else None
        sta = tid[1] if len(tid) > 1 else None
        loc = tid[2] if len(tid) > 2 else None
        peak = getattr(p, "peak_time", None) or getattr(p, "start_time", None)
        val = getattr(p, "peak_value", None)
        picks.append({
            "network": net, "station": sta, "location": loc, "channel": None,
            "phase": getattr(p, "phase", "?"),
            "time": str(peak) if peak is not None else None,
            "probability": (round(float(val), 4) if val is not None else None),
            "backend": model,
        })
    return picks


def pick(stream, model="phasenet", pretrained="original", p_thr=0.3, s_thr=0.3):
    """High-level entry: return picks from the ML model, or the classic fallback.

    Never raises on a missing dependency — if seisbench+torch are absent it prints
    the guard message and returns classic STA/LTA + AIC picks tagged
    'classic-fallback' so downstream association/location still gets picks.
    """
    if HAVE_SEISBENCH:
        return classify(stream, model=model, pretrained=pretrained,
                        p_thr=p_thr, s_thr=s_thr), model
    # ---- fallback ----
    print(f"[phasenet_pick] {GUARD_MSG}", file=sys.stderr)
    slp = _load_classic()
    picks = slp.pick_stream(stream, method="recursive", refine=True)
    for p in picks:
        p["probability"] = None
        p["backend"] = "classic-fallback"
    return picks, "classic-fallback"


def picks_to_csv(picks, path=None):
    """Write/return picks as CSV (columns: network,station,location,channel,phase,
    time,probability,backend)."""
    cols = ["network", "station", "location", "channel", "phase", "time",
            "probability", "backend"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for p in picks:
        w.writerow({c: p.get(c) for c in cols})
    text = buf.getvalue()
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mseed", required=True, help="miniSEED file")
    ap.add_argument("--model", default="phasenet", choices=["phasenet", "eqtransformer"])
    ap.add_argument("--pretrained", default="original", help="SeisBench weights name")
    ap.add_argument("--p-thr", type=float, default=0.3)
    ap.add_argument("--s-thr", type=float, default=0.3)
    ap.add_argument("--out", help="write picks CSV to this path")
    args = ap.parse_args(argv)

    import obspy
    st = obspy.read(args.mseed)
    picks, backend = pick(st, model=args.model, pretrained=args.pretrained,
                          p_thr=args.p_thr, s_thr=args.s_thr)
    csv_text = picks_to_csv(picks, args.out)
    print(json.dumps({
        "backend": backend,
        "seisbench_available": HAVE_SEISBENCH,
        "n_picks": len(picks),
        "out": args.out,
        "note": None if HAVE_SEISBENCH else GUARD_MSG,
    }, indent=2))
    if not args.out:
        print(csv_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
