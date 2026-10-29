#!/usr/bin/env python3
"""Coarse-grain spectral/granular style transfer: rebuild a song from a corpus of other music.

Runs all four stages; each can also be run (and re-run with other settings) alone:

    python -m cgt.analyze corpus library/ --out work/corpus.npz     # 1. corpus units + descriptors (cached per track)
    python -m cgt.analyze target song/    --out work/target.npz      # 2. target units + descriptors
    python -m cgt.match  --corpus work/corpus.npz --target work/target.npz --out work/plan.json   # 3. unit selection
    python -m cgt.synth  --plan work/plan.json --out output/mosaic.wav                           # 4. render

Examples
    .venv/bin/python run.py                                  # library/ + song/ -> output/<song>.wav
    .venv/bin/python run.py --match "--jump-cost 2"          # longer Four Tet phrases
    .venv/bin/python run.py --match "--jump-cost 0 --jitter 0.3 --seed 7" --synth "--morph 0.7"
    .venv/bin/python run.py --from match                     # reuse analysis, redo selection + render
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from cgt import analyze, audio, match, synth

STAGES = ["corpus", "target", "match", "synth"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="library")
    ap.add_argument("--song", default="song", help="target file, or a directory holding exactly one")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default=None, help="output wav (default output/<song>__mosaic.wav)")
    ap.add_argument("--from", dest="start", choices=STAGES, default="corpus", help="first stage to run")
    ap.add_argument("--to", dest="stop", choices=STAGES, default="synth", help="last stage to run")
    ap.add_argument("--analyze", default="", help="extra args for cgt.analyze (both corpus and target)")
    ap.add_argument("--match", default="", help="extra args for cgt.match, quoted")
    ap.add_argument("--synth", default="", help="extra args for cgt.synth, quoted")
    a = ap.parse_args()

    song = audio.find_audio(a.song)[0]
    work = Path(a.work)
    out = a.out or f"output/{song.stem}__mosaic.wav"
    cache = ["--cache", str(work / "cache")]
    steps = {
        "corpus": lambda: analyze.main(["corpus", a.library, "--out", str(work / "corpus.npz"), *cache, *shlex.split(a.analyze)]),
        "target": lambda: analyze.main(["target", str(song), "--out", str(work / "target.npz"), *cache, *shlex.split(a.analyze)]),
        "match": lambda: match.main(["--corpus", str(work / "corpus.npz"), "--target", str(work / "target.npz"),
                                     "--out", str(work / "plan.json"), *shlex.split(a.match)]),
        "synth": lambda: synth.main(["--plan", str(work / "plan.json"), "--out", out, *shlex.split(a.synth)]),
    }
    for s in STAGES[STAGES.index(a.start):STAGES.index(a.stop) + 1]:
        print(f"\n== {s} ==")
        steps[s]()


if __name__ == "__main__":
    main()
