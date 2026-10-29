"""Stage 1 & 2: analyse a corpus library and a target song into unit databases.

    python -m cgt.analyze corpus library/ --out work/corpus.npz
    python -m cgt.analyze target "song/02 Variatio 1 a 1 Clav.flac" --out work/target.npz

Per-track results are cached (keyed on path, size, mtime and segmentation
params), so re-running after adding an album only analyses the new files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import audio, features

UNIT_KEYS = ["start", "dur", "chroma", "mfcc", "loud", "bright", "noisy", "perc", "attack", "env"]


def _cache_key(path: Path, min_dur: float, max_dur: float) -> str:
    st = path.stat()
    s = f"{path.resolve()}|{st.st_size}|{st.st_mtime}|{min_dur}|{max_dur}|v1"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def analyze_file(path: str, min_dur: float, max_dur: float, cache_dir: str | None) -> tuple[str, dict, float]:
    p = Path(path)
    cache = Path(cache_dir) / f"{_cache_key(p, min_dur, max_dur)}.npz" if cache_dir else None
    if cache and cache.exists():
        d = np.load(cache)
        return path, {k: d[k] for k in UNIT_KEYS}, float(d["length"])
    y = audio.load(p, features.SR, mono=True)
    units = features.analyze(y, min_dur, max_dur)
    length = len(y) / features.SR
    if cache:
        np.savez(cache, length=length, **units)
    return path, units, length


def build(paths: list[Path], out: Path, min_dur: float, max_dur: float, workers: int, cache_dir: Path | None) -> None:
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[dict, float]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_file, str(p), min_dur, max_dur, str(cache_dir) if cache_dir else None): p for p in paths}
        for k, f in enumerate(as_completed(futs), 1):
            path, units, length = f.result()
            results[path] = (units, length)
            print(f"  [{k}/{len(paths)}] {len(units['start']):5d} units  {length/60:5.1f} min  {Path(path).name}  ({time.time()-t0:.0f}s)", flush=True)

    tracks, cols = [], {k: [] for k in UNIT_KEYS}
    track_idx, contig = [], []
    for ti, p in enumerate(paths):
        units, length = results[str(p)]
        tracks.append({"path": str(p), "length": length})
        n = len(units["start"])
        for k in UNIT_KEYS:
            cols[k].append(units[k])
        track_idx.append(np.full(n, ti))
        # contig[i]: unit i+1 directly continues unit i in the same track
        c = np.zeros(n, bool)
        if n > 1:
            c[:-1] = np.isclose(units["start"][:-1] + units["dur"][:-1], units["start"][1:])
        contig.append(c)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["track"] = np.concatenate(track_idx)
    data["contig"] = np.concatenate(contig)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tracks=json.dumps(tracks), **data)
    total = sum(t["length"] for t in tracks)
    print(f"wrote {out}: {len(data['start'])} units from {len(tracks)} files ({total/3600:.2f} h)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["corpus", "target"])
    ap.add_argument("src", help="directory (searched recursively) or a single audio file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dur", type=float, default=0.06, help="shortest unit in seconds")
    ap.add_argument("--max-dur", type=float, default=None, help="split longer units (default 1.0 corpus / 0.75 target)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--cache", default="work/cache", help="per-track cache dir ('' to disable)")
    a = ap.parse_args(argv)

    paths = audio.find_audio(a.src)
    if not paths:
        sys.exit(f"no audio found in {a.src}")
    if a.kind == "target" and len(paths) != 1:
        sys.exit(f"target must be a single file; found {len(paths)} in {a.src}")
    max_dur = a.max_dur or (1.0 if a.kind == "corpus" else 0.75)
    print(f"analysing {len(paths)} file(s) as {a.kind} (units {a.min_dur}-{max_dur}s)")
    build(paths, Path(a.out), a.min_dur, max_dur, a.workers, Path(a.cache) if a.cache else None)


if __name__ == "__main__":
    main()
