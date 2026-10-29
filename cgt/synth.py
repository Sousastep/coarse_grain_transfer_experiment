"""Stage 4: render a plan into audio the exact length of the target song.

    python -m cgt.synth --plan work/plan.json --out output/mosaic.wav

1. Concatenation: each selected corpus grain is read from its source onset,
   resampled for its transposition (varispeed, so pitch and speed move
   together), gain-matched to the target unit's loudness and placed at the
   target unit's onset, with smoothstep fades and a fade-out tail overlapping the next grain.
2. Spectral envelope morph (optional, off by default, `--morph`): frame by frame, the mosaic's
   mel-band envelope is pulled toward the target's by a fraction `--morph`
   (0 = raw mosaic, 1 = full envelope imprint). This is the "spectral" half:
   it tightens note timing, dynamics and register beyond what grain choice
   alone achieves, while the fine structure (timbre) stays Four Tet's.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import soxr

from . import audio

SR = 44100


def smoothstep(x: np.ndarray, order: int) -> np.ndarray:
    """Generalised smoothstep S_N (N=1 cubic, N=2 'smootherstep', N=3 7th-degree); N=0 is linear."""
    s = sum(comb(order + k, k) * comb(2 * order + 1, order - k) * (-x) ** k for k in range(order + 1))
    return x ** (order + 1) * s


def place_grains(plan: dict, n: int, tail: float, fade_in: float, fade_order: int,
                 loud_follow: float, max_boost: float) -> np.ndarray:
    out = np.zeros((n, 2), np.float32)
    by_track = defaultdict(list)
    for e in plan["units"]:
        by_track[e["track"]].append(e)
    tail_n = int(tail * SR)
    for k, (ti, entries) in enumerate(sorted(by_track.items()), 1):
        path = plan["tracks"][ti]
        src = audio.load(path, SR, mono=False)
        print(f"  [{k}/{len(by_track)}] {len(entries):4d} grains from {Path(path).name}", flush=True)
        for e in entries:
            o0 = int(round(e["t_start"] * SR))
            if o0 >= n:
                continue
            body = min(int(round(e["t_dur"] * SR)), n - o0)
            length = min(body + tail_n, n - o0)
            r = 2.0 ** (e["transpose"] / 12)
            s0 = int(round(e["src_start"] * SR))
            seg = src[s0:s0 + int(np.ceil(length * r)) + 64]
            if len(seg) < 16:
                continue
            if r != 1.0:
                seg = soxr.resample(seg, SR * r, SR)
            g = np.zeros((length, 2), np.float32)
            g[:min(length, len(seg))] = seg[:length]
            # gain-match the grain body to the target unit loudness
            g_db = 20 * np.log10(np.sqrt(np.mean(g[:body] ** 2)) + 1e-8)
            gain_db = np.clip((e["t_loud"] - g_db) * loud_follow, -40, max_boost)
            g *= 10 ** (gain_db / 20)
            fi = min(int(fade_in * SR), body)
            g[:fi] *= smoothstep(np.linspace(0, 1, fi), fade_order)[:, None]
            fo = length - body
            if fo <= 0:  # grain clipped by the end of the song: short fade inside the body
                fo = min(int(0.005 * SR), length)
            g[length - fo:] *= smoothstep(np.linspace(1, 0, fo), fade_order)[:, None]
            out[o0:o0 + length] += g
        del src
    return out


def spectral_morph(mosaic: np.ndarray, target: np.ndarray, amount: float, max_db: float, n_bands: int) -> np.ndarray:
    n_fft, hop = 2048, 512
    M = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_bands)
    Xt = librosa.stft(target.mean(1), n_fft=n_fft, hop_length=hop)
    Xo = librosa.stft(mosaic.T, n_fft=n_fft, hop_length=hop)  # (2, bins, frames)
    Et = M @ (np.abs(Xt) ** 2)
    Eo = M @ (np.abs(Xo) ** 2).mean(0)
    floor = 1e-9 * max(Et.max(), Eo.max())
    g_db = 10 * amount * np.log10((Et + floor) / (Eo + floor))
    g_db = np.clip(g_db, -max_db, max_db)
    g_db = (np.roll(g_db, 1, 1) + 2 * g_db + np.roll(g_db, -1, 1)) / 4  # light temporal smoothing
    norm = M.sum(0) + 1e-9
    g_bin = 10 ** ((M.T @ g_db) / norm[:, None] / 20)
    Y = Xo * g_bin[None]
    return librosa.istft(Y, hop_length=hop, n_fft=n_fft, length=len(mosaic)).T.astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="work/plan.json")
    ap.add_argument("--out", default="output/mosaic.wav")
    ap.add_argument("--tail", type=float, default=0.06, help="fade-out: seconds each grain rings past the next onset")
    ap.add_argument("--fade-in", type=float, default=0.008, help="grain fade-in seconds")
    ap.add_argument("--fade-order", type=int, default=3, help="smoothstep order for fades (0 linear, 1 cubic, 3 = 7th-degree)")
    ap.add_argument("--loud-follow", type=float, default=0.85, help="how strongly grain gain tracks target loudness (0-1)")
    ap.add_argument("--max-boost", type=float, default=18.0, help="max gain (dB) applied to a quiet grain")
    ap.add_argument("--morph", type=float, default=0.0, help="spectral envelope morph amount (0 = off; >0 lets the target bleed through)")
    ap.add_argument("--morph-max-db", type=float, default=24.0)
    ap.add_argument("--morph-bands", type=int, default=40)
    ap.add_argument("--dry", type=float, default=0.0, help="mix in the original target (0-1)")
    ap.add_argument("--save-raw", action="store_true", help="also write the un-morphed mosaic as *_raw.wav")
    a = ap.parse_args(argv)

    plan = json.loads(Path(a.plan).read_text())
    target = audio.load(plan["target"]["path"], SR, mono=False)
    n = len(target)
    print(f"rendering {len(plan['units'])} grains into {n / SR:.2f}s")
    mix = place_grains(plan, n, a.tail, a.fade_in, a.fade_order, a.loud_follow, a.max_boost)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if a.save_raw:
        raw = mix / (np.abs(mix).max() + 1e-9) * 0.89
        sf.write(out_path.with_name(out_path.stem + "_raw.wav"), raw, SR, subtype="PCM_24")
    if a.morph > 0:
        print(f"  spectral morph {a.morph}")
        mix = spectral_morph(mix, target, a.morph, a.morph_max_db, a.morph_bands)
    if a.dry > 0:
        mix = (1 - a.dry) * mix / (np.abs(mix).max() + 1e-9) + a.dry * target / (np.abs(target).max() + 1e-9)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89  # -1 dBFS
    assert len(mix) == n
    sf.write(out_path, mix, SR, subtype="PCM_24")
    print(f"wrote {out_path} ({len(mix) / SR:.3f}s, target {n / SR:.3f}s)")


if __name__ == "__main__":
    main()
