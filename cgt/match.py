"""Stage 3: unit selection. For every target unit pick a corpus unit (+ transposition).

    python -m cgt.match --corpus work/corpus.npz --target work/target.npz --out work/plan.json

Follows the data-driven unit-selection formulation of Schwarz (CATERPILLAR,
DAFx 2000/2003): total cost = sum of target costs (descriptor distance between
target unit and corpus unit) + concatenation costs (between consecutive
selections), minimised with Viterbi over a top-K candidate lattice.

The concatenation cost is zero when the next selection is the unit that
directly follows the previous one in its source track (Zils & Pachet's "title
continuity"), so `--jump-cost` controls the grain scale: low values give a
note-by-note glitch mosaic, high values let whole Four Tet phrases run on as
long as they keep tracking the target.

Transposition: corpus units may be resampled by +/- `--max-transpose`
semitones; the chroma vector is rotated accordingly before comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .features import SCALAR_GROUPS


def load_db(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    db = {k: d[k] for k in d.files if k != "tracks"}
    db["tracks"] = json.loads(str(d["tracks"]))
    return db


class Coster:
    """Vectorised target cost between target units and (corpus unit, transposition) pairs."""

    def __init__(self, corpus: dict, target: dict, w: dict):
        self.w = w
        # z-score scalar descriptors with corpus statistics, then fold the group
        # weights in so a plain squared euclidean distance gives the weighted cost
        zc, zt = [], []
        for g, dim in SCALAR_GROUPS.items():
            mu = corpus[g].mean(0)
            sd = corpus[g].std(0) + 1e-6
            s = np.sqrt(w[g] / dim)
            zc.append((corpus[g] - mu) / sd * s)
            zt.append((target[g] - mu) / sd * s)
        self.zc = np.hstack(zc).astype(np.float32)
        self.zt = np.hstack(zt).astype(np.float32)
        self.zc2 = (self.zc ** 2).sum(1)
        self.zt2 = (self.zt ** 2).sum(1)
        self.cc = corpus["chroma"].astype(np.float32)
        self.ct = target["chroma"].astype(np.float32)
        self.ct_rot = np.stack([np.roll(self.ct, -t, axis=1) for t in range(12)], 1)  # [i, t % 12]
        self.ldc = np.log(corpus["dur"]).astype(np.float32)
        self.ldt = np.log(target["dur"]).astype(np.float32)
        # chroma is meaningless for near-silent target units: fade its weight out below -40 dB
        self.tonal = np.clip((target["loud"][:, 0] + 60) / 20, 0, 1).astype(np.float32) * w["chroma"]

    def full(self, rows: np.ndarray, t: int) -> np.ndarray:
        """Cost matrix (len(rows), n_corpus) for transposition t (grain shifted up t semitones)."""
        scal = self.zt2[rows, None] + self.zc2[None, :] - 2 * self.zt[rows] @ self.zc.T
        ct = np.roll(self.ct[rows], -t, axis=1)  # target pitch classes expressed in the un-shifted grain's frame
        chroma = (1 - ct @ self.cc.T) * self.tonal[rows, None]
        dur = (self.ldc[None, :] - t * np.log(2) / 12 - self.ldt[rows, None]) ** 2 * self.w["dur"]
        return np.maximum(scal, 0) + chroma + dur + self.w["transpose"] * abs(t)

    def pairs(self, i: int, J: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Cost for one target unit i against arrays of corpus units J with transpositions T."""
        d = self.zt[i] - self.zc[J]
        scal = (d * d).sum(1)
        ct = self.ct_rot[i, T % 12]
        chroma = (1 - (ct * self.cc[J]).sum(1)) * self.tonal[i]
        dur = (self.ldc[J] - T * np.log(2) / 12 - self.ldt[i]) ** 2 * self.w["dur"]
        return scal + chroma + dur + self.w["transpose"] * np.abs(T)

    def breakdown(self, i: int, j: int, t: int) -> dict:
        out, k = {}, 0
        d = self.zt[i] - self.zc[j]
        for g, dim in SCALAR_GROUPS.items():
            out[g] = float((d[k:k + dim] ** 2).sum())
            k += dim
        out["chroma"] = float((1 - np.roll(self.ct[i], -t) @ self.cc[j]) * self.tonal[i])
        out["dur"] = float((self.ldc[j] - t * np.log(2) / 12 - self.ldt[i]) ** 2 * self.w["dur"])
        out["transpose"] = self.w["transpose"] * abs(t)
        return out


def topk(coster: Coster, n_target: int, k: int, max_t: int, chunk: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = list(range(-max_t, max_t + 1))
    J = np.zeros((n_target, k), int)
    T = np.zeros((n_target, k), int)
    C = np.zeros((n_target, k), np.float32)
    for r0 in range(0, n_target, chunk):
        rows = np.arange(r0, min(r0 + chunk, n_target))
        bj, bt, bc = [], [], []
        for t in ts:
            M = coster.full(rows, t)
            idx = np.argpartition(M, k, axis=1)[:, :k]
            bj.append(idx)
            bt.append(np.full_like(idx, t))
            bc.append(np.take_along_axis(M, idx, 1))
        bj, bt, bc = np.hstack(bj), np.hstack(bt), np.hstack(bc)
        sel = np.argsort(bc, axis=1)[:, :k]
        J[rows] = np.take_along_axis(bj, sel, 1)
        T[rows] = np.take_along_axis(bt, sel, 1)
        C[rows] = np.take_along_axis(bc, sel, 1)
    return J, T, C


def viterbi(coster, corpus, J, T, jump_cost, repeat_cost, usage_pen, rng, jitter):
    n = len(J)
    contig = corpus["contig"]
    states, back, acc = [], [], None
    for i in range(n):
        cj, ct = J[i], T[i]
        if i > 0:
            pj, pt = states[-1]
            # offer continuations of the best previous states only (beam), else the lattice snowballs
            ok = contig[pj] & (acc <= np.partition(acc, min(len(J[i]), len(acc)) - 1)[min(len(J[i]), len(acc)) - 1])
            cj = np.concatenate([cj, pj[ok] + 1])
            ct = np.concatenate([ct, pt[ok]])
            key = cj * 100 + (ct + 50)
            _, u = np.unique(key, return_index=True)
            cj, ct = cj[u], ct[u]
        cost = coster.pairs(i, cj, ct) + usage_pen.get_many(cj)
        if jitter:
            cost = cost + rng.exponential(jitter, len(cost))
        if i == 0:
            acc = cost
            back.append(np.zeros(len(cj), int))
        else:
            pj, pt = states[-1]
            cont = (cj[None, :] == pj[:, None] + 1) & (ct[None, :] == pt[:, None]) & contig[pj][:, None]
            trans = np.where(cont, 0.0, jump_cost)
            trans = trans + np.where(cj[None, :] == pj[:, None], repeat_cost, 0.0)
            tot = acc[:, None] + trans
            b = tot.argmin(0)
            acc = tot[b, np.arange(len(cj))] + cost
            back.append(b)
        states.append((cj, ct))
    path = [int(acc.argmin())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path.reverse()
    return [(int(states[i][0][s]), int(states[i][1][s])) for i, s in enumerate(path)]


class UsagePenalty:
    def __init__(self, n, pen):
        self.count = np.zeros(n)
        self.pen = pen

    def get_many(self, j):
        return self.pen * np.maximum(self.count[j] - 1, 0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="work/corpus.npz")
    ap.add_argument("--target", default="work/target.npz")
    ap.add_argument("--out", default="work/plan.json")
    ap.add_argument("--k", type=int, default=60, help="candidates per target unit")
    ap.add_argument("--max-transpose", type=int, default=4, help="semitones (0 disables)")
    ap.add_argument("--jump-cost", type=float, default=0.6, help="concatenation cost for a non-continuous jump (phrase-ness)")
    ap.add_argument("--repeat-cost", type=float, default=1.5, help="extra cost for the same unit twice in a row")
    ap.add_argument("--reuse-penalty", type=float, default=0.4, help="per-extra-use cost, applied over --passes")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=0.0, help="random cost noise for variation")
    ap.add_argument("--seed", type=int, default=0)
    for g, d in {"chroma": 6.0, "mfcc": 1.5, "loud": 0.3, "bright": 0.6, "noisy": 0.4, "perc": 0.8, "attack": 0.6,
                 "env": 0.6, "dur": 0.5, "transpose": 0.05}.items():
        ap.add_argument(f"--w-{g}", type=float, default=d, help=f"weight of {g} (default {d})")
    a = ap.parse_args(argv)
    w = {g: getattr(a, f"w_{g}") for g in ["chroma", *SCALAR_GROUPS, "dur", "transpose"]}

    corpus, target = load_db(a.corpus), load_db(a.target)
    nt, nc = len(target["dur"]), len(corpus["dur"])
    print(f"target {nt} units, corpus {nc} units, transpose ±{a.max_transpose}")
    coster = Coster(corpus, target, w)
    J, T, _ = topk(coster, nt, a.k, a.max_transpose)
    rng = np.random.default_rng(a.seed)
    usage = UsagePenalty(nc, a.reuse_penalty)
    for p in range(a.passes):
        sel = viterbi(coster, corpus, J, T, a.jump_cost, a.repeat_cost, usage, rng, a.jitter)
        cnt = Counter(j for j, _ in sel)
        for j, c in cnt.items():  # cumulative, so units that keep winning get progressively pricier
            usage.count[j] += c
        print(f"  pass {p+1}: {len(cnt)} distinct units, max reuse {max(cnt.values())}")

    tracks = corpus["tracks"]
    entries, run, prev = [], 0, None
    parts = Counter()
    for i, (j, t) in enumerate(sel):
        cont = prev is not None and j == prev[0] + 1 and t == prev[1] and corpus["contig"][prev[0]]
        run += 0 if cont or i == 0 else 1
        bd = coster.breakdown(i, j, t)
        parts.update(bd)
        entries.append({
            "i": i, "t_start": float(target["start"][i]), "t_dur": float(target["dur"][i]),
            "t_loud": float(target["loud"][i, 0]),
            "track": int(corpus["track"][j]), "unit": j, "src_start": float(corpus["start"][j]),
            "src_dur": float(corpus["dur"][j]), "src_loud": float(corpus["loud"][j, 0]),
            "transpose": t, "cost": float(sum(bd.values())), "run": run, "continues": bool(cont),
        })
        prev = (j, t)

    plan = {"target": target["tracks"][0], "tracks": [t["path"] for t in tracks], "params": vars(a), "units": entries}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=1))

    by_track = Counter(Path(tracks[e["track"]]["path"]).stem for e in entries)
    runs = Counter(e["run"] for e in entries)
    print(f"wrote {a.out}")
    print(f"  mean cost {np.mean([e['cost'] for e in entries]):.3f}; mean run length {nt/len(runs):.2f} units (longest {max(runs.values())})")
    print("  mean cost by term: " + ", ".join(f"{k} {v/nt:.3f}" for k, v in parts.most_common()))
    print("  transpositions: " + ", ".join(f"{t:+d}:{c}" for t, c in sorted(Counter(e['transpose'] for e in entries).items())))
    print("  top sources: " + "; ".join(f"{s} ({c})" for s, c in by_track.most_common(6)))


if __name__ == "__main__":
    main()
