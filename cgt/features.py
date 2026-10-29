"""Onset segmentation + per-unit descriptors, shared by corpus and target analysis.

Units are spans between detected onsets (Jehan 2004, "event-synchronous"
segmentation), so a unit is roughly a note / chord / drum hit. Long units are
split into chunks so sustained material still yields several candidates.

Descriptor groups (each unit):
  chroma   12   harmonic-part pitch-class energy (HPSS removes drums first), L2-normalised
  mfcc     12   timbre (MFCC 1..12 mean, c0 excluded)
  loud      1   mean RMS in dB
  bright    1   log spectral centroid
  noisy     1   mean log spectral flatness
  perc      1   percussive / total energy ratio
  attack    1   onset strength at unit start
  env       4   RMS contour (dB relative to unit max) at 4 points
  dur       1   duration in seconds (log taken at match time)
"""

from __future__ import annotations

import numpy as np
import librosa

SR = 22050
HOP = 512
NFFT = 2048

SCALAR_GROUPS = {"mfcc": 12, "loud": 1, "bright": 1, "noisy": 1, "perc": 1, "attack": 1, "env": 4}


def frame_features(y: np.ndarray) -> dict:
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    H, Pc = librosa.decompose.hpss(S, kernel_size=17)
    P = S ** 2
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=SR, n_mels=64))
    eps = 1e-10
    return {
        "chroma": librosa.feature.chroma_stft(S=H ** 2, sr=SR, norm=None),
        "mfcc": librosa.feature.mfcc(S=logmel, n_mfcc=13)[1:],
        "rms": librosa.feature.rms(S=S, frame_length=NFFT)[0],
        "centroid": librosa.feature.spectral_centroid(S=S, sr=SR)[0],
        "flatness": librosa.feature.spectral_flatness(S=S + eps)[0],
        "perc": (Pc ** 2).sum(0) / (P.sum(0) + eps),
        "onset_env": librosa.onset.onset_strength(S=logmel, sr=SR),
    }


def segment(onset_env: np.ndarray, n_frames: int, min_dur: float, max_dur: float) -> list[tuple[int, int]]:
    on = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP, backtrack=True, units="frames")
    bounds = sorted(set([0, *on.tolist(), n_frames]))
    min_f = max(1, int(round(min_dur * SR / HOP)))
    max_f = max(min_f + 1, int(round(max_dur * SR / HOP)))
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_f:
            merged.append(b)
    if merged[-1] != n_frames:
        merged[-1] = n_frames
    spans = []
    for a, b in zip(merged[:-1], merged[1:]):
        k = int(np.ceil((b - a) / max_f))
        cuts = np.linspace(a, b, k + 1).round().astype(int)
        spans.extend(zip(cuts[:-1].tolist(), cuts[1:].tolist()))
    return [(a, b) for a, b in spans if b > a]


def unit_features(ff: dict, spans: list[tuple[int, int]]) -> dict:
    n = len(spans)
    out = {
        "start": np.zeros(n), "dur": np.zeros(n),
        "chroma": np.zeros((n, 12)), "mfcc": np.zeros((n, 12)),
        "loud": np.zeros((n, 1)), "bright": np.zeros((n, 1)), "noisy": np.zeros((n, 1)),
        "perc": np.zeros((n, 1)), "attack": np.zeros((n, 1)), "env": np.zeros((n, 4)),
    }
    eps = 1e-8
    for i, (a, b) in enumerate(spans):
        rms = ff["rms"][a:b]
        w = rms + eps
        out["start"][i] = a * HOP / SR
        out["dur"][i] = (b - a) * HOP / SR
        c = ff["chroma"][:, a:b].sum(1)
        out["chroma"][i] = c / (np.linalg.norm(c) + eps)
        out["mfcc"][i] = (ff["mfcc"][:, a:b] * w).sum(1) / w.sum()
        out["loud"][i] = 20 * np.log10(rms.mean() + eps)
        out["bright"][i] = np.log((ff["centroid"][a:b] * w).sum() / w.sum() + 1.0)
        out["noisy"][i] = np.log(ff["flatness"][a:b] + eps).mean()
        out["perc"][i] = (ff["perc"][a:b] * w).sum() / w.sum()
        out["attack"][i] = np.log1p(ff["onset_env"][a:min(a + 3, b)].max())
        db = 20 * np.log10(rms + eps)
        out["env"][i] = np.interp(np.linspace(0, len(db) - 1, 4), np.arange(len(db)), db) - db.max()
    return out


def analyze(y: np.ndarray, min_dur: float, max_dur: float) -> dict:
    ff = frame_features(y)
    spans = segment(ff["onset_env"], ff["rms"].shape[0], min_dur, max_dur)
    return unit_features(ff, spans)
