<div align="center">

[![Linktree](https://img.shields.io/badge/linktree-1de9b6?logo=linktree&style=flat-square&color=029900)](https://www.sousastep.quest/links/)

</div>


<a href="#studio_microphone-introduction">![rainbow](./media/rainbow.png)</a>

# :studio_microphone: Prompt

Let's create a few scripts that can do a sort of spectral-granular style transfer that reconstructs one song from a corpus of coarse grains built from a library of other music albums.

These scripts should have a separation of duties that can all be run in one go from one main script, or run separately to allow parts of the process to be examined. Or it could just be all in one script with many flags, I suppose. Your choice.

To start, I've added a "song" directory with one song, to be reconstructed by a corpus created with the albums in the "library" directory.

Ideally, each note, or chord, or phrase in the song, would be replaced with a similar note or chord or phrase from the corpus, but it's alright if it's glitchy and not perfect as long as it's interesting.

Please use '/sousa_fx_generator/tools/search.py' to search for relevant papers to read.

The final output should be a .wav file the same length as the song being reconstructed.


<a href="#books-coarse-grain-spectral-granular-style-transfer">![rainbow](./media/rainbow.png)</a>

# :books: Coarse-grain spectral/granular style transfer

Rebuilds a target song (`song/`) out of note-, chord- and phrase-sized grains cut from a library of other music (`library/`). Output is a 24-bit WAV exactly as long as the target.

```bash
.venv/bin/python run.py            # -> output/<song>__mosaic.wav
```

The first run analyses the library, which takes about 2 minutes for 8.5 h of audio on 8 workers. Results are cached per track in `work/cache`, so later runs take about 15 s.

## Stages

| stage | module | output | what it does |
|---|---|---|---|
| 1 corpus | `cgt.analyze corpus` | `work/corpus.npz` | onset segmentation into units (split at 1 s), plus descriptors: HPSS-harmonic chroma, MFCC, loudness, brightness, noisiness, percussiveness, attack, 4-point envelope, duration |
| 2 target | `cgt.analyze target` | `work/target.npz` | the same analysis on the song |
| 3 match | `cgt.match` | `work/plan.json` | unit selection: top-K candidates per target unit across ±4 semitone transpositions, then Viterbi with a concatenation cost; a cumulative reuse penalty over 3 passes spreads selections across the corpus |
| 4 synth | `cgt.synth` | `output/*.wav` | places each grain at its target onset (varispeed transposition, loudness-matched, 8 ms fade-in / 60 ms fade-out tail, 7th-degree smoothstep curves); optional mel-band spectral-envelope morph toward the target |

Each stage is its own CLI with `--help`. `work/plan.json` is human-readable and lists, for every target unit, the source file, time, transposition and cost.

## Useful knobs

Pass these through `run.py --match "..." --synth "..."`, and use `--from match` to skip re-analysis.

- `--jump-cost` (match, default 0.6): cost of jumping elsewhere in the corpus instead of continuing the current source passage. Raise it (2–5) for longer Four Tet phrases, or set it to 0 for maximum glitch.
- `--w-chroma`, `--w-mfcc`, … (match): descriptor weights. Raise chroma to favour harmony and MFCC to favour timbre.
- `--max-transpose` (match): semitone range for resampling grains; 0 disables it.
- `--jitter 0.3 --seed N` (match): randomised variations.
- `--morph` (synth, default 0, off): >0 pulls the mosaic's spectral envelope toward the target, which tightens timing/dynamics but lets the original bleed through.
- `--fade-in`, `--tail`, `--fade-order` (synth): grain fade lengths and smoothstep order (0 linear, 1 cubic, 2 smootherstep, 3 default).
- `--loud-follow`, `--dry`, `--save-raw` (synth).

## Measured against the target (Bach, Variatio 1)

| | chroma cos | log-RMS corr | onset corr |
|---|---|---|---|
| random grains, same onsets | 0.68 | 0.62 | 0.22 |
| matched (default, `--morph 0`) | 0.83 | 0.86 | 0.52 |
| matched + `--morph 0.4` | 0.85 | 0.94 | 0.70 |

## References (from the papers library)

- Schwarz, *A System for Data-Driven Concatenative Sound Synthesis* (DAFx 2000) and *The Caterpillar system* (DAFx 2003): target and concatenation costs, unit selection.
- Zils & Pachet, *Musical Mosaicing* (DAFx 2001): imitation mode and "title continuity", which is the zero-cost continuation used here.
- Jehan, *Event-Synchronous Music Analysis/Synthesis* (DAFx 2004): onset-based segmentation of polyphonic music.


<a href="#lock_with_ink_pen-licenses">![rainbow](./media/rainbow.png)</a>

# :lock_with_ink_pen: License

In place of a legal notice, here is a blessing:

> May you do good and not evil.

> May you find forgiveness for yourself and forgive others.

> May you share freely, never taking more than you give.


<a href="#studio_microphone-introduction">![rainbow](./media/rainbow.png)</a>
