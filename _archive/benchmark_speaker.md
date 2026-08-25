# Speaker-ID Benchmark

Record date: **2026-08-24** — probe on a LibriSpeech dev-clean subset (14
speakers / 1193 clips, CC-BY-4.0, single speaker per file, 16 kHz) in the Linux
container; product wiring landed the same day. The transcribe / translate
records live in `benchmark_transcribe.md` / `benchmark_translate.md`.

Status: **done (2026-08-24)** — plan picks validated; CAM++ dropped; zh model
needs zh audio. Product wiring landed the same day (`src/interpreter/speaker.py`,
listen-mode auto-assignment). Data: LibriSpeech dev-clean subset, 14 speakers /
1193 clips (CC-BY-4.0, single speaker per file, 16 kHz). Assets live under
`data/benchmark/speaker/` (gitignored); results JSON in `data/benchmark/speaker/results/`.

## Why probe at all

Phase 3's speaker ID is a *new task* (not STT/translate), so the Phase 1 rule —
pick by measurement on the container, not by reputation — applies. But the probe
is deliberately **lightweight**: no 15-model sweep. WeSpeaker en/zh are the
obvious candidates (sherpa-onnx first-class, CC-BY-4.0, direct download), so the
probe validates *they run here and separate speakers well*, plus an operating
threshold — it does not re-derive the model registry.

## Task + data

The `speaker` task was added to `src/interpreter/benchmark.py`
(`--task speaker`). Enrollment-based identification via sherpa-onnx
`SpeakerEmbeddingExtractor` + `SpeakerEmbeddingManager`:

- **Enroll** = mean embedding of each speaker's first 3 clips.
- **Test** = next 10 clips per speaker (140 total); each test embedding is
  scored (cosine) against all 14 enrolled speakers.
- **Metrics**: top-1 identification accuracy (argmax, no threshold); EER over a
  similarity-threshold sweep (genuine = same-speaker pairs, impostor =
  different-speaker pairs).

## Results (container, 8 GB / 8 cores)

| model | top-1 acc | EER | thr@EER | ms/clip |
|-------|-----------|-----|---------|---------|
| wespeaker_en_voxceleb_resnet34 | **100.0%** | **1.6%** | 0.74 | 282 |
| wespeaker_en_voxceleb_CAM++    | 21.4%     | 39.2% | 0.68    | 85   |
| wespeaker_zh_cnceleb_resnet34  | — (smoke: loads + embeds; accuracy not measurable on English data) | | | |

### CAM++ check (not a harness artifact)

A 21% top-1 is surprising for a modern model, so it was verified with direct
cosine distributions (60 same + 60 different-speaker pairs, LibriSpeech):

- **resnet34**: same cos mean **0.856** (min 0.634) vs diff mean 0.499 (max 0.701) → clean gap.
- **CAM++**: same cos mean 0.637 (**min 0.113**) vs diff mean 0.509 (**max 0.891**) → heavy overlap, erratic.

The sherpa-released `wespeaker_en_voxceleb_CAM++.onnx` does not discriminate on
this data. **Dropped.** resnet34 is the clear pick.

## Verdict

- **en (listen): `wespeaker_en_voxceleb_resnet34`** — 100% top-1 / EER 1.6% on
  clean audio; threshold **0.74** (EER point) is the product default.
- **zh (dictate/mixed): `wespeaker_zh_cnceleb_resnet34`** — loads + computes on
  the container, but accuracy is **untested** (English-only data). Needs a
  zh-labeled eval — record real zh clips on the host (a zh speaker-ID eval set)
  and re-run before relying on it.
- **Caveat**: LibriSpeech dev-clean is clean studio audio with matched
  enroll/test conditions — these numbers are **upper bounds**. Real meetings
  (noise, cross-channel, overlapping speech) will be worse; if one speaker
  fragments into several, lower `NEW_SPEAKER_THRESHOLD`.

## Product integration (same day)

- `src/interpreter/speaker.py`: `SpeakerAssigner` — online auto-expanding
  registry, no enrollment step. First voice = `self`, second = `other`, then
  `speaker 3`, `speaker 4`, ... Each segment's centroid is a rolling mean; a
  match absorbs the segment, a miss creates a speaker. Listen mode only → en
  model only.
- `transcribe.py`: `RealTimeTranscribe(speaker_id=...)` loads the assigner
  concurrently with VAD/STT/translator; `_ingest_segment` assigns the newest
  segment and tags each committed chunk; listen lines render `[self]` / `[other]`
  / `[speaker N]` prefixes.
- CLI: `listen` has speaker ID **on by default**; `--no-speaker` disables it.

## Addendum (same day): uncertainty band → relative acceptance

The user's first real 2-voice test (22:00 listen session, one laptop mic) showed the
0.74 threshold **silently absorbed the second voice into `self`** — cross-speaker
cosine measured 0.80–0.84 raw, overlapping the same-speaker range, so no threshold
separates them. The product moved through three passes: (1) a **threshold band**
(`< 0.74` new speaker, `≥ 0.84` confident, in between renders `[?]` — never
mislabels, absorbs into no centroid); (2) **gray-zone cluster promotion**
(≥3 tight segments → named speaker, retroactively relabeling its `[?]` chunks) +
speaker-homogeneous windows — the 22:22 device voice (0.77) became `[other]`;
(3) the 22:37 test showed the SAME device voice measuring 0.84–0.87 vs the user —
no fixed bar is reliable — so acceptance became **relative to each cluster's own
measured internal consistency** (μ − 0.05, early floor 0.87, measured floor 0.84),
and promotion guards likewise. Cross-session matches confirmed the 14:14 "review
stretch" is the same device voice (not the user reading), which invalidated pass 2's
0.78 guard and freed the correct promotions. Details in `PLAN.md` Phase 3 +
`speaker.py` docstring.

## Repro

```bash
UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync python -m interpreter benchmark --task speaker --list
UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync python -m interpreter benchmark --task speaker
```
