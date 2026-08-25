"""Online speaker assignment for listen mode (Phase 3, 2026-08-24).

Auto-assigns each VAD segment to a speaker via WeSpeaker embeddings
(sherpa-onnx `SpeakerEmbeddingExtractor`). No enrollment step: the registry
starts empty and names speakers as they appear — first voice = `self`,
second = `other`, then `speaker 3`, `speaker 4`, ... (auto-expanding).
Listen only → the en model only.

Assignment semantics (relative acceptance, 2026-08-24): a segment is tagged
with the best-matching speaker when its cosine similarity to that centroid
meets the speaker's own internal-consistency bar — within
`CONFIDENCE_DELTA` of the cluster's mean member similarity, floored at
`CONFIDENT_THRESHOLD` (`_accept_threshold`); below the bar minus
`NEW_SPEAKER_MARGIN` it starts a new speaker. Between the two it returns
`UNCERTAIN` ("?") — genuinely ambiguous — and is absorbed into no centroid.
Rationale: on one laptop mic, distinct voices measure at 0.74–0.90 cosine
(same-speaker and cross-speaker ranges overlap), and the same second voice
measures 0.76–0.79 vs the primary voice in one session but 0.84 in another —
so any FIXED threshold is arbitrary and silently absorbing marginal segments
mislabeled the second voice as `self` and drifted the centroid toward the
blended voice (self-reinforcing). The relative bar adapts per session; "?"
is the honest label. Gray-zone segments that later form a tight
self-consistent cluster (`PENDING_TIGHTNESS`, >= `MIN_PENDING_SEGMENTS`,
stray outliers dropped greedily) are promoted to a named speaker — a
genuinely distinct-but-close voice still becomes `other`. Promotion is
refused while the cluster's mean per-segment similarity to an existing
speaker clears that speaker's own acceptance bar — a new voice must be
clearly farther from a speaker than that speaker's own members typically are,
otherwise the speaker's own variance would be mislabeled as a new voice.

Model pick (Phase 3 speaker-ID probe, 2026-08-24): `wespeaker_en_voxceleb_resnet34`
(en, VoxCeleb) — top-1 100% / EER 1.6% @ threshold 0.74 on a LibriSpeech
dev-clean subset; CAM++ dropped (EER 39%, poor discrimination on the same
data). Full record: `_archive/benchmark_speaker.md`. Caveat: LibriSpeech is
clean studio audio — real-meeting accuracy will be lower; the constants are
starting points (0.15 new-speaker margin / 0.84 floor / 0.05 acceptance delta
/ 0.85 cluster tightness, calibrated on recorded listen sessions), tune if
one speaker fragments into several or the second voice is mislabeled.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from interpreter import DATA_DIR

MODELS_DIR = DATA_DIR / "benchmark" / "speaker" / "models"
EN_MODEL = MODELS_DIR / "wespeaker_en_voxceleb_resnet34.onnx"

_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"
)

# A segment that misses its best speaker's acceptance bar by more than this
# margin starts a NEW speaker ("strongly different voice"); between the bar
# and the bar-minus-margin lies the ambiguous band -> UNCERTAIN (pending
# cluster). Relative, not a fixed cosine: the bar itself adapts to the
# cluster's measured internal consistency. The new-speaker path is also gated
# on segment quality (MIN_NEW_SPEAKER_DURATION_S / MIN_NEW_SPEAKER_RMS): a
# near-silent blip must not claim a speaker slot — it is buffered as
# uncertain instead. Calibrated on recorded sessions: a 0.93 s / rms 0.009
# noise burst scored 0.37 vs everything and stole `other` (23:00 session);
# rms 0.02 let a 1.5 s / rms 0.020 fragment claim `speaker 4` in a
# 3-speaker session (23:08 — the "too many speakers" report) — so 0.03 is
# the rms floor.
NEW_SPEAKER_MARGIN = 0.15
MIN_NEW_SPEAKER_DURATION_S = 1.0
MIN_NEW_SPEAKER_RMS = 0.03

# Acceptance bars (relative, not a fixed band): once a cluster's internal
# consistency is measured (first absorption), a segment must score within
# CONFIDENCE_DELTA of it (floored at CONFIDENT_THRESHOLD) — a tight speaker
# (e.g. 0.93) stops absorbing marginal segments that would drag its centroid
# toward a second voice, while a genuinely loose speaker keeps its own
# members. Until then, EARLY_CONFIDENT_THRESHOLD applies — it must sit above
# the closest observed second-voice matches (0.85-0.87 vs the primary voice's
# own segments in the 23:58 session) so a marginal early segment can't anchor
# into `self` and start the drift-absorption cascade. Values are calibrated
# in kNN space (KNN_WINDOW), which runs higher than centroid space.
EARLY_CONFIDENT_THRESHOLD = 0.89
CONFIDENT_THRESHOLD = 0.84
CONFIDENCE_DELTA = 0.05

# Gray-zone promotion: buffered ambiguous embeddings are promoted to a named
# speaker only when they form a tight self-consistent cluster — mean pairwise
# cosine >= PENDING_TIGHTNESS with every member at least
# PENDING_MEMBER_TIGHTNESS from the others (so one off-voice segment can't
# ride along inside an otherwise tight cluster), at least MIN_PENDING_SEGMENTS
# of them. Outliers are dropped greedily before the tightness check, so one
# stray segment can't poison the buffer. Buffer is capped at
# MAX_PENDING_SEGMENTS (drop oldest).
PENDING_TIGHTNESS = 0.85
PENDING_MEMBER_TIGHTNESS = 0.80
MIN_PENDING_SEGMENTS = 3
MAX_PENDING_SEGMENTS = 16

# Sentinel returned for ambiguous segments (rendered "[?]" in listen lines).
UNCERTAIN = "?"

# First two speaker slots get fixed names; further distinct voices expand.
_DEFAULT_NAMES = ("self", "other")

# Margin rescue: with two or more speakers, a segment that scores just below
# its best speaker's bar (within MARGIN_RESCUE_LOOSENING) is still accepted if
# it beats every other speaker by at least MARGIN_RESCUE_GAP. This rescues the
# primary voice's weak segments (their margin over the second cluster is large,
# e.g. 0.10) while the genuinely ambiguous ones stay "?" — the device voice's
# fragments measure within 0.02-0.04 of BOTH clusters (23:58 / 00:52 sessions).
MARGIN_RESCUE_LOOSENING = 0.03
MARGIN_RESCUE_GAP = 0.08

# Hard ceiling on the number of speakers a session can name: the user's setup
# is their voice + a device/second voice — "just self and other" (their words).
# Once reached, the new-speaker path and the gray-zone promotion refuse:
# further distinct segments stay UNCERTAIN ("[?]") instead of creating
# speaker 3, 4, ... (sub-clusters of one voice share the same margin geometry
# as distinct voices, so extra buckets were fragments of the review voice).
# Fewer, stable names.
MAX_SPEAKERS = 2

# "Looking back" window: each speaker keeps its KNN_WINDOW most recent member
# embeddings, and a segment is scored against a speaker by its MAX cosine to
# those recent segments — not the centroid. Why: centroid averaging blends the
# shared mic channel and inflates cross-speaker similarity (the review voice
# measured 0.84-0.94 vs the primary voice's centroid in the 23:58 session but
# only 0.76-0.87 against the primary voice's individual segments, while both
# voices match their own members at 0.90+). kNN preserves that margin — the
# self-contained version of a diarization system's "look back at recent audio
# to detect a speaker change".
KNN_WINDOW = 6


@dataclass
class _Speaker:
    name: str
    centroid: np.ndarray
    count: int = 1
    # Rolling mean of each absorbed segment's kNN similarity to the cluster —
    # the cluster's measured internal consistency. None until the first
    # absorption; drives the relative acceptance threshold.
    mean_sim: float | None = None
    # Recent member embeddings for kNN matching (KNN_WINDOW, drop oldest).
    recent: list[np.ndarray] = field(default_factory=list)


def _ensure_onnxruntime_runtime() -> None:
    """Self-heal sherpa-onnx's onnxruntime dlopen on both platforms:

    - macOS: sherpa wheels don't bundle onnxruntime — copy the dylibs into the
      sherpa package's lib dir (the first @rpath search location). dyld reads
      DYLD_* at exec time, so a runtime env tweak can't fix this.
    - Linux: sherpa dlopens `libonnxruntime.so`; the PyPI wheel ships only the
      versioned soname — symlink it and put it on the loader path.
    """
    if sys.platform == "darwin":
        try:
            import onnxruntime

            spec = importlib.util.find_spec("sherpa_onnx")
            if spec is None or not spec.submodule_search_locations:
                return
            sherpa_lib = Path(spec.submodule_search_locations[0]) / "lib"
            sherpa_lib.mkdir(parents=True, exist_ok=True)
            capi = Path(onnxruntime.__file__).parent / "capi"
            for src in capi.glob("libonnxruntime*.dylib"):
                dest = sherpa_lib / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
        except Exception:  # noqa: S110, BLE001 - best-effort fix; surface the real import error
            pass
        return
    try:
        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        if not capi.exists():
            return
        plain = capi / "libonnxruntime.so"
        if not plain.exists():
            libs = sorted(capi.glob("libonnxruntime.so.*"))
            if libs:
                plain.symlink_to(libs[-1].name)
        path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{capi}{os.pathsep}{path}"
    except Exception:  # noqa: S110, BLE001 - best-effort env fix for sherpa-onnx
        pass


def _ensure_en_model() -> Path:
    if EN_MODEL.exists():
        return EN_MODEL
    import urllib.request

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading speaker-ID model (WeSpeaker en, 25 MB)...", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, EN_MODEL)
    return EN_MODEL


class SpeakerAssigner:
    """Online auto-expanding speaker registry.

    `assign()` embeds a 16 kHz segment and matches it against known speaker
    centroids (rolling mean); a match absorbs the segment, a miss creates a
    new speaker. Names: self, other, speaker 3, ... Thread-safe.

    Acceptance is RELATIVE, not a fixed band: a segment is tagged with the
    best-matching speaker when its similarity meets that speaker's own
    internal-consistency bar (`_accept_threshold` — within CONFIDENCE_DELTA
    of the cluster's mean member similarity, floored at CONFIDENT_THRESHOLD);
    below the bar minus NEW_SPEAKER_MARGIN it starts a new speaker; between
    the two it returns UNCERTAIN ("?") and absorbs into no centroid
    (gray-zone embeddings are buffered and promoted to a named speaker when
    they form a tight self-consistent cluster — see module docstring).
    """

    def __init__(self, model_path: Path | str = EN_MODEL) -> None:
        _ensure_onnxruntime_runtime()
        _ensure_en_model()
        import sherpa_onnx  # heavy — import only when speaker ID is enabled

        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path))
        )
        self._lock = threading.Lock()
        self._speakers: list[_Speaker] = []
        self._pending: list[np.ndarray] = []
        self._pending_ids: list[int] = []
        self._next_pending_id = 0
        self._promoted: tuple[str, list[int]] | None = None

    @property
    def speaker_names(self) -> list[str]:
        with self._lock:
            return [s.name for s in self._speakers]

    def assign(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Embed 16 kHz audio and assign it to a speaker (embed + assign_embedding)."""
        return self.assign_embedding(
            self.embed(audio, sample_rate),
            duration_s=len(audio) / sample_rate,
            rms=float(np.sqrt(np.mean(audio**2))),
        )

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Speaker embedding of 16 kHz audio. Embed RAW audio — noise reduction
        and normalization compress the embedding space (distinct voices merge)."""
        if sample_rate != 16000:
            raise ValueError(f"speaker embedder expects 16 kHz, got {sample_rate}")
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("speaker extractor not ready")
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)

    def _accept_threshold(self, spk: _Speaker) -> float:
        """Relative acceptance bar for `spk`: a segment must score within
        CONFIDENCE_DELTA of the cluster's measured internal consistency
        (floored at CONFIDENT_THRESHOLD). The stricter EARLY_CONFIDENT_THRESHOLD
        applies until the cluster has absorbed TWO members (count >= 3): a
        single-absorption mean is a noisy sample and lowers the bar too fast —
        in the 23:58 session a second-voice segment at 0.87 vs the anchor
        slipped in past a bar seeded from one 0.90 absorption and started the
        drift-absorption cascade. A fixed absolute bar is too arbitrary — the
        same second voice measured 0.76-0.79 vs the primary voice in one
        session but 0.84 in another."""
        if spk.mean_sim is None or spk.count < 3:
            return EARLY_CONFIDENT_THRESHOLD
        return max(CONFIDENT_THRESHOLD, spk.mean_sim - CONFIDENCE_DELTA)

    def _new_speaker_bar(self, spk: _Speaker) -> float:
        """Below this, the segment is a strongly different voice -> new
        speaker (relative to `spk`'s acceptance bar; see NEW_SPEAKER_MARGIN)."""
        return self._accept_threshold(spk) - NEW_SPEAKER_MARGIN

    def _knn_sim(self, emb: np.ndarray, spk: _Speaker) -> float:
        """ "Look back": similarity of `emb` to `spk` = its MAX cosine to the
        speaker's KNN_WINDOW most recent member embeddings. Sharper than the
        centroid (which blends the shared mic channel and inflates
        cross-speaker similarity — see KNN_WINDOW)."""
        return max((_cosine(e, emb) for e in spk.recent), default=-1.0)

    def assign_embedding(
        self,
        emb: np.ndarray,
        duration_s: float | None = None,
        rms: float | None = None,
    ) -> str:
        """Match a precomputed embedding against known speakers.

        Returns the speaker name, or UNCERTAIN ("?") when the best match sits
        below the winner's relative acceptance threshold (see module
        docstring). A match absorbs the segment into the winner's centroid
        (rolling mean) and kNN window; an ambiguous one updates nothing; a
        miss (below the winner's bar minus NEW_SPEAKER_MARGIN) creates a new
        speaker — gated on segment quality (`duration_s`/`rms`, see
        MIN_NEW_SPEAKER_*), so noise blips can't claim a speaker slot. A
        promotion (pending cluster -> named speaker) is reported via
        `consume_promotion()`.
        """
        with self._lock:
            self._promoted = None
            ranked = sorted(
                ((s.name, self._knn_sim(emb, s)) for s in self._speakers),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if not ranked:
                spk = _Speaker(name="self", centroid=emb.copy(), recent=[emb.copy()])
                self._speakers.append(spk)
                return "self"
            name, sim = ranked[0]
            second_sim = ranked[1][1] if len(ranked) > 1 else -1.0
            spk = next(s for s in self._speakers if s.name == name)
            if sim < self._new_speaker_bar(spk):
                if len(self._speakers) >= MAX_SPEAKERS:
                    return self._buffer_uncertain(emb)
                if (
                    duration_s is not None and duration_s < MIN_NEW_SPEAKER_DURATION_S
                ) or (rms is not None and rms < MIN_NEW_SPEAKER_RMS):
                    return self._buffer_uncertain(emb)
                self._speakers.append(
                    _Speaker(
                        name=self._new_name(), centroid=emb.copy(), recent=[emb.copy()]
                    )
                )
                return self._speakers[-1].name
            if sim >= self._accept_threshold(spk) or (
                len(self._speakers) >= 2
                and sim >= self._accept_threshold(spk) - MARGIN_RESCUE_LOOSENING
                and sim - second_sim >= MARGIN_RESCUE_GAP
            ):
                spk.centroid = (spk.centroid * spk.count + emb) / (spk.count + 1)
                spk.count += 1
                spk.recent.append(emb.copy())
                if len(spk.recent) > KNN_WINDOW:
                    spk.recent.pop(0)
                if spk.mean_sim is None:
                    spk.mean_sim = float(sim)
                else:
                    spk.mean_sim = (spk.mean_sim * (spk.count - 2) + sim) / (
                        spk.count - 1
                    )
                return name
            return self._buffer_uncertain(emb)

    def pending_count(self) -> int:
        """Number of ambiguous embeddings currently buffered (gray-zone
        promotion candidates)."""
        with self._lock:
            return len(self._pending)

    def consume_promotion(self) -> tuple[str, list[int]] | None:
        """(name, pending ids) of the speaker created by a pending-cluster
        promotion on the last `assign_embedding` call, if any (cleared on
        read). The ids let the display layer retroactively relabel exactly
        the earlier "?" chunks whose segments joined the promoted cluster —
        outliers dropped during promotion stay "?"."""
        with self._lock:
            promoted = self._promoted
            self._promoted = None
            return promoted

    def _buffer_uncertain(self, emb: np.ndarray) -> str:
        """Buffer an ambiguous embedding; promote the buffer to a new speaker
        when it forms a tight self-consistent cluster (see module docstring).
        Returns the promoted speaker name or UNCERTAIN."""
        self._pending.append(emb.copy())
        self._pending_ids.append(self._next_pending_id)
        self._next_pending_id += 1
        if len(self._pending) > MAX_PENDING_SEGMENTS:
            self._pending.pop(0)
            self._pending_ids.pop(0)
        promoted = self._promotable_centroid()
        if promoted is None or len(self._speakers) >= MAX_SPEAKERS:
            return UNCERTAIN
        centroid, ids, kept = promoted
        name = self._new_name()
        # Seed the promoted speaker's internal consistency from the cluster's
        # mean pairwise — otherwise its acceptance bar falls back to the early
        # floor and a second sub-cluster of the SAME voice can slip past the
        # promotion guard (fragmenting one voice into speaker 3 + speaker 4).
        tightness = float(
            np.mean(
                [
                    _cosine(kept[i], kept[j])
                    for i in range(len(kept))
                    for j in range(i + 1, len(kept))
                ]
            )
        )
        self._speakers.append(
            _Speaker(
                name=name,
                centroid=centroid,
                count=len(ids),
                mean_sim=tightness,
                recent=[e.copy() for e in kept],
            )
        )
        # Remove only the promoted segments from the pending buffer — a second
        # gray-zone voice keeps its accumulated evidence and can promote after
        # the first one does (3+-speaker sessions); clearing the whole buffer
        # would wipe it and leave the later voice stuck at "?" forever.
        promoted_ids = set(ids)
        self._pending = [
            e
            for e, pid in zip(self._pending, self._pending_ids)
            if pid not in promoted_ids
        ]
        self._pending_ids = [
            pid for pid in self._pending_ids if pid not in promoted_ids
        ]
        self._promoted = (name, ids)
        return name

    def _promotable_centroid(
        self,
    ) -> tuple[np.ndarray, list[int], list[np.ndarray]] | None:
        """Greedily drop the least-self-consistent pending embedding until the
        remainder is tight (mean pairwise >= PENDING_TIGHTNESS, every member
        >= PENDING_MEMBER_TIGHTNESS) or too few remain; the cluster is
        promoted only if its mean per-segment kNN similarity to EVERY existing
        speaker stays below that speaker's own relative acceptance threshold (a
        new voice must be clearly farther from a speaker than that speaker's
        own members typically are). Returns the promoted centroid, the ids and
        the kept embeddings of the segments in it."""
        pool = self._pending.copy()
        ids = self._pending_ids.copy()
        while len(pool) >= MIN_PENDING_SEGMENTS:
            stacked = np.stack(pool)
            pairwise = np.array(
                [
                    [float(_cosine(stacked[i], stacked[j])) for j in range(len(pool))]
                    for i in range(len(pool))
                ]
            )
            tri = [
                pairwise[i, j]
                for i in range(len(pool))
                for j in range(i + 1, len(pool))
            ]
            member_means = np.array(
                [
                    np.mean([pairwise[i, j] for j in range(len(pool)) if j != i])
                    for i in range(len(pool))
                ]
            )
            if (
                float(np.mean(tri)) >= PENDING_TIGHTNESS
                and float(member_means.min()) >= PENDING_MEMBER_TIGHTNESS
            ):
                centroid = stacked.mean(axis=0)
                for spk in self._speakers:
                    mean_sim = float(
                        np.mean([self._knn_sim(pool[i], spk) for i in range(len(pool))])
                    )
                    if mean_sim >= self._accept_threshold(spk):
                        return None
                return centroid, ids, [e.copy() for e in pool]
            worst = int(np.argmin(member_means))
            pool.pop(worst)
            ids.pop(worst)
        return None

    def _new_name(self) -> str:
        n = len(self._speakers)
        if n < len(_DEFAULT_NAMES):
            return _DEFAULT_NAMES[n]
        return f"speaker {n + 1}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))
