"""
Local STT benchmark harness — Phase 1 research gate.

Benchmarks local STT models against gold-reference transcripts:
- WER / CER (jiwer), overall and per language block (code-switching WER)
- Wall-clock decode time and RTF (decode time / audio duration)
- Peak RSS
- Segments (start/end/text) per sample, when the model exposes them
- Streaming decode (--stream) for models that support it (moonshine v2):
  records per-update (stream_time, wall_time, text) for tail-latency analysis

Usage:
  uv run python -m interpreter.benchmark --list                          # stt models + samples
  uv run python -m interpreter.benchmark                                 # all stt models
  uv run python -m interpreter.benchmark whispercpp-large-v3-turbo-q5_0 moonshine-streaming-medium
  uv run python -m interpreter.benchmark --samples sample_a1
  uv run python -m interpreter.benchmark --stream moonshine-streaming-medium
  uv run python -m interpreter.benchmark --task translate                # all en->zh models
  uv run python -m interpreter.benchmark --task translate opus-mt-en-zh
  uv run python -m interpreter.benchmark --record mode_b_1.wav 30  # host only (mic)

Default runs load one model per subprocess (load -> benchmark -> write JSON
-> exit, memory fully released -> next model). A model killed by OOM
(exit 137) is recorded as "excluded: OOM" and the run continues — this is the
container's 4 GB memory budget acting as the model filter: anything that does
not fit is not considered. Use --in-process to disable isolation (debugging).

RTF (Real-Time Factor) = decode wall time / audio duration; < 1.0 means the
model processes faster than real time. The container numbers are relative
(roughly 13-18x slower than the M4 host); the ranking is what matters here.

Results are written to data/benchmark/transcribe/results/<model>.json (STT) and
data/benchmark/translate/results/<model>.json (en->zh); a summary table
is printed to stdout.

Environment notes:
- Linux aarch64 container: the sherpa-onnx adapter auto-plumbs the
  onnxruntime lib path (see _ensure_onnxruntime_lib) — no env fiddling.
- sherpa-onnx is pinned to 1.13.0 + onnxruntime 1.24.4: newer sherpa
  wheels link a VERS_1.27.1 symbol that no PyPI onnxruntime exports
  (see PLAN.md Phase 1 deps item).
- Moonshine STT is the `moonshine-voice` package (PyPI `moonshine` is an
  unrelated remote-sensing library; `moonshine-ml` no longer exists).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from interpreter import DATA_DIR

BENCH_DIR = DATA_DIR / "benchmark"
TRANSCRIBE_DIR = BENCH_DIR / "transcribe"
TRANSLATE_DIR = BENCH_DIR / "translate"
MODELS_DIR = TRANSCRIBE_DIR / "models"
TRANSCRIBE_RESULTS_DIR = TRANSCRIBE_DIR / "results"
TRANSLATE_RESULTS_DIR = TRANSLATE_DIR / "results"
TRANSCRIBE_MANIFEST = TRANSCRIBE_DIR / "manifest.json"
TRANSLATE_MANIFEST = TRANSLATE_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest / samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    id: str
    mode: str  # "A" = en-only audio, "B" = contains zh
    category: str  # "en" | "zh" | "mixed"
    path: str
    ref: str
    lang: str = "en"
    blocks: list[dict] = field(
        default_factory=list
    )  # [{lang, start, end}] word idx into ref

    def duration(self) -> float:
        return sf.info(str(TRANSCRIBE_DIR / self.path)).duration


def load_manifest(path: Path = TRANSCRIBE_MANIFEST) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Sample(**s) for s in data["samples"]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return text.lower().strip()


def _wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    import jiwer

    return jiwer.wer(_norm(ref), _norm(hyp)), jiwer.cer(_norm(ref), _norm(hyp))


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _block_wer(ref: str, hyp: str, blocks: list[dict]) -> list[dict]:
    """Per-language-block error via word-level alignment (code-switching).

    en blocks use word-level WER; zh blocks use char-level CER (zh has no
    word boundaries, and the reference is word-segmented while model output
    is not). zh hypothesis words are assigned to the nearest zh block (by
    aligned reference position) so unspaced/glued zh output is still scored.
    """
    if not blocks:
        return []
    import copy

    import jiwer

    ref_words = _norm(ref).split()
    hyp_words = _norm(hyp).split()
    out = jiwer.process_words(_norm(ref), _norm(hyp))
    blocks = copy.deepcopy(blocks)

    zh_blocks = [b for b in blocks if b.get("metric") == "cer"]
    if zh_blocks:
        hyp_to_ref: dict[int, int] = {}
        for sentence in out.alignments:
            for chunk in sentence:
                if chunk.type == "insert":
                    continue
                for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    hyp_to_ref[chunk.hyp_start_idx + k] = chunk.ref_start_idx + k
        for i, word in enumerate(hyp_words):
            if not _has_cjk(word):
                continue
            r = hyp_to_ref.get(i)
            if r is None:
                continue
            best = next((b for b in zh_blocks if b["start"] <= r < b["end"]), None)
            if best is None:
                best = min(
                    zh_blocks,
                    key=lambda b: min(abs(r - b["start"]), abs(r - b["end"] - 1)),
                )
            best.setdefault("_hyp", []).append(i)

    res = []
    for block in blocks:
        start, end = block["start"], block["end"]
        metric = block.get("metric", "wer")
        if metric == "cer":
            ref_span = "".join(ref_words[start:end])
            idx = sorted(block.get("_hyp", []))
            hyp_span = "".join(hyp_words[i] for i in idx)
            score = jiwer.cer(ref_span, hyp_span)
        else:
            errs = hits = 0
            for sentence in out.alignments:
                for chunk in sentence:
                    if chunk.type == "insert":
                        continue
                    overlap = min(chunk.ref_end_idx, end) - max(
                        chunk.ref_start_idx, start
                    )
                    if overlap <= 0:
                        continue
                    if chunk.type == "equal":
                        hits += overlap
                    else:
                        errs += overlap
            total = hits + errs
            score = errs / total if total else 0.0
        res.append(
            {
                "lang": block["lang"],
                "start": start,
                "end": end,
                "metric": metric,
                "score": score,
            }
        )
    return res


def peak_rss_mb() -> float:
    """Peak RSS of this process in MB (Linux ru_maxrss is KB, macOS bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 if sys.platform == "linux" else 1024 * 1024)


def env_info() -> dict:
    import platform

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


class Adapter:
    name = ""
    tier = ""  # en-only | multilingual
    modes: tuple[str, ...] = ("A", "B")  # sample modes this model can transcribe
    stream_capable = False
    weight_note = ""

    def __init__(self) -> None:
        self.load_s = 0.0

    def transcribe(self, sample: Sample, use_stream: bool = False) -> tuple[str, dict]:
        raise NotImplementedError

    def stream_info(self, sample: Sample) -> dict:
        raise NotImplementedError


class _WhisperCpp(Adapter):
    def __init__(self, model_size: str) -> None:
        super().__init__()
        from pywhispercpp.model import Model

        t0 = time.perf_counter()
        self.model = Model(model_size, n_threads=4, print_progress=False)
        self.load_s = time.perf_counter() - t0

    def transcribe(self, sample: Sample, use_stream: bool = False):
        t0 = time.perf_counter()
        segs = self.model.transcribe(str(TRANSCRIBE_DIR / sample.path), temperature=0.0)
        wall = time.perf_counter() - t0
        segments = [(s.t0, s.t1, s.text.strip()) for s in segs]
        text = " ".join(t for _, _, t in segments)
        return text, {"segments": segments, "wall": wall}


class _FasterWhisper(Adapter):
    def __init__(self, model_size: str = "large-v3") -> None:
        super().__init__()
        import faster_whisper

        t0 = time.perf_counter()
        self.model = faster_whisper.WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
        self.load_s = time.perf_counter() - t0

    def transcribe(self, sample: Sample, use_stream: bool = False):
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            str(TRANSCRIBE_DIR / sample.path),
            beam_size=5,
            language=sample.lang if sample.lang == "en" else None,
        )
        segs = list(segments)
        wall = time.perf_counter() - t0
        out = [(s.start, s.end, s.text.strip()) for s in segs]
        text = " ".join(t for _, _, t in out)
        return text, {"segments": out, "wall": wall}


class _OpenAIWhisper(Adapter):
    """Original openai-whisper (torch, fp32) — the model_comparison.py classic."""

    def __init__(self, model_size: str = "small") -> None:
        super().__init__()
        import whisper

        t0 = time.perf_counter()
        self.model = whisper.load_model(model_size)
        self.load_s = time.perf_counter() - t0

    def transcribe(self, sample: Sample, use_stream: bool = False):
        audio, sr = sf.read(str(TRANSCRIBE_DIR / sample.path), dtype="float32")
        assert sr == 16000, (sample.id, sr)
        t0 = time.perf_counter()
        result = self.model.transcribe(audio, temperature=0.0, beam_size=1)
        wall = time.perf_counter() - t0
        text = result["text"].strip()
        segs = [(s["start"], s["end"], s["text"].strip()) for s in result["segments"]]
        return text, {"segments": segs, "wall": wall}


class _Moonshine(Adapter):
    """Moonshine v2 streaming-medium via moonshine-voice."""

    name = "moonshine-streaming-medium"
    tier = "en-only"
    modes = ("A",)
    stream_capable = True
    weight_note = "downloads from download.moonshine.ai; arch MEDIUM_STREAMING"

    def __init__(self) -> None:
        super().__init__()
        from moonshine_voice import get_model_for_language
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        t0 = time.perf_counter()
        self.arch = ModelArch.MEDIUM_STREAMING
        self.model_path, _ = get_model_for_language("en", self.arch)
        self.transcriber = Transcriber(model_path=self.model_path, model_arch=self.arch)
        self.load_s = time.perf_counter() - t0

    def _lines_to_segments(self, transcript) -> list[tuple[float, float, str]]:
        return [
            (l.start_time, l.start_time + l.duration, (l.text or "").strip())
            for l in transcript.lines
        ]

    def transcribe(self, sample: Sample, use_stream: bool = False):
        audio, sr = sf.read(str(TRANSCRIBE_DIR / sample.path), dtype="float32")
        audio_list = audio.tolist()
        t0 = time.perf_counter()
        transcript = self.transcriber.transcribe_without_streaming(audio_list, sr)
        wall = time.perf_counter() - t0
        segs = self._lines_to_segments(transcript)
        text = " ".join(t for _, _, t in segs)
        return text, {"segments": segs, "wall": wall}

    def stream_info(self, sample: Sample) -> dict:
        """Feed audio in 0.25 s chunks; record (stream_time, wall, text) per update."""
        audio, sr = sf.read(str(TRANSCRIBE_DIR / sample.path), dtype="float32")
        stream = self.transcriber.create_stream()
        stream.start()
        updates: list[dict] = []
        chunk = int(0.25 * sr)
        stream_time = 0.0
        last_text = ""
        t0 = time.perf_counter()
        for i in range(0, len(audio), chunk):
            stream.add_audio(audio[i : i + chunk].tolist(), sr)
            stream_time += len(audio[i : i + chunk]) / sr
            transcript = stream.update_transcription()
            text = " ".join((l.text or "").strip() for l in transcript.lines)
            if text != last_text:
                updates.append(
                    {
                        "stream_time": stream_time,
                        "wall": time.perf_counter() - t0,
                        "text": text,
                    }
                )
                last_text = text
        transcript = stream.stop()
        wall = time.perf_counter() - t0
        return {
            "updates": updates,
            "segments": self._lines_to_segments(transcript),
            "wall": wall,
        }


def _onnxruntime_capi_dir() -> Path | None:
    """Path to the onnxruntime capi dir (Linux), or None if unavailable."""
    if sys.platform != "linux":
        return None
    try:
        import onnxruntime
    except Exception:  # noqa: BLE001 - missing dep
        return None
    capi = Path(onnxruntime.__file__).parent / "capi"
    return capi if capi.exists() else None


def _ensure_onnxruntime_lib() -> None:
    """Linux only: sherpa-onnx dlopens `libonnxruntime.so`; the PyPI wheel
    ships only the versioned soname. Symlink it and put it on the loader path."""
    capi = _onnxruntime_capi_dir()
    if capi is None:
        return
    try:
        plain = capi / "libonnxruntime.so"
        if not plain.exists():
            libs = sorted(capi.glob("libonnxruntime.so.*"))
            if libs:
                plain.symlink_to(libs[-1].name)
        path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{capi}{os.pathsep}{path}"
    except Exception:  # noqa: S110, BLE001 - best-effort env fix for sherpa-onnx
        pass


class _Sherpa(Adapter):
    """Offline sherpa-onnx models (Parakeet TDT, Dolphin CTC, Fun-ASR-Nano,
    SenseVoice, Qwen3-ASR)."""

    def __init__(
        self, name: str, files: dict, factory: str, factory_kwargs: dict | None = None
    ) -> None:
        super().__init__()
        self.name = name
        _ensure_onnxruntime_lib()
        import sherpa_onnx

        t0 = time.perf_counter()
        kwargs = dict(factory_kwargs or {})
        for key, rel in files.items():
            kwargs[key] = str(MODELS_DIR / self.name / rel)
        kwargs["num_threads"] = 4
        self.recognizer = getattr(sherpa_onnx.OfflineRecognizer, factory)(**kwargs)
        self.load_s = time.perf_counter() - t0

    def transcribe(self, sample: Sample, use_stream: bool = False):
        audio, sr = sf.read(str(TRANSCRIBE_DIR / sample.path), dtype="float32")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        t0 = time.perf_counter()
        self.recognizer.decode_stream(stream)
        wall = time.perf_counter() - t0
        result = stream.result
        segments = None
        if result.timestamps:
            starts = result.timestamps
            starts.append(sample.duration())
            segments = [
                (starts[i], starts[i + 1], word)
                for i, word in enumerate(result.tokens)
                if word.strip()
            ]
        text = result.text
        return text, {"segments": segments, "wall": wall}


class _ArkAsr(Adapter):
    """ARK-ASR-0.6B (AutoArk, 2026) — multilingual zh/en ASR via transformers.

    Apache-2.0; published size 0.6B (decoder LLM) — the ≤1B policy counts
    published size, so the ~1.2B end-to-end (0.6B decoder + 0.6B audio
    encoder) is flagged, not excluded (docs/benchmark.md). trust_remote_code;
    audio via the chat template, greedy generation with bad-token filtering.
    """

    name = "ark-asr-0.6b"
    tier = "multilingual"
    modes = ("A", "B")
    weight_note = "Apache-2.0; 0.6B decoder + 0.6B encoder (e2e ~1.2B, policy counts published size)"

    def __init__(self) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            "AutoArk-AI/ARK-ASR-0.6B", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "AutoArk-AI/ARK-ASR-0.6B", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "AutoArk-AI/ARK-ASR-0.6B",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self._bad_words_ids = self._build_bad_words_ids(self.tokenizer)
        self.load_s = time.perf_counter() - t0

    @staticmethod
    def _build_bad_words_ids(tokenizer) -> list[list[int]]:
        eos_ids = tokenizer.eos_token_id
        keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        bad_ids = set(tokenizer.all_special_ids) - keep_ids
        bad_ids.update(
            token_id
            for token, token_id in tokenizer.get_added_vocab().items()
            if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
        )
        return [[token_id] for token_id in sorted(bad_ids)]

    def transcribe(self, sample: Sample, use_stream: bool = False):
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(TRANSCRIBE_DIR / sample.path)},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=30 * 16000,
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=self._bad_words_ids,
            )
        wall = time.perf_counter() - t0
        decoded = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return (decoded[0] if decoded else "").strip(), {"segments": None, "wall": wall}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SHERPA_HF = "https://huggingface.co"
GITHUB_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


@dataclass
class _SherpaSpec:
    repo: str | None
    files: dict[str, str]
    factory: str
    tier: str
    weight_note: str
    url: str | None = None
    factory_kwargs: dict | None = None


_MODEL_FILES: dict[str, _SherpaSpec] = {
    "parakeet-tdt-0.6b-v2": _SherpaSpec(
        repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        files={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        factory="from_transducer",
        factory_kwargs={"model_type": "nemo_transducer"},
        tier="en-only",
        weight_note="int8 NeMo transducer",
    ),
    "parakeet-tdt-0.6b-v3": _SherpaSpec(
        repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        files={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        factory="from_transducer",
        factory_kwargs={"model_type": "nemo_transducer"},
        tier="en-only",
        weight_note="int8 NeMo transducer; 25 EU langs, auto lang-detect (no zh)",
    ),
    "dolphin-small": _SherpaSpec(
        repo=None,
        url=f"{GITHUB_RELEASES}/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02.tar.bz2",
        files={
            "model": "sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/model.int8.onnx",
            "tokens": "sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/tokens.txt",
        },
        factory="from_dolphin_ctc",
        tier="multilingual",
        weight_note="offline CTC (sherpa port; the Dataocean Dolphin streaming variant needs funasr)",
    ),
    "funasr-nano-2512": _SherpaSpec(
        repo="csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30",
        files={
            "encoder_adaptor": "encoder_adaptor.int8.onnx",
            "llm": "llm.int8.onnx",
            "embedding": "embedding.int8.onnx",
            "tokenizer": "Qwen3-0.6B",
        },
        factory="from_funasr_nano",
        tier="multilingual",
        weight_note="Qwen3-0.6B-based LLM ASR, int8",
    ),
    "sensevoice": _SherpaSpec(
        repo="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        files={
            "model": "model.int8.onnx",
            "tokens": "tokens.txt",
        },
        factory="from_sense_voice",
        factory_kwargs={"use_itn": True},
        tier="multilingual",
        weight_note="non-streaming, fastest zh/en",
    ),
    "qwen3-asr-0.6b": _SherpaSpec(
        repo="csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        files={
            "conv_frontend": "conv_frontend.onnx",
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "tokenizer": "tokenizer",
        },
        factory="from_qwen3_asr",
        tier="multilingual",
        weight_note="int8 sherpa port",
    ),
}


def _download_model_files(name: str) -> None:
    spec = _MODEL_FILES[name]
    dest = MODELS_DIR / name
    if (dest / ".complete").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    if spec.repo:
        import json
        import urllib.error
        import urllib.request

        from huggingface_hub import hf_hub_download

        files: list[str] = []
        for rel in spec.files.values():
            url = f"https://huggingface.co/api/models/{spec.repo}/tree/main/{rel}"
            try:
                entries = json.load(urllib.request.urlopen(url))
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                entries = None
            if entries is None:
                files.append(rel)
            else:
                files.extend(e["path"] for e in entries if e["type"] == "file")
        for rel in files:
            hf_hub_download(spec.repo, rel, repo_type="model", local_dir=dest)
    else:
        import tarfile
        import urllib.request

        tarball = dest / "model.tar.bz2"
        assert spec.url is not None
        urllib.request.urlretrieve(spec.url, tarball)
        with tarfile.open(tarball, "r:bz2") as tf:
            tf.extractall(dest)
        tarball.unlink()
    (dest / ".complete").touch()


def make_adapter(name: str) -> Adapter:
    t0 = time.perf_counter()
    if name.startswith("whispercpp-"):
        sizes = {
            "whispercpp-large-v3-turbo-q5_0": "large-v3-turbo-q5_0",  # STT baseline
            "whispercpp-base": "base",
            "whispercpp-small": "small",
            "whispercpp-medium-en": "medium.en",
        }
        en_only = name == "whispercpp-medium-en"
        adapter = _WhisperCpp(sizes[name])
        adapter.name = name
        adapter.tier = "en-only" if en_only else "multilingual"
        adapter.modes = ("A",) if en_only else ("A", "B")
        adapter.weight_note = (
            "auto-downloaded via pywhispercpp (547 MB for q5_0)"
            if name == "whispercpp-large-v3-turbo-q5_0"
            else ""
        )
    elif name in (
        "faster-whisper-small",
        "faster-whisper-medium",
    ):
        adapter = _FasterWhisper(name.removeprefix("faster-whisper-"))
        adapter.name = name
        adapter.tier = "multilingual"
        adapter.weight_note = "int8, beam 5"
    elif name == "whisper-small":
        adapter = _OpenAIWhisper("small")
        adapter.name = name
        adapter.tier = "multilingual"
        adapter.weight_note = "openai-whisper (torch, fp32)"
    elif name == "moonshine-streaming-medium":
        adapter = _Moonshine()
    elif name == "ark-asr-0.6b":
        adapter = _ArkAsr()
    elif name in _MODEL_FILES:
        spec = _MODEL_FILES[name]
        _download_model_files(name)
        adapter = _Sherpa(name, spec.files, spec.factory, spec.factory_kwargs)
        adapter.name = name
        adapter.tier = spec.tier
        adapter.modes = ("A",) if spec.tier == "en-only" else ("A", "B")
        adapter.weight_note = spec.weight_note
    else:
        raise ValueError(f"unknown model: {name}")
    adapter.load_s = max(adapter.load_s, time.perf_counter() - t0)
    return adapter


def available_models() -> list[str]:
    return [
        "whispercpp-large-v3-turbo-q5_0",  # STT baseline (product default)
        "whispercpp-base",
        "whispercpp-small",
        "whispercpp-medium-en",
        "whisper-small",
        "faster-whisper-small",
        "faster-whisper-medium",
        "moonshine-streaming-medium",
        "parakeet-tdt-0.6b-v2",
        "parakeet-tdt-0.6b-v3",
        "dolphin-small",
        "funasr-nano-2512",
        "sensevoice",
        "qwen3-asr-0.6b",
        "ark-asr-0.6b",
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_model(model_name: str, samples: list[Sample], use_stream: bool) -> dict:
    print(f"\n=== {model_name} ===")
    try:
        adapter = make_adapter(model_name)
    except Exception as e:  # noqa: BLE001 - any model load failure must not stop the benchmark
        print(f"  LOAD FAILED: {e!r}")
        return {"model": model_name, "error": repr(e)}
    print(f"  load: {adapter.load_s:.1f}s  tier: {adapter.tier}  {adapter.weight_note}")

    results = {"model": model_name, "tier": adapter.tier, "env": env_info()}
    rows: list[dict] = []
    for sample in samples:
        if sample.mode not in adapter.modes:
            print(
                f"  {sample.id:12s} skipped ({adapter.tier} model, mode {sample.mode})"
            )
            rows.append(
                {
                    "sample": sample.id,
                    "note": f"skipped ({adapter.tier} model, mode {sample.mode})",
                }
            )
            continue
        try:
            if use_stream and adapter.stream_capable:
                info = adapter.stream_info(sample)
                segments = info["segments"]
                wall = info["wall"]
                hyp = " ".join(t for _, _, t in (segments or []))
            else:
                text, info = adapter.transcribe(sample)
                hyp = text
                wall = info["wall"]
                segments = info["segments"]
            wer, cer = _wer_cer(sample.ref, hyp)
            rows.append(
                {
                    "sample": sample.id,
                    "category": sample.category,
                    "duration_s": sample.duration(),
                    "wer": wer,
                    "cer": cer,
                    "block_wer": _block_wer(sample.ref, hyp, sample.blocks),
                    "wall_s": wall,
                    "rtf": wall / sample.duration() if sample.duration() else None,
                    "hyp": hyp,
                    "segments": segments,
                    "updates": info.get("updates") if use_stream else None,
                }
            )
            print(
                f"  {sample.id:12s} WER {wer:.3f}  CER {cer:.3f}  "
                f"RTF {wall / sample.duration():.3f}  ({wall:.1f}s)"
            )
        except Exception as e:  # noqa: BLE001 - one bad sample must not abort the run
            print(f"  {sample.id}: FAILED {e!r}")
            rows.append({"sample": sample.id, "error": repr(e)})
    results["samples"] = rows
    results["peak_rss_mb"] = peak_rss_mb()
    return results


def _avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _merged_table() -> list[dict]:
    """One row per model across all result files (the merged benchmark view).

    Columns: en-only WER (word), zh-only CER (char), en+zh mixed error (mean
    of the per-sample block scores — en WER + zh CER), RTF avg, peak RSS MB.
    """
    rows: list[dict] = []
    for p in sorted(TRANSCRIBE_RESULTS_DIR.glob("*.json")):
        if p.name == "merged.json":
            continue
        r = json.loads(p.read_text())
        if "samples" not in r:
            rows.append({"model": r["model"], "error": r.get("error", "no results")})
            continue
        en_rows = [s for s in r["samples"] if s.get("category") == "en" and "wer" in s]
        zh_rows = [s for s in r["samples"] if s.get("category") == "zh" and "wer" in s]
        mx_rows = [
            s for s in r["samples"] if s.get("category") == "mixed" and "wer" in s
        ]

        def zh_block_score(s: dict) -> float | None:
            for b in s.get("block_wer", []):
                if b.get("metric") == "cer":
                    return b["score"]
            return None

        rows.append(
            {
                "model": r["model"],
                "en_wer": _avg([s["wer"] for s in en_rows]),
                "zh_cer": _avg(
                    [x for x in (zh_block_score(s) for s in zh_rows) if x is not None]
                ),
                "mixed_err": _avg(
                    [
                        v
                        for v in (
                            _avg([float(b["score"]) for b in s["block_wer"]])
                            for s in mx_rows
                            if s.get("block_wer")
                        )
                        if v is not None
                    ]
                ),
                "rtf": _avg([s["rtf"] for s in r["samples"] if "rtf" in s]),
                "rss_mb": r.get("peak_rss_mb"),
            }
        )
    return rows


def _print_merged(rows: list[dict]) -> None:
    print("\n=== Merged benchmark (container) ===")
    print(
        f"{'model':30s} {'en-only WER':>11s} {'zh-only CER':>11s} "
        f"{'en+zh mixed':>11s} {'RTF avg':>8s} {'RSS MB':>7s}"
    )
    for r in rows:
        if "error" in r:
            print(f"{r['model']:30s} {r['error'][:55]:55s}")
            continue
        fmt = lambda v: f"{v:.3f}" if v is not None else "—"
        print(
            f"{r['model']:30s} {fmt(r['en_wer']):>11s} {fmt(r['zh_cer']):>11s} "
            f"{fmt(r['mixed_err']):>11s} {fmt(r['rtf']):>8s} "
            f"{fmt(r['rss_mb']):>7s}"
        )


def _run_isolated(
    name: str, task: str, sample_ids: list[str], use_stream: bool
) -> tuple[dict, int]:
    """Run one model in a fresh subprocess so its memory is fully released."""
    import subprocess

    if task == "translate":
        out = TRANSLATE_RESULTS_DIR / f"{name}.json"
    else:
        out = TRANSCRIBE_RESULTS_DIR / f"{name}.json"
    out.unlink(missing_ok=True)  # never trust a stale file from a previous run

    cmd = [
        sys.executable,
        "-m",
        "interpreter.benchmark",
        "--task",
        task,
        name,
        "--in-process",
        "--no-merged",  # the parent owns merged.json / the merged table
    ]
    if task == "stt":
        cmd += ["--samples", *sample_ids]
    if use_stream:
        cmd.append("--stream")
    env = None
    capi = _onnxruntime_capi_dir()
    if capi is not None:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{capi}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(cmd, check=False, env=env)
    if proc.returncode in (-9, 137):
        # SIGKILL (exit -9 / 137): OOM-killed — does not fit the benchmark budget.
        result = {
            "model": name,
            "error": "excluded: OOM (SIGKILL) — does not fit the benchmark budget",
            "env": env_info(),
        }
        out.write_text(json.dumps(result, indent=2))
        return result, proc.returncode
    if not out.exists():
        return {
            "model": name,
            "error": f"no results written (exit {proc.returncode})",
        }, proc.returncode
    return json.loads(out.read_text()), proc.returncode


# ---------------------------------------------------------------------------
# Translation benchmark (en -> zh)
# ---------------------------------------------------------------------------


def load_translate_manifest() -> list[dict]:
    data = json.loads(TRANSLATE_MANIFEST.read_text(encoding="utf-8"))
    return data["sentences"]


class _HfSeq2SeqTranslator:
    """Dedicated NMT via transformers (opus-mt / M2M100)."""

    def __init__(self, model_id: str, forced_bos: str | None = None) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        t0 = time.perf_counter()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
        self.model: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.forced_bos = (
            self.tokenizer.lang_code_to_id[forced_bos] if forced_bos else None
        )
        self.load_s = time.perf_counter() - t0

    def translate(self, src: str) -> str:
        inputs = self.tokenizer(src, return_tensors="pt", truncation=True)
        kwargs = {"forced_bos_token_id": self.forced_bos} if self.forced_bos else {}
        out = self.model.generate(**inputs, max_length=256, **kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def make_translate_adapter(name: str):
    if name == "opus-mt-en-zh":
        return _HfSeq2SeqTranslator("Helsinki-NLP/opus-mt-en-zh")
    if name == "m2m100-418m":
        return _HfSeq2SeqTranslator("facebook/m2m100_418M", forced_bos="zh")
    raise ValueError(f"unknown translate model: {name}")


def translate_models() -> list[str]:
    return [
        "opus-mt-en-zh",
        "m2m100-418m",
    ]


def run_translate_model(name: str, sentences: list[dict]) -> dict:
    print(f"\n=== {name} ===")
    try:
        adapter = make_translate_adapter(name)
    except Exception as e:  # noqa: BLE001 - any model load failure must not stop the benchmark
        print(f"  LOAD FAILED: {e!r}")
        return {"model": name, "error": repr(e)}
    print(f"  load: {adapter.load_s:.1f}s")

    hyps: list[str] = []
    refs: list[str] = []
    walls: list[float] = []
    rows: list[dict] = []
    for s in sentences:
        t0 = time.perf_counter()
        try:
            hyp = adapter.translate(s["src"])
        except Exception as e:  # noqa: BLE001 - one bad sentence must not abort
            rows.append({"id": s["id"], "error": repr(e)})
            continue
        wall = time.perf_counter() - t0
        hyps.append(hyp)
        refs.append(s["ref"])
        walls.append(wall)
        rows.append(
            {
                "id": s["id"],
                "src": s["src"],
                "ref": s["ref"],
                "hyp": hyp,
                "wall_s": wall,
            }
        )

    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score if hyps else None
    result = {
        "model": name,
        "direction": "en->zh",
        "bleu_zh": bleu,
        "ms_per_sentence": 1000.0 * sum(walls) / len(walls) if walls else None,
        "sentences": rows,
        "peak_rss_mb": peak_rss_mb(),
        "env": env_info(),
    }
    print(f"  BLEU(zh) {bleu:.2f}  {result['ms_per_sentence']:.0f} ms/sentence")
    return result


def _translate_merged_table() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(TRANSLATE_RESULTS_DIR.glob("*.json")):
        if p.name == "merged.json":
            continue
        r = json.loads(p.read_text())
        if "bleu_zh" not in r:
            rows.append({"model": r["model"], "error": r.get("error", "no results")})
            continue
        rows.append(
            {
                "model": r["model"],
                "bleu": r["bleu_zh"],
                "ms": r["ms_per_sentence"],
                "rss_mb": r.get("peak_rss_mb"),
            }
        )
    return rows


def _print_translate_merged(rows: list[dict]) -> None:
    print("\n=== Merged translation benchmark (en->zh) ===")
    print(f"{'model':26s} {'BLEU(zh)':>9s} {'ms/sent':>9s} {'RSS MB':>7s}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:26s} {r['error'][:55]:55s}")
            continue
        fmt = lambda v: f"{v:.2f}" if v is not None else "—"
        print(
            f"{r['model']:26s} {fmt(r['bleu']):>9s} {fmt(r['ms']):>9s} "
            f"{fmt(r['rss_mb']):>7s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="model names (default: all)")
    parser.add_argument(
        "--task",
        choices=("stt", "translate"),
        default="stt",
        help="benchmark task (default: stt)",
    )
    parser.add_argument("--samples", nargs="*", help="sample ids (default: all)")
    parser.add_argument(
        "--stream", action="store_true", help="streaming decode where supported"
    )
    parser.add_argument("--list", action="store_true", help="list models and samples")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="load models in this process (default: one subprocess per model)",
    )
    parser.add_argument(
        "--no-merged",
        action="store_true",
        help="skip the merged table/merged.json (used by per-model subprocesses)",
    )
    parser.add_argument(
        "--record", metavar="WAV", help="record <SECONDS> of audio (host mic)"
    )
    parser.add_argument("seconds", nargs="?", type=float, default=30.0)
    args = parser.parse_args()

    if args.record:
        import sounddevice as sd

        fs = 16000
        print(f"Recording {args.seconds}s to {args.record} ... speak now")
        audio = sd.rec(int(args.seconds * fs), fs, channels=1, dtype="float32")
        sd.wait()
        sf.write(args.record, audio, fs)
        print(f"Saved {args.record} — add it + a reference to data/benchmark/transcribe/manifest.json")
        return

    samples = load_manifest()
    if args.samples:
        samples = [s for s in samples if s.id in args.samples]

    if args.list:
        print("Task:", args.task)
        if args.task == "translate":
            print("Models:", ", ".join(translate_models()))
            print(
                "Sentences:",
                len(load_translate_manifest()),
                "en->zh (wmt19 validation)",
            )
            return
        print("Models:", ", ".join(available_models()))
        print("Samples:")
        for s in samples:
            print(f"  {s.id}  mode={s.mode}  {s.duration():.1f}s  lang={s.lang}")
        return

    if args.task == "translate":
        sentences = load_translate_manifest()
        names = args.models or translate_models()
        TRANSLATE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        single = len(names) == 1
        for name in names:
            if args.in_process or single:
                result = run_translate_model(name, sentences)
                if "error" in result and "bleu_zh" not in result:
                    continue
                out = TRANSLATE_RESULTS_DIR / f"{name}.json"
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"\n=== {name} (isolated subprocess) ===")
                result, code = _run_isolated(name, "translate", [], args.stream)
                if code in (-9, 137):
                    print(f"  {result['error']}")
            print(f"  -> {TRANSLATE_RESULTS_DIR / f'{name}.json'}")
        merged = _translate_merged_table()
        (TRANSLATE_RESULTS_DIR / "merged.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False)
        )
        if not args.no_merged:
            _print_translate_merged(merged)
        return

    names = args.models or available_models()
    TRANSCRIBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    single = len(names) == 1
    for name in names:
        if args.in_process or single:
            result = run_model(name, samples, args.stream)
            if "error" in result and "samples" not in result:
                continue
            out = TRANSCRIBE_RESULTS_DIR / f"{name}.json"
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n=== {name} (isolated subprocess) ===")
            result, code = _run_isolated(
                name, "stt", [s.id for s in samples], args.stream
            )
            if code in (-9, 137):
                print(f"  {result['error']}")
        print(f"  -> {TRANSCRIBE_RESULTS_DIR / f'{name}.json'}")

    merged = _merged_table()
    (TRANSCRIBE_RESULTS_DIR / "merged.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False)
    )
    if not args.no_merged:
        _print_merged(merged)


if __name__ == "__main__":
    main()
