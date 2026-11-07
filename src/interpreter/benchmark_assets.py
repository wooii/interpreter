"""Materialize the benchmark corpus under ``data/benchmark/``.

The manifests are tracked in ``benchmark/``; the heavy audio is a
gitignored cache that can be deleted and rebuilt. Sources are public and need
no authentication:

- STT English: LibriSpeech ``test-clean`` (CC-BY-4.0, OpenSLR SLR12)
- STT Chinese: AISHELL-1 ``S0002`` speaker archive (Apache-2.0, Hugging Face)
- STT mixed clips: generated deterministically from the en/zh clips (0.4 s gaps)
- Speaker: LibriSpeech ``dev-clean`` clips listed in the tracked speaker manifest

LibriSpeech's ``test-clean``/``dev-clean`` archives are ~346/338 MB and contain
the whole split; only the needed members are extracted. Archives stay in
``data/benchmark/.downloads/`` until a task finishes, then are removed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from interpreter import DATA_DIR, PROJECT_ROOT

BENCH_DIR = DATA_DIR / "benchmark"
TRANSCRIBE_DIR = BENCH_DIR / "transcribe"
TRANSLATE_DIR = BENCH_DIR / "translate"
TRANSCRIBE_SAMPLES_DIR = TRANSCRIBE_DIR / "samples"
TRANSCRIBE_RESULTS_DIR = TRANSCRIBE_DIR / "results"
TRANSLATE_RESULTS_DIR = TRANSLATE_DIR / "results"
SPEAKER_DIR = BENCH_DIR / "speaker"
SPEAKER_SAMPLES_DIR = SPEAKER_DIR / "samples"
SPEAKER_RESULTS_DIR = SPEAKER_DIR / "results"
SPEAKER_BUILT_MANIFEST = SPEAKER_DIR / "manifest.json"
DOWNLOADS_DIR = BENCH_DIR / ".downloads"

MANIFEST_DIR = PROJECT_ROOT / "benchmark"
TRANSCRIBE_MANIFEST = MANIFEST_DIR / "transcribe.json"
TRANSLATE_MANIFEST = MANIFEST_DIR / "translate.json"
SPEAKER_MANIFEST = MANIFEST_DIR / "speaker.json"

SAMPLE_RATE = 16000
MIX_GAP_SECONDS = 0.4

_OPENSLR = "https://www.openslr.org/resources/12"
_TEST_CLEAN = (
    f"{_OPENSLR}/test-clean.tar.gz",
    "32fa31d27d2e1cad72775fee3f4849a9",
    346_663_984,
)
_DEV_CLEAN = (
    f"{_OPENSLR}/dev-clean.tar.gz",
    "42e2234ba48799c1f50f24a7926300a1",
    337_926_286,
)
_AISHELL_REPO = "AISHELL/AISHELL-1"
_AISHELL_FILE = "data_aishell/wav/S0002.tar.gz"
_AISHELL_SHA256 = "5700ffa081f42c5a1be701147e680490fa494db959ca3c10ae205c3658af159f"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, *, md5: str | None = None, sha256: str | None = None) -> None:
    if md5 is not None and _digest(path, "md5") != md5:
        raise RuntimeError(f"checksum mismatch (md5) for {path}")
    if sha256 is not None and _digest(path, "sha256") != sha256:
        raise RuntimeError(f"checksum mismatch (sha256) for {path}")


def _matches(path: Path, *, md5: str | None = None, sha256: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        _verify(path, md5=md5, sha256=sha256)
        return True
    except RuntimeError:
        return False


def _download(
    url: str,
    dest: Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    sha256: str | None = None,
) -> Path:
    if _matches(dest, md5=md5, sha256=sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = dest.name
    if size:
        print(f"Downloading {label} ({size / 1_000_000:.0f} MB) ...", flush=True)
    else:
        print(f"Downloading {label} ...", flush=True)
    try:
        downloaded = 0
        last_pct = -1
        with urllib.request.urlopen(url, timeout=120) as src, dest.open("wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if size:
                    pct = int(downloaded * 100 / size)
                    if pct < 100 and pct // 10 != last_pct // 10:
                        print(f"  {pct}%", end="\r", flush=True)
                        last_pct = pct
        if size:
            print("  100%", flush=True)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {url}: {exc}") from exc
    _verify(dest, md5=md5, sha256=sha256)
    return dest


def _download_hf(filename: str, dest: Path, sha256: str) -> Path:
    if _matches(dest, sha256=sha256):
        return dest
    dest.unlink(missing_ok=True)
    from huggingface_hub import hf_hub_download

    print(f"Downloading {Path(filename).name} (AISHELL-1) ...", flush=True)
    try:
        got = Path(
            hf_hub_download(
                repo_id=_AISHELL_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(DOWNLOADS_DIR),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"download failed: {_AISHELL_REPO}/{filename}: {exc}"
        ) from exc
    if got.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(got, dest)
    _verify(dest, sha256=sha256)
    return dest


def _extract_tar(
    tar_path: Path, wanted: dict[str, Path], *, match_basename: bool = False
) -> None:
    """Extract only ``wanted`` members (name -> destination) from a tarball."""
    remaining = dict(wanted)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key = Path(member.name).name if match_basename else member.name
            target = remaining.get(key)
            if target is None:
                continue
            src = tf.extractfile(member)
            if src is None:
                raise RuntimeError(f"cannot read {member.name} from {tar_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            with target.open("wb") as out:
                shutil.copyfileobj(src, out)
            del remaining[key]
            if not remaining:
                break
    if remaining:
        sample = ", ".join(list(remaining)[:3])
        raise RuntimeError(
            f"{tar_path.name}: {len(remaining)} expected members missing ({sample})"
        )


def _transcribe_samples() -> list[dict]:
    return _load_json(TRANSCRIBE_MANIFEST)["samples"]


def _sample_path(sample: dict) -> Path:
    return TRANSCRIBE_DIR / sample["path"]


def _all_present(paths: list[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _fetch_librispeech_stt() -> None:
    samples = {
        s["source"][len("librispeech:") :]: s
        for s in _transcribe_samples()
        if s.get("source", "").startswith("librispeech:")
    }
    url, md5, size = _TEST_CLEAN
    tar_path = _download(url, DOWNLOADS_DIR / "test-clean.tar.gz", size=size, md5=md5)

    with tempfile.TemporaryDirectory(dir=DOWNLOADS_DIR) as tmp:
        tmp_dir = Path(tmp)
        wanted = {
            f"LibriSpeech/{rel}": tmp_dir / f"{rel.rsplit('/', 1)[-1]}"
            for rel in samples
        }
        _extract_tar(tar_path, wanted)
        for rel, sample in samples.items():
            flac = wanted[f"LibriSpeech/{rel}"]
            audio, sr = sf.read(str(flac), dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"{flac.name}: expected {SAMPLE_RATE} Hz, got {sr}")
            target = _sample_path(sample)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(target), audio, sr, subtype="PCM_16")


def _fetch_aishell_stt() -> None:
    samples = {
        s["source"][len("aishell:") :]: s
        for s in _transcribe_samples()
        if s.get("source", "").startswith("aishell:")
    }
    tar_path = _download_hf(
        _AISHELL_FILE,
        DOWNLOADS_DIR / _AISHELL_FILE,
        sha256=_AISHELL_SHA256,
    )
    wanted = {
        f"{utterance}.wav": _sample_path(sample)
        for utterance, sample in samples.items()
    }
    _extract_tar(tar_path, wanted, match_basename=True)


def _build_mixed() -> None:
    samples = _transcribe_samples()
    by_id = {s["id"]: s for s in samples}
    gap = np.zeros(int(MIX_GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)

    for sample in samples:
        parts = sample.get("parts")
        if not parts:
            continue
        pieces: list[np.ndarray] = []
        for index, part_id in enumerate(parts):
            if index:
                pieces.append(gap)
            source = by_id.get(part_id)
            if source is None:
                raise RuntimeError(f"{sample['id']}: unknown source sample {part_id}")
            audio, sr = sf.read(
                str(_sample_path(source)), dtype="float32", always_2d=False
            )
            if sr != SAMPLE_RATE:
                raise RuntimeError(
                    f"{source['id']}: expected {SAMPLE_RATE} Hz, got {sr}"
                )
            pieces.append(audio)
        target = _sample_path(sample)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), np.concatenate(pieces), SAMPLE_RATE, subtype="PCM_16")


def _cleanup_downloads(*names: str) -> None:
    for name in names:
        (DOWNLOADS_DIR / name).unlink(missing_ok=True)


def ensure_transcribe_assets(force: bool = False) -> None:
    """Download/build all STT audio into ``data/benchmark/transcribe/samples``."""
    marker = TRANSCRIBE_DIR / ".complete"
    samples = _transcribe_samples()
    paths = [_sample_path(s) for s in samples]
    if not force and marker.is_file() and _all_present(paths):
        return

    print("Preparing STT benchmark assets ...", flush=True)
    TRANSCRIBE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _fetch_librispeech_stt()
    _fetch_aishell_stt()
    _build_mixed()

    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} STT sample files still missing under {TRANSCRIBE_SAMPLES_DIR}"
        )
    marker.touch()
    _cleanup_downloads("test-clean.tar.gz", _AISHELL_FILE)
    print(f"STT benchmark assets ready: {len(paths)} samples", flush=True)


def _speaker_paths() -> list[str]:
    manifest = _load_json(SPEAKER_MANIFEST)["speakers"]
    return [
        rel
        for speaker in manifest.values()
        for key in ("enroll", "test")
        for rel in speaker[key]
    ]


def ensure_speaker_assets(force: bool = False) -> None:
    """Download the LibriSpeech dev-clean subset listed in the speaker manifest."""
    marker = SPEAKER_DIR / ".complete"
    rel_paths = _speaker_paths()
    targets = [SPEAKER_SAMPLES_DIR / rel for rel in rel_paths]
    if not force and marker.is_file() and _all_present(targets):
        return

    print("Preparing speaker benchmark assets ...", flush=True)
    url, md5, size = _DEV_CLEAN
    tar_path = _download(url, DOWNLOADS_DIR / "dev-clean.tar.gz", size=size, md5=md5)

    wanted = {
        f"LibriSpeech/dev-clean/{rel}": SPEAKER_SAMPLES_DIR / rel for rel in rel_paths
    }
    _extract_tar(tar_path, wanted)

    missing = [p for p in targets if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} speaker clips still missing under {SPEAKER_SAMPLES_DIR}"
        )
    marker.touch()
    _cleanup_downloads("dev-clean.tar.gz")
    print(f"Speaker benchmark assets ready: {len(targets)} clips", flush=True)


def ensure_assets(task: str, force: bool = False) -> None:
    """Ensure the local corpus for ``task``; translate needs only its manifest."""
    if task == "stt":
        ensure_transcribe_assets(force=force)
    elif task == "speaker":
        ensure_speaker_assets(force=force)
    elif task == "translate":
        print(
            "Translate benchmark uses the tracked text manifest; no audio assets.",
            flush=True,
        )
    else:
        raise ValueError(f"unknown benchmark task: {task}")
