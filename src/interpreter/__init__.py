from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR: Path = DATA_DIR / "models"
TRANSCRIBE_MODELS_DIR: Path = MODELS_DIR / "transcribe"
TRANSLATE_MODELS_DIR: Path = MODELS_DIR / "translate"
SPEAKER_MODELS_DIR: Path = MODELS_DIR / "speaker"
