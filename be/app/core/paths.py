from pathlib import Path

# Resolve runtime paths from the source tree instead of the shell working directory.
# This keeps generated files stable whether the app is launched from the repo root or `be/`.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent

STORAGE_ROOT = PROJECT_ROOT / "storage"
UPLOADS_DIR = STORAGE_ROOT / "uploads"
LOGS_DIR = STORAGE_ROOT / "logs"

OMR_DIR = UPLOADS_DIR / "omr"
ANSWER_KEYS_DIR = UPLOADS_DIR / "answer_keys"
OMR_ANSWER_KEYS_DIR = ANSWER_KEYS_DIR / "omr"
TEMP_DIR = UPLOADS_DIR / "temp"
OMR_TEMPLATE_DIR = UPLOADS_DIR / "omr_templates"
OMR_DATA_DIR = UPLOADS_DIR / "omr_data"
OMR_PROFILE_DIR = OMR_DATA_DIR / "profiles"


def ensure_runtime_dirs() -> None:
    for path in [
        STORAGE_ROOT,
        UPLOADS_DIR,
        LOGS_DIR,
        OMR_DIR,
        ANSWER_KEYS_DIR,
        OMR_ANSWER_KEYS_DIR,
        TEMP_DIR,
        OMR_TEMPLATE_DIR,
        OMR_DATA_DIR,
        OMR_PROFILE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
