"""
Configuration module for Biology RAG project.
Loads environment variables and provides shared configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _path_from_env(env_name: str, default: Path, base_dir: Path = PROJECT_ROOT) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _database_child_path_from_env(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    parts = path.parts
    if parts and parts[0].lower() == "database":
        path = Path(*parts[1:]) if len(parts) > 1 else Path()
    return (PERSIST_DIR / path).resolve()


DATA_DIR = _path_from_env("RAG_DATA_DIR", PROJECT_ROOT / "datasources")
PERSIST_DIR = _path_from_env("RAG_DATABASE_DIR", PROJECT_ROOT / "database")
IMAGES_DIR = PERSIST_DIR / "images"
PROCESSED_FILES_LOG = PERSIST_DIR / "processed_files.txt"
PROCESSED_IMAGES_LOG = PERSIST_DIR / "processed_images.txt"

HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract").strip() or "tesseract"
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip() or None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# --- Gemini Multi-Key Rotation & Rate Limiting ---
_raw_gemini_keys = os.getenv("GEMINI_API_KEYS", "").strip()
if _raw_gemini_keys:
    GEMINI_API_KEYS = [k.strip() for k in _raw_gemini_keys.split(",") if k.strip()]
else:
    # Fallback to single GEMINI_API_KEY
    GEMINI_API_KEYS = [GEMINI_API_KEY] if GEMINI_API_KEY else []
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "15"))

# --- OCR Strategy ---
OCR_STRATEGY = os.getenv("OCR_STRATEGY", "hybrid").strip().lower()
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
GEMINI_OCR_ENABLED = os.getenv("GEMINI_OCR_ENABLED", "true").lower() == "true"
GEMINI_OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "").strip() or GEMINI_MODEL
GEMINI_OCR_CACHE_PATH = _database_child_path_from_env(
    "GEMINI_OCR_CACHE_PATH",
    PERSIST_DIR / "gemini_ocr_cache.json",
)
# Minimum characters from PDF text layer to consider it "good enough" (skip OCR)
PDF_TEXT_MIN_CHARS = int(os.getenv("PDF_TEXT_MIN_CHARS", "200"))
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch16")
OWL_VIT_MODEL = os.getenv("OWL_VIT_MODEL", "google/owlvit-base-patch32")
OWL_VIT_CONFIDENCE_THRESHOLD = float(
    os.getenv("OWL_VIT_CONFIDENCE_THRESHOLD", "0.1"))
IMAGE_EXTRACTION_VERSION = os.getenv(
    "IMAGE_EXTRACTION_VERSION", "v15_per_variant")
IMAGE_CAPTION_ENABLED = os.getenv(
    "IMAGE_CAPTION_ENABLED", "true").lower() == "true"
IMAGE_CAPTION_MODEL = os.getenv(
    "IMAGE_CAPTION_MODEL",
    "5CD-AI/Vintern-1B-v2",
)
IMAGE_CAPTION_MAX_NEW_TOKENS = int(
    os.getenv("IMAGE_CAPTION_MAX_NEW_TOKENS", "96"))
IMAGE_CAPTION_CACHE_PATH = _database_child_path_from_env(
    "IMAGE_CAPTION_CACHE_PATH",
    PERSIST_DIR / "image_caption_cache.json",
)
IMAGE_REVIEW_MANIFEST_PATH = _database_child_path_from_env(
    "IMAGE_REVIEW_MANIFEST_PATH",
    PERSIST_DIR / "image_review_manifest.jsonl",
)

TEXT_COLLECTION_NAME = "biology_text"
IMAGE_COLLECTION_NAME = "biology_images"
IMAGE_METADATA_COLLECTION_NAME = "biology_image_metadata"
STATUS_COLLECTION_NAME = "processing_status"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 120

LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 500
LLM_TOP_P = 0.75

RETRIEVER_K = 3
# How many candidates to pull from the vector store before relevance gating.
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "8"))
# Max chunks kept after gating (upper bound on context size).
RETRIEVER_MAX_K = int(os.getenv("RETRIEVER_MAX_K", "4"))
# Relative distance gate: keep a chunk only if its distance is within
# (1 + margin) * best_distance. Drops off-topic chunks that are far from the
# best match (e.g. unrelated pages bleeding into the answer).
RETRIEVER_DISTANCE_MARGIN = float(os.getenv("RETRIEVER_DISTANCE_MARGIN", "0.3"))
IMAGE_RETRIEVER_K = int(os.getenv("IMAGE_RETRIEVER_K", "5"))
IMAGE_RETRIEVER_FETCH_K = int(os.getenv("IMAGE_RETRIEVER_FETCH_K", "48"))
IMAGE_METADATA_FETCH_K = int(os.getenv("IMAGE_METADATA_FETCH_K", "64"))
IMAGE_RELEVANCE_THRESHOLD = float(
    os.getenv("IMAGE_RELEVANCE_THRESHOLD", "0.36"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Tesseract PSM mode (page segmentation) ---
TESSERACT_PSM = os.getenv("TESSERACT_PSM", "3")
