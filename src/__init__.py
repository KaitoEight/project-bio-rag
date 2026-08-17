"""Biology RAG package."""

# Monkeypatch numpy for compatibility with older libraries (like imgaug and paddleocr) under NumPy 2.x
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, str, bytes]
    }
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "typeDict"):
    if hasattr(np, "sctypeDict"):
        np.typeDict = np.sctypeDict
    else:
        np.typeDict = {}



from src.config import (
    DATA_DIR,
    PERSIST_DIR,
    IMAGES_DIR,
    POPPLER_PATH,
    TESSERACT_CMD,
    HF_TOKEN,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CLIP_MODEL,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    IMAGE_METADATA_COLLECTION_NAME,
    STATUS_COLLECTION_NAME,
)

__all__ = [
    "DATA_DIR",
    "PERSIST_DIR",
    "IMAGES_DIR",
    "POPPLER_PATH",
    "TESSERACT_CMD",
    "HF_TOKEN",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "CLIP_MODEL",
    "TEXT_COLLECTION_NAME",
    "IMAGE_COLLECTION_NAME",
    "IMAGE_METADATA_COLLECTION_NAME",
    "STATUS_COLLECTION_NAME",
]
