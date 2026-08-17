"""Self-hosted Vietnamese OCR using RapidOCR (ONNX detection) + VietOCR (PyTorch-GPU recognition).

Pipeline:
  1. RapidOCR (ONNX) detects text regions (bounding boxes) in the page image
  2. Each detected region is cropped and fed to VietOCR (PyTorch) for high-accuracy Vietnamese text recognition
  3. Results are sorted by reading order (top-to-bottom, left-to-right)
  4. OCR results are cached by image hash for efficiency

This runs entirely local: RapidOCR on CPU (fast, stable, ONNX) and VietOCR on GPU (PyTorch).
No API calls, no package conflicts.
"""

import hashlib
import io
import json
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..config import (
    GEMINI_OCR_CACHE_PATH,
    USE_GPU,
)

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = GEMINI_OCR_CACHE_PATH


class PaddleVietOCR:
    """Vietnamese OCR engine combining RapidOCR (ONNX) detection with VietOCR (PyTorch) recognition.

    RapidOCR handles text detection (finding where text is) without PaddlePaddle conflicts.
    VietOCR handles text recognition (reading Vietnamese text accurately) on GPU.

    Args:
        use_gpu: Whether to use GPU for VietOCR.
        cache_path: Path to JSON cache file for OCR results.
        vietocr_model: VietOCR model weight name ('vgg_transformer' or 'vgg_seq2seq').
    """

    def __init__(
        self,
        use_gpu: bool = USE_GPU,
        cache_path: Optional[Path] = None,
        vietocr_model: str = "vgg_transformer",
    ):
        self._use_gpu = use_gpu
        self.cache_path = Path(cache_path or _DEFAULT_CACHE_PATH)
        self._vietocr_model_name = vietocr_model

        self._rapid_ocr = None
        self._vietocr_predictor = None

        self._cache: Dict[str, str] = self._load_cache()

    # ── Cache ─────────────────────────────────────────────────────────

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded {len(data)} cached OCR results")
            return data
        except Exception as e:
            logger.warning(f"Could not load OCR cache: {e}")
            return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save OCR cache: {e}")

    def flush_cache(self) -> None:
        """Force save cache to disk."""
        self._save_cache()
        logger.info(f"OCR cache saved: {len(self._cache)} entries")

    # ── Hashing ───────────────────────────────────────────────────────

    @staticmethod
    def _image_hash(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()[:24]

    # ── Model Loading ─────────────────────────────────────────────────

    def _load_rapid_ocr(self):
        """Lazy-load RapidOCR (ONNX detector)."""
        if self._rapid_ocr is not None:
            return

        from rapidocr_onnxruntime import RapidOCR
        logger.info("Loading RapidOCR (ONNX) engine...")
        # RapidOCR loads ONNX models. Since it's ONNX, it is stable and doesn't conflict.
        self._rapid_ocr = RapidOCR()
        logger.info("RapidOCR loaded successfully")

    def _load_vietocr(self):
        """Lazy-load VietOCR predictor on GPU (or CPU)."""
        if self._vietocr_predictor is not None:
            return

        import torch
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        logger.info(f"Loading VietOCR model: {self._vietocr_model_name}...")

        config = Cfg.load_config_from_name(self._vietocr_model_name)
        # Load onto GPU if available and configured
        config["device"] = "cuda:0" if (self._use_gpu and torch.cuda.is_available()) else "cpu"
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = False

        self._vietocr_predictor = Predictor(config)
        logger.info(f"VietOCR loaded successfully (device={config['device']})")

    # ── Core Pipeline ─────────────────────────────────────────────────

    def _detect_boxes(self, image: Image.Image) -> List[dict]:
        """Detect text bounding boxes using RapidOCR.

        Returns a list of dicts: {'box': [[x1,y1],...], 'rapid_text': str, 'conf': float}
        """
        self._load_rapid_ocr()

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        result, elapse = self._rapid_ocr(img_array)

        regions = []
        if result:
            for item in result:
                box, text, conf = item
                # box is a list of 4 points, each can be float from ONNX -> cast to int
                box_ints = [[int(p[0]), int(p[1])] for p in box]
                regions.append({
                    "box": box_ints,
                    "rapid_text": text,
                    "conf": float(conf)
                })
        return regions

    def _crop_box(self, image: Image.Image, box: List[List[int]], padding: int = 3) -> Image.Image:
        """Crop a text region from the image."""
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(image.width, max(xs) + padding)
        y_max = min(image.height, max(ys) + padding)
        return image.crop((x_min, y_min, x_max, y_max))

    def _recognize(self, crop: Image.Image) -> str:
        """Recognize Vietnamese text from a cropped text-line image using VietOCR."""
        self._load_vietocr()
        try:
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            return self._vietocr_predictor.predict(crop).strip()
        except Exception as e:
            logger.debug(f"VietOCR recognition error: {e}")
            return ""

    def _sort_reading_order(self, regions: List[dict]) -> List[dict]:
        """Sort regions top-to-bottom, left-to-right with line grouping."""
        if not regions:
            return []

        # Annotate with center_y, min_x, height
        items = []
        for region in regions:
            box = region["box"]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            center_y = sum(ys) / len(ys)
            min_x = min(xs)
            height = max(ys) - min(ys)
            items.append((center_y, min_x, height, region))

        items.sort(key=lambda b: b[0])

        # Group into lines by Y proximity
        lines = [[items[0]]]
        for i in range(1, len(items)):
            prev_y = lines[-1][-1][0]
            curr_y = items[i][0]
            threshold = max(lines[-1][-1][2], items[i][2]) * 0.5
            if abs(curr_y - prev_y) < threshold:
                lines[-1].append(items[i])
            else:
                lines.append([items[i]])

        # Sort within each line by X
        result = []
        for line in lines:
            line.sort(key=lambda b: b[1])
            result.extend([b[3] for b in line])
        return result

    def ocr_page(self, image: Image.Image, page_info: str = "") -> str:
        """OCR a full page: RapidOCR detect → VietOCR recognize.

        Args:
            image: PIL Image of the page.
            page_info: Optional context string for logging.

        Returns:
            Full page text, lines separated by newlines.
        """
        img_hash = self._image_hash(image)
        cache_key = f"paddle_vietocr:{img_hash}"

        if cache_key in self._cache:
            logger.debug(f"OCR cache hit: {page_info or cache_key[:20]}")
            return self._cache[cache_key]

        logger.debug(f"OCR processing: {page_info}")

        # Step 1: Detect
        regions = self._detect_boxes(image)
        if not regions:
            logger.debug(f"No text detected: {page_info}")
            self._cache[cache_key] = ""
            return ""

        # Step 2: Sort reading order
        sorted_regions = self._sort_reading_order(regions)

        # Step 3: Crop + VietOCR recognize
        lines = []
        for region in sorted_regions:
            crop = self._crop_box(image, region["box"])
            
            # Try VietOCR first, fallback to RapidOCR's internal text
            try:
                text = self._recognize(crop)
            except Exception:
                text = ""
                
            if not text:
                text = region.get("rapid_text", "")
                
            if text:
                lines.append(text)

        full_text = "\n".join(lines)

        # Cache
        self._cache[cache_key] = full_text
        if len(self._cache) % 10 == 0:
            self._save_cache()

        return full_text

    def ocr_pages(
        self,
        images: List[Image.Image],
        page_infos: Optional[List[str]] = None,
    ) -> List[str]:
        """OCR multiple pages."""
        if page_infos is None:
            page_infos = [f"page {i+1}" for i in range(len(images))]

        results = []
        for image, info in zip(images, page_infos):
            results.append(self.ocr_page(image, page_info=info))

        self._save_cache()
        return results
