"""Document loaders for PDF files."""

import glob
import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..config import (
    OCR_DPI,
    OCR_STRATEGY,
    PDF_TEXT_MIN_CHARS,
    POPPLER_PATH,
    TESSERACT_CMD,
    TESSERACT_PSM,
)
from .cleaner import clean_vietnamese_text

logger = logging.getLogger(__name__)


class SimpleLoader:
    """Load PDFs using PyPDFLoader with image extraction."""

    def load_pdf(self, pdf_file: str) -> List[Document]:
        docs = PyPDFLoader(pdf_file, extract_images=True).load()
        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)
        return docs

    def load_dir(self, dir_path: str) -> List[Document]:
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        if not pdf_files:
            raise ValueError(f"No PDF files found in {dir_path}")

        all_docs = []
        for pdf_file in pdf_files:
            try:
                all_docs.extend(self.load_pdf(pdf_file))
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        return all_docs


class RobustOCRLoader:
    """Load PDFs using OCR (Tesseract) for better Vietnamese support."""

    def load_pdf(self, pdf_file: str) -> List[Document]:
        import time
        from pdf2image import convert_from_path
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        docs = []
        try:
            images = convert_from_path(pdf_file, poppler_path=POPPLER_PATH)
            total_pages = len(images)
            logger.info(f"[{Path(pdf_file).name}] Starting OCR on {total_pages} pages")

            for i, img in enumerate(images):
                start_time = time.time()
                raw_text = pytesseract.image_to_string(img, lang="vie")
                elapsed = time.time() - start_time

                logger.info(
                    f"[{Path(pdf_file).name}] Page {i + 1}/{total_pages} completed in {elapsed:.2f}s"
                )

                cleaned_text = clean_vietnamese_text(raw_text)
                if cleaned_text and len(cleaned_text) > 10:
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={"source": Path(pdf_file).name, "page": i + 1},
                    )
                    docs.append(doc)
        except Exception as e:
            logger.error(f"OCR error for {pdf_file}: {e}")
        return docs

    def load_dir(self, dir_path: str) -> List[Document]:
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        all_docs = []
        for pdf_file in pdf_files:
            all_docs.extend(self.load_pdf(pdf_file))
        return all_docs


class HybridOCRLoader:
    """Hybrid OCR loader with 3-tier strategy for maximum accuracy.

    Tier 1: PDF Text Layer (pymupdf) — fast, free, if text layer exists
    Tier 2: PaddleOCR + VietOCR — best Vietnamese accuracy, runs on local GPU
    Tier 3: Enhanced Tesseract — fallback with DPI 300 + image preprocessing

    Usage:
        loader = HybridOCRLoader(strategy="hybrid")
        docs = loader.load_pdf("path/to/textbook.pdf")

    Strategies:
        "hybrid"    : Tier 1 → 2 → 3 (default)
        "paddle"    : Only PaddleOCR + VietOCR
        "tesseract" : Only enhanced Tesseract
        "pdf_text"  : Only PDF text layer extraction
    """

    def __init__(
        self,
        strategy: str = OCR_STRATEGY,
        dpi: int = OCR_DPI,
        min_text_chars: int = PDF_TEXT_MIN_CHARS,
    ):
        self.strategy = strategy.lower()
        self.dpi = dpi
        self.min_text_chars = min_text_chars
        self._paddle_ocr = None  # Lazy loaded

    def _get_paddle_ocr(self):
        """Lazy-load PaddleVietOCR engine."""
        if self._paddle_ocr is None:
            from .paddle_vietocr import PaddleVietOCR
            self._paddle_ocr = PaddleVietOCR()
            logger.info("PaddleVietOCR engine loaded")
        return self._paddle_ocr

    def _extract_pdf_text(self, pdf_file: str) -> List[Optional[str]]:
        """Tier 1: Extract text from PDF text layer using pymupdf.

        Returns a list of text per page (None if page has no usable text).
        """
        try:
            import fitz  # pymupdf

            doc = fitz.open(pdf_file)
            texts = []
            for page in doc:
                text = page.get_text("text")
                cleaned = clean_vietnamese_text(text) if text else ""
                if cleaned and len(cleaned) >= self.min_text_chars:
                    texts.append(cleaned)
                else:
                    texts.append(None)  # Not enough text, needs OCR
            doc.close()
            return texts
        except Exception as e:
            logger.warning(f"pymupdf text extraction failed: {e}")
            return []

    def _ocr_tesseract_enhanced(self, image) -> str:
        """Tier 3: Enhanced Tesseract OCR with preprocessing.

        Improvements over basic Tesseract:
        - Image preprocessing: grayscale, denoise, adaptive threshold
        - Custom PSM mode for better page segmentation
        """
        import pytesseract
        import numpy as np

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        try:
            # Convert to numpy array for preprocessing
            img_array = np.array(image)

            # Convert to grayscale if color
            if len(img_array.shape) == 3:
                import cv2
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply light denoising
            import cv2
            denoised = cv2.fastNlMeansDenoising(gray, h=10)

            # Adaptive threshold for better binarization
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Convert back to PIL for pytesseract
            from PIL import Image as PILImage
            processed = PILImage.fromarray(binary)

            # OCR with custom config
            custom_config = f"--psm {TESSERACT_PSM} --oem 3"
            raw_text = pytesseract.image_to_string(
                processed, lang="vie", config=custom_config
            )
            return clean_vietnamese_text(raw_text)
        except Exception as e:
            logger.warning(f"Enhanced Tesseract failed, trying basic: {e}")
            # Fallback to basic Tesseract
            try:
                raw_text = pytesseract.image_to_string(image, lang="vie")
                return clean_vietnamese_text(raw_text)
            except Exception as e2:
                logger.error(f"Basic Tesseract also failed: {e2}")
                return ""

    def load_pdf(self, pdf_file: str) -> List[Document]:
        """Load a PDF with the configured OCR strategy.

        Args:
            pdf_file: Path to PDF file.

        Returns:
            List of Documents, one per page with usable text.
        """
        filename = Path(pdf_file).name
        logger.info(f"[{filename}] Loading with strategy: {self.strategy}")

        # ── Strategy: pdf_text only ──
        if self.strategy == "pdf_text":
            return self._load_pdf_text_only(pdf_file)

        # ── Get PDF text layer for all strategies ──
        pdf_texts = []
        if self.strategy in ("hybrid",):
            pdf_texts = self._extract_pdf_text(pdf_file)
            good_pages = sum(1 for t in pdf_texts if t is not None)
            logger.info(
                f"[{filename}] PDF text layer: {good_pages}/{len(pdf_texts)} pages have usable text"
            )

        # ── Convert pages to images for OCR ──
        from pdf2image import convert_from_path

        logger.info(f"[{filename}] Converting PDF to images (DPI={self.dpi})...")
        images = convert_from_path(
            pdf_file, dpi=self.dpi, poppler_path=POPPLER_PATH
        )
        total_pages = len(images)
        logger.info(f"[{filename}] {total_pages} pages converted")

        # Ensure pdf_texts list matches image count
        while len(pdf_texts) < total_pages:
            pdf_texts.append(None)

        docs = []
        ocr_engine_name = self.strategy

        for i, img in enumerate(images):
            page_num = i + 1
            start_time = time.time()
            text = None
            method_used = "none"

            # ── Tier 1: PDF text layer ──
            if self.strategy == "hybrid" and pdf_texts[i] is not None:
                text = pdf_texts[i]
                method_used = "pdf_text"

            # ── Tier 2: PaddleOCR + VietOCR ──
            if text is None and self.strategy in ("hybrid", "paddle"):
                try:
                    engine = self._get_paddle_ocr()
                    text = engine.ocr_page(
                        img, page_info=f"{filename} trang {page_num}"
                    )
                    if text and len(text.strip()) > 10:
                        text = clean_vietnamese_text(text)
                        method_used = "paddle+vietocr"
                    else:
                        text = None
                except Exception as e:
                    logger.warning(
                        f"[{filename}] PaddleOCR+VietOCR failed page {page_num}: {e}"
                    )
                    text = None

            # ── Tier 3: Enhanced Tesseract ──
            if text is None and self.strategy in ("hybrid", "tesseract"):
                text = self._ocr_tesseract_enhanced(img)
                if text and len(text.strip()) > 10:
                    method_used = "tesseract"
                else:
                    text = None

            elapsed = time.time() - start_time
            logger.info(
                f"[{filename}] Page {page_num}/{total_pages} "
                f"({method_used}) {elapsed:.2f}s"
            )

            if text and len(text.strip()) > 10:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "ocr_method": method_used,
                    },
                )
                docs.append(doc)

        # Flush OCR cache
        if self._paddle_ocr is not None:
            self._paddle_ocr.flush_cache()

        logger.info(
            f"[{filename}] Completed: {len(docs)}/{total_pages} pages extracted"
        )
        return docs

    def _load_pdf_text_only(self, pdf_file: str) -> List[Document]:
        """Load using only PDF text layer (no OCR)."""
        filename = Path(pdf_file).name
        texts = self._extract_pdf_text(pdf_file)
        docs = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 10:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": i + 1,
                        "ocr_method": "pdf_text",
                    },
                )
                docs.append(doc)
        logger.info(
            f"[{filename}] PDF text extraction: {len(docs)}/{len(texts)} pages"
        )
        return docs

    def load_dir(self, dir_path: str) -> List[Document]:
        """Load all PDFs in a directory."""
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        if not pdf_files:
            raise ValueError(f"No PDF files found in {dir_path}")

        all_docs = []
        for pdf_file in sorted(pdf_files):
            try:
                all_docs.extend(self.load_pdf(pdf_file))
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        return all_docs

