"""Image ETL pipeline for scanned PDFs with OWL-ViT region detection."""

import hashlib
import io
import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from langchain_core.documents import Document
from pdf2image import convert_from_path
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from tqdm import tqdm

from ..config import (
    HF_TOKEN,
    IMAGE_EXTRACTION_VERSION,
    IMAGE_REVIEW_MANIFEST_PATH,
    IMAGES_DIR,
    OWL_VIT_CONFIDENCE_THRESHOLD,
    OWL_VIT_MODEL,
    POPPLER_PATH,
    TESSERACT_CMD,
    USE_GPU,
)
from .image_captioner import ImageCaptioner
from .processing_status import ProcessingStatus, compute_file_hash

logger = logging.getLogger(__name__)

OWL_VIT_TEXT_QUERIES = [
    "a scientific formula",
    "a biology diagram",
    "a textbook illustration",
    "a data table",
    "a chart or graph",
    "a science experiment setup",
    "a microscope image",
    "a photo in a textbook",
    "a framed textbook picture",
    "a material sample photo",
    "an object photo",
    "a framed object photo",
    "a product-style object photo",
    "an isolated object on white background",
    "a bowl of liquid",
    "a sample of raw material",
    "a coil of wire",
    "a textbook information panel",
    "a colored background panel with text",
    "a callout box with portrait photo",
]

# Patterns indicating a question/activity prompt (NOT a real figure to extract).
QUESTION_PROMPT_PATTERNS = (
    r"h[aãâ]y\s+quan\s+s[aá]t",
    # Bare "Quan sát …" observation prompt (no leading "Hãy") — anchored at the
    # line start so it never matches "quan sát" buried in body text. Common on
    # CD pages (e.g. page 131 "Quan sát hình 23.11, …").
    r"^quan\s+s[aá]t\b",
    r"h[aãâ]y\s+t[iìí]m",
    r"h[aãâ]y\s+cho\s+bi[eế]t",
    r"h[aãâ]y\s+nh[aậ]n\s+x[eé]t",
    r"h[aãâ]y\s+m[oô]\s+t[aả]",
    r"tr[aả]\s+l[oờ]i\s+c[aâ]u\s+h[oỏ]i",
    r"em\s+h[aãâ]y",
    r"th[aả]o\s+lu[aậ]n",
)

# Patterns that mark Vietnamese figure/table captions.
FIGURE_CAPTION_REGEX = re.compile(
    r"^\s*(H[iì]nh|B[aả]ng)\s+\d+", re.IGNORECASE)

# Sub-figure letter labels: "a)", "b)" ... up to "h)".
# Vietnamese OCR often misreads "d" as "đ" and "g" as "ø", so accept both.
SUB_FIGURE_LABEL_REGEX = re.compile(
    r"^\s*[a-hđøA-HĐØ]\s*[\)\.]\s+", flags=re.UNICODE)

# Section markers (info boxes / activities) typical for Vietnamese textbooks.
INFO_BOX_TITLE_REGEX = re.compile(
    r"\b(Em\s+c[oó]\s+bi[eế]t|T[iì]m\s+hi[eể]u\s+th[eê]m|"
    r"M[oở]\s+r[oộ]ng|Ki[eế]n\s+th[uứ]c\s+m[oớ]i|"
    r"Th[uự]c\s+h[aà]nh|V[aậ]n\s+d[uụ]ng|Luy[eệ]n\s+t[aậ]p)\b",
    flags=re.IGNORECASE,
)

# v7 anchor-first patterns ------------------------------------------------

# Strict figure caption: "Hình 1.1." / "Hình 1.1 abc"
FIG_CAPTION_STRICT_REGEX = re.compile(
    r"^\s*H[iì]nh\s+\d+(?:\.\d+)?\s*[\.:]?\s*\S",
    flags=re.IGNORECASE,
)

# Strict table caption: "Bảng 1.1." / "Bảng 1.1 abc"  -> ALWAYS rejected.
TABLE_CAPTION_STRICT_REGEX = re.compile(
    r"^\s*B[aả]ng\s+\d+(?:\.\d+)?\s*[\.:]?",
    flags=re.IGNORECASE,
)

# Section label patterns that precede dashed-only tool grids on page 13/25/etc.
# "Dụng cụ đo chiều dài", "Một số dụng cụ", "Hộp dụng cụ".
TOOL_GROUP_LABEL_REGEX = re.compile(
    r"^\s*(D[uụ]ng\s+c[uụ]\s+(?:[ad][oơ]|trong)|"
    r"M[oộ]t\s+s[oố]\s+d[uụ]ng\s+c[uụ]|"
    r"H[oộ]p\s+d[uụ]ng\s+c[uụ]|"
    r"Chu[aẩ]n\s+b[iị][:\s]|Ti[eế]n\s+h[aà]nh[:\s]|"
    r"D[uụ]ng\s+c[uụ][:\s])",
    flags=re.IGNORECASE,
)

VIETNAMESE_STOPWORDS = {
    "anh",
    "bang",
    "cac",
    "cho",
    "co",
    "cua",
    "duoc",
    "hay",
    "hinh",
    "khi",
    "la",
    "lam",
    "mot",
    "nay",
    "nhung",
    "o",
    "quan",
    "sat",
    "sgk",
    "the",
    "thi",
    "trang",
    "trong",
    "tu",
    "va",
    "ve",
    "voi",
}


class ImageProcessor:
    """Extract, filter, and enrich images from scanned PDFs using Vietnamese page context.

    This base class implements the Cánh Diều (CD) conventions. The shared
    geometry / OCR helpers are reused by every publisher; the parts that
    genuinely differ per publisher are isolated behind the tuning attributes
    below and a small number of overridable seams
    (``_classify_text_anchors``, ``_match_recovered_caption``,
    ``_INFO_BOX_TITLE_KEYS``). Subclasses (``CtsstImageProcessor``,
    ``KnttImageProcessor``) keep their distinct logic separate by overriding
    only those seams — see the per-variant notes in
    ``skills/etl-textbook-images``.
    """

    # ── Region-build tuning (override per publisher to keep logic separate) ──
    # How far above a caption (fraction of page height) a visual cell may sit
    # and still be claimed by it. Kept moderate so a bottom caption never grabs
    # an unrelated photo near the top of the page (the main "dính text" cause).
    _FIG_ASSIGN_MAX_VGAP: float = 0.20
    # Top-growth through a figure's own NARROW cell labels. Body paragraphs are
    # never absorbed: the gap budget is small and the max line width is well
    # under a single-column body line.
    _FIG_TOP_GROW_MAX_GAP: float = 0.045
    _FIG_TOP_GROW_MAX_WIDTH: float = 0.34
    # Info/activity panels: when True, a panel anchored only by a title is kept
    # ONLY if it has a coloured background or an embedded picture (visual score
    # >= _INFO_MIN_VIS). This drops bare section headers on white
    # ("Tìm hiểu về …", "Thí nghiệm 1:") that are body text, not images.
    _INFO_REQUIRE_VISUAL: bool = False
    _INFO_MIN_VIS: float = 0.045
    # Recover a figure when its picture is detected but the caption anchor was
    # missed by OCR: re-OCR the strip directly below the picture (upscaled) and,
    # if it reads "Hình X.Y", emit the figure. Robust to mangled scans.
    _RECOVER_CAPTIONS_BELOW_PHOTOS: bool = True
    # Minimum coloured-content score for a detected visual region to qualify as
    # a real picture worth recovering a caption for (filters text/icon blobs).
    _RECOVER_MIN_VIS: float = 0.06
    # Whether a figure caption may sit ABOVE its figure (KNTT pill labels do;
    # CD/CTST captions are always below the figure).
    _FIG_CAPTION_ABOVE_OK: bool = False
    # Emit one sub_figure crop per visual cell when a figure carries ≥2
    # 'a)'/'b)'/… labels over ≥2 cells (parent composite is kept too).
    _SPLIT_SUBFIGURES: bool = True
    # Also split a captioned grid whose cells carry NO 'a)/b)' labels but each
    # has a short centred TITLE line directly below it (CD page 131
    # "Con cá heo / Con trâu / …"). Gated per-variant: on for CD, off for
    # CTST/KNTT until each is tuned and smoke-tested separately.
    _SPLIT_SUBFIGURES_BY_TITLE: bool = True
    # Max rows of TITLE anchors a split may span. CD grids stack (page 131 is
    # 2 rows of animal titles) so the base allows many; a publisher whose
    # figures are single-row photo strips but whose diagrams carry scattered
    # internal labels (CTST biogas) sets this to 1 to reject the diagrams.
    _SUBFIG_TITLE_MAX_ROWS: int = 99
    # Opt-in extra detector for LARGE PALE photos (beige building, sketches)
    # that are not colour-saturated, so OWL-ViT and the colour-blob detector
    # both miss them (CTST page 59 "Hình 11.9"). Text blocks it also picks up
    # are removed afterwards by `_filter_text_visual_regions`.
    _DETECT_TEXTURED_PHOTOS: bool = False
    # Opt-in detector for SOLID photo rectangles (a dark photo on black, thin
    # line-drawings, image cells OWL only partially detects or the colour-blob
    # detector merges with adjacent text). Non-white content is morphologically
    # OPENED to delete thin text strokes while keeping solid photo fills.
    _DETECT_PHOTO_RECTANGLES: bool = False

    def __init__(self, status_tracker: Optional[ProcessingStatus] = None):
        self.status_tracker = status_tracker or ProcessingStatus()
        self._owlvit_model: Optional[OwlViTForObjectDetection] = None
        self._owlvit_processor: Optional[OwlViTProcessor] = None
        self._owlvit_device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        self.captioner = ImageCaptioner()
        self.image_extraction_version = IMAGE_EXTRACTION_VERSION
        self.review_manifest_path = IMAGE_REVIEW_MANIFEST_PATH

    @property
    def owlvit_model(self) -> OwlViTForObjectDetection:
        if self._owlvit_model is None:
            logger.info(f"Loading OWL-ViT detector: {OWL_VIT_MODEL}")
            self._owlvit_model = OwlViTForObjectDetection.from_pretrained(
                OWL_VIT_MODEL,
                token=HF_TOKEN if HF_TOKEN else None,
            )
            self._owlvit_processor = OwlViTProcessor.from_pretrained(
                OWL_VIT_MODEL,
                token=HF_TOKEN if HF_TOKEN else None,
            )
            self._owlvit_model.to(self._owlvit_device)
            self._owlvit_model.eval()
            logger.info(f"OWL-ViT detector loaded on {self._owlvit_device}")
        return self._owlvit_model

    @property
    def owlvit_processor(self) -> OwlViTProcessor:
        if self._owlvit_processor is None:
            _ = self.owlvit_model
        return self._owlvit_processor

    def _detect_contour_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Open-vocabulary region proposal. Kept for compatibility with older callers."""
        pil_image = Image.fromarray(image).convert("RGB")
        return self._detect_regions_with_owlvit(pil_image, OWL_VIT_TEXT_QUERIES)

    def _detect_regions_with_owlvit(
        self,
        image: Image.Image,
        text_queries: List[str],
        threshold: float = OWL_VIT_CONFIDENCE_THRESHOLD,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect textbook visual regions with OWL-ViT zero-shot object detection."""
        try:
            rgb_image = image.convert("RGB")
            inputs = self.owlvit_processor(
                text=[text_queries],
                images=rgb_image,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._owlvit_device) if hasattr(
                    value, "to") else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)

            target_sizes = torch.tensor(
                [rgb_image.size[::-1]],
                dtype=torch.float,
                device=self._owlvit_device,
            )
            post_process_object_detection = getattr(
                self.owlvit_processor, "post_process_object_detection", None)
            if post_process_object_detection:
                results = post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold,
                )[0]
            else:
                results = self.owlvit_processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    threshold=threshold,
                    target_sizes=target_sizes,
                    text_labels=[text_queries],
                )[0]

            page_width, page_height = rgb_image.size
            page_area = page_width * page_height
            regions: List[Tuple[int, int, int, int]] = []
            labels = results.get("labels", results.get("text_labels", []))
            for score, label, box in zip(
                results.get("scores", []),
                labels,
                results.get("boxes", []),
            ):
                x0, y0, x1, y1 = [int(round(value)) for value in box.tolist()]
                bbox = (
                    max(0, min(page_width, x0)),
                    max(0, min(page_height, y0)),
                    max(0, min(page_width, x1)),
                    max(0, min(page_height, y1)),
                )
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                if self._bbox_area(bbox) > page_area * 0.82:
                    continue
                regions.append(bbox)
                label_value = label.item() if hasattr(label, "item") else label
                if isinstance(label_value, int) and label_value < len(text_queries):
                    query = text_queries[label_value]
                else:
                    query = str(label_value)
                logger.debug(
                    f"OWL-ViT detection: query={query!r}, score={float(score):.3f}, bbox={bbox}"
                )

            return regions
        except Exception as e:
            logger.warning(f"OWL-ViT detection failed: {e}")
            return []

    def _detect_framed_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find textbook picture panels from cyan/green frame strokes that OWL-ViT can miss."""
        try:
            page_height, page_width = image.shape[0], image.shape[1]
            page_area = page_width * page_height
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Vietnamese textbooks often use thin cyan/green rounded rectangles around sub-figures.
            frame_mask = cv2.inRange(
                hsv,
                np.array([70, 35, 65]),
                np.array([105, 255, 255]),
            )
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
            frame_mask = cv2.dilate(frame_mask, close_kernel, iterations=1)
            frame_mask = cv2.morphologyEx(
                frame_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

            return self._regions_from_frame_mask(
                frame_mask,
                page_width,
                page_height,
                page_area,
                min_width=110,
                min_height=70,
                min_area=9000,
                max_area_ratio=0.35,
                min_border_fill=0.015,
                expand_ratio=0.008,
            )
        except Exception as e:
            logger.warning(f"Frame-based region detection failed: {e}")
            return []

    def _detect_dashed_frame_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find sub-figure panels outlined by thin grey dashed strokes (common on some SGK pages).

        Cyan/green framed detection misses these; OWL-ViT is often empty on dense grids.
        """
        try:
            page_height, page_width = image.shape[0], image.shape[1]
            page_area = page_width * page_height
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Grey dashed borders: low saturation, mid brightness (not page white).
            grey_mask = cv2.inRange(
                hsv,
                np.array([0, 0, 95]),
                np.array([180, 70, 225]),
            )
            # Drop cyan/green strokes handled by _detect_framed_regions.
            cyan_mask = cv2.inRange(
                hsv,
                np.array([70, 35, 65]),
                np.array([105, 255, 255]),
            )
            grey_mask = cv2.bitwise_and(grey_mask, cv2.bitwise_not(cyan_mask))

            # Tighter morphology than cyan frames so dashed cells stay separate.
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            grey_mask = cv2.dilate(grey_mask, close_kernel, iterations=1)
            grey_mask = cv2.morphologyEx(
                grey_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

            return self._regions_from_frame_mask(
                grey_mask,
                page_width,
                page_height,
                page_area,
                min_width=95,
                min_height=65,
                min_area=7500,
                max_area_ratio=0.32,
                min_border_fill=0.012,
                expand_ratio=0.006,
            )
        except Exception as e:
            logger.warning(f"Dashed frame region detection failed: {e}")
            return []

    def _regions_from_frame_mask(
        self,
        frame_mask: np.ndarray,
        page_width: int,
        page_height: int,
        page_area: int,
        *,
        min_width: int,
        min_height: int,
        min_area: int,
        max_area_ratio: float,
        min_border_fill: float,
        expand_ratio: float,
    ) -> List[Tuple[int, int, int, int]]:
        """Shared contour filter for cyan and grey dashed frame masks."""
        contours, _ = cv2.findContours(
            frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if width < min_width or height < min_height:
                continue
            if area < min_area or area > page_area * max_area_ratio:
                continue

            aspect_ratio = width / height if height else 0
            if aspect_ratio < 0.42 or aspect_ratio > 4.5:
                continue

            is_wide_header = width > page_width * 0.62 and height < page_height * 0.14
            if is_wide_header and y < page_height * 0.22:
                continue

            border_band = frame_mask[y:y + height, x:x + width]
            if border_band.size == 0 or float(np.mean(border_band > 0)) < min_border_fill:
                continue

            regions.append(self._expand_bbox(
                (x, y, x + width, y + height),
                page_width,
                page_height,
                ratio=expand_ratio,
            ))

        return regions

    def frame_stroke_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Pixel ratios for QA scans — cyan vs grey dashed stroke coverage."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        page_pixels = float(image.shape[0] * image.shape[1]) or 1.0

        cyan_mask = cv2.inRange(
            hsv,
            np.array([70, 35, 65]),
            np.array([105, 255, 255]),
        )
        grey_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 95]),
            np.array([180, 70, 225]),
        )
        grey_mask = cv2.bitwise_and(grey_mask, cv2.bitwise_not(cyan_mask))

        return {
            "cyan_stroke_ratio": float(np.count_nonzero(cyan_mask)) / page_pixels,
            "grey_dashed_stroke_ratio": float(np.count_nonzero(grey_mask)) / page_pixels,
        }

    def _detect_colored_panel_regions(
        self,
        image: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect filled pastel info panels via HSV bands.

        Useful as a coarse fallback when the panel has a strong colored
        background (e.g. some question prompts). 'Em có biết' style panels
        with mostly-white interior are handled by the OCR-anchored detector
        `_detect_info_boxes_via_titles` instead.
        """
        try:
            page_height, page_width = image.shape[:2]
            page_area = page_width * page_height
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            color_ranges: List[Tuple[np.ndarray, np.ndarray]] = [
                # Pink / peach (some question prompts have heavier saturation).
                (np.array([0, 35, 215]), np.array([18, 140, 255])),
                (np.array([160, 35, 215]), np.array([180, 140, 255])),
                # Light blue / cyan
                (np.array([85, 30, 215]), np.array([115, 120, 255])),
                # Light yellow
                (np.array([18, 35, 215]), np.array([40, 130, 255])),
                # Light green
                (np.array([40, 30, 215]), np.array([85, 110, 255])),
            ]

            combined_mask = np.zeros(
                (page_height, page_width), dtype=np.uint8)
            for lower, upper in color_ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, color_mask)

            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (25, 15))
            combined_mask = cv2.morphologyEx(
                combined_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions: List[Tuple[int, int, int, int]] = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if w < page_width * 0.45 or h < 80:
                    continue
                if area < 40000 or area > page_area * 0.72:
                    continue
                if y < page_height * 0.02:
                    continue
                if (y + h) > page_height * 0.985:
                    continue

                roi_mask = combined_mask[y:y + h, x:x + w]
                if roi_mask.size == 0:
                    continue
                coverage = float(np.mean(roi_mask > 0))
                # Require strong fill so we don't catch white gaps between figs.
                if coverage < 0.40:
                    continue

                regions.append((x, y, x + w, y + h))

            return regions
        except Exception as e:
            logger.warning(f"Colored panel detection failed: {e}")
            return []

    def _detect_info_boxes_via_titles(
        self,
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Anchor info boxes to their Vietnamese title text.

        For each OCR line whose text contains a known info-box title
        ('Em có biết', 'Tìm hiểu thêm', 'Vận dụng', etc.) we build a panel
        bbox by:
          - top = title.y0 (minus small padding)
          - bottom = last continuous text line below (gap budget ~6.5% page H)
          - left/right = page text margins (5%..95% of width)

        This works even when the panel background is too pale for HSV
        detection because the panel is identified by its anchor title.
        Returns list of (bbox, panel_label) pairs.
        """
        if not text_lines:
            return []

        title_indexes: List[int] = []
        for index, line in enumerate(text_lines):
            text = str(line["text"]).strip()
            if INFO_BOX_TITLE_REGEX.search(text):
                title_indexes.append(index)

        if not title_indexes:
            return []

        panels: List[Tuple[Tuple[int, int, int, int], str]] = []
        max_gap = int(page_height * 0.065)
        pad_y_top = int(page_height * 0.012)
        pad_y_bottom = int(page_height * 0.012)
        x_left = max(0, int(page_width * 0.04))
        x_right = min(page_width, int(page_width * 0.96))

        title_set = set(title_indexes)
        for current_pos, title_index in enumerate(title_indexes):
            title_line = text_lines[title_index]
            tx0, ty0, tx1, ty1 = title_line["bbox"]  # type: ignore[misc]

            panel_y_bottom = ty1
            prev_y_bottom = ty1
            for next_index in range(title_index + 1, len(text_lines)):
                if next_index in title_set:
                    break
                next_line = text_lines[next_index]
                nx0, ny0, nx1, ny1 = next_line["bbox"]  # type: ignore[misc]

                gap = ny0 - prev_y_bottom
                if gap > max_gap:
                    break
                # Must stay roughly within the page text frame.
                if nx1 < page_width * 0.04 or nx0 > page_width * 0.96:
                    continue
                # Hop over very tall non-text blocks just in case.
                if (ny1 - ny0) > page_height * 0.10:
                    continue

                panel_y_bottom = max(panel_y_bottom, ny1)
                prev_y_bottom = max(prev_y_bottom, ny1)

            panel_bbox = (
                x_left,
                max(0, int(ty0) - pad_y_top),
                x_right,
                min(page_height, int(panel_y_bottom) + pad_y_bottom),
            )

            # Drop tiny panels (likely false title detection).
            if (panel_bbox[3] - panel_bbox[1]) < page_height * 0.04:
                continue

            label = self._classify_panel_label(str(title_line["text"]))
            if not label:
                label = "textbook_info_box"
            panels.append((panel_bbox, label))

        return panels

    def _collect_page_text_lines(
        self,
        pil_img: Image.Image,
    ) -> List[Dict[str, object]]:
        """Run page-level OCR ONCE and return text lines with bounding boxes.

        Used by caption-aware expansion so we can stretch a figure crop to
        cover its caption ("Hình 1.1.") or sub-figure label ("a) Tìm hiểu ...")
        WITHOUT calling tesseract per-region.
        """
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            data = pytesseract.image_to_data(
                pil_img, lang="vie", output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"Page-level OCR (image_to_data) failed: {e}")
            return []

        page_width = pil_img.width
        # Split a tesseract line when two consecutive words are separated by a
        # horizontal gap wider than this — i.e. the 2-column gutter. Without
        # this, a left-column cell label and a right-column question prompt on
        # the same scan row get merged into one full-width line, which then
        # makes figure/info boxes span both columns.
        gutter_gap = int(page_width * 0.055)

        word_count = len(data.get("text", []))
        grouped: Dict[Tuple[int, int, int], List[Dict[str, int]]] = {}
        order: List[Tuple[int, int, int]] = []
        for index in range(word_count):
            text = (data["text"][index] or "").strip()
            try:
                conf = int(float(data["conf"][index]))
            except (TypeError, ValueError):
                conf = -1
            if not text or conf < 25:
                continue

            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])

            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append({
                "text_idx": index,
                "x0": left,
                "y0": top,
                "x1": left + width,
                "y1": top + height,
                "word": text,  # type: ignore[dict-item]
            })

        lines: List[Dict[str, object]] = []
        for key in order:
            words = sorted(grouped[key], key=lambda w: w["x0"])
            # Break into segments at the column gutter.
            segments: List[List[Dict[str, int]]] = [[]]
            prev_x1: Optional[int] = None
            for word in words:
                if prev_x1 is not None and (word["x0"] - prev_x1) > gutter_gap:
                    segments.append([])
                segments[-1].append(word)
                prev_x1 = word["x1"]

            for segment in segments:
                if not segment:
                    continue
                joined = " ".join(str(w["word"]) for w in segment)
                bbox = (
                    min(w["x0"] for w in segment),
                    min(w["y0"] for w in segment),
                    max(w["x1"] for w in segment),
                    max(w["y1"] for w in segment),
                )
                lines.append({"text": joined, "bbox": bbox})

        lines.sort(key=lambda line: line["bbox"][1])
        return lines

    def _is_question_prompt_text(self, text: str) -> bool:
        normalized = self._normalize_text(text or "")
        return any(re.search(pattern, normalized) for pattern in QUESTION_PROMPT_PATTERNS)

    def _find_lines_in_band(
        self,
        text_lines: List[Dict[str, object]],
        y_top: int,
        y_bottom: int,
        x_left: int,
        x_right: int,
        min_x_overlap_ratio: float = 0.25,
    ) -> List[Dict[str, object]]:
        """Return text lines whose vertical band sits between y_top and y_bottom."""
        results: List[Dict[str, object]] = []
        bbox_width = max(1, x_right - x_left)
        for line in text_lines:
            lx0, ly0, lx1, ly1 = line["bbox"]  # type: ignore[misc]
            line_center_y = (ly0 + ly1) / 2
            if line_center_y < y_top or line_center_y > y_bottom:
                continue
            overlap = min(x_right, lx1) - max(x_left, lx0)
            if overlap < bbox_width * min_x_overlap_ratio:
                continue
            results.append(line)
        return results

    def _expand_region_to_caption(
        self,
        bbox: Tuple[int, int, int, int],
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
        is_composite: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Stretch bbox downward to include figure caption or sub-figure label.

        For composite figures (Hình X.X.) we look up to ~7% of page height
        below the bbox. For sub-figures (a/b/c labels) we look up to ~4.5%.
        """
        if not text_lines:
            return bbox

        x0, y0, x1, y1 = bbox
        max_gap = int(page_height * (0.075 if is_composite else 0.045))
        scan_bottom = min(page_height, y1 + max_gap)

        candidates = self._find_lines_in_band(
            text_lines,
            y_top=y1 - 4,
            y_bottom=scan_bottom,
            x_left=x0,
            x_right=x1,
            min_x_overlap_ratio=0.18,
        )

        matched_lines: List[Dict[str, object]] = []
        for line in candidates:
            text = str(line["text"]).strip()
            if not text:
                continue
            if FIGURE_CAPTION_REGEX.match(text):
                if is_composite:
                    # Composite parent SHOULD include 'Hình X.Y. ...' caption.
                    matched_lines.append(line)
                    continue
                # Sub-figure should NOT reach into the main figure caption;
                # stop scanning here so its bbox stays within its own row.
                break
            if not is_composite and SUB_FIGURE_LABEL_REGEX.match(text):
                matched_lines.append(line)
                # Sub-figure caption may wrap to a 2nd line; keep scanning a bit.
                continue
            if matched_lines and not is_composite:
                # Continuation line of a sub-figure caption (no "a)" prefix).
                prev = matched_lines[-1]
                prev_bottom = int(prev["bbox"][3])  # type: ignore[index]
                ly0 = int(line["bbox"][1])  # type: ignore[index]
                if ly0 - prev_bottom <= int(page_height * 0.025):
                    matched_lines.append(line)

        if not matched_lines:
            return bbox

        pad_y = max(4, int(page_height * 0.005))
        pad_x = max(4, int(page_width * 0.005))
        new_y1 = max(int(line["bbox"][3]) for line in matched_lines)
        line_x0 = min([x0] + [int(line["bbox"][0]) for line in matched_lines])
        line_x1 = max([x1] + [int(line["bbox"][2]) for line in matched_lines])

        if is_composite:
            # Composite parent can safely expand to cover the full caption line.
            new_x0 = line_x0
            new_x1 = line_x1
        else:
            # Sub-figure caption may be wrongly OCR-merged with siblings on the
            # same row ("a) Tìm hiểu ... b) Tìm hiểu ..."). Clip x expansion to
            # the original bbox plus a small margin so the crop doesn't leak
            # into a neighbouring sub-figure.
            bbox_width = max(40, x1 - x0)
            x_margin = max(12, int(bbox_width * 0.18))
            new_x0 = max(line_x0, x0 - x_margin)
            new_x1 = min(line_x1, x1 + x_margin)

        return (
            max(0, new_x0 - pad_x),
            y0,
            min(page_width, new_x1 + pad_x),
            min(page_height, new_y1 + pad_y),
        )

    def _trim_region_top_to_exclude_prompt(
        self,
        bbox: Tuple[int, int, int, int],
        text_lines: List[Dict[str, object]],
        page_height: int,
    ) -> Tuple[int, int, int, int]:
        """If the top of a bbox contains a question prompt, trim it off.

        Used for both composite parents and individual sub-figures so a
        framed-frame detection that swallowed the question-prompt panel
        on top of the figure doesn't make the final crop start too high.

        Also follows wrap-around continuation lines (e.g. "tự nhiên.")
        that don't match a prompt pattern on their own but immediately
        follow a prompt line and contain no caption marker.
        """
        if not text_lines:
            return bbox

        x0, y0, x1, y1 = bbox
        scan_top = y0
        scan_bottom = min(y1, y0 + int(page_height * 0.22))
        # Use a low x-overlap threshold so short wrap-around lines (e.g.
        # "tự nhiên.") that take up only a fraction of the bbox width still
        # contribute to the trim decision.
        lines_in_top = self._find_lines_in_band(
            text_lines,
            y_top=scan_top,
            y_bottom=scan_bottom,
            x_left=x0,
            x_right=x1,
            min_x_overlap_ratio=0.05,
        )
        if not lines_in_top:
            return bbox

        lines_sorted = sorted(
            lines_in_top, key=lambda line: int(line["bbox"][1]))
        last_prompt_bottom: Optional[int] = None
        wrap_gap_budget = int(page_height * 0.025)

        for line in lines_sorted:
            text = str(line["text"]).strip()
            line_top = int(line["bbox"][1])
            line_bottom = int(line["bbox"][3])

            if self._is_question_prompt_text(text):
                last_prompt_bottom = max(last_prompt_bottom or 0, line_bottom)
                continue

            if last_prompt_bottom is None:
                # Haven't found a prompt yet; stop scanning once we hit a real
                # figure caption or sub-figure label so we don't false-trim.
                if (
                    FIGURE_CAPTION_REGEX.match(text)
                    or SUB_FIGURE_LABEL_REGEX.match(text)
                ):
                    break
                continue

            # Already inside a prompt block: include wrap-around continuation
            # lines that are close to the previous prompt bottom and don't
            # look like a caption.
            if (
                FIGURE_CAPTION_REGEX.match(text)
                or SUB_FIGURE_LABEL_REGEX.match(text)
            ):
                break
            if line_top - last_prompt_bottom <= wrap_gap_budget:
                last_prompt_bottom = max(last_prompt_bottom, line_bottom)
            else:
                break

        if last_prompt_bottom is None:
            return bbox

        new_y0 = min(y1 - 10, last_prompt_bottom +
                     int(page_height * 0.008))
        if new_y0 <= y0 + 4 or new_y0 >= y1 - 40:
            return bbox
        return (x0, new_y0, x1, y1)

    def _classify_panel_label(self, text: str) -> str:
        """Return a coarse label for colored panels: info_box / activity / prompt / ''."""
        normalized = self._normalize_text(text or "")
        if "em co biet" in normalized:
            return "textbook_info_box"
        if "tim hieu them" in normalized or "mo rong" in normalized:
            return "activity_box"
        if any(re.search(pattern, normalized) for pattern in QUESTION_PROMPT_PATTERNS):
            return "question_prompt"
        if "thuc hanh" in normalized or "van dung" in normalized or "luyen tap" in normalized:
            return "activity_box"
        return ""

    # ------------------------------------------------------------------
    # v7 anchor-first deterministic detection
    # ------------------------------------------------------------------

    _INFO_BOX_TITLE_KEYS: List[Tuple[str, str]] = [
        ("em co biet", "textbook_info_box"),
        ("tim hieu them", "activity_box"),
        ("mo rong", "activity_box"),
        ("kien thuc moi", "activity_box"),
        ("thuc hanh", "activity_box"),
        ("van dung", "activity_box"),
        ("luyen tap", "activity_box"),
        ("thi nghiem", "activity_box"),
    ]

    # A figure marker is "Hình <number>" NOT immediately followed by a letter,
    # so "hình 3R" / "hình 3D" (mô hình 3R) is never mistaken for a caption and
    # never triggers a spurious split (CTST page 59).
    _FIGURE_MARKER_REGEX = re.compile(
        r"H[iì]nh\s+\d+(?:\.\d+)?(?![A-Za-z])", flags=re.IGNORECASE)

    def _split_merged_figure_caption(
        self,
        entry: Dict[str, object],
    ) -> List[Dict[str, object]]:
        """If a single OCR line carries multiple `Hình X.Y` markers, split it.

        bbox is partitioned linearly between markers based on character index
        — good enough because OCR line bboxes are tight to the rendered text.
        """
        text = str(entry["text"])
        matches = list(self._FIGURE_MARKER_REGEX.finditer(text))
        if len(matches) <= 1:
            return [entry]

        x0, y0, x1, y1 = entry["bbox"]  # type: ignore[misc]
        line_width = max(1, x1 - x0)
        text_length = max(1, len(text))

        parts: List[Dict[str, object]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + \
                1 < len(matches) else len(text)
            slice_text = text[start:end].strip()
            if not slice_text:
                continue
            part_x0 = x0 + int(start / text_length * line_width)
            part_x1 = x0 + int(end / text_length * line_width)
            parts.append({
                "index": entry["index"],
                "text": slice_text,
                "bbox": (part_x0, y0, part_x1, y1),
            })
        return parts or [entry]

    def _match_info_box_title(self, text: str) -> str:
        """Return panel label if `text` is a STAND-ALONE info-box title.

        The strict regex `INFO_BOX_TITLE_REGEX` is too eager because
        Vietnamese textbook body text frequently contains phrases like
        "trong phòng thực hành", "vận dụng vào...". Real info-box headers
        always:
          1. start at the line beginning, and
          2. are short (header, no body sentence).
        """
        norm = self._normalize_text(text or "").strip()
        if not norm:
            return ""
        # "em co the" (Em có thể — learning objectives header) must NOT match
        # "em co biet". Guard explicitly.
        if norm.startswith("em co the"):
            return ""
        for key, label in self._INFO_BOX_TITLE_KEYS:
            if not norm.startswith(key):
                continue
            tail = norm[len(key):].strip()
            # Header line: nothing after, optional ':' / '!' or a short
            # 1-3 word subtitle. Reject when the line is clearly body text.
            if len(tail) <= 30 and not re.search(r"\s(la|nay|cua|cho|va|hoac|trong|ben)\s", tail):
                return label
        return ""

    def _fuzzy_info_label(self, text: str) -> str:
        """Loose match of a (possibly noisy) re-OCR header to a panel label."""
        norm = self._normalize_text(text or "")
        norm = re.sub(r"\s+", " ", norm).strip()
        if not norm:
            return ""
        if "em co the" in norm:        # learning-objectives header, not a panel
            return ""
        for key, label in self._INFO_BOX_TITLE_KEYS:
            # allow the key to appear anywhere (tab OCR may add junk chars).
            if key in norm:
                return label
            # tolerate one missing space / char by collapsing spaces.
            if key.replace(" ", "") in norm.replace(" ", ""):
                return label
        return ""

    def _detect_colored_info_headers(
        self,
        pil_img: Image.Image,
    ) -> List[Dict[str, object]]:
        """Find info-box headers by their coloured tab / coloured bold text.

        Vietnamese SGK marks "Em có biết", "Tìm hiểu thêm", "Vận dụng" etc.
        with a saturated pink/red or blue/teal header. When the header is a
        filled tab with WHITE text, the page-level OCR usually fails to read
        it (page 70). Here we locate the coloured header region directly, then
        re-OCR it with white-text-on-colour pre-processing to recover the
        label. Returns anchor entries shaped like OCR info_titles
        ``{"index": -1, "text", "bbox", "label"}``.
        """
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        except Exception:
            return []

        rgb = np.array(pil_img.convert("RGB"))
        page_height, page_width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Pink/red (hue wraps) OR blue/teal — the two header colour families.
        pink = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 158))
                & (sat > 70) & (val > 110))
        blue = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 120)
                & (sat > 70) & (val > 110))
        mask = (pink | blue).astype(np.uint8) * 255

        # Connect header glyphs / tab fill into a blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        headers: List[Dict[str, object]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Header-like geometry: short, wider-than-tall, not a full panel.
            if w < int(page_width * 0.06) or w > int(page_width * 0.55):
                continue
            if h < 16 or h > int(page_height * 0.05):
                continue
            if w < h * 1.3:
                continue
            # Re-OCR the header with white-text-on-colour pre-processing.
            pad = 4
            cx0 = max(0, x - pad)
            cy0 = max(0, y - pad)
            cx1 = min(page_width, x + w + pad)
            cy1 = min(page_height, y + h + pad)
            label = self._reocr_colored_header(rgb[cy0:cy1, cx0:cx1])
            if not label:
                continue
            headers.append({
                "index": -1,
                "text": label[1],
                "bbox": (cx0, cy0, cx1, cy1),
                "label": label[0],
            })
        return headers

    def _reocr_colored_header(self, crop_rgb: np.ndarray) -> Optional[Tuple[str, str]]:
        """Re-OCR a coloured header crop. Returns (label, raw_text) or None."""
        try:
            import pytesseract
        except Exception:
            return None
        if crop_rgb.size == 0:
            return None
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        candidates: List[np.ndarray] = []
        # (a) white text on colour -> bright text.
        _, bright = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
        candidates.append(cv2.bitwise_not(bright))
        # (b) dark coloured text on white -> dark text (e.g. blue bold text).
        _, dark = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        candidates.append(dark)
        for binary in candidates:
            up = cv2.resize(binary, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
            try:
                text = pytesseract.image_to_string(
                    up, lang="vie", config="--psm 7")
            except Exception:
                continue
            label = self._fuzzy_info_label(text)
            if label:
                return (label, " ".join(text.split())[:40])
        return None

    def _classify_text_anchors(
        self,
        text_lines: List[Dict[str, object]],
    ) -> Dict[str, List[Dict[str, object]]]:
        """Bucket OCR lines into Hình caption / Bảng caption / info title /
        sub-figure label / question prompt / tool-group label.

        Each bucket entry is a dict ``{"index", "text", "bbox"}``.
        """
        figure_caps: List[Dict[str, object]] = []
        table_caps: List[Dict[str, object]] = []
        info_titles: List[Dict[str, object]] = []
        sub_labels: List[Dict[str, object]] = []
        question_prompts: List[Dict[str, object]] = []
        tool_labels: List[Dict[str, object]] = []

        for index, line in enumerate(text_lines):
            text = str(line["text"]).strip()
            bbox = tuple(int(value)
                         for value in line["bbox"])  # type: ignore[misc]
            entry = {"index": index, "text": text, "bbox": bbox}

            if TABLE_CAPTION_STRICT_REGEX.match(text):
                table_caps.append(entry)
                continue
            if FIG_CAPTION_STRICT_REGEX.match(text):
                # Tesseract often merges two side-by-side captions
                # ("Hình 14.1 ...    Hình 14.2 ...") into a single OCR line.
                # Split the line into one entry per "Hình X.Y" marker so each
                # figure gets its own composite.
                split_entries = self._split_merged_figure_caption(entry)
                figure_caps.extend(split_entries)
                continue
            info_label = self._match_info_box_title(text)
            if info_label:
                entry["label"] = info_label
                info_titles.append(entry)
                continue
            if SUB_FIGURE_LABEL_REGEX.match(text):
                sub_labels.append(entry)
                # sub labels often co-occur with body text below; not exclusive
            if self._is_question_prompt_text(text):
                question_prompts.append(entry)
                continue
            if TOOL_GROUP_LABEL_REGEX.match(text):
                tool_labels.append(entry)
                continue

        return {
            "figure_captions": figure_caps,
            "table_captions": table_caps,
            "info_titles": info_titles,
            "sub_labels": sub_labels,
            "question_prompts": question_prompts,
            "tool_group_labels": tool_labels,
        }

    def _detect_object_blobs(
        self,
        image: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Connected-component fallback for isolated object photos on white.

        OWL-ViT under-detects product-style object photos (a chair, a bucket,
        a beaker on a white background — page 107). Here we threshold coloured
        / dark ink, close it into blobs, and keep components whose size and
        fill look like a picture rather than a text paragraph.
        """
        try:
            page_height, page_width = image.shape[:2]
            page_area = page_width * page_height
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            # Coloured OR dark foreground (covers greyscale objects too).
            colored = (sat > 55) & (val > 60) & (val < 250)
            dark = val < 90
            mask = (colored | dark).astype(np.uint8) * 255

            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (19, 19))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            blobs: List[Tuple[int, int, int, int]] = []
            for i in range(1, num):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])
                bbox_area = w * h
                if bbox_area < page_area * 0.004 or bbox_area > page_area * 0.40:
                    continue
                if w < page_width * 0.04 or h < page_height * 0.035:
                    continue
                aspect = w / max(1, h)
                if aspect < 0.18 or aspect > 7.0:
                    continue
                # Solidity: a picture fills much of its bbox; a text paragraph
                # (sparse strokes) does not.
                fill = area / max(1, bbox_area)
                if fill < 0.30:
                    continue
                blobs.append((x, y, x + w, y + h))
            return blobs
        except Exception as e:
            logger.warning(f"Object-blob detection failed: {e}")
            return []

    def _detect_textured_photo_regions(
        self,
        image: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect large photo regions by local TEXTURE (not colour).

        A pale photo (beige building + sky, a pencil sketch) has high local
        pixel variance everywhere but low colour saturation, so OWL-ViT and the
        colour/dark blob detector both miss it (CTST page 59 "Hình 11.9"). Here
        we threshold local std-dev into blobs. Text paragraphs are also textured
        and get caught — they are dropped afterwards by
        `_filter_text_visual_regions` (high OCR text-line coverage), leaving the
        captionless picture.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            page_height, page_width = gray.shape
            page_area = page_width * page_height
            kernel_size = 9
            mean = cv2.boxFilter(gray, -1, (kernel_size, kernel_size))
            mean_sq = cv2.boxFilter(gray * gray, -1, (kernel_size, kernel_size))
            std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
            textured = (std > 18).astype(np.uint8) * 255

            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
            textured = cv2.morphologyEx(
                textured, cv2.MORPH_CLOSE, close_kernel, iterations=2)
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            textured = cv2.morphologyEx(
                textured, cv2.MORPH_OPEN, open_kernel, iterations=1)

            num, _labels, stats, _ = cv2.connectedComponentsWithStats(
                textured, 8)
            regions: List[Tuple[int, int, int, int]] = []
            for i in range(1, num):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])
                bbox_area = w * h
                # Big rectangular blobs only — a real picture, not a stray line.
                if bbox_area < page_area * 0.02 or bbox_area > page_area * 0.55:
                    continue
                if w < page_width * 0.12 or h < page_height * 0.06:
                    continue
                if area / max(1, bbox_area) < 0.40:
                    continue
                regions.append((x, y, x + w, y + h))
            return regions
        except Exception as e:
            logger.warning(f"Textured-photo detection failed: {e}")
            return []

    def _detect_photo_rectangles(
        self,
        image: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect SOLID photo rectangles by deleting thin text strokes.

        Non-white page content is masked, then morphologically OPENED: thin
        text strokes (a few px wide) vanish while solid photo fills survive.
        This recovers photos OWL detects only partially or that the colour-blob
        detector merges with the dark text above them — a dark photo on black
        (KNTT page 131 bat), thick line-drawings (page 13 warning triangles),
        and the diagram cells of a figure sitting under a title (page 80). A
        solid colour heading bar can also survive; it is removed downstream by
        the figure builder's text-coverage gate / `_filter_text_visual_regions`.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            page_height, page_width = gray.shape
            page_area = page_width * page_height
            # Anything darker than the near-white page background is "content".
            content = (gray < 238).astype(np.uint8) * 255
            # Opening removes thin text strokes, keeps solid fills.
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            solid = cv2.morphologyEx(content, cv2.MORPH_OPEN, open_k,
                                     iterations=1)
            # Close only TINY interior gaps so a photo is one component — a
            # bigger kernel would bridge the gutter between a coloured question
            # box and the figure below it (page 13).
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, close_k,
                                     iterations=1)

            num, _labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
            regions: List[Tuple[int, int, int, int]] = []
            for i in range(1, num):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])
                bbox_area = w * h
                if bbox_area < page_area * 0.01 or bbox_area > page_area * 0.75:
                    continue
                if w < page_width * 0.06 or h < page_height * 0.04:
                    continue
                # Solid fill = a picture, not a sparse paragraph block.
                if area / max(1, bbox_area) < 0.45:
                    continue
                regions.append((x, y, x + w, y + h))
            return regions
        except Exception as e:
            logger.warning(f"Photo-rectangle detection failed: {e}")
            return []

    def _text_line_coverage(
        self,
        region: Tuple[int, int, int, int],
        text_lines: List[Dict[str, object]],
    ) -> float:
        """Fraction of `region` area covered by OCR text-line boxes.

        A paragraph / heading / objectives box is almost entirely covered by
        text lines; a figure (photo, drawing) — even one with scattered
        labels — is mostly non-text, so its coverage is low.
        """
        rx0, ry0, rx1, ry1 = region
        region_area = max(1, (rx1 - rx0) * (ry1 - ry0))
        covered = 0
        for line in text_lines:
            lx0, ly0, lx1, ly1 = line["bbox"]  # type: ignore[misc]
            ix0, iy0 = max(rx0, lx0), max(ry0, ly0)
            ix1, iy1 = min(rx1, lx1), min(ry1, ly1)
            if ix1 > ix0 and iy1 > iy0:
                covered += (ix1 - ix0) * (iy1 - iy0)
        return min(1.0, covered / region_area)

    def _filter_text_visual_regions(
        self,
        regions: List[Tuple[int, int, int, int]],
        pil_img: Image.Image,
        text_lines: Optional[List[Dict[str, object]]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Drop detector regions that are actually text blocks.

        Two false-positive families to remove:
          1. OWL-ViT fires on a heading / paragraph (page 85 "I. VÌ SAO …") —
             no colour, OCRs to several words.
          2. A framed/colour detector fires on a coloured info/objectives box
             (page 45 "MỤC TIÊU") — high colour from its heading, but the box
             is wall-to-wall TEXT.

        A region is dropped when OCR text-line boxes cover most of it
        (text-line coverage), or when it is colourless yet OCRs to words.
        A real figure — even a labelled diagram — has low text-line coverage.
        """
        text_lines = text_lines or []
        kept: List[Tuple[int, int, int, int]] = []
        for region in regions:
            crop = pil_img.crop(region)
            if crop.width < 4 or crop.height < 4:
                continue

            # (1) Mostly-text box (paragraph / objectives / heading panel).
            if text_lines and self._text_line_coverage(region, text_lines) >= 0.45:
                continue

            if self._visual_content_score(crop) >= 0.03:
                kept.append(region)
                continue

            # (2) Colourless region that OCRs to real words = text block.
            text = self._ocr_crop_text(crop)
            words = [
                w for w in self._normalize_text(text).split()
                if len(w) >= 3 and w.isalpha()
            ]
            if len(words) >= 2:
                continue
            kept.append(region)
        return kept

    def _dedupe_visual_regions(
        self,
        regions: List[Tuple[int, int, int, int]],
        min_area: int = 1500,
        iou_threshold: float = 0.55,
    ) -> List[Tuple[int, int, int, int]]:
        candidates = [tuple(bbox) for bbox in regions
                      if self._bbox_area(bbox) >= min_area]
        candidates.sort(key=lambda bbox: self._bbox_area(bbox), reverse=True)
        kept: List[Tuple[int, int, int, int]] = []
        for bbox in candidates:
            skip = False
            for other in kept:
                if self._iou(bbox, other) > iou_threshold:
                    skip = True
                    break
                if self._coverage_ratio(bbox, other) > 0.85:
                    skip = True
                    break
            if not skip:
                kept.append(bbox)
        return kept

    def _build_table_zones(
        self,
        table_caps: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Bảng X.Y caption + the rows immediately below → exclusion zone."""
        zones: List[Tuple[int, int, int, int]] = []
        if not table_caps:
            return zones

        max_gap = int(page_height * 0.055)
        col_tol = int(page_width * 0.05)
        for cap in table_caps:
            cx0, cy0, cx1, cy1 = cap["bbox"]  # type: ignore[misc]
            zone_y_bottom = cy1
            prev_y_bottom = cy1
            zone_x_left = cx0
            zone_x_right = cx1
            # Accumulated column — stops a right-column table caption from
            # absorbing the left column's figure text into the zone (which
            # would corrupt every downstream gutter/exclusion check).
            col_x_left = cx0
            col_x_right = cx1

            for line in text_lines:
                lx0, ly0, lx1, ly1 = (
                    int(line["bbox"][0]),  # type: ignore[index]
                    int(line["bbox"][1]),  # type: ignore[index]
                    int(line["bbox"][2]),  # type: ignore[index]
                    int(line["bbox"][3]),  # type: ignore[index]
                )
                if ly0 <= cy1:
                    continue

                text = str(line["text"]).strip()
                if (FIG_CAPTION_STRICT_REGEX.match(text)
                        or TABLE_CAPTION_STRICT_REGEX.match(text)
                        or self._match_info_box_title(text)):
                    break

                gap = ly0 - prev_y_bottom
                if gap > max_gap:
                    break
                # Column guard: skip text from the other column.
                if lx1 < col_x_left - col_tol or lx0 > col_x_right + col_tol:
                    continue

                zone_y_bottom = max(zone_y_bottom, ly1)
                zone_x_left = min(zone_x_left, lx0)
                zone_x_right = max(zone_x_right, lx1)
                prev_y_bottom = max(prev_y_bottom, ly1)
                col_x_left = min(col_x_left, lx0)
                col_x_right = max(col_x_right, lx1)

            pad_x = int(page_width * 0.02)
            pad_y = int(page_height * 0.012)
            zones.append((
                max(0, zone_x_left - pad_x),
                max(0, cy0 - pad_y),
                min(page_width, zone_x_right + pad_x),
                min(page_height, zone_y_bottom + pad_y),
            ))

        return zones

    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def _assign_regions_to_captions(
        self,
        figure_caps: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
        exclusion_zones: List[Tuple[int, int, int, int]],
        column_separators: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Assign each visual region to the figure caption directly below it.

        SGK convention: the caption sits UNDER its figure. So a region is
        assigned to the NEAREST caption whose top is below the region and
        whose column (horizontal centre) matches. This naturally:
          * keeps tall single-column figures whole (page 70 Hình 12.6 —
            cells span the full column height, all share one caption at the
            bottom), and
          * splits side-by-side / stacked figures cleanly (each region picks
            its own nearest caption).
        """
        assignments: Dict[int, List[Tuple[int, int, int, int]]] = {
            idx: [] for idx in range(len(figure_caps))
        }
        if not figure_caps:
            return assignments

        # A region belongs to a caption when (a) they horizontally overlap, or
        # (b) the region is reasonably centred over the caption AND no
        # other-column anchor (info-box header / table caption) sits at the
        # region's height between it and the caption. Rule (b) lets a single
        # full-width figure claim its edge cells (page 6) while preventing a
        # 2-column page from pulling the other column's content onto a
        # page-centre caption (page 70).
        center_tol = page_width * 0.42
        # Moderate vertical reach: a caption only claims cells reasonably near
        # it. A tall list-figure whose interior cells were missed by OWL-ViT is
        # rescued afterwards by text-based top-growth, not by an enormous reach
        # here (which would let a bottom caption grab an unrelated illustration
        # at the top of the page — page 85).
        max_vgap = page_height * self._FIG_ASSIGN_MAX_VGAP
        separators = column_separators or []

        for region in visual_regions:
            rx0, ry0, rx1, ry1 = region
            # Reject visual regions sitting inside a table zone.
            if any(self._coverage_ratio(region, zone) > 0.45
                   for zone in exclusion_zones):
                continue

            rcx = (rx0 + rx1) / 2.0
            rcy = (ry0 + ry1) / 2.0
            # Two-tier preference. Tier 1: captions the region horizontally
            # OVERLAPS (true column match) — this splits side-by-side figures
            # correctly (page 85). Tier 2 (only if no overlap exists): a
            # centred caption within tolerance and not separated by an
            # other-column anchor — this lets a single short caption claim the
            # edge cells of a full-width figure (page 100).
            overlap_best_idx: Optional[int] = None
            overlap_best_vgap = float("inf")
            center_best_idx: Optional[int] = None
            center_best_vgap = float("inf")

            for idx, cap in enumerate(figure_caps):
                cx0, cy0, cx1, cy1 = cap["bbox"]  # type: ignore[misc]
                # Caption must be roughly BELOW the region. Allow the region to
                # dip a little past the caption top — object blobs and cells
                # often include the caption row's leading edge.
                if cy0 < ry1 - int(page_height * 0.045):
                    continue
                vgap = cy0 - ry1
                if vgap > max_vgap:
                    continue
                ccx = (cx0 + cx1) / 2.0
                hov = min(rx1, cx1) - max(rx0, cx0)
                if hov > 0:
                    if vgap < overlap_best_vgap:
                        overlap_best_vgap = vgap
                        overlap_best_idx = idx
                    continue
                if abs(rcx - ccx) > center_tol:
                    continue
                lo, hi = sorted((rcx, ccx))
                separated = False
                for sx0, sy0, sx1, sy1 in separators:
                    scx = (sx0 + sx1) / 2.0
                    if lo < scx < hi and sy0 <= rcy <= sy1:
                        separated = True
                        break
                if separated:
                    continue
                if vgap < center_best_vgap:
                    center_best_vgap = vgap
                    center_best_idx = idx

            best_idx = overlap_best_idx if overlap_best_idx is not None \
                else center_best_idx
            if best_idx is not None:
                assignments[best_idx].append(region)

        return assignments

    def _prompt_blocker_bboxes(
        self,
        question_prompts: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Expand each question-prompt anchor down through its wrapped lines.

        A prompt like "Quan sát hình 23.11, …" often wraps to a 2nd line
        ("… của các động vật trong hình.") that carries no prompt keyword. The
        first line is the only anchor, so a figure ceiling computed from it
        sits ABOVE the wrapped tail, letting top-growth absorb that tail into
        the crop (page 131 "dính text"). Here we grow each prompt bbox downward
        through continuous same-column lines (small gap, not another anchor) so
        the ceiling sits below the WHOLE prompt paragraph.
        """
        blockers: List[Tuple[int, int, int, int]] = []
        if not question_prompts:
            return blockers
        max_gap = int(page_height * 0.02)
        ordered = sorted(text_lines, key=lambda ln: int(ln["bbox"][1]))
        for prompt in question_prompts:
            px0, py0, px1, py1 = prompt["bbox"]  # type: ignore[misc]
            bottom = int(py1)
            prev_bottom = int(py1)
            for line in ordered:
                lx0, ly0, lx1, ly1 = (
                    int(line["bbox"][0]),  # type: ignore[index]
                    int(line["bbox"][1]),  # type: ignore[index]
                    int(line["bbox"][2]),  # type: ignore[index]
                    int(line["bbox"][3]),  # type: ignore[index]
                )
                if ly0 <= int(py1):
                    continue
                if (ly0 - prev_bottom) > max_gap:
                    break
                # Same column as the prompt (must horizontally overlap).
                if min(lx1, int(px1)) - max(lx0, int(px0)) <= 0:
                    continue
                text = str(line["text"]).strip()
                if (FIG_CAPTION_STRICT_REGEX.match(text)
                        or TABLE_CAPTION_STRICT_REGEX.match(text)
                        or self._match_info_box_title(text)
                        or SUB_FIGURE_LABEL_REGEX.match(text)):
                    break
                bottom = max(bottom, ly1)
                prev_bottom = max(prev_bottom, ly1)
            blockers.append((int(px0), int(py0), int(px1), bottom))
        return blockers

    def _build_figure_composites(
        self,
        figure_caps: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        question_prompts: List[Dict[str, object]],
        info_titles: List[Dict[str, object]],
        sub_labels: List[Dict[str, object]],
        exclusion_zones: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """For each Hình caption, build a bbox that wraps the figure(s) above it."""
        outputs: List[Dict[str, object]] = []
        if not figure_caps:
            return outputs

        # Other-column anchors that can separate a region from a caption.
        column_separators = (
            [t["bbox"] for t in info_titles]
            + list(exclusion_zones)
        )
        assignments = self._assign_regions_to_captions(
            figure_caps, visual_regions, page_width, page_height,
            exclusion_zones, column_separators=column_separators,
        )

        # Per-caption upward ceiling: prev caption / info-title / question prompt / page top.
        # Prompt bboxes are expanded through their wrapped continuation lines so
        # the ceiling sits below the whole prompt paragraph (page 131).
        prompt_blockers = self._prompt_blocker_bboxes(
            question_prompts, text_lines, page_width, page_height)
        all_blockers = sorted(
            [cap["bbox"] for cap in figure_caps]
            + [cap["bbox"] for cap in info_titles]
            + prompt_blockers,
            key=lambda b: b[1],  # type: ignore[index]
        )

        used_region_keys = set()
        for cap_idx, cap in enumerate(figure_caps):
            assigned = [tuple(r) for r in assignments.get(cap_idx, [])]
            if not assigned:
                continue

            cx0, cy0, cx1, cy1 = cap["bbox"]  # type: ignore[misc]

            # Upward ceiling: the highest blocker bbox-bottom that is still
            # ABOVE the assigned region top. This is what stops the composite
            # from absorbing the question prompt or previous caption.
            assigned_top_y = min(r[1] for r in assigned)
            ceiling_y = 0
            for blocker in all_blockers:
                bx0, by0, bx1, by1 = blocker
                if by1 >= assigned_top_y - 2:
                    continue
                # Horizontal overlap with caption region must exist
                cap_left = min(cx0, min(r[0] for r in assigned))
                cap_right = max(cx1, max(r[2] for r in assigned))
                hov = min(bx1, cap_right) - max(bx0, cap_left)
                if hov <= 0:
                    continue
                ceiling_y = max(ceiling_y, by1 + int(page_height * 0.005))

            x0 = min([cx0] + [r[0] for r in assigned])
            x1 = max([cx1] + [r[2] for r in assigned])
            y0 = max(ceiling_y, min(r[1] for r in assigned))
            y1 = cy1

            # Visual ceiling: walk UP from the assigned cells through visual
            # regions that are CONTIGUOUS (gap-connected) in this figure's
            # column, and stop at the first big vertical gap. Text-based
            # top-growth may not climb above this ceiling.
            #   * list-figure (page 70): cells are gap-connected up the whole
            #     column → ceiling reaches the top.
            #   * single photo with a coloured text box above it (page 45
            #     "MỤC TIÊU"): the box is separated from the photo by a wide
            #     gap → ceiling stays at the photo, so growth cannot climb
            #     into the box.
            col_x0, col_x1 = x0, x1
            col_tol = int(page_width * 0.03)
            max_cell_gap = int(page_height * 0.16)
            column_cells = sorted(
                [vr for vr in visual_regions
                 if col_x0 - col_tol <= (vr[0] + vr[2]) / 2.0 <= col_x1 + col_tol
                 and vr[1] < y0 + 2],
                key=lambda vr: vr[1], reverse=True,  # nearest-above first
            )
            visual_top = y0
            for vr in column_cells:
                # vr sits above the current ceiling; bridge only a small gap.
                if visual_top - vr[3] > max_cell_gap:
                    break
                visual_top = min(visual_top, vr[1])
            # Pull the top up to INCLUDE the gap-connected upper visual cells
            # themselves (not merely use them as a growth ceiling). A 2-row
            # grid whose bottom caption is too far for direct cell assignment
            # (page 6 "Hình 1.1") otherwise drops its entire upper row of
            # photos. Bounded by the nearest blocker above (`ceiling_y`) so it
            # never climbs across a question prompt or previous caption — AND
            # only when the bridged band is image-like (low text coverage). The
            # latter is what separates page 6 (photo rows between the cells)
            # from page 100 (a section-objectives text block / chapter banner
            # sitting above a single figure, which must NOT be absorbed).
            candidate_top = max(ceiling_y, visual_top)
            if candidate_top < y0 - 2:
                band = (x0, candidate_top, x1, y0)
                if self._text_line_coverage(band, text_lines) < 0.22:
                    y0 = candidate_top
            # Only a small overshoot above the topmost contiguous cell, so the
            # growth can pick up a cell's own label row but never a section
            # header that sits well above a single photo (page 45).
            grow_ceiling = max(ceiling_y, visual_top -
                               int(page_height * 0.015))

            # Extend the top UP through the figure column's own NARROW labels
            # (page 70 "Tế bào ..."), bounded by the visual ceiling.
            x0, y0, x1 = self._grow_figure_top(
                x0, y0, x1, grow_ceiling, text_lines, page_width, page_height,
            )

            # Snap the figure horizontally to its column's text (e.g. the
            # left "Tế bào ..." labels that sit beside each cell image),
            # bounded by the gutter to the tall info/table zones so it never
            # bleeds into an info box / table in the other column. Question
            # prompts are NOT used as gutters — they are often small margin
            # notes that would wrongly truncate a wide figure.
            other_anchors = list(exclusion_zones)
            caption_cx = (cx0 + cx1) / 2.0
            x0, x1 = self._snap_figure_to_column(
                (x0, y0, x1, y1), caption_cx, text_lines, other_anchors,
                page_width, page_height,
            )

            pad_x = int(page_width * 0.008)
            pad_y = int(page_height * 0.006)
            bbox = (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(page_width, x1 + pad_x),
                min(page_height, y1 + pad_y),
            )
            # A real composite must carry sub-figure letter labels INSIDE
            # the bbox. Otherwise (single illustration with caption) it is
            # a single_figure and we must not slice it further.
            sub_label_inside = False
            for sub in sub_labels:
                sx0, sy0, sx1, sy1 = sub["bbox"]  # type: ignore[misc]
                if sx0 >= bbox[0] - 5 and sx1 <= bbox[2] + 5 \
                        and sy0 >= bbox[1] - 5 and sy1 <= bbox[3] + 12:
                    sub_label_inside = True
                    break
            label = (
                "composite_figure"
                if sub_label_inside and len(assigned) >= 2
                else "single_figure"
            )
            outputs.append({
                "bbox": bbox,
                "image_type": label,
                "caption_text": cap["text"],
                "caption_bbox": cap["bbox"],
                "assigned_regions": assigned,
            })
            for region in assigned:
                used_region_keys.add(region)

        return outputs

    def _snap_figure_to_column(
        self,
        bbox: Tuple[int, int, int, int],
        caption_cx: float,
        text_lines: List[Dict[str, object]],
        other_anchors: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> Tuple[int, int]:
        """Clip a figure's x-range to its column, then widen left-only.

        1. Gutter clip — infers column from other anchors (info-box, table).
        2. Left-label widen — absorbs short row-labels to the LEFT of the
           figure (e.g. page 70 "Tế bào ..."). NEVER widens right, which
           prevents absorbing right-margin question prompts (CTST page 45).
        """
        x0, y0, x1, y1 = bbox
        fig_height = max(1, y1 - y0)
        margin = int(page_width * 0.01)
        right_limit = page_width
        left_limit = 0
        inside_lo = y0 + 0.12 * fig_height
        inside_hi = y1 - 0.05 * fig_height
        for ax0, ay0, ax1, ay1 in other_anchors:
            acy = (ay0 + ay1) / 2.0
            if acy < inside_lo or acy > inside_hi:
                continue
            acx = (ax0 + ax1) / 2.0
            if acx > caption_cx:
                right_limit = min(right_limit, ax0 - margin)
            elif acx < caption_cx:
                left_limit = max(left_limit, ax1 + margin)

        # Clip to gutter limits.
        if left_limit < x1:
            x0 = max(x0, left_limit)
        if right_limit > x0:
            x1 = min(x1, right_limit)

        # Left-label widen: short row-labels (≤25% page width) that start
        # to the left of x0 and reach close to (or into) the figure's x0
        # edge. Only extends x0 LEFTWARD — x1 is untouched.
        new_x0 = x0
        adj_gap = int(page_width * 0.04)
        max_line_width = int(page_width * 0.25)
        for line in text_lines:
            lx0, ly0, lx1, ly1 = (
                int(line["bbox"][0]),  # type: ignore[index]
                int(line["bbox"][1]),  # type: ignore[index]
                int(line["bbox"][2]),  # type: ignore[index]
                int(line["bbox"][3]),  # type: ignore[index]
            )
            lcy = (ly0 + ly1) / 2.0
            if lcy < y0 or lcy > y1:
                continue
            if (lx1 - lx0) > max_line_width:
                continue
            if lx1 < left_limit or lx0 > x1 + adj_gap:
                continue
            # Must start to the left of x0 and reach close to it.
            if lx0 >= new_x0 or lx1 < new_x0 - adj_gap:
                continue
            text = str(line["text"]).strip()
            if (FIG_CAPTION_STRICT_REGEX.match(text)
                    or TABLE_CAPTION_STRICT_REGEX.match(text)
                    or self._match_info_box_title(text)
                    or self._is_question_prompt_text(text)):
                continue
            new_x0 = min(new_x0, lx0)

        return new_x0, x1

    def _grow_figure_top(
        self,
        x0: int,
        y0: int,
        x1: int,
        ceiling_y: int,
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> Tuple[int, int, int]:
        """Extend a composite's top edge up through contiguous column text.

        Walks text lines that (a) sit above the current top, (b) horizontally
        overlap the current column, (c) are gap-connected, and (d) are NOT an
        anchor line (figure / table / info title / question prompt). Each
        absorbed line raises the top and may widen the column.

        The gap budget is generous enough to bridge the whitespace between the
        rows of a list-figure (page 70 has ~150 px between cell rows).
        """
        max_gap = int(page_height * self._FIG_TOP_GROW_MAX_GAP)
        cur_top = y0
        cur_x0 = x0
        cur_x1 = x1
        # A figure's own cell labels are short. A full-width body paragraph is
        # not part of the figure — never absorb it (this keeps side-by-side
        # figures from each ballooning to the whole page, page 85).
        max_line_width = int(page_width * self._FIG_TOP_GROW_MAX_WIDTH)

        # Candidate lines above the current top, sorted nearest-first.
        candidates = [
            line for line in text_lines
            if int(line["bbox"][3]) <= cur_top + 4  # type: ignore[index]
            and int(line["bbox"][3]) >= ceiling_y - 2  # type: ignore[index]
        ]
        candidates.sort(key=lambda line: int(line["bbox"][3]), reverse=True)

        for line in candidates:
            lx0, ly0, lx1, ly1 = (
                int(line["bbox"][0]),  # type: ignore[index]
                int(line["bbox"][1]),  # type: ignore[index]
                int(line["bbox"][2]),  # type: ignore[index]
                int(line["bbox"][3]),  # type: ignore[index]
            )
            # Horizontal overlap with the current column (small tolerance).
            tol = int(page_width * 0.02)
            if lx1 < cur_x0 - tol or lx0 > cur_x1 + tol:
                continue
            if (lx1 - lx0) > max_line_width:
                break
            text = str(line["text"]).strip()
            # Anchor lines bound the figure — never absorb them.
            if (FIG_CAPTION_STRICT_REGEX.match(text)
                    or TABLE_CAPTION_STRICT_REGEX.match(text)
                    or self._match_info_box_title(text)
                    or self._is_question_prompt_text(text)):
                break
            gap = cur_top - ly1
            if gap > max_gap:
                break
            cur_top = min(cur_top, ly0)
            # Track the column internally so the overlap test follows the
            # labels leftward, but do NOT return a widened x — horizontal
            # extent is decided by `_snap_figure_to_column`, which has the
            # gutter limits. (Returning a widened x here let a wide header row
            # blow the figure into the margin notes — page 45.)
            cur_x0 = min(cur_x0, lx0)
            cur_x1 = max(cur_x1, lx1)

        return x0, cur_top, x1

    def _build_info_panels(
        self,
        info_titles: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        all_blockers: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Anchor on info-box title; extend downward through text + adjacent visuals."""
        outputs: List[Dict[str, object]] = []
        if not info_titles:
            return outputs

        max_gap = int(page_height * 0.055)
        for title in info_titles:
            tx0, ty0, tx1, ty1 = title["bbox"]  # type: ignore[misc]

            # Downward ceiling: nearest blocker top below the title.
            floor_y = page_height
            for blocker in all_blockers:
                bx0, by0, bx1, by1 = blocker
                if by0 <= ty1 + 2:
                    continue
                floor_y = min(floor_y, by0 - int(page_height * 0.004))

            panel_y_bottom = ty1
            prev_y_bottom = ty1
            text_x_left = tx0
            text_x_right = tx1
            # Accumulated panel column — used to reject text from the OTHER
            # column on a 2-column page (page 70: a right-column "Em có biết"
            # must not absorb the left-column figure's cell labels).
            col_x_left = tx0
            col_x_right = tx1
            col_tol = int(page_width * 0.05)

            for line in text_lines:
                lx0, ly0, lx1, ly1 = (
                    int(line["bbox"][0]),  # type: ignore[index]
                    int(line["bbox"][1]),  # type: ignore[index]
                    int(line["bbox"][2]),  # type: ignore[index]
                    int(line["bbox"][3]),  # type: ignore[index]
                )
                if ly0 <= ty1:
                    continue
                if ly1 > floor_y:
                    break

                text = str(line["text"]).strip()
                if FIG_CAPTION_STRICT_REGEX.match(text):
                    break
                if TABLE_CAPTION_STRICT_REGEX.match(text):
                    break
                if (self._match_info_box_title(text)
                        and line["index"] != title["index"]):  # type: ignore[index]
                    break

                gap = ly0 - prev_y_bottom
                if gap > max_gap:
                    break
                # Column guard: skip lines that don't overlap the panel column.
                if lx1 < col_x_left - col_tol or lx0 > col_x_right + col_tol:
                    continue

                panel_y_bottom = max(panel_y_bottom, ly1)
                prev_y_bottom = max(prev_y_bottom, ly1)
                text_x_left = min(text_x_left, lx0)
                text_x_right = max(text_x_right, lx1)
                col_x_left = min(col_x_left, lx0)
                col_x_right = max(col_x_right, lx1)

            # Pull in any visual region that sits inside the panel y-band AND
            # the panel's horizontal column (e.g. Marie Curie portrait next to
            # the text). The column guard stops a right-column info box from
            # swallowing a figure cell in the left column (page 70).
            region_y_bottom = panel_y_bottom
            region_x_left = text_x_left
            region_x_right = text_x_right
            # Only pull in a visual whose horizontal CENTRE lies within the
            # panel's text column. Touching the column edge is not enough — on
            # tight 2-column layouts (page 70) the figure's rightmost cell sits
            # just left of the info box and must not be absorbed.
            col_left = min(tx0, text_x_left)
            col_right = max(tx1, text_x_right)
            for region in visual_regions:
                rx0, ry0, rx1, ry1 = region
                if ry0 > floor_y or ry1 < ty0:
                    continue
                if ry0 < ty0 - int(page_height * 0.02):
                    continue
                if ry1 > panel_y_bottom + int(page_height * 0.04):
                    continue
                rcx = (rx0 + rx1) / 2.0
                if rcx < col_left or rcx > col_right:
                    continue
                region_x_left = min(region_x_left, rx0)
                region_x_right = max(region_x_right, rx1)
                region_y_bottom = max(region_y_bottom, ry1)

            pad_x = int(page_width * 0.012)
            pad_y = int(page_height * 0.010)

            # Column-aware width. A full-width info box (title hugging the left
            # margin, page-6 style) is clamped to the page text frame. A
            # right/left-column info box (title indented, page-70 style) stays
            # within its own column so it does NOT overlap a figure in the
            # other column.
            is_full_width = tx0 < page_width * 0.20
            if is_full_width:
                outer_left = int(page_width * 0.035)
                outer_right = int(page_width * 0.965)
                box_left = max(0, min(outer_left, region_x_left - pad_x))
                box_right = min(page_width, max(
                    outer_right, region_x_right + pad_x))
            else:
                box_left = max(0, region_x_left - pad_x)
                box_right = min(page_width, region_x_right + pad_x)
                # Keep a single-column box inside its own column: never bleed
                # across the central gutter into the other column (page 56
                # "Em có biết" was grabbing the left-column body + figure cells).
                col_margin = int(page_width * 0.03)
                title_cx = (tx0 + tx1) / 2.0
                mid = page_width * 0.5
                if title_cx >= mid:           # right-column box
                    box_left = max(box_left, int(mid - col_margin))
                else:                          # left-column box
                    box_right = min(box_right, int(mid + col_margin))
            bbox = (
                box_left,
                max(0, ty0 - pad_y),
                box_right,
                min(page_height, region_y_bottom + pad_y),
            )
            outputs.append({
                "bbox": bbox,
                "image_type": title.get("label", "textbook_info_box"),
                "caption_text": title["text"],
                "caption_bbox": title["bbox"],
            })

        return outputs

    def _build_dashed_tool_groups(
        self,
        dashed_regions: List[Tuple[int, int, int, int]],
        framed_regions: List[Tuple[int, int, int, int]],
        owlvit_regions: List[Tuple[int, int, int, int]],
        tool_labels: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        exclusion_zones: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Tool-group panels anchored on a "Dụng cụ đo ..." label.

        Many SGK pages render this as a dashed-border row, but not always —
        some pages just put 3-4 instrument photos in a row below the label
        without a visible border. We therefore anchor on the LABEL and
        collect every visual region in the row directly below it.

        The dashed/framed detections are used as a positive signal to
        widen the panel bbox when present.
        """
        outputs: List[Dict[str, object]] = []
        if not tool_labels:
            return outputs

        # Per-cell instrument detections (rulers, scales, etc.). We must NOT
        # use the deduped visual_regions because individual cells are fully
        # contained in any dashed-border outer box and the coverage filter
        # would kill them. Light intra-cell dedupe only.
        min_cell_area = int(page_width * page_height * 0.0025)
        raw_cells = [tuple(r) for r in owlvit_regions
                     if self._bbox_area(r) >= min_cell_area]
        raw_cells.sort(key=lambda b: self._bbox_area(b), reverse=True)
        all_cells: List[Tuple[int, int, int, int]] = []
        for cell in raw_cells:
            duplicate = any(self._iou(cell, kept) > 0.55 for kept in all_cells)
            if not duplicate:
                all_cells.append(cell)

        used_cells: set = set()
        # type: ignore[index]
        for label in sorted(tool_labels, key=lambda l: l["bbox"][1]):
            lx0, ly0, lx1, ly1 = label["bbox"]  # type: ignore[misc]

            # Find every cell sitting in the row directly below the label.
            row_cells: List[Tuple[int, int, int, int]] = []
            for cell in all_cells:
                if cell in used_cells:
                    continue
                cx0, cy0, cx1, cy1 = cell
                vgap = cy0 - ly1
                if vgap < -10 or vgap > page_height * 0.10:
                    continue
                # Row must not extend further than ~25% page height down.
                if cy1 - ly1 > page_height * 0.28:
                    continue
                # Horizontal: cell must intersect label band or be near it.
                hov = min(lx1, cx1) - max(lx0, cx0)
                if hov <= 0 and (cx0 > lx1 + page_width * 0.18 or
                                 cx1 < lx0 - page_width * 0.18):
                    continue
                row_cells.append(cell)

            # Need at least 2 instruments to count as a "group".
            if len(row_cells) < 2:
                continue

            # Reject group when label sits inside an existing anchor zone
            # (page header, table caption, etc.) — those rule it out.
            if any(self._coverage_ratio(label["bbox"], zone) > 0.50  # type: ignore[arg-type]
                   for zone in exclusion_zones):
                continue

            group_x0 = min(lx0, min(c[0] for c in row_cells))
            group_x1 = max(lx1, max(c[2] for c in row_cells))
            group_y0 = ly0
            group_y1 = max(c[3] for c in row_cells)

            # If a dashed/framed outer box overlaps the row strongly, widen
            # the bbox to its full extent — OWL-ViT often misses the
            # last cell at the right edge, and the dashed border preserves it.
            cell_band = (group_x0, group_y0, group_x1, group_y1)
            for outer in list(dashed_regions) + list(framed_regions):
                ox0, oy0, ox1, oy1 = outer
                # Outer must sit in the row band.
                if oy0 > group_y1 + int(page_height * 0.02):
                    continue
                if oy1 < group_y0 - int(page_height * 0.02):
                    continue
                # Outer must overlap the cell band horizontally.
                hov = min(ox1, group_x1) - max(ox0, group_x0)
                if hov <= 0:
                    continue
                # And contain at least one cell.
                if not any(self._coverage_ratio(c, outer) > 0.6
                           for c in row_cells):
                    continue
                group_x0 = min(group_x0, ox0)
                group_x1 = max(group_x1, ox1)
                group_y0 = min(group_y0, oy0)
                group_y1 = max(group_y1, oy1)

            # Extend down to include per-tool name labels ("Thước cuộn",
            # "Cân đồng hồ", ...). These are short text lines on the row
            # immediately below the cells. We also use them to widen the
            # bbox horizontally — they reliably span the full instrument
            # row even when OWL-ViT misses cells at the edges.
            for line in text_lines:
                tx0, ty0, tx1, ty1 = (
                    int(line["bbox"][0]),  # type: ignore[index]
                    int(line["bbox"][1]),  # type: ignore[index]
                    int(line["bbox"][2]),  # type: ignore[index]
                    int(line["bbox"][3]),  # type: ignore[index]
                )
                if ty0 < group_y1 - 4 or ty0 > group_y1 + int(page_height * 0.045):
                    continue
                txt = str(line["text"]).strip()
                if not txt or len(txt) > 60:
                    continue
                if FIG_CAPTION_STRICT_REGEX.match(txt):
                    continue
                if TABLE_CAPTION_STRICT_REGEX.match(txt):
                    continue
                if self._match_info_box_title(txt):
                    continue
                if SUB_FIGURE_LABEL_REGEX.match(txt):
                    continue
                # Must be roughly within the page text frame.
                if tx0 < int(page_width * 0.03) or tx1 > int(page_width * 0.97):
                    continue
                group_y1 = max(group_y1, ty1)
                group_x0 = min(group_x0, tx0)
                group_x1 = max(group_x1, tx1)

            group_bbox = (
                max(0, group_x0 - int(page_width * 0.008)),
                max(0, group_y0 - int(page_height * 0.006)),
                min(page_width, group_x1 + int(page_width * 0.008)),
                min(page_height, group_y1 + int(page_height * 0.008)),
            )

            # Re-check exclusion of final bbox against figure/info zones.
            if any(self._coverage_ratio(group_bbox, zone) > 0.45
                   for zone in exclusion_zones):
                continue

            for cell in row_cells:
                used_cells.add(cell)

            outputs.append({
                "bbox": group_bbox,
                "image_type": "tool_group",
                "caption_text": str(label["text"]),
                "caption_bbox": label["bbox"],
            })

        return outputs

    def _split_region_sub_figures(
        self,
        region_bbox: Tuple[int, int, int, int],
        visual_regions: List[Tuple[int, int, int, int]],
        sub_labels: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        pil_img: Image.Image,
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Split a figure / tool group into per-photo sub-figures.

        The strategy is **caption-rows + pixel-columns**, which is robust to the
        detector merging a whole grid into one (or a few) cells — the common CD
        failure where OWL-ViT returns one box per coloured row (page 8) or one
        box for the whole grid (page 131):
          * ROWS come from the caption lines below each photo — letter labels
            ('a)/b)/…') when present, else (CD only, via
            `_SPLIT_SUBFIGURES_BY_TITLE`) the short centred titles below each
            cell ("Con cá heo / …", "Thước cuộn / …").
          * COLUMNS come from the VERTICAL WHITE GUTTERS in each row's photo
            band (`_detect_columns_by_projection`) — so a row detected as one
            wide cell still splits into its photos, and a ragged grid (3 photos
            over 2) splits correctly per row.
        A caption row with NO photo band above it (page 40 internal apparatus
        labels "Dung dịch / Nến") is rejected by the photo-band guard.
        """
        cells: List[Tuple[int, int, int, int]] = []
        for raw in visual_regions:
            vr = tuple(int(v) for v in raw)
            if self._coverage_ratio(vr, region_bbox) < 0.70:
                continue
            cells.append(vr)
        if not cells:
            return []
        ux0 = min(c[0] for c in cells)
        uy0 = min(c[1] for c in cells)
        ux1 = max(c[2] for c in cells)
        uy1 = max(c[3] for c in cells)

        anchors, is_label_mode = self._collect_subfig_anchors(
            (ux0, uy0, ux1, uy1), region_bbox, sub_labels,
            text_lines, page_width, page_height)
        if len(anchors) < 2:
            return []

        # Cluster caption anchors into rows (by top-y).
        row_tol = int(page_height * 0.03)
        anchors.sort(key=lambda a: a["y0"])
        rows: List[List[Dict[str, float]]] = []
        for anchor in anchors:
            if rows and (anchor["y0"] - rows[-1][-1]["y0"]) <= row_tol:
                rows[-1].append(anchor)
            else:
                rows.append([anchor])

        # Title-mode is inherently riskier than letter labels (legend lists,
        # internal diagram labels). Require strong grid evidence: ≥2 distinct
        # title columns AND at least one row holding ≥2 side-by-side titles —
        # this rejects a vertically-stacked legend beside a single chart
        # (page 40 "Hình 7.3" pie + legend).
        if not is_label_mode:
            col_tol = int(page_width * 0.10)
            col_clusters = self._cluster_sorted_values(
                sorted(a["cx"] for a in anchors), col_tol)
            if len(col_clusters) < 2 or max(len(r) for r in rows) < 2:
                return []
            # Reject diagrams with labels scattered over many rows (CTST biogas
            # flow chart) for publishers whose titled figures are single-row.
            if len(rows) > self._SUBFIG_TITLE_MAX_ROWS:
                return []

        # Build each row's photo band; reject a caption row with no photo above
        # it (page 40 "Hình 7.2" internal apparatus labels "Dung dịch / Nến").
        min_photo_h = int(page_height * 0.05)
        bands: List[Tuple[int, int, int, List[Dict[str, float]]]] = []
        prev_bottom = int(uy0)
        for row in rows:  # top → bottom
            title_top = int(min(a["y0"] for a in row))
            title_bottom = int(max(a["y1"] for a in row))
            if title_top - prev_bottom < min_photo_h:
                return []
            bands.append((prev_bottom, title_top, title_bottom, row))
            prev_bottom = title_bottom

        # A cell is "granular" when it fits inside a single row's photo band
        # (the detector separated the photos — page 6, 109). When the only
        # cells span multiple rows / a whole coloured row (page 8, 131) there
        # are none, and we fall back to pixel-gutter column detection.
        min_col_w = int(page_width * 0.05)
        min_cell_h = int(page_height * 0.03)
        fit_tol = int(page_height * 0.02)
        granular: List[Tuple[int, int, int, int]] = []
        for cell in cells:
            cw, ch = cell[2] - cell[0], cell[3] - cell[1]
            if cw < min_col_w or ch < min_cell_h:
                continue
            for ptop, ttop, _tbot, _row in bands:
                if cell[1] >= ptop - fit_tol and cell[3] <= ttop + fit_tol:
                    granular.append(cell)
                    break
        granular = self._dedupe_visual_regions(
            granular, min_area=int(page_width * page_height * 0.003),
            iou_threshold=0.35)

        pad_x = int(page_width * 0.004)
        pad_y = int(page_height * 0.004)
        outputs: List[Dict[str, object]] = []

        if len(granular) >= 2 and len(granular) >= len(rows):
            # CELL MODE — one crop per detected cell, caption attached below.
            for cell in sorted(granular, key=lambda c: (c[1], c[0])):
                cx0, cy0, cx1, cy1 = cell
                bottom = cy1
                caption_text = ""
                for anchor in anchors:
                    if not (cx0 <= anchor["cx"] <= cx1):
                        continue
                    if anchor["y0"] < cy1 - fit_tol \
                            or anchor["y0"] > cy1 + page_height * 0.06:
                        continue
                    bottom = max(bottom, int(anchor["y1"]))
                    caption_text = str(anchor["text"])
                    break
                outputs.append({
                    "bbox": (
                        max(0, cx0 - pad_x), max(0, cy0 - pad_y),
                        min(page_width, cx1 + pad_x),
                        min(page_height, bottom + pad_y),
                    ),
                    "image_type": "sub_figure",
                    "caption_text": caption_text,
                })
        else:
            # PROJECTION MODE — split each row's band at vertical white gutters.
            for ptop, ttop, tbot, row in bands:
                columns = self._detect_columns_by_projection(
                    pil_img, (ux0, ptop, ux1, ttop), page_width, page_height)
                if not columns:
                    continue
                for cseg0, cseg1 in columns:
                    caption_text = ""
                    for anchor in row:
                        if cseg0 <= anchor["cx"] <= cseg1:
                            caption_text = str(anchor["text"])
                            break
                    outputs.append({
                        "bbox": (
                            max(0, cseg0 - pad_x), max(0, ptop - pad_y),
                            min(page_width, cseg1 + pad_x),
                            min(page_height, tbot + pad_y),
                        ),
                        "image_type": "sub_figure",
                        "caption_text": caption_text,
                    })

        if len(outputs) < 2:
            return []
        return outputs

    @staticmethod
    def _cluster_sorted_values(values: List[float], tol: float) -> List[float]:
        """Cluster ascending values whose neighbour-gap ≤ tol; return centroids."""
        clusters: List[float] = []
        current: List[float] = []
        for value in values:
            if current and (value - current[-1]) > tol:
                clusters.append(sum(current) / len(current))
                current = []
            current.append(value)
        if current:
            clusters.append(sum(current) / len(current))
        return clusters

    def _collect_subfig_anchors(
        self,
        band: Tuple[int, int, int, int],
        region_bbox: Tuple[int, int, int, int],
        sub_labels: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> Tuple[List[Dict[str, float]], bool]:
        """Caption anchors below each cell, plus a flag for which kind.

        Returns ``(anchors, is_label_mode)``. Each anchor =
        ``{"cx", "x0", "x1", "y0", "y1", "text"}``. Letter labels win when
        present (the reliable per-cell markers, ``is_label_mode=True``); only
        when a region carries NONE does the CD title path collect short centred
        captions ("Con cá heo", "Thước cuộn") so unlabelled photo grids and
        tool rows still split (``is_label_mode=False``).
        """
        ux0, uy0, ux1, uy1 = band
        rx0, ry0, rx1, ry1 = region_bbox
        label_margin = int(page_height * 0.05)
        anchors: List[Dict[str, float]] = []
        for label in sub_labels:
            lx0, ly0, lx1, ly1 = (int(v) for v in label["bbox"])  # type: ignore[misc]
            if lx0 >= rx0 - 5 and lx1 <= rx1 + 5 \
                    and ly0 >= ry0 - 5 and ly0 <= ry1 + label_margin:
                anchors.append({
                    "cx": (lx0 + lx1) / 2.0,
                    "x0": float(lx0), "x1": float(lx1),
                    "y0": float(ly0), "y1": float(ly1),
                    "text": str(label["text"])})  # type: ignore[dict-item]
        if anchors:
            return anchors, True
        if not self._SPLIT_SUBFIGURES_BY_TITLE:
            return [], False

        x_tol = int(page_width * 0.01)
        below_limit = uy1 + int(page_height * 0.04)
        max_title_width = int(page_width * 0.22)
        for line in text_lines:
            lx0, ly0, lx1, ly1 = (int(v) for v in line["bbox"])  # type: ignore[misc]
            cx = (lx0 + lx1) / 2.0
            if cx < ux0 - x_tol or cx > ux1 + x_tol:
                continue
            if ly0 < uy0 or ly0 > below_limit:
                continue
            if (lx1 - lx0) > max_title_width:
                continue
            text = str(line["text"]).strip()
            if (FIG_CAPTION_STRICT_REGEX.match(text)
                    or TABLE_CAPTION_STRICT_REGEX.match(text)
                    or SUB_FIGURE_LABEL_REGEX.match(text)
                    or self._match_info_box_title(text)
                    or self._is_question_prompt_text(text)
                    or TOOL_GROUP_LABEL_REGEX.match(text)):
                continue
            # Drop OCR junk (stray glyphs like 'La', '"') — a real caption has
            # at least a few letters.
            if len(re.sub(r"[^a-z]", "", self._normalize_text(text))) < 3:
                continue
            anchors.append({
                "cx": cx, "x0": float(lx0), "x1": float(lx1),
                "y0": float(ly0), "y1": float(ly1),
                "text": text})  # type: ignore[dict-item]
        return anchors, False

    def _detect_columns_by_projection(
        self,
        pil_img: Image.Image,
        band: Tuple[int, int, int, int],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[int, int]]:
        """Split a photo band into column x-ranges at vertical white gutters.

        A real inter-photo gutter is a near-white vertical strip spanning the
        band height. Thin white columns INSIDE a photo are bridged so a single
        picture is never sliced.
        """
        bx0, by0, bx1, by1 = (int(band[0]), int(band[1]),
                              int(band[2]), int(band[3]))
        if by1 - by0 < 12 or bx1 - bx0 < int(page_width * 0.06):
            return []
        crop = np.array(
            pil_img.crop((bx0, by0, bx1, by1)).convert("L")).astype(np.float32)
        if crop.size == 0:
            return []
        # Per-column variance down the band: a gutter is a vertically-uniform
        # strip (white page OR a flat coloured row background — page 8 cells sit
        # on blue/green panels), so its std is low; a photo column varies a lot.
        col_std = crop.std(axis=0)
        is_content = col_std >= 14.0
        # Bridge thin gutters (white runs narrower than min_gutter) so an
        # internal white stripe inside a photo doesn't split it.
        min_gutter = max(6, int(page_width * 0.012))
        width = len(is_content)
        index = 0
        while index < width:
            if is_content[index]:
                index += 1
                continue
            run_start = index
            while index < width and not is_content[index]:
                index += 1
            if (index - run_start) < min_gutter:
                is_content[run_start:index] = True
        # Collect content runs wide enough to be a photo column.
        min_col_w = int(page_width * 0.05)
        columns: List[Tuple[int, int]] = []
        index = 0
        while index < width:
            if not is_content[index]:
                index += 1
                continue
            run_start = index
            while index < width and is_content[index]:
                index += 1
            if (index - run_start) >= min_col_w:
                columns.append((bx0 + run_start, bx0 + index))
        return columns

    def detect_regions_anchor_first(
        self,
        pil_img: Image.Image,
        img_array: np.ndarray,
        text_lines: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Top-level v7 detector. Returns dict with 'regions' (list) and 'debug' fields.

        Each region is ``{"bbox", "image_type", "caption_text", "caption_bbox?"}``.
        """
        page_width = pil_img.width
        page_height = pil_img.height
        if text_lines is None:
            text_lines = self._collect_page_text_lines(pil_img)

        anchors = self._classify_text_anchors(text_lines)

        # Augment OCR-derived info titles with colour-detected headers.
        # Vietnamese SGK marks info boxes with a pink/blue header; when the
        # header is a filled tab with white text, page-level OCR misses it
        # entirely (page 70 "Em có biết"). Recover those here and merge.
        colored_headers = self._detect_colored_info_headers(pil_img)
        if colored_headers:
            existing = anchors["info_titles"]
            for header in colored_headers:
                hx0, hy0, hx1, hy1 = header["bbox"]  # type: ignore[misc]
                hcy = (hy0 + hy1) / 2.0
                # Skip if an OCR title already covers this header position.
                duplicate = False
                for title in existing:
                    tx0, ty0, tx1, ty1 = title["bbox"]  # type: ignore[misc]
                    if abs((ty0 + ty1) / 2.0 - hcy) < page_height * 0.03 \
                            and min(hx1, tx1) - max(hx0, tx0) > 0:
                        duplicate = True
                        break
                if not duplicate:
                    existing.append(header)
            existing.sort(key=lambda e: e["bbox"][1])  # type: ignore[index]

        # Raw visual detections (used by all builders, never returned as-is).
        owlvit_regions = self._detect_regions_with_owlvit(
            pil_img, OWL_VIT_TEXT_QUERIES)
        framed_regions = self._detect_framed_regions(img_array)
        dashed_regions = self._detect_dashed_frame_regions(img_array)
        # Object-blob fallback ONLY fills gaps where OWL-ViT / frame detectors
        # found nothing — it must not override the per-cell detections that
        # already work (otherwise a blob spanning several sub-figures would
        # suppress them).
        detector_regions = (list(owlvit_regions) + list(framed_regions)
                            + list(dashed_regions))
        # Pale-photo fallback (publisher opt-in): large textured regions OWL and
        # the colour-blob detector miss. Text blocks are dropped below by
        # `_filter_text_visual_regions`.
        if self._DETECT_TEXTURED_PHOTOS:
            detector_regions += list(
                self._detect_textured_photo_regions(img_array))
        # Solid photo-rectangle fallback (opt-in): recovers full photos OWL only
        # partially detects or the blob detector merges with text.
        if self._DETECT_PHOTO_RECTANGLES:
            detector_regions += list(
                self._detect_photo_rectangles(img_array))
        blob_regions = [
            blob for blob in self._detect_object_blobs(img_array)
            if not any(self._coverage_ratio(blob, d) > 0.35
                       or self._coverage_ratio(d, blob) > 0.55
                       for d in detector_regions)
        ]
        visual_regions = self._dedupe_visual_regions(
            detector_regions + blob_regions,
            min_area=1500,
            iou_threshold=0.55,
        )
        # Drop OWL-ViT false positives that are really text blocks before they
        # can be assigned to a figure caption.
        visual_regions = self._filter_text_visual_regions(
            visual_regions, pil_img, text_lines)

        # 1. Table zones (exclusion-only, never emitted as regions).
        table_zones = self._build_table_zones(
            anchors["table_captions"], text_lines, page_width, page_height,
        )

        # 2. Info-box panels FIRST. Their full (tall) bboxes are then used as
        # exclusion zones + column separators when assigning cells to figures,
        # which is far more robust than the thin info-box header line (page 70:
        # the "Tìm hiểu thêm" panel keeps its body's stray OWL-ViT detections
        # out of the left-column figure).
        info_blockers = (
            [cap["bbox"] for cap in anchors["figure_captions"]]
            + [cap["bbox"] for cap in anchors["table_captions"]]
            + [cap["bbox"] for cap in anchors["info_titles"]]
        )
        info_outputs = self._build_info_panels(
            anchors["info_titles"], text_lines, visual_regions,
            info_blockers, page_width, page_height,
        )
        info_zones = [out["bbox"] for out in info_outputs]

        # 3. Figure composites. Exclusion = table zones + info panels.
        figure_outputs = self._build_figure_composites(
            anchors["figure_captions"], text_lines, visual_regions,
            anchors["question_prompts"], anchors["info_titles"],
            anchors["sub_labels"],
            list(table_zones) + list(info_zones), page_width, page_height,
        )

        # 4. Dashed-only tool groups.
        anchor_zones = (
            table_zones
            + [out["bbox"] for out in figure_outputs]
            + info_zones
        )
        tool_outputs = self._build_dashed_tool_groups(
            dashed_regions, framed_regions, owlvit_regions,
            anchors["tool_group_labels"], text_lines, anchor_zones,
            page_width, page_height,
        )

        # Clean up internal fields before returning.
        for fig in figure_outputs:
            fig.pop("assigned_regions", None)

        regions = figure_outputs + info_outputs + tool_outputs

        # ── Recover figures the caption-first builder missed ──────────────
        # (a) Caption anchor present but no visual region was assigned to it
        #     (B&W line drawings OWL-ViT skipped, side-by-side singles, or a
        #     caption whose cells were stolen by an over-grown neighbour).
        #     Build a region from the visual/whitespace band directly above it.
        covered_caps = {
            tuple(f["caption_bbox"]) for f in figure_outputs  # type: ignore[arg-type]
            if f.get("caption_bbox")
        }
        uncovered_caps = [
            c for c in anchors["figure_captions"]
            if tuple(c["bbox"]) not in covered_caps  # type: ignore[arg-type]
        ]
        if uncovered_caps:
            extra = self._build_uncovered_caption_regions(
                uncovered_caps, visual_regions, regions, text_lines,
                list(table_zones) + list(info_zones),
                page_width, page_height,
            )
            regions = regions + extra

        # ── Drop text-only info/activity panels (publisher opt-in) ────────
        #    A real info box has a coloured background or an embedded picture;
        #    a bare section header on white is body text, not an image. Done
        #    BEFORE photo recovery so a dropped text box doesn't shadow a real
        #    picture sitting inside it (CTST page 174).
        if self._INFO_REQUIRE_VISUAL:
            regions = [r for r in regions
                       if not self._is_text_only_panel(r, pil_img)]

        # (b) Picture detected but its caption anchor was missed by OCR
        #     (e.g. CTST '▲ Hình' mangled to 'Aình403'): re-OCR the strip
        #     directly below the picture. Publisher opt-in.
        if self._RECOVER_CAPTIONS_BELOW_PHOTOS:
            emitted = [tuple(r["bbox"]) for r in regions]  # type: ignore[arg-type]
            recovered = self._recover_captions_below_photos(
                pil_img, visual_regions, emitted, page_width, page_height,
            )
            regions = regions + recovered

        # ── Final overlap suppression (fixes duplicate / "đè mất" crops) ──
        regions = self._suppress_overlapping_regions(
            regions, page_width, page_height)

        # ── Split multi-cell figures into per-cell sub-figures (a/b/c/d) ──
        #    Runs AFTER recovery so recovered composites (KNTT caption-above
        #    mushroom rows, CD a/b line drawings) split too. Parent is kept and
        #    relabelled composite_figure; sub-figures are added alongside it.
        if self._SPLIT_SUBFIGURES:
            sub_outputs: List[Dict[str, object]] = []
            for region in regions:
                rtype = region.get("image_type")
                if rtype not in (
                        "single_figure", "composite_figure", "tool_group"):
                    continue
                subs = self._split_region_sub_figures(
                    tuple(region["bbox"]), visual_regions,  # type: ignore[arg-type]
                    anchors["sub_labels"], text_lines, pil_img,
                    page_width, page_height,
                )
                if len(subs) >= 2:
                    # A figure that splits is a composite; a tool group keeps
                    # its type (the per-tool crops are added alongside it).
                    if rtype != "tool_group":
                        region["image_type"] = "composite_figure"
                    sub_outputs.extend(subs)
            regions = regions + sub_outputs

        return {
            "regions": regions,
            "anchors": anchors,
            "table_zones": table_zones,
            "visual_regions": visual_regions,
            "owlvit_regions": owlvit_regions,
            "framed_regions": framed_regions,
            "dashed_regions": dashed_regions,
        }

    # ------------------------------------------------------------------
    # Recovery / cleanup helpers (shared; tuned via the class attributes)
    # ------------------------------------------------------------------

    def _match_recovered_caption(self, text: str) -> str:
        """Return cleaned 'Hình X.Y …' if a figure marker appears in `text`.

        Used on re-OCR'd caption strips, so it SEARCHES anywhere in the line
        (the strip may carry a leading marker such as the CTST '▲' OCR'd as
        'A'). Subclasses with different caption vocabulary may override.
        """
        match = re.search(
            r"H[iì]nh\s+\d+(?:\.\d+)?[a-h]?", text, flags=re.IGNORECASE)
        if not match:
            return ""
        tail = text[match.start():].strip()
        if TABLE_CAPTION_STRICT_REGEX.match(tail):
            return ""
        return tail

    def _reocr_caption_below(
        self,
        pil_img: Image.Image,
        picture_bbox: Tuple[int, int, int, int],
        page_width: int,
        page_height: int,
    ) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
        """Re-OCR (upscaled) the strip directly below a picture.

        Returns (clean_caption_text, strip_bbox) when the strip reads as a
        figure caption, else None. Robust to mangled page-level OCR because it
        crops tight and upscales 3× before thresholding.
        """
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        except Exception:
            return None

        px0, py0, px1, py1 = picture_bbox
        pad_x = int((px1 - px0) * 0.06)
        sx0 = max(0, px0 - pad_x)
        sx1 = min(page_width, px1 + pad_x)
        # Start slightly INSIDE the picture bottom: a detector box often
        # over-extends to swallow the caption row sitting at its lower edge
        # (CTST page 59 building), so a strict "below the bottom" strip would
        # miss it. The "Hình X.Y" match still anchors on the caption text.
        sy0 = max(0, py1 - int(page_height * 0.03))
        sy1 = min(page_height, py1 + int(page_height * 0.065))
        if sy1 - sy0 < 12 or sx1 - sx0 < 30:
            return None

        strip = np.array(pil_img.convert("RGB"))[sy0:sy1, sx0:sx1]
        if strip.size == 0:
            return None
        gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
        up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(up, 150, 255, cv2.THRESH_BINARY)
        for psm in ("7", "6"):
            try:
                text = pytesseract.image_to_string(
                    bw, lang="vie", config=f"--psm {psm}")
            except Exception:
                continue
            caption = self._match_recovered_caption(" ".join(text.split()))
            if caption:
                return caption, (sx0, sy0, sx1, sy1)
        return None

    def _recover_captions_below_photos(
        self,
        pil_img: Image.Image,
        visual_regions: List[Tuple[int, int, int, int]],
        emitted: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Emit a figure for each real picture whose caption OCR was missed.

        Only fires on pictures NOT already inside an emitted region, that look
        like real photos (visual score), and only when the re-OCR'd strip below
        actually reads 'Hình X.Y'. This is what rescues CTST pages whose
        '▲ Hình' caption was mangled by page-level OCR.
        """
        outputs: List[Dict[str, object]] = []
        for raw in visual_regions:
            vr = tuple(int(v) for v in raw)
            if any(self._coverage_ratio(vr, e) > 0.5 for e in emitted):
                continue
            width = vr[2] - vr[0]
            height = vr[3] - vr[1]
            if width < page_width * 0.10 or height < page_height * 0.06:
                continue
            if self._visual_content_score(pil_img.crop(vr)) < self._RECOVER_MIN_VIS:
                continue
            result = self._reocr_caption_below(
                pil_img, vr, page_width, page_height)
            if not result:
                continue
            caption_text, (cx0, cy0, cx1, cy1) = result
            pad_x = int(page_width * 0.006)
            pad_y = int(page_height * 0.005)
            bbox = (
                max(0, min(vr[0], cx0) - pad_x),
                max(0, vr[1] - pad_y),
                min(page_width, max(vr[2], cx1) + pad_x),
                min(page_height, cy1 + pad_y),
            )
            outputs.append({
                "bbox": bbox,
                "image_type": "single_figure",
                "caption_text": caption_text,
            })
            emitted.append(bbox)
        return outputs

    def _build_uncovered_caption_regions(
        self,
        uncovered_caps: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        existing_regions: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        exclusion_zones: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Build a figure for each 'Hình X.Y' caption that got no region.

        Prefers the visual cells sitting directly above the caption in its
        column (the picture). Honours 'wider context' — captures the picture
        fully + the caption. Falls back to a bounded band above the caption
        only when no visual was detected (faint line drawing).
        """
        outputs: List[Dict[str, object]] = []
        existing = [tuple(r["bbox"]) for r in existing_regions]  # type: ignore[arg-type]

        for cap in sorted(uncovered_caps, key=lambda c: c["bbox"][1]):  # type: ignore[index]
            cx0, cy0, cx1, cy1 = cap["bbox"]  # type: ignore[misc]
            cap_cx = (cx0 + cx1) / 2.0

            # Ceiling: bottom of the nearest region/caption above in this column.
            ceiling = 0
            for ex in existing:
                if ex[3] <= cy0 - 2 and min(ex[2], cx1) - max(ex[0], cx0) > 0:
                    ceiling = max(ceiling, ex[3])
            for other in uncovered_caps:
                ob = other["bbox"]  # type: ignore[index]
                if ob is cap["bbox"]:
                    continue
                if ob[3] < cy0 - 2 and min(ob[2], cx1) - max(ob[0], cx0) > 0:
                    ceiling = max(ceiling, ob[3])
            top_limit = max(ceiling, cy0 - int(page_height * 0.26))

            # Picture sits ABOVE the caption (the normal SGK convention). Some
            # publishers (KNTT pill labels) put the caption ABOVE the figure
            # row — `_FIG_CAPTION_ABOVE_OK` enables that direction too.
            below_reach = int(page_height * 0.32)
            col_cells: List[Tuple[int, int, int, int]] = []
            for raw in visual_regions:
                vr = tuple(int(v) for v in raw)
                vx0, vy0, vx1, vy1 = vr
                above = (top_limit - 2) <= vy0 and vy1 <= cy0 + int(page_height * 0.04)
                below = (self._FIG_CAPTION_ABOVE_OK
                         and vy0 >= cy1 - int(page_height * 0.04)
                         and vy1 <= cy1 + below_reach)
                if not (above or below):
                    continue
                vcx = (vx0 + vx1) / 2.0
                hov = min(vx1, cx1) - max(vx0, cx0)
                if abs(vcx - cap_cx) > page_width * 0.30 and hov <= 0:
                    continue
                if any(self._coverage_ratio(vr, z) > 0.5 for z in exclusion_zones):
                    continue
                if any(self._coverage_ratio(vr, e) > 0.6 for e in existing):
                    continue
                col_cells.append(vr)

            if col_cells:
                x0 = min([cx0] + [c[0] for c in col_cells])
                x1 = max([cx1] + [c[2] for c in col_cells])
                y0 = min([int(cy0)] + [c[1] for c in col_cells])
                y1 = max([int(cy1)] + [c[3] for c in col_cells])
                bbox = (
                    max(0, x0 - 4), max(0, y0 - 4),
                    min(page_width, x1 + 4), min(page_height, y1 + 4),
                )
            else:
                # No detected visual cell. This is either a faint line drawing
                # OWL-ViT skipped (recover it) or a body-text *reference* to the
                # figure ("Hình 9.5 là một mô hình …", recover NOTHING). They are
                # told apart by the area above the caption: a figure area is
                # almost text-free, a paragraph reference is not.
                band_h = int(page_height * 0.16)
                y0 = max(top_limit, int(cy0) - band_h)
                cw = max(40, cx1 - cx0)
                bbox = (
                    max(0, int(cx0 - cw * 0.15)), max(0, y0 - 4),
                    min(page_width, int(cx1 + cw * 0.15)),
                    min(page_height, int(cy1) + 4),
                )
                if (bbox[3] - bbox[1]) < page_height * 0.05:
                    continue
                band_only = (bbox[0], bbox[1], bbox[2], int(cy0))
                if self._text_line_coverage(band_only, text_lines) >= 0.18:
                    continue

            outputs.append({
                "bbox": bbox,
                "image_type": "single_figure",
                "caption_text": cap.get("text", ""),
                "caption_bbox": cap["bbox"],
            })
            existing.append(bbox)
        return outputs

    def _is_text_only_panel(
        self,
        region: Dict[str, object],
        pil_img: Image.Image,
    ) -> bool:
        """True when an info/activity panel is bare text on white (not a real box)."""
        if region.get("image_type") not in ("textbook_info_box", "activity_box"):
            return False
        crop = pil_img.crop(tuple(region["bbox"]))  # type: ignore[arg-type]
        return self._visual_content_score(crop) < self._INFO_MIN_VIS

    def _suppress_overlapping_regions(
        self,
        regions: List[Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """Drop duplicate / heavily-overlapping emitted regions.

        Preserves the legitimate composite→sub_figure hierarchy; only removes
        same-tier near-duplicates and a region almost entirely contained by
        another (keeps the larger, caption-bearing one).
        """
        if len(regions) <= 1:
            return regions
        order = sorted(
            regions,
            key=lambda r: self._bbox_area(tuple(r["bbox"])),  # type: ignore[arg-type]
            reverse=True,
        )
        kept: List[Dict[str, object]] = []
        for region in order:
            rb = tuple(region["bbox"])  # type: ignore[arg-type]
            rtype = region.get("image_type")
            drop = False
            for keeper in kept:
                kb = tuple(keeper["bbox"])  # type: ignore[arg-type]
                ktype = keeper.get("image_type")
                # A sub_figure legitimately nests inside its composite parent.
                if {rtype, ktype} == {"sub_figure", "composite_figure"}:
                    continue
                if self._iou(rb, kb) > 0.60:
                    drop = True
                    break
                if (self._coverage_ratio(rb, kb) > 0.85
                        or self._coverage_ratio(kb, rb) > 0.85):
                    drop = True
                    break
            if not drop:
                kept.append(region)
        kept.sort(key=lambda r: tuple(r["bbox"])[1])  # type: ignore[arg-type]
        return kept

    def _bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        x0, y0, x1, y1 = bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
        ratio: float = 0.012,
    ) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        pad_x = max(4, int((x1 - x0) * ratio))
        pad_y = max(4, int((y1 - y0) * ratio))
        return (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(width, x1 + pad_x),
            min(height, y1 + pad_y),
        )

    def _iou(self, left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
        inter = self._intersection_area(left, right)
        if inter == 0:
            return 0.0
        union = self._bbox_area(left) + self._bbox_area(right) - inter
        return inter / union if union > 0 else 0.0

    def _intersection_area(self, left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> int:
        lx0, ly0, lx1, ly1 = left
        rx0, ry0, rx1, ry1 = right
        ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
        ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
        return max(0, ix1 - ix0) * max(0, iy1 - iy0)

    def _contains(self, outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
        ox0, oy0, ox1, oy1 = outer
        ix0, iy0, ix1, iy1 = inner
        return ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1

    def _is_hierarchical_overlap(
        self,
        left: Tuple[int, int, int, int],
        right: Tuple[int, int, int, int],
        min_area_ratio: float = 2.0,
    ) -> bool:
        """Return True when boxes are likely parent/child figures, not duplicates."""
        left_area = self._bbox_area(left)
        right_area = self._bbox_area(right)
        smaller_area = max(1, min(left_area, right_area))
        larger_area = max(left_area, right_area)
        if larger_area / smaller_area < min_area_ratio:
            return False

        overlap_of_smaller = self._intersection_area(
            left, right) / smaller_area
        return overlap_of_smaller >= 0.86

    def _contained_region_count(
        self,
        bbox: Tuple[int, int, int, int],
        regions: List[Tuple[int, int, int, int]],
        min_child_area_ratio: float = 0.035,
    ) -> int:
        area = self._bbox_area(bbox)
        count = 0
        for other in regions:
            if other == bbox:
                continue
            if not self._is_hierarchical_overlap(bbox, other):
                continue
            if self._bbox_area(other) < area and self._bbox_area(other) >= area * min_child_area_ratio:
                count += 1
        return count

    def _classify_region_hierarchy(
        self,
        bbox: Tuple[int, int, int, int],
        regions: List[Tuple[int, int, int, int]],
    ) -> str:
        if self._contained_region_count(bbox, regions) >= 2:
            return "composite_figure"

        bbox_area = self._bbox_area(bbox)
        for other in regions:
            if other == bbox or self._bbox_area(other) <= bbox_area:
                continue
            if self._is_hierarchical_overlap(other, bbox):
                return "sub_figure"

        return ""

    def _deduplicate_regions(
        self,
        regions: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        candidates = [
            self._expand_bbox(raw_bbox, page_width, page_height)
            for raw_bbox in regions
        ]
        candidates = [
            bbox for bbox in candidates
            if self._bbox_area(bbox) >= 1200
        ]
        candidates.sort(key=lambda bbox: self._bbox_area(bbox), reverse=True)

        kept: List[Tuple[int, int, int, int]] = []
        for bbox in candidates:
            bbox_area = self._bbox_area(bbox)
            should_drop = False
            for kept_bbox in kept:
                kept_area = self._bbox_area(kept_bbox)
                smaller_area = max(1, min(bbox_area, kept_area))
                contained_ratio = self._intersection_area(
                    bbox, kept_bbox) / smaller_area
                if self._iou(bbox, kept_bbox) > 0.45 or contained_ratio > 0.80:
                    should_drop = True
                    break

            if not should_drop:
                kept.append(bbox)

        return kept

    def _region_gap(
        self,
        left: Tuple[int, int, int, int],
        right: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        x_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]))
        y_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]))
        return x_gap, y_gap

    def _group_composite_figures(
        self,
        regions: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
        margin_ratio: float = 0.22,
        text_lines: Optional[List[Dict[str, object]]] = None,
        exclude_regions: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Add synthetic parent boxes for nearby sub-figures while keeping children.

        When text_lines is supplied, the composite parent is also stretched
        downward to cover its Vietnamese figure caption ("Hình X.Y. ...") and
        trimmed at the top to drop any question prompt that happens to sit
        above the figure cluster.

        ``exclude_regions`` lists regions that must NOT participate in component
        connectivity (e.g. OCR-anchored info boxes). They are already a parent
        themselves, so pulling them into a composite would create a union that
        spans the whole page and gets filtered out.
        """
        if len(regions) < 2:
            return regions

        exclude_set = {tuple(bbox) for bbox in (exclude_regions or [])}
        groupable_indexes = [
            index for index, bbox in enumerate(regions)
            if tuple(bbox) not in exclude_set
        ]

        margin_x = max(24, int(page_width * margin_ratio))
        margin_y = max(24, int(page_width * margin_ratio))
        visited = set()
        components: List[List[Tuple[int, int, int, int]]] = []

        for start_index in groupable_indexes:
            if start_index in visited:
                continue

            visited.add(start_index)
            component_indexes = [start_index]
            stack = [start_index]
            while stack:
                current_index = stack.pop()
                current_bbox = regions[current_index]
                for candidate_index in groupable_indexes:
                    if candidate_index in visited:
                        continue
                    candidate_bbox = regions[candidate_index]
                    x_gap, y_gap = self._region_gap(
                        current_bbox, candidate_bbox)
                    if x_gap <= margin_x and y_gap <= margin_y:
                        visited.add(candidate_index)
                        component_indexes.append(candidate_index)
                        stack.append(candidate_index)

            if len(component_indexes) >= 2:
                components.append([regions[index]
                                  for index in component_indexes])

        composite_regions: List[Tuple[int, int, int, int]] = []
        page_area = page_width * page_height
        for component in components:
            union_bbox = (
                min(bbox[0] for bbox in component),
                min(bbox[1] for bbox in component),
                max(bbox[2] for bbox in component),
                max(bbox[3] for bbox in component),
            )
            union_bbox = self._expand_bbox(
                union_bbox, page_width, page_height, ratio=0.025)

            if text_lines:
                union_bbox = self._trim_region_top_to_exclude_prompt(
                    union_bbox, text_lines, page_height)
                union_bbox = self._expand_region_to_caption(
                    union_bbox,
                    text_lines,
                    page_width,
                    page_height,
                    is_composite=True,
                )

            union_area = self._bbox_area(union_bbox)
            if union_area > page_area * 0.75:
                continue

            largest_child_area = max(self._bbox_area(bbox)
                                     for bbox in component)
            if union_area < largest_child_area * 1.20:
                continue

            duplicate_parent = False
            for existing in regions + composite_regions:
                existing_area = self._bbox_area(existing)
                overlap_of_union = self._intersection_area(
                    union_bbox, existing) / max(1, union_area)
                if (
                    self._iou(union_bbox, existing) > 0.90
                    or (existing_area >= union_area * 0.90 and overlap_of_union > 0.95)
                ):
                    duplicate_parent = True
                    break

            if not duplicate_parent:
                composite_regions.append(union_bbox)

        grouped = composite_regions + regions
        grouped.sort(key=lambda bbox: self._bbox_area(bbox), reverse=True)
        return grouped

    def _suppress_container_regions(
        self,
        regions: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        if len(regions) <= 1:
            return regions

        kept: List[Tuple[int, int, int, int]] = []
        for bbox in regions:
            area = self._bbox_area(bbox)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            overlap_children = 0

            for other in regions:
                if other == bbox:
                    continue
                overlap_ratio = self._iou(bbox, other)
                if overlap_ratio >= 0.18 and self._bbox_area(other) < area * 0.7:
                    overlap_children += 1

            is_wide_container = (
                width / max(1, page_width)) > 0.88 and (height / max(1, page_height)) < 0.26
            is_page_chrome = is_wide_container and (
                bbox[1] < page_height * 0.18 or bbox[3] > page_height * 0.92)
            if is_page_chrome and overlap_children >= 2:
                continue

            # Cyan false-positive: one large box swallowing several sub-figure cells.
            nested_children = 0
            for other in regions:
                if other == bbox:
                    continue
                if self._bbox_area(other) >= area * 0.45:
                    continue
                if self._coverage_ratio(other, bbox) >= 0.82:
                    nested_children += 1
            if nested_children >= 3:
                continue

            kept.append(bbox)

        return kept

    def _coverage_ratio(
        self,
        inner: Tuple[int, int, int, int],
        outer: Tuple[int, int, int, int],
    ) -> float:
        """Share of inner bbox area that lies inside outer."""
        inner_area = float(self._bbox_area(inner))
        if inner_area <= 0:
            return 0.0
        x0 = max(inner[0], outer[0])
        y0 = max(inner[1], outer[1])
        x1 = min(inner[2], outer[2])
        y1 = min(inner[3], outer[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return ((x1 - x0) * (y1 - y0)) / inner_area

    def _limit_regions_for_extraction(
        self,
        regions: List[Tuple[int, int, int, int]],
        max_regions: int = 24,
    ) -> List[Tuple[int, int, int, int]]:
        if len(regions) <= max_regions:
            return [(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in regions]

        ranked = sorted(
            [(int(x0), int(y0), int(x1), int(y1))
             for x0, y0, x1, y1 in regions],
            key=lambda bbox: self._bbox_area(bbox),
            reverse=True,
        )
        selected: List[Tuple[int, int, int, int]] = []
        for bbox in ranked:
            if any(
                self._iou(bbox, kept) > 0.58
                and not self._is_hierarchical_overlap(bbox, kept)
                for kept in selected
            ):
                continue
            selected.append(bbox)
            if len(selected) >= max_regions:
                break

        if len(selected) < max_regions:
            for bbox in ranked:
                if bbox in selected:
                    continue
                selected.append(bbox)
                if len(selected) >= max_regions:
                    break
        return selected

    def _check_color_variance(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a region has sufficient color variance to be a real image."""
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)

        if x1 <= x0 or y1 <= y0:
            return False

        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])

        total_variance = h_std + s_std + v_std
        return total_variance > 15

    def _refine_regions(self, regions: List[Tuple[int, int, int, int]], image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Recall-first filtering: keep candidate image blocks, remove obvious noise."""
        page_height, page_width = image.shape[0], image.shape[1]
        page_area = page_width * page_height
        refined = []
        for bbox in regions:
            x0, y0, x1, y1 = bbox
            width = x1 - x0
            height = y1 - y0
            area = width * height

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.12 or aspect_ratio > 12:
                continue

            if width < 45 or height < 45:
                continue

            # Reject very tiny icons and over-large whole-page text blocks.
            if area < 1200:
                continue
            if area > page_area * 0.75:
                continue

            edge_margin = int(min(page_width, page_height) * 0.012)
            if x0 <= edge_margin and y0 <= edge_margin and width < 170 and height < 170:
                continue

            refined.append(bbox)

        refined.sort(key=lambda item: self._bbox_area(item), reverse=True)
        deduped = self._deduplicate_regions(refined, page_width, page_height)
        return self._suppress_container_regions(deduped, page_width, page_height)

    def _visual_content_score(self, image: Image.Image) -> float:
        """Estimate how much non-background colored visual content a crop contains."""
        img_array = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        colored_pixels = (saturation > 45) & (value > 40) & (value < 252)
        return float(np.mean(colored_pixels))

    def _extract_page_image(self, pdf_path: str, page_num: int) -> Optional[Tuple[np.ndarray, Image.Image]]:
        """Render a PDF page as image for extraction."""
        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,
                poppler_path=POPPLER_PATH,
            )
            if images:
                img_array = np.array(images[0])
                return img_array, images[0]
        except Exception as e:
            logger.warning(f"Failed to render page {page_num}: {e}")
        return None

    def _get_context_text(self, pdf_path: str, page_num: int, bbox: Tuple[int, int, int, int], page_text: str) -> str:
        """Extract text within region around the image using OCR."""
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,
                poppler_path=POPPLER_PATH,
            )
            if not images:
                return page_text[:500] if page_text else ""

            img = images[0]
            x0, y0, x1, y1 = bbox

            h, w = img.height, img.width
            y0_pad = max(0, y0 - int(h * 0.05))
            y1_pad = min(h, y1 + int(h * 0.05))

            padded_img = img.crop((0, y0_pad, w, y1_pad))
            context = pytesseract.image_to_string(padded_img, lang="vie")
            return context.strip()[:500]
        except Exception:
            return page_text[:500] if page_text else ""

    def _ocr_crop_text(self, image: Image.Image) -> str:
        """OCR only the crop itself so text-heavy regions can be identified."""
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            text = pytesseract.image_to_string(image, lang="vie")
            return self._clean_text(text, max_chars=300)
        except Exception:
            return ""

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(
            char for char in text if unicodedata.category(char) != "Mn")
        return re.sub(r"[^a-z0-9\s]+", " ", text)

    def _clean_text(self, text: str, max_chars: int = 1200) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[:max_chars]

    def _extract_section_title(self, page_text: str) -> str:
        """Find a compact lesson or section title from page OCR/PDF text."""
        lines = [line.strip()
                 for line in (page_text or "").splitlines() if line.strip()]
        title_candidates = []
        for line in lines[:30]:
            normalized = self._normalize_text(line)
            if len(line) > 90 or len(line) < 4:
                continue
            if re.match(r"^(\d+[\.\)]|[ivx]+\.)\s+", normalized) or line.isupper():
                title_candidates.append(line)
            elif any(keyword in normalized for keyword in ("bai ", "chu de", "cac lop", "su da dang")):
                title_candidates.append(line)

        return title_candidates[-1] if title_candidates else ""

    def _extract_lesson_title(self, page_text: str) -> str:
        """Find a coarse lesson title for page-level metadata."""
        lines = [line.strip()
                 for line in (page_text or "").splitlines() if line.strip()]
        for index, line in enumerate(lines[:35]):
            normalized = self._normalize_text(line)
            if re.search(r"\bbai\s+\d+", normalized):
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                combined = f"{line} {next_line}".strip()
                return self._clean_text(combined, max_chars=140)
        return ""

    def _extract_figure_label(self, context_text: str, page_text: str) -> str:
        """Extract labels that are meaningful in Vietnamese textbooks."""
        text = f"{context_text}\n{page_text}"
        patterns = [
            r"Em\s+c[oó]\s+bi[eế]t",
            r"T[iì]m\s+hi[eể]u\s+th[eê]m",
            r"Quan\s+s[aá]t",
            r"B[aả]ng\s+\d+(?:\.\d+)?",
            r"H[iì]nh\s+\d+(?:\.\d+)?",
            r"Th[uư]c\s+h[aà]nh",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return ""

    def _extract_figure_caption(self, context_text: str, page_text: str) -> str:
        """Extract a compact figure/table caption instead of using the whole page context."""
        text = f"{context_text}\n{page_text}"
        patterns = [
            r"(H[iì]nh\s+\d+(?:\.\d+)?[\.:]?\s*[^\n]{0,180})",
            r"(B[aả]ng\s+\d+(?:\.\d+)?[\.:]?\s*[^\n]{0,180})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1), max_chars=220)
        return ""

    def _infer_image_type(
        self,
        figure_label: str,
        context_text: str,
        crop_text: str = "",
        hierarchy_type: str = "",
    ) -> str:
        if hierarchy_type:
            return hierarchy_type

        normalized = self._normalize_text(f"{figure_label} {context_text}")
        crop_normalized = self._normalize_text(crop_text)
        crop_tokens = [token for token in crop_normalized.split()
                       if len(token) > 1]
        if len(crop_tokens) >= 2 and len(crop_tokens) <= 12 and not figure_label:
            return "text_crop"
        if "em co biet" in normalized:
            return "textbook_info_box"
        if "tim hieu them" in normalized or "quan sat" in normalized or "thuc hanh" in normalized:
            return "activity_box"
        if "bang" in normalized:
            return "table"
        if "hinh" in normalized:
            return "figure"
        return "image_region"

    def _is_text_dominant_crop(
        self,
        crop: Image.Image,
        crop_text: str,
    ) -> bool:
        """Reject standalone headings/labels that detector can mistake for figures."""
        normalized = self._normalize_text(crop_text)
        tokens = [token for token in normalized.split() if len(token) > 1]
        if len(tokens) <= 15:
            return False

        aspect_ratio = crop.width / crop.height if crop.height else 0
        crop_area = crop.width * crop.height
        image_keyword_pattern = (
            r"[\d=+\-*/^√≤≥<>]|"
            r"\b([a-d]\)|bang|hinh|anh|cong thuc|bieu do|so do|"
            r"thi nghiem|quan sat|mo hinh|vat mau|mau vat|"
            r"coc|nhiet ke|day|tuong|nhom|nen|the ran|the long)\b"
        )
        if re.search(image_keyword_pattern, normalized):
            return False

        visual_score = self._visual_content_score(crop)
        foreground_score = self._foreground_content_score(crop)
        has_visual_contrast = visual_score >= 0.035 or foreground_score >= 0.10
        if crop_area > 15000:
            return not has_visual_contrast and aspect_ratio < 9

        weak_visual = visual_score < 0.025 and foreground_score < 0.065
        return weak_visual and aspect_ratio < 9

    def _foreground_content_score(self, image: Image.Image) -> float:
        """Estimate non-white/non-background content so low-saturation objects survive."""
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        foreground = gray < 238
        return float(np.mean(foreground))

    def _extract_keywords(self, *texts: str, limit: int = 18) -> str:
        """Extract lightweight Vietnamese keyword metadata without calling another model."""
        normalized = self._normalize_text(" ".join(texts))
        tokens = [token for token in normalized.split() if len(
            token) > 1 and token not in VIETNAMESE_STOPWORDS]

        scores: Dict[str, int] = {}
        for token in tokens:
            scores[token] = scores.get(token, 0) + 1

        for left, right in zip(tokens, tokens[1:]):
            phrase = f"{left} {right}"
            scores[phrase] = scores.get(phrase, 0) + 2

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(keyword for keyword, _ in ranked[:limit])

    def _save_page_snapshot(self, output_dir: Path, page_num: int, page_image: Image.Image) -> str:
        """Save a deterministic page snapshot for fallback/debug metadata."""
        snapshot_dir = output_dir / "pages"
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = snapshot_dir / f"page_{page_num}_snapshot.png"
        if not snapshot_path.exists():
            page_image.save(snapshot_path, format="PNG")
        return str(snapshot_path)

    def _compute_image_hash(self, image_data: bytes) -> str:
        """Compute MD5 hash for deduplication."""
        return hashlib.md5(image_data).hexdigest()

    def _resolve_image_path(self, output_dir: Path, page_num: int, img_index: int, img_hash: str) -> Path:
        """Return a deterministic image path and avoid duplicate files on reprocessing."""
        filepath = output_dir / f"page_{page_num}_img_{img_index}.png"
        if not filepath.exists():
            return filepath

        try:
            existing_hash = self._compute_image_hash(filepath.read_bytes())
            if existing_hash == img_hash:
                return filepath
        except Exception as e:
            logger.debug(f"Could not hash existing image {filepath}: {e}")

        hash_path = output_dir / \
            f"page_{page_num}_img_{img_index}_{img_hash[:8]}.png"
        if not hash_path.exists():
            return hash_path

        try:
            existing_hash = self._compute_image_hash(hash_path.read_bytes())
            if existing_hash == img_hash:
                return hash_path
        except Exception as e:
            logger.debug(f"Could not hash existing image {hash_path}: {e}")

        attempt = 1
        while True:
            candidate = output_dir / \
                f"page_{page_num}_img_{img_index}_{img_hash[:8]}_{attempt}.png"
            if not candidate.exists():
                return candidate
            attempt += 1

    def _build_image_id(
        self,
        pdf_hash: str,
        page_num: int,
        bbox: Tuple[int, int, int, int],
        image_hash: str,
    ) -> str:
        payload = f"{pdf_hash}:{page_num}:{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}:{image_hash}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _append_review_manifest(self, metadata: Dict[str, object]) -> None:
        try:
            self.review_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(metadata)
            record["manifest_written_at"] = datetime.now().isoformat()
            with self.review_manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not append image manifest: {e}")

    def _build_image_search_text(self, metadata: Dict[str, object]) -> str:
        """Build Vietnamese-first text used to retrieve this image later."""
        caption_text = metadata.get("caption_vi_manual", "") or metadata.get(
            "caption_vi", "") or metadata.get("caption", "")
        keywords_text = metadata.get(
            "keywords_vi_manual", "") or metadata.get("keywords_vi", "")
        parts = [
            metadata.get("figure_label", ""),
            metadata.get("figure_caption", ""),
            metadata.get("section_title", ""),
            metadata.get("image_type", ""),
            keywords_text,
            caption_text,
            metadata.get("context_text", ""),
            f"Trang {metadata.get('page_number', '')}",
            metadata.get("pdf_filename", ""),
        ]
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def extract_images_from_pdf(
        self,
        pdf_path: str,
        pdf_hash: str,
        pdf_filename: str,
        ocr_text_per_page: Dict[int, str],
        force: bool = False,
    ) -> List[Document]:
        """
        Full image ETL pipeline for a single PDF.

        Phase 1: OWL-ViT open-vocabulary object detection for region discovery
        Phase 2: Region refinement, dedupe, and caps
        Phase 3: Vietnamese OCR/context metadata + storage
        """
        extracted_docs = []
        output_dir = IMAGES_DIR / Path(pdf_filename).stem
        os.makedirs(output_dir, exist_ok=True)

        import fitz

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        for page_num in tqdm(range(1, total_pages + 1), desc=f"[{pdf_filename}] Extracting images"):
            if not force and not self.status_tracker.needs_image_processing_versioned(
                pdf_hash,
                page_num,
                required_version=self.image_extraction_version,
            ):
                logger.debug(f"Page {page_num}: already processed, skipping")
                continue

            page_result = self._extract_page_image(pdf_path, page_num - 1)
            if not page_result:
                continue

            img_array, pil_img = page_result
            page_text = ocr_text_per_page.get(page_num, "")
            nearby_text = self._clean_text(page_text, max_chars=1600)
            lesson_title = self._extract_lesson_title(page_text)
            section_title = self._extract_section_title(page_text)
            page_snapshot_path = self._save_page_snapshot(
                output_dir, page_num, pil_img)

            # Page-level OCR (image_to_data) drives the anchor-first detector.
            page_text_lines = self._collect_page_text_lines(pil_img)
            logger.debug(
                f"Page {page_num}: collected {len(page_text_lines)} OCR text lines")

            # v7 anchor-first detection. Yields concrete (bbox, image_type)
            # tuples — no further refinement is required.
            logger.info(
                f"[v7][page={page_num}] Anchor-first detection (Hình/Bảng/info-box/dashed)"
            )
            detection = self.detect_regions_anchor_first(
                pil_img, img_array, text_lines=page_text_lines)
            # type: ignore[assignment]
            detected_regions: List[Dict[str, object]] = detection["regions"]

            logger.info(
                f"[v7][page={page_num}] anchors: "
                # type: ignore[index]
                f"{len(detection['anchors']['figure_captions'])} Hình caption, "
                # type: ignore[index]
                f"{len(detection['anchors']['table_captions'])} Bảng (rejected), "
                # type: ignore[index]
                f"{len(detection['anchors']['info_titles'])} info-titles, "
                # type: ignore[index]
                f"{len(detection['anchors']['tool_group_labels'])} tool-labels"
                f" → {len(detected_regions)} final regions"
            )

            # Build a lookup so the metadata loop knows the panel/figure label
            # without having to re-run the panel classifier.
            panel_lookup: Dict[Tuple[int, int, int, int], str] = {}
            for region in detected_regions:
                region_bbox = tuple(region["bbox"])  # type: ignore[arg-type]
                region_type = str(region["image_type"])
                panel_lookup[region_bbox] = region_type

            img_index = 0
            page_seen_hashes = set()
            for region in detected_regions:
                bbox = tuple(region["bbox"])  # type: ignore[arg-type]
                x0, y0, x1, y1 = bbox
                crop = pil_img.crop((x0, y0, x1, y1))

                if crop.width < 50 or crop.height < 50:
                    continue

                panel_label = panel_lookup.get(bbox, "")

                # Phase 3: Vietnamese OCR/context metadata and storage.
                logger.info(
                    f"[Phase 3][page={page_num}][candidate={img_index}] Building OCR/context metadata"
                )
                crop_text = self._ocr_crop_text(crop)

                # Every v7 region is whitelisted by an anchor (Hình caption,
                # info-box title, or labelled dashed-frame). Text-dominance
                # filter is intentionally not applied here.

                img_bytes = io.BytesIO()
                crop.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                img_data = img_bytes.read()

                img_hash = self._compute_image_hash(img_data)
                if img_hash in page_seen_hashes:
                    logger.debug(
                        f"Page {page_num} img {img_index}: duplicate hash, skipping")
                    img_index += 1
                    continue
                page_seen_hashes.add(img_hash)

                image_id = self._build_image_id(
                    pdf_hash, page_num, bbox, img_hash)

                filepath = self._resolve_image_path(
                    output_dir, page_num, img_index, img_hash)

                with open(filepath, "wb") as f:
                    f.write(img_data)

                context_text = self._get_context_text(
                    pdf_path, page_num, bbox, page_text)
                context_text = self._clean_text(context_text, max_chars=1200)
                local_text = "\n".join(part for part in (
                    context_text, crop_text) if part)
                figure_label = self._extract_figure_label(local_text, "")
                figure_caption = self._extract_figure_caption(local_text, "")
                detected_bboxes = [tuple(r["bbox"]) for r in detected_regions]
                hierarchy_type = self._classify_region_hierarchy(
                    bbox, detected_bboxes)

                # v7: the anchor-first detector already labelled this region.
                # Trust its label; fall back to keyword inference only when
                # the detector returned the generic "panel".
                if panel_label and panel_label not in ("panel", ""):
                    image_type = panel_label
                else:
                    image_type = self._infer_image_type(
                        figure_label, context_text, crop_text, hierarchy_type)
                caption_context = {
                    "pdf_filename": pdf_filename,
                    "page_number": page_num,
                    "lesson_title": lesson_title,
                    "section_title": section_title,
                    "figure_label": figure_label,
                    "figure_caption": figure_caption,
                    "image_type": image_type,
                    "region_hierarchy": hierarchy_type or "standalone",
                    "context_text": context_text,
                    "crop_text": crop_text,
                    "nearby_text": nearby_text,
                }
                visual_metadata = self.captioner.caption(
                    crop, img_hash, context=caption_context)
                visual_caption = visual_metadata.get("visual_caption_vi", "")
                visual_keywords = visual_metadata.get("visual_keywords_vi", "")
                visual_objects = visual_metadata.get("visual_objects_vi", "")
                keywords_vi = self._extract_keywords(
                    lesson_title,
                    section_title,
                    figure_label,
                    figure_caption,
                    visual_caption,
                    visual_keywords,
                    visual_objects,
                    context_text,
                    crop_text,
                )
                caption_text = visual_caption or figure_caption or context_text[:240]

                metadata = {
                    "image_id": image_id,
                    "image_path": str(filepath),
                    "page_snapshot_path": page_snapshot_path,
                    "image_hash": img_hash,
                    "page_number": page_num,
                    "pdf_hash": pdf_hash,
                    "pdf_filename": pdf_filename,
                    "lesson_title": lesson_title,
                    "section_title": section_title,
                    "figure_label": figure_label,
                    "figure_caption": figure_caption,
                    **visual_metadata,
                    "image_type": image_type,
                    "region_hierarchy": hierarchy_type or "standalone",
                    "keywords_vi": keywords_vi,
                    "caption": caption_text,
                    "caption_vi": caption_text,
                    "caption_vi_manual": "",
                    "context_text": context_text,
                    "crop_text": crop_text,
                    "nearby_text": nearby_text,
                    "review_status": "pending",
                    "is_active": True,
                    "review_notes": "",
                    "reviewed_by": "",
                    "reviewed_at": "",
                    "keywords_vi_manual": "",
                    "final_caption_vi": caption_text,
                    "final_keywords_vi": keywords_vi,
                    "extraction_version": self.image_extraction_version,
                    "bbox": ",".join(str(value) for value in bbox),
                    "image_width": crop.width,
                    "image_height": crop.height,
                    "visual_content_score": round(self._visual_content_score(crop), 4),
                    "clip_positive_score": 0.0,
                    "clip_negative_score": 0.0,
                    "detector_model": OWL_VIT_MODEL,
                    "detector_threshold": OWL_VIT_CONFIDENCE_THRESHOLD,
                }
                search_text = self._build_image_search_text(metadata)
                metadata["search_text"] = search_text

                doc = Document(page_content=search_text, metadata=metadata)
                extracted_docs.append(doc)
                self._append_review_manifest(metadata)

                logger.debug(
                    f"Saved: {filepath} (label={figure_label or image_type}, keywords={keywords_vi[:80]})")
                img_index += 1

            self.status_tracker.mark_image_extracted(
                pdf_hash,
                page_num,
                pdf_filename,
                image_extraction_version=self.image_extraction_version,
            )

        logger.info(
            f"[{pdf_filename}] Extracted {len(extracted_docs)} images from {total_pages} pages")
        return extracted_docs


# ===========================================================================
# Publisher routing
# ===========================================================================

# Filename keywords that identify each publisher variant.
_VARIANT_CTST = "ctst"     # Chân Trời Sáng Tạo
_VARIANT_KNTT = "kntt"     # Kết Nối Tri Thức → shares layout conventions with CD for now
_VARIANT_CD = "cd"      # Cánh Diều


def get_pdf_variant(pdf_filename: str) -> str:
    """Return 'ctst', 'kntt', or 'cd' based on the PDF filename.

    Matches the keyword anywhere in the stem, handling both space-separated
    (``SGK KHTN 6 CTST.pdf``) and underscore-separated filenames.
    """
    stem = Path(pdf_filename).stem.lower().replace(" ", "_")
    if "_ctst" in stem:
        return _VARIANT_CTST
    if "_kntt" in stem:
        return _VARIANT_KNTT
    return _VARIANT_CD


def make_image_processor(
    pdf_filename: str = "",
    status_tracker=None,
) -> "ImageProcessor":
    """Factory that returns the right processor for a given PDF filename."""
    variant = get_pdf_variant(pdf_filename)
    if variant == _VARIANT_CTST:
        return CtsstImageProcessor(status_tracker=status_tracker)
    if variant == _VARIANT_KNTT:
        return KnttImageProcessor(status_tracker=status_tracker)
    return ImageProcessor(status_tracker=status_tracker)


# ---------------------------------------------------------------------------
# CTST — Chân Trời Sáng Tạo publisher
# ---------------------------------------------------------------------------

# CTST figures are captioned: "▲ Hình X.Y. description"
# Tesseract reads the filled-triangle ▲ inconsistently: as À/Á/A, as a dot,
# a pipe/bracket, TWO junk chars (".A Hình 8.1"), OR — when the marker is a
# RIGHT-pointing ▶ (page 59 "▶ Hình 11.8") — as "}>" / "{>" / "›". Allow a short
# leading run of any of those artefact characters before "Hình".
_CTST_FIG_PREFIX = r"[ÀÁAa▲▶∆.,'`|()<>{}»›\[\]\s]{0,4}"
_CTST_FIG_CAPTION_REGEX = re.compile(
    r"^" + _CTST_FIG_PREFIX + r"H[iì]nh\s+\d+(?:\.\d+)?\s*[\.:]?\s*\S",
    flags=re.IGNORECASE,
)

# Strip the leading triangle/OCR-artefact so the rest of the pipeline
# receives a clean "Hình X.Y. ..." text for metadata.
_CTST_TRIANGLE_PREFIX = re.compile(
    r"^" + _CTST_FIG_PREFIX + r"(?=H[iì]nh)",
    flags=re.IGNORECASE,
)

# CTST-specific info-box panel titles (different vocabulary from CD).
_CTST_INFO_BOX_TITLE_KEYS: List[Tuple[str, str]] = [
    # Exercise/practice boxes.
    ("bai tap",      "activity_box"),
    ("luyen tap",    "activity_box"),
    ("van dung",     "activity_box"),
    # Discovery / exploration boxes.
    ("kham pha",     "activity_box"),
    # NOTE: "tim hieu" intentionally NOT a key — in CTST "Tìm hiểu về …" is a
    # section header on white (body text), not a coloured info box. It used to
    # spawn text-only activity boxes (page 174). Real CTST boxes are coloured
    # and matched by the keys above.
    # Summary / overview.
    ("tong ket",     "activity_box"),
    ("on tap",       "activity_box"),
    # Lab / experiment.
    ("thuc hanh",    "activity_box"),
    ("thi nghiem",   "activity_box"),
    # "Em có biết" style appears occasionally.
    ("em co biet",   "textbook_info_box"),
    ("mo rong",      "activity_box"),
]


class CtsstImageProcessor(ImageProcessor):
    """Image ETL for SGK CTST (Chân Trời Sáng Tạo) publisher.

    CTST differs from CD in ways that need their OWN logic (kept here, not in
    the shared base, so CD stays untouched):

    1. Caption format ``▲ Hình X.Y`` — the filled triangle is OCR-mangled
       (`_classify_text_anchors` override + `_CTST_FIG_CAPTION_REGEX`).
    2. Caption is **left-aligned under a figure that often spans the full
       content width**. CD's centred-caption assignment clips the right half of
       such a figure (page 64 biogas, page 135 food-chain), so CTST gets its
       own band-based builder ``_build_figure_composites`` below.
    3. Sub-figure splitting (per-photo crops) IS wanted for CTST grids
       (food-chain circles, stopwatch a/b) — enabled here.
    """

    # Override: accept CTST caption format.
    _INFO_BOX_TITLE_KEYS: List[Tuple[str, str]] = _CTST_INFO_BOX_TITLE_KEYS

    # ── CTST-specific tuning (kept separate from CD/KNTT) ──
    # CTST info/activity boxes are always coloured panels; a bare title on
    # white (section header / experiment prose) must NOT become an image.
    _INFO_REQUIRE_VISUAL: bool = True
    _INFO_MIN_VIS: float = 0.045
    # CTST captions are '▲ Hình X.Y' and the triangle frequently corrupts the
    # page-level OCR of the whole line; recovering the caption by re-OCRing the
    # strip below each detected photo is essential here.
    _RECOVER_CAPTIONS_BELOW_PHOTOS: bool = True
    # CTST figures are captioned grids (food-chain circles "Cỏ / Châu chấu / …",
    # stopwatch a)/b)) — split them into per-photo sub-figures.
    _SPLIT_SUBFIGURES_BY_TITLE: bool = True
    # CTST titled figures are single-row photo strips; its multi-component
    # diagrams (biogas flow chart, page 64) carry internal labels at many
    # heights — cap title-split to one row so those diagrams stay whole.
    _SUBFIG_TITLE_MAX_ROWS: int = 1
    # CTST has pale building/landscape photos OWL misses entirely (page 59
    # "Hình 11.9") — enable the texture-based photo fallback.
    _DETECT_TEXTURED_PHOTOS: bool = True
    # A CTST figure sits in the vertical BAND above its caption; cap that band
    # so the first caption on a page can't reach decorative header art.
    _CTST_FIG_MAX_BAND_FRAC: float = 0.55

    def _build_figure_composites(
        self,
        figure_caps: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        question_prompts: List[Dict[str, object]],
        info_titles: List[Dict[str, object]],
        sub_labels: List[Dict[str, object]],
        exclusion_zones: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """CTST figure builder — band-based, full-width (separate from CD).

        CTST's ``▲ Hình X.Y`` caption is left-aligned at the BOTTOM of a figure
        that frequently spans the full content width. CD's centred-caption
        assignment then clips the right half (page 64, 135). Here each caption
        instead claims EVERY visual cell in the vertical band between it and the
        previous caption / info title above it, and the figure bbox is the union
        of those cells — so a full-width figure is captured whole, while a
        half-width figure still bounds tightly to its own cells.

        Question prompts are deliberately NOT used as the band ceiling: a CTST
        right-column prompt frequently sits at the same height as a left-column
        figure and would wrongly truncate the band (page 135 Hình 29.3).
        """
        outputs: List[Dict[str, object]] = []
        if not figure_caps:
            return outputs

        caps = sorted(figure_caps, key=lambda c: int(c["bbox"][1]))  # type: ignore[index]
        # Ceilings: bottoms of OTHER captions / info titles / exclusion zones
        # above a caption. (Prompts excluded on purpose — see docstring.)
        ceiling_bottoms = (
            [int(c["bbox"][3]) for c in figure_caps]  # type: ignore[index]
            + [int(t["bbox"][3]) for t in info_titles]  # type: ignore[index]
            + [int(z[3]) for z in exclusion_zones]
        )
        max_band = int(page_height * self._CTST_FIG_MAX_BAND_FRAC)

        # Each caption's vertical band [band_top, caption_top] and x-centre.
        band_info: List[Dict[str, float]] = []
        for cap in caps:
            cx0, cy0, cx1, cy1 = (int(v) for v in cap["bbox"])  # type: ignore[misc]
            ceiling = 0
            for bottom in ceiling_bottoms:
                if bottom < cy0 - 2:
                    ceiling = max(ceiling, bottom)
            band_info.append({
                "top": float(max(ceiling, cy0 - max_band)),
                "cap_top": float(cy0), "ccx": (cx0 + cx1) / 2.0})

        # Assign each visual cell to the NEAREST caption (by x-centre) among the
        # captions whose band contains it. Side-by-side figures sharing a y-band
        # (page 58 Hình 11.4 ship + 11.5 gears) thus split by column, while a
        # lone caption over a full-width figure (food chain) claims every cell.
        assigned_by_cap: List[List[Tuple[int, int, int, int]]] = [
            [] for _ in caps]
        for raw in visual_regions:
            vx0, vy0, vx1, vy1 = (int(v) for v in raw)
            vcy = (vy0 + vy1) / 2.0
            vcx = (vx0 + vx1) / 2.0
            if any(self._coverage_ratio((vx0, vy0, vx1, vy1), zone) > 0.45
                   for zone in exclusion_zones):
                continue
            candidates = [i for i, b in enumerate(band_info)
                          if b["top"] <= vcy <= b["cap_top"]]
            if not candidates:
                continue
            best = min(candidates,
                       key=lambda i: abs(vcx - band_info[i]["ccx"]))
            assigned_by_cap[best].append((vx0, vy0, vx1, vy1))

        for cap_index, cap in enumerate(caps):
            cx0, cy0, cx1, cy1 = (int(v) for v in cap["bbox"])  # type: ignore[misc]
            assigned = assigned_by_cap[cap_index]
            if not assigned:
                continue

            x0 = min([cx0] + [r[0] for r in assigned])
            y0 = min(r[1] for r in assigned)
            x1 = max([cx1] + [r[2] for r in assigned])
            pad_x = int(page_width * 0.008)
            pad_y = int(page_height * 0.006)
            bbox = (
                max(0, x0 - pad_x), max(0, y0 - pad_y),
                min(page_width, x1 + pad_x), min(page_height, cy1 + pad_y),
            )
            # A prompt that slipped into the figure top is trimmed off.
            bbox = self._trim_region_top_to_exclude_prompt(
                bbox, text_lines, page_height)

            sub_inside = sum(
                1 for s in sub_labels
                if bbox[0] - 5 <= int(s["bbox"][0])  # type: ignore[index]
                and int(s["bbox"][2]) <= bbox[2] + 5  # type: ignore[index]
                and bbox[1] - 5 <= int(s["bbox"][1]) <= bbox[3] + 12)  # type: ignore[index]
            label = ("composite_figure"
                     if sub_inside >= 1 and len(assigned) >= 2
                     else "single_figure")
            outputs.append({
                "bbox": bbox,
                "image_type": label,
                "caption_text": cap["text"],
                "caption_bbox": cap["bbox"],
                "assigned_regions": assigned,
            })
        return outputs

    def detect_regions_anchor_first(
        self,
        pil_img: Image.Image,
        img_array: np.ndarray,
        text_lines: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """CTST override: recover ``▲ Hình`` captions the page OCR missed.

        The filled-triangle prefix makes Tesseract drop whole caption lines on
        some pages (page 29: only 'Hình 6.4' is read, so 'Hình 6.2/6.3' vanish
        and the three figures collapse into one). Before building, re-OCR the
        strip below each detected cell and inject any recovered ``Hình X.Y`` as
        a synthetic caption line, so the band builder sees all three captions.
        """
        if text_lines is None:
            text_lines = self._collect_page_text_lines(pil_img)
        text_lines = self._inject_recovered_captions(
            pil_img, img_array, text_lines)
        return super().detect_regions_anchor_first(
            pil_img, img_array, text_lines=text_lines)

    def _inject_recovered_captions(
        self,
        pil_img: Image.Image,
        img_array: np.ndarray,
        text_lines: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        page_width, page_height = pil_img.size

        def fig_number(text: str) -> str:
            match = re.search(r"H[iì]nh\s+(\d+(?:\.\d+)?)", text,
                              flags=re.IGNORECASE)
            return match.group(1) if match else ""

        # Figure numbers the page OCR already produced — never re-inject one,
        # so a caption read by both passes can't spawn a duplicate figure
        # (page 10 "Hình 2.10").
        seen_numbers = {fig_number(str(ln["text"]))
                        for ln in text_lines
                        if _CTST_FIG_CAPTION_REGEX.match(str(ln["text"]).strip())}
        seen_numbers.discard("")
        # Cheap CV cell candidates (no OWL): framed panels + object blobs +
        # textured pale photos (so a building photo whose caption the page OCR
        # dropped — page 59 "Hình 11.9" — gets its strip re-OCR'd).
        cells = self._dedupe_visual_regions(
            list(self._detect_framed_regions(img_array))
            + list(self._detect_object_blobs(img_array))
            + list(self._detect_textured_photo_regions(img_array)),
            min_area=int(page_width * page_height * 0.004),
            iou_threshold=0.5,
        )
        recovered: List[Dict[str, object]] = []
        for cell in cells:
            result = self._reocr_caption_below(
                pil_img, cell, page_width, page_height)
            if not result:
                continue
            caption_text, strip_bbox = result
            number = fig_number(caption_text)
            if not number or number in seen_numbers:
                continue
            seen_numbers.add(number)
            recovered.append({"text": caption_text, "bbox": strip_bbox})
        if recovered:
            text_lines = list(text_lines) + recovered
            text_lines.sort(key=lambda ln: int(ln["bbox"][1]))
        return text_lines

    def _classify_text_anchors(
        self,
        text_lines: List[Dict[str, object]],
    ) -> Dict[str, List[Dict[str, object]]]:
        """Same as parent but treats '▲/À/Á Hình X.Y' as a figure caption."""
        figure_caps: List[Dict[str, object]] = []
        table_caps: List[Dict[str, object]] = []
        info_titles: List[Dict[str, object]] = []
        sub_labels: List[Dict[str, object]] = []
        question_prompts: List[Dict[str, object]] = []
        tool_labels: List[Dict[str, object]] = []

        for index, line in enumerate(text_lines):
            text = str(line["text"]).strip()
            bbox = tuple(int(v) for v in line["bbox"])  # type: ignore[misc]

            # CTST figure caption (with or without triangle prefix).
            if _CTST_FIG_CAPTION_REGEX.match(text):
                clean = _CTST_TRIANGLE_PREFIX.sub("", text)
                # Keep the clean text in the entry so metadata shows "Hình…"
                entry: Dict[str, object] = {
                    "index": index,
                    "text": clean,
                    "bbox": bbox,
                }
                split_entries = self._split_merged_figure_caption(entry)
                # Table caption check on clean text.
                for se in split_entries:
                    if TABLE_CAPTION_STRICT_REGEX.match(str(se["text"])):
                        table_caps.append(se)
                    else:
                        figure_caps.append(se)
                continue

            if TABLE_CAPTION_STRICT_REGEX.match(text):
                table_caps.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

            info_label = self._match_info_box_title(text)
            if info_label:
                info_titles.append({
                    "index": index, "text": text,
                    "bbox": bbox, "label": info_label})
                continue

            if SUB_FIGURE_LABEL_REGEX.match(text):
                sub_labels.append({
                    "index": index, "text": text, "bbox": bbox})

            if self._is_question_prompt_text(text):
                question_prompts.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

            if TOOL_GROUP_LABEL_REGEX.match(text):
                tool_labels.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

        return {
            "figure_captions":   figure_caps,
            "table_captions":    table_caps,
            "info_titles":       info_titles,
            "sub_labels":        sub_labels,
            "question_prompts":  question_prompts,
            "tool_group_labels": tool_labels,
        }


# ---------------------------------------------------------------------------
# KNTT — Kết Nối Tri Thức publisher
# ---------------------------------------------------------------------------

# KNTT figure captions are rendered inside a coloured pill / badge.
# Tesseract OCR may:
#   - miss the pill entirely → "Hình" absent from OCR text
#   - produce garbage chars before "Hình" (pipe, bracket, digit, etc.)
#   - wrap in parentheses: "(Hình 42.1a)"
#   - drop the SPACE and the DECIMAL POINT inside the tight pill, so
#     "Hình 19.3" reads as "Hình193" and "Hình 2.2" as "Hình22".
# The permissive regex below matches "Hình X.Y" *anywhere* in the line by
# scanning past an arbitrary prefix; `\s*` and `[.,]?` make the space and the
# dot optional so the mangled pill forms still match.
_KNTT_FIG_CAPTION_REGEX = re.compile(
    r".*?(?:H[iì]nh\s*\d+(?:[.,]\d+)?[a-h]?)",
    flags=re.IGNORECASE,
)

# A KNTT caption line STARTS with "Hình X.Y" (after at most a short pill-OCR
# junk prefix like "| ", "[ ", "("). This rejects body-text references where
# the marker is buried mid-sentence ("… (Hình 36.15).") — those used to spawn
# spurious duplicate figures (page 131).
_KNTT_FIG_START_REGEX = re.compile(
    r"^[\s(\[|>}.,'`]{0,4}H[iì]nh\s*\d+(?:[.,]\d+)?[a-h]?",
    flags=re.IGNORECASE,
)

# A real KNTT caption is "Hình X.Y <description noun-phrase>". A body sentence
# "(Hình 42.1a) thì lò xo dãn ra" continues with a function word — if the word
# right after the number is one of these, the line is a reference, not a caption.
_KNTT_REF_FUNCTION_WORDS = {
    "thi", "la", "va", "khi", "co", "duoc", "cho", "nen", "hoac", "cua",
    "trong", "thuong", "se", "da", "nay", "do", "ta",
}

# Strip everything before (and including) the first "Hình" so the pipeline
# receives a clean "Hình X.Y. ..." text for metadata.  Also strips a
# leading open-parenthesis that Tesseract may add: "(Hình 42.1a)".
_KNTT_FIG_CLEAN_PREFIX = re.compile(
    r"^[^(]*?(?:\(\s*)?(?=H[iì]nh)",
    flags=re.IGNORECASE,
)

# Extract "Hình X.Y" plus clean description text.  After the figure number
# (and optional sub-figure letter), optionally skip a closing parenthesis
# and capture the description (no parens allowed in description).
# "(Hình 42.1a) thì lò xo dãn ra (" → "Hình 42.1a thì lò xo dãn ra"
# "(Hình 42.1b)." → "Hình 42.1b"
_KNTT_FIG_EXTRACT = re.compile(
    r"(H[iì]nh\s*\d+(?:[.,]\d+)?[a-h]?)[)]?"
    r"(?:\s+([^\s\(\)]\S*(?:\s+[^\s\(\)]\S*)*))?",
    flags=re.IGNORECASE,
)


def _normalise_kntt_caption(raw: str) -> str:
    """Clean a (possibly mangled) pill OCR into "Hình X.Y" best-effort.

    The tight pill drops the space and dot, so "Hình193" arrives without them.
    We re-insert the space, turn a comma into a dot, and — when the dot is
    missing but ≥2 digits remain — split off the last 1-2 digits as the figure
    number ("193"→"19.3", "22"→"2.2", "3615"→"36.15"). The exact split is a
    heuristic; its job is to anchor the figure, not to be authoritative.
    """
    match = re.search(r"H[iì]nh\s*([0-9]+(?:[.,][0-9]+)?)\s*([a-h]?)",
                      raw, flags=re.IGNORECASE)
    if not match:
        return ""
    number = match.group(1).replace(",", ".")
    letter = (match.group(2) or "").lower()
    if "." not in number and len(number) >= 2:
        if len(number) <= 3:
            number = number[:-1] + "." + number[-1]
        else:
            number = number[:2] + "." + number[2:]
    return f"Hình {number}{letter}".strip()


class KnttImageProcessor(ImageProcessor):
    """Image ETL for SGK KNTT (Kết Nối Tri Thức) publisher.

    The key difference from CD is the figure caption format.  KNTT renders
    "Hình X.Y" inside a coloured pill / badge with white text on an orange
    (or similar) background.  Tesseract OCR either misses the pill entirely,
    adds garbage prefix characters, or wraps the text in parentheses.

    This subclass:
    1. Pre-processes the page image to make pill text readable — detects
       coloured regions via HSV, thresholds white-on-colour text to
       black-on-white, and pastes the enhanced crops back onto the page.
    2. Uses a permissive regex that finds "Hình X.Y" anywhere in the OCR
       line and strips any prefix to produce clean caption text.
    """

    # ── KNTT-specific tuning (kept separate from CD/CTST) ──
    # KNTT pill captions frequently sit ABOVE the row of sub-figures they label
    # (e.g. page 109 "Hình 32.1" above the a/b/c/d mushroom row), so caption
    # recovery must also look below the caption.
    _FIG_CAPTION_ABOVE_OK: bool = True
    # KNTT sub-figure crops come out partial (the per-cell OWL boxes are
    # incomplete and miss the a)/b) titles — pages 69, 109), so they add noise
    # rather than value. Keep the (correct) composite whole and skip splitting;
    # re-enable once KNTT cell detection is good enough for clean sub-crops.
    _SPLIT_SUBFIGURES: bool = False
    _SPLIT_SUBFIGURES_BY_TITLE: bool = False
    # Hard cap on how far a pill caption's figure band may reach (runaway
    # backstop). The real bound is the contiguity walk in the builder; a tall
    # connected diagram (page 80) can legitimately fill most of the page.
    _KNTT_FIG_MAX_BAND_FRAC: float = 0.85
    # Recover full photo rectangles OWL only partially detects — a dark photo
    # on black (page 131 bat), warning triangles (page 13), the diagram cells
    # under a title (page 80).
    _DETECT_PHOTO_RECTANGLES: bool = True

    def _detect_pill_figure_captions(
        self,
        pil_img: Image.Image,
    ) -> List[Dict[str, object]]:
        """Detect figure captions hidden inside coloured pills / badges.

        KNTT renders "Hình X.Y" as white text on an orange pill.  Standard
        page-level OCR misses these.  This method:
        1. Finds orange pill regions via HSV colour detection.
        2. Crops each pill, thresholds to isolate white text, and OCRs it.
        3. Returns anchor entries for any line matching "Hình \\d+".

        Returns entries shaped like ``{"index": -1, "text", "bbox"}`` where
        bbox is the pill's bounding box on the full page.
        """
        try:
            import cv2
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        except Exception:
            return []

        rgb = np.array(pil_img.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        page_h, page_w = rgb.shape[:2]

        # KNTT pill colour VARIES, so detect several hue families (orange,
        # yellow, green, teal, blue, purple, pink/red). We do NOT use an
        # all-hue mask: that also catches the colourful PHOTOS and bridges
        # adjacent pills into one wide blob (page 95). The strict "must OCR to
        # Hình X.Y" gate rejects the remaining non-caption colour blobs.
        bands = (
            ((5, 70, 70), (40, 255, 255)),     # orange / yellow
            ((40, 60, 70), (90, 255, 255)),    # green / teal
            ((90, 60, 70), (135, 255, 255)),   # blue
            ((135, 50, 70), (165, 255, 255)),  # purple
            ((0, 80, 80), (5, 255, 255)),      # red
            ((165, 50, 120), (180, 200, 255)),  # pink
        )
        pill_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in bands:
            pill_mask = cv2.bitwise_or(
                pill_mask, cv2.inRange(hsv, lower, upper))

        # Connect glyphs into a solid blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(pill_mask, cv2.MORPH_CLOSE, kernel,
                                  iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        captions: List[Dict[str, object]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Pill-like geometry: wider than tall, reasonable size.
            if w < int(page_w * 0.03) or w > int(page_w * 0.4):
                continue
            if h < 12 or h > int(page_h * 0.05):
                continue
            if w < h * 1.2:
                continue

            # Extract and preprocess the pill crop for OCR.
            pad = 4
            cx0 = max(0, x - pad)
            cy0 = max(0, y - pad)
            cx1 = min(page_w, x + w + pad)
            cy1 = min(page_h, y + h + pad)
            crop = rgb[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue

            # White text on a coloured pill. Threshold to isolate the glyphs
            # AND upscale 4× — the pills are small (~30 px), and without the
            # upscale Tesseract drops the space and the decimal point
            # ("Hình 19.3" → "Hình193").
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            # Upscale the GRAYSCALE first (upscaling the binary adds ringing
            # that hurts OCR), then try a few thresholds for white-on-colour.
            up_gray = cv2.resize(gray, None, fx=3, fy=3,
                                 interpolation=cv2.INTER_CUBIC)
            text = ""
            for thresh in (190, 160, 130, 110):
                _, bw = cv2.threshold(up_gray, thresh, 255, cv2.THRESH_BINARY)
                try:
                    candidate = pytesseract.image_to_string(
                        bw, config="--psm 7 -l vie").strip()
                except Exception:
                    continue
                if re.search(r"H[iì]nh\s*\d", candidate, re.IGNORECASE):
                    text = candidate
                    break
            if not text:
                continue

            # Normalise the (often space/dot-stripped) pill text to "Hình X.Y".
            clean = _normalise_kntt_caption(text)
            if not clean:
                continue
            captions.append({
                "index": -1,
                "text": clean,
                "bbox": (cx0, cy0, cx1, cy1),
            })

        return captions

    def detect_regions_anchor_first(
        self,
        pil_img: Image.Image,
        img_array: np.ndarray,
        text_lines: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Override to inject pill captions into text_lines before OCR-based detection.

        KNTT renders "Hình X.Y" inside coloured pills that standard Tesseract
        misses.  This method detects those pills, OCRs them individually, and
        injects the results as synthetic text lines so the parent's anchor
        classification and region building pick them up naturally.
        """
        if text_lines is None:
            text_lines = self._collect_page_text_lines(pil_img)

        # Detect pill-based figure captions missed by standard OCR.
        pill_captions = self._detect_pill_figure_captions(pil_img)

        if pill_captions:
            # Convert pill detections to synthetic text_lines entries.
            # Use index=-1 so they sort to the top and are easy to identify.
            for pc in pill_captions:
                text_lines.append({
                    "text": pc["text"],
                    "bbox": pc["bbox"],
                    "conf": 99,
                    "line_num": -1,
                    "block_num": -1,
                })

        return super().detect_regions_anchor_first(
            pil_img, img_array, text_lines=text_lines)

    def _classify_text_anchors(
        self,
        text_lines: List[Dict[str, object]],
    ) -> Dict[str, List[Dict[str, object]]]:
        """Same as parent but uses permissive regex for KNTT pill captions.

        Even after preprocessing, some pill text may arrive with prefix
        artefacts (parentheses, digits).  The permissive regex strips
        everything before "Hình" to produce clean caption text. Captions are
        then deduped by figure number (`_dedupe_kntt_captions`).
        """
        figure_caps: List[Dict[str, object]] = []
        table_caps: List[Dict[str, object]] = []
        info_titles: List[Dict[str, object]] = []
        sub_labels: List[Dict[str, object]] = []
        question_prompts: List[Dict[str, object]] = []
        tool_labels: List[Dict[str, object]] = []

        for index, line in enumerate(text_lines):
            text = str(line["text"]).strip()
            bbox = tuple(int(v) for v in line["bbox"])  # type: ignore[misc]

            # KNTT figure caption — only when the line STARTS with the marker
            # (a pill or a true caption line), never an inline body reference.
            if _KNTT_FIG_START_REGEX.match(text):
                matches = list(_KNTT_FIG_EXTRACT.finditer(text))
                # A body sentence ("(Hình 42.1a) thì lò xo dãn ra (Hình 42.1b).")
                # is rejected WHOLE when any marker is followed by a function
                # word or is wrapped with a closing ")" — those are references,
                # not captions (page 152).
                body_ref = False
                for em in matches:
                    g2 = em.group(2)
                    if g2:
                        first = self._normalize_text(g2.strip().split(" ")[0])
                        if first in _KNTT_REF_FUNCTION_WORDS:
                            body_ref = True
                            break
                    end = em.end(1)
                    if end < len(text) and text[end:end + 1] == ")":
                        body_ref = True
                        break
                if body_ref:
                    continue
                for em in matches:
                    clean = em.group(1)
                    if em.group(2):
                        clean = clean + " " + em.group(2).strip()
                    entry: Dict[str, object] = {
                        "index": index,
                        "text": clean,
                        "bbox": bbox,
                    }
                    # Table caption check on clean text.
                    if TABLE_CAPTION_STRICT_REGEX.match(clean):
                        table_caps.append(entry)
                    else:
                        figure_caps.append(entry)
                continue

            if TABLE_CAPTION_STRICT_REGEX.match(text):
                table_caps.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

            info_label = self._match_info_box_title(text)
            if info_label:
                info_titles.append({
                    "index": index, "text": text,
                    "bbox": bbox, "label": info_label})
                continue

            if SUB_FIGURE_LABEL_REGEX.match(text):
                sub_labels.append({
                    "index": index, "text": text, "bbox": bbox})

            if self._is_question_prompt_text(text):
                question_prompts.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

            if TOOL_GROUP_LABEL_REGEX.match(text):
                tool_labels.append({
                    "index": index, "text": text, "bbox": bbox})
                continue

        # A pill caption + its page-OCR'd caption line (or repeated pill
        # contours) for the same figure number otherwise spawn duplicate
        # figures (page 131 "Hình 36.15"); keep the most complete caption.
        figure_caps = self._dedupe_kntt_captions(figure_caps)

        return {
            "figure_captions":   figure_caps,
            "table_captions":    table_caps,
            "info_titles":       info_titles,
            "sub_labels":        sub_labels,
            "question_prompts":  question_prompts,
            "tool_group_labels": tool_labels,
        }

    @staticmethod
    def _dedupe_kntt_captions(
        figure_caps: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Keep one caption per figure NUMBER (the longest / most complete)."""
        def number_key(text: str) -> str:
            match = re.search(r"(\d+(?:[.,]\d+)?)[a-h]?", str(text))
            return match.group(1).replace(",", ".") if match else str(text)

        best: Dict[str, Dict[str, object]] = {}
        for cap in figure_caps:
            key = number_key(str(cap["text"]))
            if key not in best or \
                    len(str(cap["text"])) > len(str(best[key]["text"])):
                best[key] = cap
        return sorted(best.values(),
                      key=lambda c: int(c["bbox"][1]))  # type: ignore[index]

    def _build_figure_composites(
        self,
        figure_caps: List[Dict[str, object]],
        text_lines: List[Dict[str, object]],
        visual_regions: List[Tuple[int, int, int, int]],
        question_prompts: List[Dict[str, object]],
        info_titles: List[Dict[str, object]],
        sub_labels: List[Dict[str, object]],
        exclusion_zones: List[Tuple[int, int, int, int]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, object]]:
        """KNTT figure builder — pill caption ABOVE or BELOW a multi-photo figure.

        A KNTT pill sits either just below its figure (most cases — page 131
        "Hình 36.15" bat, "Hình 36.16" sheep+bees) or just above it (page 109
        mushroom row). CD's centred-caption builder both clips wide figures and
        drops the far cells of a stacked figure, so KNTT gets its own:

          * Per caption, the figure SIDE (above/below) is whichever has the
            smaller adjacent visual-cell gap.
          * Each visual cell is assigned to the nearest caption ON ITS
            figure-side (vertical gap + a small x-penalty for side-by-side),
            and the figure bbox is the UNION of the caption pill and its cells —
            so a stacked multi-photo figure is captured whole.

        Kept entirely in KnttImageProcessor; CD/CTST builders are untouched.
        """
        outputs: List[Dict[str, object]] = []
        if not figure_caps:
            return outputs

        caps = sorted(
            figure_caps,
            key=lambda c: (int(c["bbox"][1]) + int(c["bbox"][3])) / 2.0)  # type: ignore[index]
        cells: List[Tuple[int, int, int, int]] = []
        for raw in visual_regions:
            vr = tuple(int(v) for v in raw)
            if any(self._coverage_ratio(vr, zone) > 0.45
                   for zone in exclusion_zones):
                continue
            cells.append(vr)
        if not cells:
            return outputs

        def x_overlaps(cell: Tuple[int, int, int, int],
                       ax0: int, ax1: int) -> bool:
            return min(cell[2], ax1) - max(cell[0], ax0) > 0

        # Decide each caption's figure side by the smaller adjacent cell gap.
        cap_info: List[Dict[str, object]] = []
        for cap in caps:
            cx0, cy0, cx1, cy1 = (int(v) for v in cap["bbox"])  # type: ignore[misc]
            above = [cy0 - c[3] for c in cells
                     if c[3] <= cy0 + 5 and x_overlaps(c, cx0, cx1)]
            below = [c[1] - cy1 for c in cells
                     if c[1] >= cy1 - 5 and x_overlaps(c, cx0, cx1)]
            if not above and not below:  # fall back to any cell (ignore x)
                above = [cy0 - c[3] for c in cells if c[3] <= cy0 + 5]
                below = [c[1] - cy1 for c in cells if c[1] >= cy1 - 5]
            gap_above = min(above) if above else 10 ** 9
            gap_below = min(below) if below else 10 ** 9
            cap_info.append({
                "bbox": (cx0, cy0, cx1, cy1),
                "ccx": (cx0 + cx1) / 2.0,
                "figure_above": gap_above <= gap_below,
                "cap": cap,
            })

        # Group captions into ROWS (same figure-side, similar y-centre). A row
        # of N captions over one wide cell (OWL merged side-by-side figures —
        # page 95 "Hình 27.3/27.4/27.5") is split into N columns at the caption
        # x-midpoints; a lone caption in a row keeps the full-width figure.
        def y_centre(info: Dict[str, object]) -> float:
            bb = info["bbox"]  # type: ignore[index]
            return (bb[1] + bb[3]) / 2.0

        row_tol = int(page_height * 0.05)
        order = sorted(range(len(cap_info)), key=lambda i: y_centre(cap_info[i]))
        rows: List[List[int]] = []
        for i in order:
            if rows:
                j = rows[-1][-1]
                if cap_info[i]["figure_above"] == cap_info[j]["figure_above"] \
                        and y_centre(cap_info[i]) - y_centre(cap_info[j]) <= row_tol:
                    rows[-1].append(i)
                    continue
            rows.append([i])

        pad_x = int(page_width * 0.008)
        pad_y = int(page_height * 0.006)
        for row in rows:
            side_above = bool(cap_info[row[0]]["figure_above"])
            row_top = min(int(cap_info[i]["bbox"][1]) for i in row)  # type: ignore[index]
            row_bottom = max(int(cap_info[i]["bbox"][3]) for i in row)  # type: ignore[index]
            row_yc = (row_top + row_bottom) / 2.0
            # Band bounded by the nearest caption in another row.
            ceiling, floor = 0, page_height
            for i, info in enumerate(cap_info):
                if i in row:
                    continue
                if y_centre(info) < row_yc:
                    ceiling = max(ceiling, int(info["bbox"][3]))  # type: ignore[index]
                else:
                    floor = min(floor, int(info["bbox"][1]))  # type: ignore[index]
            band0, band1 = (ceiling, row_top) if side_above else (row_bottom, floor)
            # Bound the band by CONTIGUITY, not a fixed fraction: walk through
            # the cells on the figure-side bridging gaps ≤ max_gap and stop at
            # the first big gap. A tall connected diagram is kept whole (page 80
            # Hình 23.1 body + surrounding circles), while a caption's band
            # stops before an unrelated illustration far away (page 109 mushroom
            # row vs the top-right photo). A hard fraction still caps runaway.
            max_gap = int(page_height * 0.25)
            hard = int(page_height * self._KNTT_FIG_MAX_BAND_FRAC)
            # A text-heavy cell is a title bar / objectives block / paragraph,
            # not part of the photo — stop the walk before bridging into it
            # (page 80 keeps the diagram but not the "MỤC TIÊU" header), while a
            # pure-image cell far below is still reached (page 131 sheep).
            def bridgeable(c: Tuple[int, int, int, int]) -> bool:
                return self._text_line_coverage(c, text_lines) <= 0.25
            # Membership is by cell CENTRE (a KNTT pill often sits ON the photo
            # bottom edge, so a strict "whole cell above the caption" test would
            # drop the photo — page 131 bat).
            if side_above:
                frontier = row_top
                reach = row_top
                for c in sorted((c for c in cells
                                 if (c[1] + c[3]) / 2.0 < row_top),
                                key=lambda c: c[3], reverse=True):
                    if frontier - c[3] > max_gap or not bridgeable(c):
                        break
                    frontier = min(frontier, c[1])
                    reach = min(reach, c[1])
                band0 = max(ceiling, reach, row_top - hard)
            else:
                frontier = row_bottom
                reach = row_bottom
                for c in sorted((c for c in cells
                                 if (c[1] + c[3]) / 2.0 > row_bottom),
                                key=lambda c: c[1]):
                    if c[1] - frontier > max_gap or not bridgeable(c):
                        break
                    frontier = max(frontier, c[3])
                    reach = max(reach, c[3])
                band1 = min(floor, reach, row_bottom + hard)
            band_cells = [c for c in cells
                          if band0 - 2 <= (c[1] + c[3]) / 2.0 <= band1 + 2]
            if not band_cells:
                continue
            union_x0 = min(c[0] for c in band_cells)
            union_x1 = max(c[2] for c in band_cells)

            # Column boundaries from the row's caption x-centres.
            row_sorted = sorted(row, key=lambda i: cap_info[i]["ccx"])  # type: ignore[index,return-value]
            bounds = [union_x0]
            for k in range(len(row_sorted) - 1):
                left = cap_info[row_sorted[k]]["ccx"]
                right = cap_info[row_sorted[k + 1]]["ccx"]
                bounds.append(int((left + right) / 2.0))  # type: ignore[operator]
            bounds.append(union_x1)

            for k, cap_idx in enumerate(row_sorted):
                col0, col1 = bounds[k], bounds[k + 1]
                pieces = []
                for c in band_cells:
                    ix0, ix1 = max(c[0], col0), min(c[2], col1)
                    if ix1 - ix0 > 0:
                        pieces.append((ix0, c[1], ix1, c[3]))
                if not pieces:
                    continue
                cbx0, cby0, cbx1, cby1 = cap_info[cap_idx]["bbox"]  # type: ignore[misc]
                x0 = min(p[0] for p in pieces)
                x1 = max(p[2] for p in pieces)
                # Include the caption pill's x only if it sits in this column.
                if col0 - 5 <= cap_info[cap_idx]["ccx"] <= col1 + 5:  # type: ignore[operator]
                    x0 = min(x0, cbx0)
                    x1 = max(x1, cbx1)
                y0 = min([cby0] + [p[1] for p in pieces])
                y1 = max([cby1] + [p[3] for p in pieces])
                bbox = (
                    max(0, x0 - pad_x), max(0, y0 - pad_y),
                    min(page_width, x1 + pad_x), min(page_height, y1 + pad_y),
                )
                sub_inside = sum(
                    1 for s in sub_labels
                    if bbox[0] - 5 <= int(s["bbox"][0])  # type: ignore[index]
                    and int(s["bbox"][2]) <= bbox[2] + 5  # type: ignore[index]
                    and bbox[1] - 5 <= int(s["bbox"][1]) <= bbox[3] + 12)  # type: ignore[index]
                label = ("composite_figure"
                         if sub_inside >= 1 and len(pieces) >= 2
                         else "single_figure")
                cap = cap_info[cap_idx]["cap"]
                outputs.append({
                    "bbox": bbox,
                    "image_type": label,
                    "caption_text": cap["text"],  # type: ignore[index]
                    "caption_bbox": cap["bbox"],  # type: ignore[index]
                    "assigned_regions": pieces,
                })
        return outputs
