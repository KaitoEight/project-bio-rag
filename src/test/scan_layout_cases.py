"""Batch-scan SGK page snapshots and classify layout cases vs current ETL detectors.

Reads existing ``pages/page_*_snapshot.png`` files (no PDF render). For each page:
  - OCR text lines + regex counts (sub-figure labels, captions, info boxes, prompts)
  - Cyan-framed / grey-dashed bbox counts (same methods as ImageProcessor)
  - Production crop count under ``database/images/<book>/page_*_img_*.png``
  - Heuristic layout case id + gap vs expected crop estimate

Usage:
    python -m src.test_etl.scan_layout_cases
    python -m src.test_etl.scan_layout_cases --book "SGK KHTN 6 CD" --out scripts/_layout_scan.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.etl.image_processor import (  # noqa: E402
    FIGURE_CAPTION_REGEX,
    INFO_BOX_TITLE_REGEX,
    SUB_FIGURE_LABEL_REGEX,
    ImageProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("scan_layout_cases")

DEFAULT_BOOK = "SGK KHTN 6 CD"
DEFAULT_OUT = ROOT / "scripts" / "_layout_scan_SGK_KHTN_6_CD.json"

QUESTION_RE = re.compile(
    r"h[aãâ]y\s+quan\s+s[aá]t|h[aãâ]y\s+t[iìí]m|em\s+h[aãâ]y|tr[aả]\s+l[oờ]i",
    re.IGNORECASE,
)


@dataclass
class PageScan:
    page: int
    snapshot_path: str
    prod_crop_count: int
    text_line_count: int
    sub_figure_labels: int
    figure_captions: int
    info_box_titles: int
    question_prompts: int
    cyan_framed_raw: int
    grey_dashed_raw: int
    cyan_stroke_ratio: float
    grey_dashed_stroke_ratio: float
    framed_after_refine: int
    expected_crop_estimate: int
    crop_gap: int
    layout_case: str
    sample_pages: List[str] = field(default_factory=list)


def _count_prod_crops(book_dir: Path, page: int) -> int:
    pattern = f"page_{page}_img_*.png"
    return len(list(book_dir.glob(pattern)))


def _count_regex_lines(text_lines: List[Dict[str, object]], pattern: re.Pattern[str]) -> int:
    count = 0
    for line in text_lines:
        text = str(line.get("text", "")).strip()
        if pattern.search(text):
            count += 1
    return count


def _estimate_expected_crops(
    sub_labels: int,
    figure_captions: int,
    info_boxes: int,
    cyan_framed: int,
    grey_dashed: int,
) -> int:
    """Rough lower bound of distinct crops we expect on a textbook page."""
    frame_boxes = max(cyan_framed, grey_dashed)
    if sub_labels >= 2:
        # Prefer per sub-figure + optional composite when a main caption exists.
        expected = sub_labels
        if figure_captions >= 1:
            expected += 1
        expected += info_boxes
        return expected
    if frame_boxes >= 2:
        return frame_boxes + info_boxes + figure_captions
    if figure_captions >= 1:
        return figure_captions + info_boxes + max(1, frame_boxes)
    return max(info_boxes, frame_boxes, 1 if sub_labels else 0)


def classify_layout_case(
    sub_labels: int,
    figure_captions: int,
    info_boxes: int,
    cyan_framed: int,
    grey_dashed: int,
    cyan_ratio: float,
    grey_ratio: float,
    prod_crops: int,
    crop_gap: int,
) -> Tuple[str, List[str]]:
    """Return (case_id, example_page_hints)."""
    hints: List[str] = []

    if sub_labels >= 3 and cyan_framed >= 2 and grey_dashed <= cyan_framed:
        return "A_CYAN_GRID_MULTI_SUB", ["6"]

    if sub_labels >= 3 and grey_dashed >= 2 and cyan_framed < 2:
        return "B_GREY_DASHED_GRID_MULTI_SUB", ["11"]

    if sub_labels >= 2 and cyan_framed >= 1 and grey_dashed >= 1:
        return "C_MIXED_CYAN_AND_GREY_FRAMES", hints

    if figure_captions >= 1 and sub_labels <= 1 and max(cyan_framed, grey_dashed) <= 1:
        return "D_SINGLE_MAIN_FIGURE", hints

    if info_boxes >= 1 and sub_labels == 0 and figure_captions == 0:
        return "E_INFO_BOX_DOMINANT", hints

    if sub_labels >= 2 and crop_gap >= 2:
        return "Z_UNDEREXTRACT_SUBFIGURE", hints

    if prod_crops == 0 and (cyan_framed + grey_dashed) == 0:
        return "Y_TEXT_OR_DECORATIVE_ONLY", hints

    if grey_ratio > 0.002 and cyan_ratio < 0.001:
        return "F_GREY_STROKE_LOW_CYAN", hints

    if cyan_ratio > 0.003 and grey_ratio < 0.001:
        return "G_CYAN_STROKE_LOW_GREY", hints

    return "H_GENERAL_MIXED", hints


def scan_page(
    processor: ImageProcessor,
    snapshot_path: Path,
    book_dir: Path,
) -> PageScan:
    page = int(snapshot_path.stem.split("_")[1])
    pil_img = Image.open(snapshot_path).convert("RGB")
    img_array = np.array(pil_img)
    page_width, page_height = pil_img.size

    text_lines = processor._collect_page_text_lines(pil_img)
    sub_labels = _count_regex_lines(text_lines, SUB_FIGURE_LABEL_REGEX)
    figure_captions = _count_regex_lines(text_lines, FIGURE_CAPTION_REGEX)
    info_boxes = _count_regex_lines(text_lines, INFO_BOX_TITLE_REGEX)
    question_prompts = sum(
        1 for line in text_lines if QUESTION_RE.search(str(line.get("text", "")))
    )

    cyan_raw = processor._detect_framed_regions(img_array)
    grey_raw = processor._detect_dashed_frame_regions(img_array)
    stroke = processor.frame_stroke_metrics(img_array)
    refined = processor._refine_regions(cyan_raw + grey_raw, img_array)

    prod_crops = _count_prod_crops(book_dir, page)
    expected = _estimate_expected_crops(
        sub_labels, figure_captions, info_boxes, len(cyan_raw), len(grey_raw))
    crop_gap = max(0, expected - prod_crops)
    layout_case, hints = classify_layout_case(
        sub_labels,
        figure_captions,
        info_boxes,
        len(cyan_raw),
        len(grey_raw),
        stroke["cyan_stroke_ratio"],
        stroke["grey_dashed_stroke_ratio"],
        prod_crops,
        crop_gap,
    )

    return PageScan(
        page=page,
        snapshot_path=str(snapshot_path),
        prod_crop_count=prod_crops,
        text_line_count=len(text_lines),
        sub_figure_labels=sub_labels,
        figure_captions=figure_captions,
        info_box_titles=info_boxes,
        question_prompts=question_prompts,
        cyan_framed_raw=len(cyan_raw),
        grey_dashed_raw=len(grey_raw),
        cyan_stroke_ratio=round(stroke["cyan_stroke_ratio"], 6),
        grey_dashed_stroke_ratio=round(stroke["grey_dashed_stroke_ratio"], 6),
        framed_after_refine=len(refined),
        expected_crop_estimate=expected,
        crop_gap=crop_gap,
        layout_case=layout_case,
        sample_pages=hints,
    )


def run_scan(book: str, out_path: Path) -> Dict[str, object]:
    book_dir = ROOT / "database" / "images" / book
    pages_dir = book_dir / "pages"
    snapshots = sorted(pages_dir.glob("page_*_snapshot.png"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {pages_dir}")

    processor = ImageProcessor()
    records: List[PageScan] = []
    for index, snap in enumerate(snapshots, start=1):
        if index % 20 == 0 or index == 1:
            logger.info("Scanning %d/%d: %s", index, len(snapshots), snap.name)
        records.append(scan_page(processor, snap, book_dir))

    case_counts = Counter(r.layout_case for r in records)
    gap_pages = [r for r in records if r.crop_gap >= 2]
    gap_pages.sort(key=lambda r: (-r.crop_gap, -r.sub_figure_labels))

    summary = {
        "book": book,
        "snapshot_count": len(records),
        "layout_case_counts": dict(sorted(case_counts.items())),
        "pages_with_crop_gap_ge_2": len(gap_pages),
        "top_underextract_pages": [
            {
                "page": r.page,
                "crop_gap": r.crop_gap,
                "prod_crops": r.prod_crop_count,
                "expected": r.expected_crop_estimate,
                "sub_labels": r.sub_figure_labels,
                "cyan_framed": r.cyan_framed_raw,
                "grey_dashed": r.grey_dashed_raw,
                "case": r.layout_case,
            }
            for r in gap_pages[:25]
        ],
    }

    payload = {
        "summary": summary,
        "pages": [asdict(r) for r in records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d pages)", out_path, len(records))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default=DEFAULT_BOOK, help="Folder under database/images/")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON output path")
    args = parser.parse_args()

    payload = run_scan(args.book, args.out)
    summary = payload["summary"]
    print("\n=== Layout scan summary ===")
    print(f"Book           : {summary['book']}")
    print(f"Pages scanned  : {summary['snapshot_count']}")
    print(f"Under-extract  : {summary['pages_with_crop_gap_ge_2']} pages (gap >= 2)")
    print("\nLayout cases:")
    for case_id, count in summary["layout_case_counts"].items():
        print(f"  {case_id:32s} {count:4d}")
    print(f"\nFull report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
