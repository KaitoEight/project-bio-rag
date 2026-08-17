"""Anchor-first (v7) image-extraction QA on textbook PDF pages.

Runs the deterministic anchor-first detector
(`ImageProcessor.detect_regions_anchor_first`) on a single page or every page
of a PDF and dumps:

  * 00_page_snapshot.png   : the rendered page.
  * 01_anchors.png         : OCR text-anchors colour-coded by category
                             (Hình caption / Bảng caption / info-title /
                              sub-figure label / question prompt / tool label).
  * 02_visual_regions.png  : raw OWL-ViT + cyan-frame + dashed-frame regions
                             after dedupe (debug only).
  * 03_final_regions.png   : final v7 regions (composite / info / tool /
                             sub-figure) colour-coded by image_type.
  * region_<i>__<type>__<caption>.png : one crop per final region.
  * report.json            : per-page summary (anchors + final regions).

For batch mode (``--all``) the script also writes:

  * <out_dir>/_index.html  : visual index of every page (annotated thumbnail
                             + every saved crop) for fast manual review.
  * <out_dir>/_summary.csv : per-page region counts.

The script does NOT touch the status DB and does NOT call the caption LLM,
so it is safe to re-run.

Usage (PowerShell):
    python -m src.test_etl.test_image_extraction_full --page 6
    python -m src.test_etl.test_image_extraction_full --pdf "datasources\\SGK KHTN 6 CD.pdf" --page 13
    python -m src.test_etl.test_image_extraction_full --all
    python -m src.test_etl.test_image_extraction_full --all --pages 6,13,22,40,55
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf2image import convert_from_path  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from src.config import POPPLER_PATH  # noqa: E402
from src.etl.image_processor import ImageProcessor, make_image_processor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("image_etl_v7_test")

DEFAULT_PDF = ROOT / "datasources" / "SGK KHTN 6 KNTT.pdf"
DEFAULT_PAGE = 6
DEFAULT_OUT_DIR = ROOT / "scripts" / "_out_test_etl_full"

ANCHOR_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "figure_captions":   (0, 180, 0),     # green
    "table_captions":    (200, 0, 0),     # red — rejected
    "info_titles":       (255, 0, 255),   # magenta
    "sub_labels":        (255, 200, 0),   # yellow
    "question_prompts":  (160, 80, 255),  # purple — rejected
    "tool_group_labels": (0, 180, 220),   # cyan
}

REGION_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "composite_figure":  (0, 200, 0),
    "single_figure":     (0, 150, 200),
    "sub_figure":        (255, 200, 0),
    "textbook_info_box": (255, 0, 255),
    "activity_box":      (255, 0, 200),
    "tool_group":        (0, 180, 220),
}

DEFAULT_REGION_COLOUR = (220, 30, 30)


@dataclass
class PageReport:
    pdf_path: str
    page_number: int
    page_size: Tuple[int, int]
    text_line_count: int = 0
    anchor_counts: Dict[str, int] = field(default_factory=dict)
    region_counts: Dict[str, int] = field(default_factory=dict)
    final_regions: List[Dict[str, object]] = field(default_factory=list)
    out_dir: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "page_number": self.page_number,
            "page_size": list(self.page_size),
            "text_line_count": self.text_line_count,
            "anchor_counts": self.anchor_counts,
            "region_counts": self.region_counts,
            "final_count": len(self.final_regions),
            "final_regions": self.final_regions,
            "out_dir": self.out_dir,
        }


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    colour: Tuple[int, int, int],
    label: str,
    width: int = 3,
    font: Optional[ImageFont.ImageFont] = None,
) -> None:
    draw.rectangle(bbox, outline=colour, width=width)
    if label:
        tx = bbox[0] + 4
        ty = max(0, bbox[1] - 18)
        draw.rectangle(
            (tx - 2, ty - 1, tx + len(label) * 7, ty + 14), fill=colour)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)


def _save_anchors_overlay(
    pil_img: Image.Image,
    out_path: Path,
    anchors: Dict[str, List[Dict[str, object]]],
    font: ImageFont.ImageFont,
) -> None:
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)
    for bucket, entries in anchors.items():
        colour = ANCHOR_COLOURS.get(bucket, (0, 0, 0))
        for entry in entries:
            label = f"{bucket[:8]}: {str(entry['text'])[:32]}"
            _draw_box(draw, tuple(entry["bbox"]),  # type: ignore[arg-type]
                      colour, label, width=2, font=font)
    title = (
        "01 ANCHORS — green=Hình, red=Bảng(REJECT), magenta=info, "
        "yellow=sub, purple=prompt(REJECT), cyan=tool-label"
    )
    draw.rectangle((10, 10, 10 + len(title) * 7, 32), fill=(0, 0, 0))
    draw.text((14, 12), title, fill=(255, 255, 255), font=font)
    overlay.save(out_path)


def _save_visual_regions_overlay(
    pil_img: Image.Image,
    out_path: Path,
    visual_regions: List[Tuple[int, int, int, int]],
    raw: Dict[str, List[Tuple[int, int, int, int]]],
    font: ImageFont.ImageFont,
) -> None:
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)
    for bbox in raw.get("owlvit", []):
        _draw_box(draw, bbox, (0, 128, 255), "owl", width=1, font=font)
    for bbox in raw.get("framed", []):
        _draw_box(draw, bbox, (255, 140, 0), "frame", width=1, font=font)
    for bbox in raw.get("dashed", []):
        _draw_box(draw, bbox, (160, 80, 255), "dash", width=1, font=font)
    for bbox in visual_regions:
        _draw_box(draw, bbox, (220, 30, 30), "kept", width=2, font=font)
    title = "02 VISUAL REGIONS — blue=OWL, orange=cyan-frame, purple=dashed, red=kept"
    draw.rectangle((10, 10, 10 + len(title) * 7, 32), fill=(0, 0, 0))
    draw.text((14, 12), title, fill=(255, 255, 255), font=font)
    overlay.save(out_path)


def _save_final_overlay(
    pil_img: Image.Image,
    out_path: Path,
    regions: List[Dict[str, object]],
    font: ImageFont.ImageFont,
) -> None:
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(regions):
        image_type = str(region["image_type"])
        colour = REGION_COLOURS.get(image_type, DEFAULT_REGION_COLOUR)
        caption = str(region.get("caption_text", ""))[:36]
        label = f"#{index} {image_type}{(' | ' + caption) if caption else ''}"
        _draw_box(draw, tuple(region["bbox"]),  # type: ignore[arg-type]
                  colour, label, width=3, font=font)
    title = "03 FINAL — green=composite, cyan=single, yellow=sub, magenta=info, teal=tool"
    draw.rectangle((10, 10, 10 + len(title) * 7, 32), fill=(0, 0, 0))
    draw.text((14, 12), title, fill=(255, 255, 255), font=font)
    overlay.save(out_path)


def _safe_label(text: str, max_len: int = 36) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text)
    keep = keep.strip("_")
    return keep[:max_len] or "region"


def render_page(pdf_path: Path, page_number: int, dpi: int = 150) -> Image.Image:
    pages = convert_from_path(
        str(pdf_path),
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )
    if not pages:
        raise RuntimeError(
            f"Could not render page {page_number} of {pdf_path}")
    return pages[0].convert("RGB")


def run_page(
    processor: ImageProcessor,
    pdf_path: Path,
    page_number: int,
    out_dir: Path,
    keep_old: bool = False,
) -> PageReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not keep_old:
        for stale in out_dir.glob("*.png"):
            stale.unlink()
        for stale in out_dir.glob("*.json"):
            stale.unlink()

    pil_img = render_page(pdf_path, page_number)
    page_width, page_height = pil_img.size
    img_array = np.array(pil_img)
    pil_img.save(out_dir / "00_page_snapshot.png")

    font = _font(14)

    report = PageReport(
        pdf_path=str(pdf_path),
        page_number=page_number,
        page_size=(page_width, page_height),
        out_dir=str(out_dir),
    )

    text_lines = processor._collect_page_text_lines(pil_img)
    report.text_line_count = len(text_lines)

    detection = processor.detect_regions_anchor_first(
        pil_img, img_array, text_lines=text_lines)
    anchors: Dict[str, List[Dict[str, object]]
                  ] = detection["anchors"]  # type: ignore[assignment]
    # type: ignore[assignment]
    regions: List[Dict[str, object]] = detection["regions"]
    # type: ignore[assignment]
    visual_regions: List[Tuple[int, int, int, int]
                         ] = detection["visual_regions"]

    report.anchor_counts = {key: len(value) for key, value in anchors.items()}

    _save_anchors_overlay(pil_img, out_dir / "01_anchors.png", anchors, font)
    _save_visual_regions_overlay(
        pil_img, out_dir / "02_visual_regions.png",
        visual_regions,
        {
            "owlvit": detection["owlvit_regions"],  # type: ignore[arg-type]
            "framed": detection["framed_regions"],  # type: ignore[arg-type]
            "dashed": detection["dashed_regions"],  # type: ignore[arg-type]
        },
        font,
    )
    _save_final_overlay(
        pil_img, out_dir / "03_final_regions.png", regions, font)

    region_counts: Dict[str, int] = {}
    saved_crop_paths: List[str] = []
    for index, region in enumerate(regions):
        image_type = str(region["image_type"])
        region_counts[image_type] = region_counts.get(image_type, 0) + 1

        bbox = tuple(int(value)
                     for value in region["bbox"])  # type: ignore[arg-type]
        crop = pil_img.crop(bbox)
        caption = str(region.get("caption_text", ""))
        snippet = _safe_label(caption[:60])
        filename = (
            f"region_{index:02d}__{_safe_label(image_type, 24)}"
            f"__{snippet}.png"
        )
        target = out_dir / filename
        crop.save(target)
        saved_crop_paths.append(filename)

        report.final_regions.append({
            "index": index,
            "bbox": list(bbox),
            "image_type": image_type,
            "caption_text": caption,
            "crop_file": filename,
        })

    report.region_counts = region_counts

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)

    logger.info(
        "Page %d: %d text-lines, anchors=%s, regions=%s",
        page_number, report.text_line_count,
        {k: v for k, v in report.anchor_counts.items() if v},
        region_counts,
    )

    return report


def _page_count(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def _parse_pages_arg(pages_arg: Optional[str], total: int) -> List[int]:
    if not pages_arg:
        return list(range(1, total + 1))
    out: List[int] = []
    for token in pages_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(token))
    return [page for page in out if 1 <= page <= total]


def _write_summary_csv(out_dir: Path, reports: List[PageReport]) -> Path:
    csv_path = out_dir / "_summary.csv"
    fields = [
        "page", "text_lines",
        "figure_captions", "table_captions", "info_titles",
        "sub_labels", "question_prompts", "tool_group_labels",
        "composite_figure", "single_figure", "sub_figure",
        "textbook_info_box", "activity_box", "tool_group",
        "final_total",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            row: Dict[str, object] = {
                "page": report.page_number,
                "text_lines": report.text_line_count,
                "final_total": len(report.final_regions),
            }
            for key in (
                "figure_captions", "table_captions", "info_titles",
                "sub_labels", "question_prompts", "tool_group_labels",
            ):
                row[key] = report.anchor_counts.get(key, 0)
            for key in (
                "composite_figure", "single_figure", "sub_figure",
                "textbook_info_box", "activity_box", "tool_group",
            ):
                row[key] = report.region_counts.get(key, 0)
            writer.writerow(row)
    return csv_path


def _write_index_html(out_dir: Path, reports: List[PageReport]) -> Path:
    html_path = out_dir / "_index.html"
    parts: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>v7 ETL — batch QA</title>",
        "<style>",
        "body { font-family: -apple-system, Segoe UI, sans-serif; margin: 16px; }",
        "h2 { margin-top: 32px; border-top: 2px solid #ccc; padding-top: 12px; }",
        ".row { display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap; }",
        ".overlay { max-width: 460px; }",
        ".crops { display: flex; gap: 8px; flex-wrap: wrap; max-width: 900px; }",
        ".crop { border: 1px solid #ddd; padding: 4px; max-width: 220px; }",
        ".crop img { max-width: 200px; max-height: 240px; display: block; }",
        ".crop .meta { font-size: 11px; color: #444; word-break: break-all; }",
        ".badge { display: inline-block; padding: 1px 6px; border-radius: 4px;"
        " font-size: 11px; color: white; margin-right: 4px; }",
        "</style></head><body>",
        f"<h1>v7 ETL — {len(reports)} page(s)</h1>",
    ]
    for report in reports:
        rel = Path(report.out_dir).relative_to(out_dir).as_posix() \
            if Path(report.out_dir).is_relative_to(out_dir) else report.out_dir
        page_url = f"{rel}/03_final_regions.png"
        anchor_url = f"{rel}/01_anchors.png"
        parts.append(f"<h2>Page {report.page_number} "
                     f"<small>({report.text_line_count} OCR lines, "
                     f"{len(report.final_regions)} regions)</small></h2>")
        parts.append("<div class='row'>")
        parts.append(
            f"<div><img class='overlay' src='{page_url}' alt='final'>"
            f"<div><a href='{anchor_url}'>anchors</a></div></div>"
        )
        parts.append("<div class='crops'>")
        for entry in report.final_regions:
            crop_url = f"{rel}/{entry['crop_file']}"
            image_type = html.escape(str(entry["image_type"]))
            caption = html.escape(str(entry.get("caption_text", "")))
            colour = REGION_COLOURS.get(
                str(entry["image_type"]), DEFAULT_REGION_COLOUR)
            badge_style = (
                f"background: rgb({colour[0]},{colour[1]},{colour[2]});"
            )
            parts.append(
                f"<div class='crop'>"
                f"<img src='{crop_url}'>"
                f"<div class='meta'>"
                f"<span class='badge' style='{badge_style}'>{image_type}</span>"
                f"#{entry['index']} {caption}"
                f"</div></div>"
            )
        parts.append("</div></div>")
    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def run_batch(
    pdf_path: Path,
    pages: List[int],
    out_dir: Path,
) -> List[PageReport]:
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = make_image_processor(pdf_path.name)
    reports: List[PageReport] = []
    for page in pages:
        page_dir = out_dir / f"page_{page:03d}"
        try:
            report = run_page(processor, pdf_path, page,
                              page_dir, keep_old=False)
        except Exception as exc:
            logger.warning("Page %d failed: %s", page, exc)
            continue
        reports.append(report)

    csv_path = _write_summary_csv(out_dir, reports)
    html_path = _write_index_html(out_dir, reports)
    logger.info("Wrote summary: %s", csv_path)
    logger.info("Wrote index:   %s", html_path)
    return reports


def _seeded_sample(total: int, count: int, seed: int) -> List[int]:
    """Deterministic pseudo-random page sample in [1, total] (no RNG import)."""
    if total <= count:
        return list(range(1, total + 1))
    picks: List[int] = []
    state = seed % 2147483647 or 1
    while len(picks) < count:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        page = (state % total) + 1
        if page not in picks:
            picks.append(page)
    return sorted(picks)


def run_sample_all(
    datasources: Path,
    per_book: int,
    out_dir: Path,
    seed: int,
) -> None:
    """Sample `per_book` random pages from every PDF in `datasources`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(datasources.glob("*.pdf"))
    all_reports: List[PageReport] = []
    for pdf in pdfs:
        book = pdf.stem
        # Use publisher-specific processor for CTST, KNTT, CD, etc.
        processor = make_image_processor(pdf.name)
        total = _page_count(pdf)
        pages = _seeded_sample(total, per_book, seed)
        logger.info(
            "Book %s (%s, %d pages) → sampling %s",
            book, type(processor).__name__, total, pages)
        for page in pages:
            page_dir = out_dir / _safe_label(book, 40) / f"page_{page:03d}"
            try:
                report = run_page(processor, pdf, page,
                                  page_dir, keep_old=False)
            except Exception as exc:
                logger.warning("%s page %d failed: %s", book, page, exc)
                continue
            all_reports.append(report)
    _write_summary_csv(out_dir, all_reports)
    _write_index_html(out_dir, all_reports)
    logger.info("Sample index: %s", out_dir / "_index.html")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF,
                        help=f"PDF path (default: {DEFAULT_PDF})")
    parser.add_argument("--page", type=int, default=DEFAULT_PAGE,
                        help=f"1-indexed page (default: {DEFAULT_PAGE})")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma/range page list (e.g. '6,13,22-25'). "
                        "Implies --all when provided.")
    parser.add_argument("--all", action="store_true",
                        help="Process every page; with --pages limit to listed pages.")
    parser.add_argument("--sample-books", type=int, default=0, metavar="N",
                        help="Sample N random pages from EVERY pdf in --datasources.")
    parser.add_argument("--datasources", type=Path,
                        default=ROOT / "datasources",
                        help="Folder of PDFs for --sample-books.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for --sample-books page selection.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    if args.sample_books > 0:
        if not args.datasources.is_dir():
            print(f"[ERR] datasources not found: {args.datasources}")
            return 1
        run_sample_all(args.datasources, args.sample_books,
                       args.out_dir / "_sample", args.seed)
        print(
            f"\n[OK] book sample → {args.out_dir / '_sample' / '_index.html'}")
        return 0

    if not args.pdf.exists():
        print(f"[ERR] PDF not found: {args.pdf}")
        return 1

    if args.all or args.pages:
        total = _page_count(args.pdf)
        pages = _parse_pages_arg(args.pages, total)
        if not pages:
            print(
                f"[ERR] No valid pages in --pages='{args.pages}' (total {total})")
            return 1
        run_batch(args.pdf, pages, args.out_dir)
        print(f"\n[OK] batch QA → {args.out_dir / '_index.html'}")
    else:
        processor = make_image_processor(args.pdf.name)
        run_page(processor, args.pdf, args.page,
                 args.out_dir / f"page_{args.page:03d}")
        print(f"\n[OK] page {args.page} → "
              f"{args.out_dir / f'page_{args.page:03d}' / '03_final_regions.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
