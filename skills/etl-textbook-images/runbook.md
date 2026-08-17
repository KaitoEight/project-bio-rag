# Runbook — image ETL ops

Step-by-step playbook for every recurring task.

---

## A. Process a brand-new SGK book

```powershell
# 1. drop the file
copy <wherever>\<book>.pdf datasources\

# 2. quick QA — random pages from this book (deterministic by --seed)
python -m src.test.test_random_pages_etl --pdf "datasources/<book>.pdf" --num-pages 10 --seed 42

# 3. open the report in a browser and scan visually
start src/test/_out/index.html

# 4. if QA passes, run the production pipeline (writes ChromaDB)
python main.py
```

**What to look for in QA (Snapshot / Anchors / Regions overlays per page):**
- Every `Hình X.Y` page has at least one `composite_figure` / `single_figure`.
- Figure crops are the picture + caption, NOT whole paragraphs (no "dính text").
- Coloured `Em có biết` / `Tìm hiểu thêm` / `Vận dụng` boxes are captured; bare
  text section headers are NOT emitted as images.
- No `Bảng X.Y` row appears as a region.
- Multi-cell figures with `a)/b)/c)` labels are split into `sub_figure` crops.
- Tool grids (instrument rows) are captured as `tool_group`.

If anything is missing or wrong, go to runbook §D.

---

## B. Reprocess all pages after a detector change

```powershell
# 1. bump the version so the status DB invalidates per-page caches
# edit src/config.py:  IMAGE_EXTRACTION_VERSION = "v9_<your-tag>"

# 2. re-run QA: a single book first (fast), then the full random sample
python -m src.test.test_random_pages_etl --pdf "datasources/<book>.pdf" --num-pages 5
python -m src.test.test_random_pages_etl --seed 42  # all 12 books, 5 pages each

# 3. if good, run the pipeline (skips pages already at this version)
python main.py
```

---

## C. QA tooling

```powershell
# canonical visual QA — all books, N random pages each
python -m src.test.test_random_pages_etl --num-pages 5 --seed 42

# focus on one book while tuning
python -m src.test.test_random_pages_etl --pdf "datasources/SGK KHTN 6 CTST.pdf" --num-pages 5
```

Outputs in `src/test/_out/`:
- `images/<book>/page_<N>_snapshot.png` – the rendered page
- `…/page_<N>_anchors.png` – OCR text anchors colour-coded by bucket
- `…/page_<N>_regions.png` – final extracted regions
- `…/page_<N>_crop_<M>.png` – one PNG per emitted crop
- `index.html` – per-page overlays + crops + quality summary

To quantify a bad crop while debugging, compute its text-line coverage
(`_text_line_coverage`, high = mostly text = a "dính text" smell) and its
coloured visual-content score (`_visual_content_score`, low = no real picture)
in a quick python shell against the page's text lines.

---

## D. Diagnosing a missed extraction

> **Always work top-down: anchor → visual → geometry.** Open
> `page_<N>_anchors.png` first, then `page_<N>_regions.png`, then the crops.

### D.1 Anchor missing (`page_<N>_anchors.png` has no box where expected)
The OCR text didn't match the regex. Either:
- The OCR garbled the line — open the snapshot, look at the actual text.
  If the figure is a **photo** (not a line drawing), the v9
  `_recover_captions_below_photos` pass should rescue it by re-OCRing the
  strip below; check the picture's `vis` ≥ `_RECOVER_MIN_VIS` (0.06).
  Otherwise improve OCR (paddleocr/vietocr) or override `_classify_text_anchors`.
- The regex is too strict. Edit `FIG_CAPTION_STRICT_REGEX` /
  `_CTST_FIG_CAPTION_REGEX` / `_KNTT_FIG_*`, `TABLE_CAPTION_STRICT_REGEX`,
  `_INFO_BOX_TITLE_KEYS`, `TOOL_GROUP_LABEL_REGEX`, `SUB_FIGURE_LABEL_REGEX`.

### D.2 Anchor detected but visual cell missing
OWL-ViT did not return a box for the cell. Options:
- Lower `OWL_VIT_CONFIDENCE_THRESHOLD` in `.env` from `0.1` to `0.05`.
- Add a specific query to `OWL_VIT_TEXT_QUERIES` ("a line drawing of a ruler").
- A `Hình X.Y` caption with NO visual cell is recovered as a band ONLY if the
  band above it is near-text-free; a body-text *reference* line is correctly
  skipped (see `_build_uncovered_caption_regions`).

### D.3 Visual cell present but final region wrong (geometry)

| Symptom | Method | Knob / fix |
|---|---|---|
| Crop "dính text" — swallows body paragraphs | `_assign_regions_to_captions`, `_grow_figure_top` | lower `_FIG_ASSIGN_MAX_VGAP`, `_FIG_TOP_GROW_MAX_GAP`, `_FIG_TOP_GROW_MAX_WIDTH` (per-variant attributes) |
| Bare text section header emitted as info box | `_is_text_only_panel` | set `_INFO_REQUIRE_VISUAL=True` for the variant; tune `_INFO_MIN_VIS` |
| Info box bleeds across the gutter into the other column | `_build_info_panels` | gutter clamp (v9) — verify the title's column side is detected |
| Figure missing because its cells fell inside an over-grown info box | `_build_info_panels` | usually fixed by the gutter clamp; the cells are then assignable |
| Caption above the figure not picked up | `_build_uncovered_caption_regions` | set `_FIG_CAPTION_ABOVE_OK=True` (KNTT) |
| Sub-figures not split | `_split_region_sub_figures` | needs ≥2 OCR-readable `a)/b)` labels + ≥2 cells; check `sub_labels` anchor count |
| Duplicate / overlapping crops ("đè mất hình") | `_suppress_overlapping_regions` | raise/lower the IoU / containment thresholds |

### D.4 Merged OCR lines (Tesseract glues neighbours)
- "Hình 14.1. … Hình 14.2. …" merged into one caption → `_split_merged_figure_caption`.
- A caption merged with the other column's text (CTST page 174) → recovered
  instead by `_recover_captions_below_photos` (re-OCR strip below the photo).

If a new merge pattern appears, add a similar splitter or recovery pass.

---

## E. Adding a new info-box title (e.g. "Lưu ý")

```python
# src/etl/image_processor.py
_INFO_BOX_TITLE_KEYS: List[Tuple[str, str]] = [
    ...,
    ("luu y", "activity_box"),
]
```

Bump `IMAGE_EXTRACTION_VERSION` and re-run.

---

## F. Reset processing status for a single PDF

The status DB key is `pdf_hash:page_num:version`. To force re-processing:

```python
# In a python shell:
from src.etl.processing_status import ProcessingStatus
status = ProcessingStatus()
# either bump IMAGE_EXTRACTION_VERSION (preferred) or reset the table
```

The `reset_status.py` script in the project root provides a CLI for this.
