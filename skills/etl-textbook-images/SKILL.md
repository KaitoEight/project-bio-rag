---
name: etl-textbook-images
description: |
  Anchor-first, column-aware image ETL for scanned Vietnamese textbook PDFs.
  Use this skill when (a) adding a new textbook PDF to `datasources/`,
  (b) extraction quality regresses on existing books, or
  (c) tuning the detection rules in `src/etl/image_processor.py`.
metadata:
  type: etl-pipeline
  language: python
  applies_to: scanned-vietnamese-sgk
---

# ETL — Textbook image extraction (v15 per-variant, column-aware anchor-first)

## v15 changes (KNTT photo-rectangle detector)

Fixes the KNTT figure crops that were CUT because OWL only detected a partial
cell of a hard photo. New `_detect_photo_rectangles` (base method, opt-in
`_DETECT_PHOTO_RECTANGLES`, **on for KNTT**): masks non-white page content and
morphologically OPENS it (9×9) so thin text strokes vanish while solid photo
fills survive → full photo rectangles. A small close (7×7) avoids bridging the
gutter between a coloured question box and the figure below (page 13). Two
companion fixes in the KNTT builder:
- cell membership is by CENTRE, not "whole cell above the caption", because a
  KNTT pill often sits ON the photo's bottom edge (page 131 bat caption);
- the prev-caption ceiling still keeps each figure's cells separate.

Result: page 131 bat now full (was a thin sliver); page 13 warning triangles
full incl. the apexes (was cut + merged with the question box). CD/CTST
untouched (KNTT-gated). Remaining: page 80 diagram top circles are merged with
the "MỤC TIÊU" title cell so they're text-gated out (still cut); page 95 pill
number OCR; pages 13/60 extra info/activity box.

## v14 changes (KNTT crop precision)

KNTT figure crops were cut / over-grown / wrong. All KNTT-scoped:
- **Contiguity band, text-gated** (replaces the fixed `_KNTT_FIG_MAX_BAND_FRAC`
  cap, now just a runaway backstop 0.85): the figure band walks the figure-side
  cells bridging gaps ≤25% page-H, but STOPS at a text-heavy cell
  (`_text_line_coverage > 0.25`). A tall connected diagram is kept whole
  (page 80), the band reaches a pure-image cell 0.23·H away (page 131 sheep),
  but it won't swallow a title bar / objectives block (page 80 "MỤC TIÊU") or
  an unrelated far photo (page 109).
- **Body-reference rejection**: a line like "(Hình 42.1a) thì lò xo dãn ra
  (Hình 42.1b)." is dropped WHOLE when any marker is followed by a Vietnamese
  function word (`_KNTT_REF_FUNCTION_WORDS`) or a closing ")", so it no longer
  builds a figure over the question table (page 152).
- **Sub-figure splitting DISABLED for KNTT** (`_SPLIT_SUBFIGURES=False`): the
  per-cell crops came out partial and lost the a)/b) titles (pages 69, 109);
  keeping the (correct) composite whole is cleaner until KNTT cell detection
  improves.

Status: KNTT p23/p30/p120 exact; p69/p109 clean composites (no junk subs); p80
diagram (top circles lost — merged with the title cell, see limits); p131
sheep+bees correct; p152 no longer a wrong text-box. Remaining (detection
limits): p131 bat + p13 triangle-apex are cut (OWL detects only a partial cell
of a dark/thin photo); p95 pill-number OCR (2 of 3); p13/p60 emit an extra
info/activity box.

## v13 changes (KNTT gets its OWN figure builder)

KNTT now has its own `_build_figure_composites` (like CTST), so CD's centred
builder no longer clips/merges KNTT figures. A KNTT pill sits ABOVE or BELOW
its figure, which can span multiple stacked or side-by-side photos:
- Per caption, the figure **side** (above/below) = whichever has the smaller
  adjacent visual-cell gap (page 109 caption-above mushrooms vs page 131
  caption-below).
- Captions are grouped into **rows** (same side, similar y); each row's band of
  cells is split into **columns at the caption x-midpoints**, so a row of N
  captions over one OWL-merged wide cell becomes N figures (page 95 three
  disease photos), while a lone caption keeps the full-width figure (page 13
  hazards full-width, page 131 Hình 36.16 spans sheep **and** bees).
- The band height is capped (`_KNTT_FIG_MAX_BAND_FRAC=0.45`) so a bottom caption
  can't swallow unrelated art near the page top (page 109 ballooning fixed).

Also: the pill mask uses several hue bands (orange/yellow/green/teal/blue/
purple/pink-red) — NOT an all-hue mask, which caught colourful photos and
bridged adjacent pills (page 95). Status: KNTT p13 (full-width hazards), p131
(bat + sheep&bees), p69/p109 (composite + sub), p23/p152 unchanged. Partial:
p95 gets 2 of 3 (pill *number* OCR misreads on that page). CD/CTST untouched.

## v12 changes (KNTT pill-caption anchor detection)

KNTT renders every figure title "Hình X.Y" inside a **colour-varying pill**;
the page OCR misses it, so the anchor was absent (pages 13/69) or wrong
(page 131). All fixes live in `KnttImageProcessor`; CD/CTST untouched.

1. **Pill OCR robustness** (`_detect_pill_figure_captions`). The pill colour
   varies, so the mask now accepts ANY saturated hue (`inRange (0,60,60)–
   (180,255,255)`) instead of a fixed orange/green/blue/purple set; the strict
   "must OCR to Hình X.Y" gate rejects non-caption blobs. The crop is upscaled
   3× (grayscale, then threshold — upscaling the *binary* added ringing) and
   tried at several thresholds.
2. **Mangled-pill normalisation** (`_normalise_kntt_caption`). A tight pill
   drops the space and dot: "Hình 19.3" → `Hình193`, "Hình 2.2" → `Hình22`.
   The KNTT regexes now use `\s*` and `[.,]?`, and the normaliser re-inserts the
   dot (`193`→`19.3`, `22`→`2.2`, `3615`→`36.15`) — best-effort, its job is to
   anchor the figure.
3. **Caption = pill / line-start only** (`_KNTT_FIG_START_REGEX`). The old
   permissive `.*?Hình` `.match` was effectively a search, so body references
   ("… (Hình 36.15).") became spurious captions → duplicate figures (page 131).
   Now a caption must START with the marker (after ≤4 pill-junk chars).
4. **Dedupe by figure number** (`_dedupe_kntt_captions`). A pill plus its
   page-OCR'd caption line for the same number kept the longest text only.

Status: KNTT p13 (Hình 2.2 found), p69 (composite + 3 sub), p131 (2 figures,
no duplicates), p23/p109/p152 unchanged. Partial: KNTT figure *building* from a
pill that sits below/beside a multi-photo figure still under-covers (p131 Hình
36.16 catches the bees but not the sheep above; p13 width slightly short) — a
follow-up on the KNTT caption-above/full-width assignment.

## v11 changes (CTST gets its OWN figure builder + caption recovery)

## v11 changes (CTST gets its OWN figure builder + caption recovery)

v11 fixes CTST (pages 64 / 135 / 29 / 58) with logic kept **entirely inside
`CtsstImageProcessor`** — CD's builder is untouched (the user's hard
requirement: CTST must not share CD's figure logic). Three CTST-specific
pieces:

1. **Band-based figure builder** (`CtsstImageProcessor._build_figure_composites`,
   overrides CD's). CTST's ``▲ Hình X.Y`` caption is **left-aligned under a
   figure that often spans the full content width**; CD's centred-caption
   assignment clips the right half (page 64 biogas lost the generator; page 135
   food-chain lost rắn + diều). Instead each caption claims every visual cell in
   the vertical band between it and the nearest caption/info-title above, and
   each cell goes to the **nearest caption by x-centre** — so a full-width
   figure is captured whole while side-by-side figures (page 58) still split.
   Question prompts are NOT band ceilings (a right-column prompt would wrongly
   truncate a left-column figure's band).
2. **Caption pre-recovery** (`_inject_recovered_captions`, CTST
   `detect_regions_anchor_first` override). The filled triangle makes Tesseract
   drop whole caption lines (page 29: only ``Hình 6.4`` is read → the three
   stopwatch figures collapse into one). Before building, re-OCR the strip
   below each cheap CV cell (frames + blobs, no OWL) and inject any recovered
   ``Hình X.Y``; **deduped by figure NUMBER** so a caption read by both passes
   can't spawn a duplicate figure (page 10 "Hình 2.10").
3. **Sub-figure splitting enabled** (`_SPLIT_SUBFIGURES_BY_TITLE=True`) but
   capped to **one title row** (`_SUBFIG_TITLE_MAX_ROWS=1`): CTST titled figures
   are single-row photo strips (food-chain → 5 sub-figures), while its
   multi-component diagrams carry internal labels at many heights — the cap
   keeps the biogas flow chart (page 64) whole instead of slicing it.
4. **Pale-photo recall + caption robustness** (page 59, where OWL-ViT returned
   *nothing* usable and both captions were OCR-broken):
   - `_detect_textured_photo_regions` (base, opt-in via `_DETECT_TEXTURED_PHOTOS`,
     **on for CTST**) finds large photos by local std-dev, so a pale beige
     building OWL + the colour-blob detector both miss is detected; text blocks
     it also catches are dropped by `_filter_text_visual_regions`. It also feeds
     the caption pre-recovery, so the building's strip is re-OCR'd.
   - `_CTST_FIG_PREFIX` now also absorbs a **▶** marker OCR'd as `}>`/`{>`/`›`.
   - `_FIGURE_MARKER_REGEX` requires the figure number NOT be followed by a
     letter, so "mô hình **3R**" no longer counts as a marker / triggers a split.
   - `_reocr_caption_below` starts its strip slightly INSIDE the picture bottom
     (a detector box often swallows the caption row at its lower edge).

Status: CTST p64 (whole biogas), p135 (food-chain full-width + 5 sub-figures,
29.3 + 2), p58 (side-by-side split), p59 (building Hình 11.9 recovered via the
texture detector despite OWL=0), p36/p174 unchanged. Partial: p29 recovers
2 of 3 ``▲`` captions (the 3rd is below OCR threshold); p59 Hình 11.8 (3R icons
with a left-side caption) still not built; side-by-side photos that
OWL merges into one cell still merge (p58 top 11.4+11.5).

## v10 changes (CD figure-bound + caption-row/pixel-column sub-figures)

v10 fixes the CD failures found on pages 6 / 8 / 13 / 131 (and the false-split
risks on 40 / 100 / 30). All target the base `ImageProcessor`; the shared
pieces (1, 2) help every variant, the title-anchored half of the splitter (3)
is **CD-only for now** (`_SPLIT_SUBFIGURES_BY_TITLE` = False on CTST/KNTT until
each is smoke-tested).

**1. "Quan sát …" prompts + multi-line prompt ceilings.** `QUESTION_PROMPT_PATTERNS`
now matches a bare `^Quan sát …` lead-in (no "Hãy"), and
`_prompt_blocker_bboxes` expands each prompt anchor DOWN through its wrapped
continuation lines. A figure's upward ceiling then sits below the WHOLE prompt
paragraph, so top-growth can't absorb a prompt's tail line (page 131
"… của các động vật trong hình." was being swallowed into the crop).

**2. Composite top includes gap-connected upper cells** (`_build_figure_composites`).
A 2-row grid whose bottom `Hình X.Y` caption is too far for direct cell
assignment (page 6 "Hình 1.1") used to lose its entire upper photo row. The
top is now pulled up to the topmost gap-connected visual cell (`visual_top`) —
but ONLY when the bridged band is image-like (`_text_line_coverage < 0.22`).
That guard is what separates page 6 (photo rows between cells) from page 100
(a "em có thể" objectives block / chapter banner above a single figure, which
must NOT be absorbed — without the guard the composite ballooned to y=0).

**3. Sub-figure splitting = caption-rows + pixel-columns**
(`_split_region_sub_figures`, rewritten). The old per-cell splitter emitted one
crop per OWL cell — but OWL-ViT routinely merges a grid into one box per
coloured row (page 8: 5 fields → 2 row boxes) or one box for the whole grid
(page 131: 2×2 animals → 1 box), so it produced "dính chung" row-crops. The new
splitter derives structure independently of the detector:
  - **Rows** come from the caption lines below each photo —
    `_collect_subfig_anchors` returns letter labels `a)/b)/…` when present
    (`is_label_mode=True`, all variants), else (CD only) the short centred
    titles below each cell ("Con cá heo", "Thước cuộn"). Anchors cluster into
    rows by y.
  - **Columns** come from `_detect_columns_by_projection`: per-column *variance*
    down each row's photo band — a gutter is a vertically-uniform strip (white
    page OR a flat coloured row background), a photo column varies a lot. Thin
    internal white stripes are bridged so one picture is never sliced.
  - **Dispatch:** a cell that fits inside ONE row's photo band is "granular"
    (the detector separated the photos — page 6, 109) → CELL MODE, one crop per
    cell. When the only cells span multiple rows / a whole coloured row
    (page 8, 131) there are none → PROJECTION MODE.
  Three guards keep it precise: every caption row needs a real photo band above
  it (`≥5% page H` — rejects internal apparatus labels, page 40 "Dung dịch/Nến");
  and TITLE-mode additionally requires ≥2 title columns AND ≥1 row with ≥2
  titles (rejects a vertical legend beside one chart — page 40 "Hình 7.3" pie;
  and the thermometer internal scale labels, page 30). Tool groups are split
  too (page 13: per-tool crops alongside the kept `tool_group`).

## v9 changes (per-variant separation + recovery + clean crops)

v9 keeps the v8 anchor-first, column-aware core but makes the per-publisher
logic **separate** (CD / CTST / KNTT) and fixes the cross-variant failures that
made crops "dính text" (swallow body paragraphs), miss figures, and emit
text-only boxes. The base `ImageProcessor` (= Cánh Diều) owns all shared
geometry/OCR; each variant overrides only the seams that genuinely differ.

**1. Per-variant tuning attributes (the separation mechanism).** Instead of
hard-coded constants buried in the region builders, the knobs are class
attributes on `ImageProcessor`; subclasses override them. Keep new
publisher-specific behaviour here, not in forked copies of the builders.

| Attribute | Base (CD) | CTST | KNTT | Meaning |
|---|---|---|---|---|
| `_FIG_ASSIGN_MAX_VGAP` | 0.20 | 0.20 | 0.20 | how far (frac page H) above a caption a cell may sit and still be claimed — was 0.34, the main "dính text" cause |
| `_FIG_TOP_GROW_MAX_GAP` | 0.045 | ″ | ″ | gap budget when growing a figure top through its own narrow labels |
| `_FIG_TOP_GROW_MAX_WIDTH` | 0.34 | ″ | ″ | max line width absorbed as a figure label (never a body paragraph) |
| `_INFO_REQUIRE_VISUAL` | False | **True** | False | drop info/activity panels that are bare text on white (no colour/picture) |
| `_INFO_MIN_VIS` | 0.045 | 0.045 | 0.045 | visual-score threshold for "is a real coloured box" |
| `_RECOVER_CAPTIONS_BELOW_PHOTOS` | True | True | True | re-OCR the strip below an uncaptioned picture to recover its `Hình X.Y` |
| `_RECOVER_MIN_VIS` | 0.06 | ″ | ″ | min visual score for a picture to be worth recovering |
| `_FIG_CAPTION_ABOVE_OK` | False | False | **True** | caption may sit ABOVE its figure (KNTT pill labels) |
| `_SPLIT_SUBFIGURES` | True | True | True | emit one `sub_figure` per cell when ≥2 `a)/b)` labels confirm a composite |
| `_SPLIT_SUBFIGURES_BY_TITLE` | **True** | **True** | False | (v10) split a titled photo grid with NO `a)/b)` labels from the centred titles below each cell (CD page 131, CTST food-chain); KNTT off until tuned |
| `_SUBFIG_TITLE_MAX_ROWS` | 99 | **1** | 99 | (v11) max title-anchor rows a split may span; CTST=1 keeps multi-label diagrams (biogas) whole |
| `_CTST_FIG_MAX_BAND_FRAC` | — | **0.55** | — | (v11) CTST-only: cap the band height a single caption can claim above it |
| `_DETECT_TEXTURED_PHOTOS` | False | **True** | False | (v11) detect large PALE photos by local texture (OWL/colour-blob miss them — page 59 building); text blocks dropped by the text filter |

**2. Caption-anchored recovery (two passes, after the caption-first builder).**
   - `_build_uncovered_caption_regions` — a `Hình X.Y` caption that got no
     assigned cell builds a region from the real visual cells in its column
     (above, or below when `_FIG_CAPTION_ABOVE_OK`). With no visual cell it
     falls back to a bounded band ONLY if that band is near-text-free
     (`text_line_coverage < 0.18`) — this distinguishes a faint line drawing
     (recover) from a body-text *reference* "Hình 9.5 là một mô hình …" (skip).
   - `_recover_captions_below_photos` — a detected picture with NO caption
     anchor (CTST `▲ Hình` OCR-mangled to `Aình403`) is rescued by re-OCRing
     the strip directly below it, upscaled 3× + thresholded
     (`_reocr_caption_below` → `_match_recovered_caption`).

**3. Text-only panel drop** (`_is_text_only_panel`, opt-in via
   `_INFO_REQUIRE_VISUAL`). A real info box is a coloured panel or has an
   embedded picture; a bare section header on white ("Tìm hiểu về …",
   "Thí nghiệm 1:") is body text and is dropped. Done BEFORE photo recovery so
   a dropped text box doesn't shadow a real picture sitting inside it (CTST
   page 174). NOTE: CTST also drops the `tim hieu` info-box key entirely.

**4. Column-clamped info panels.** `_build_info_panels` now clamps a
   single-column box to its own side of the central gutter, so a right-column
   "Em có biết" no longer grabs the left column's body text + figure cells
   (which previously excluded them and made the figure disappear — CD page 56).

**5. General sub-figure split** (`_split_region_sub_figures`) runs AFTER
   recovery on any figure region, sourcing cells from `visual_regions` (not the
   builder's `assigned_regions`), so recovered composites split too (KNTT
   caption-above mushroom rows). ≥2 `a)/b)` labels confirm the composite; then
   one crop per cell is emitted, label text attached when it pairs.

**6. Overlap suppression** (`_suppress_overlapping_regions`) removes duplicate /
   near-contained crops at the end (fixes CTST "đè mất hình"), while preserving
   the legitimate composite→sub_figure nesting.

Smoke status after v9 (probe set, seed 42): CD KHTN7 p9 (Hình 2/4), p56 (Hình
9.3 + 9.4 recovered, Em-có-biết clamped), p60; CTST KHTN6 p174 (Hình 40.3
recovered from mangled OCR), p58, p36; KNTT KHTN6 p23, p109 (caption-above
composite split into 4 sub-figures), p152. Remaining limits: side-by-side
single figures sharing one OWL cell can merge (CD p9 Hình 2+3); sub-figure
splitting needs ≥2 OCR-readable `a)/b)` labels (KNTT p23 clock labels unread).

## v8 changes (column-awareness + robust info boxes)

v8 keeps the v7 anchor-first philosophy but fixes the multi-column and
robustness problems found on dense pages (e.g. page 70):

1. **OCR gutter split** (`_collect_page_text_lines`) — a tesseract line is
   split wherever two words are separated by a gap wider than ~5.5 % page
   width. Without this, a left-column cell label and a right-column question
   prompt on the same scan row merged into one full-width line, which made
   figures/info boxes span both columns.
2. **Colour-detected info headers** (`_detect_colored_info_headers`) — info
   boxes are marked by a pink/blue header. When the header is a filled tab
   with white text, page-level OCR misses it entirely. v8 finds the coloured
   header directly and re-OCRs it (white-text-on-colour preprocessing) to
   recover "Em có biết" / "Tìm hiểu thêm". This is merged with the OCR-title
   anchors.
3. **Column-aware panels & tables** — `_build_info_panels` and
   `_build_table_zones` accumulate a column and refuse text from the other
   column, so a right-column box never swallows the left column.
4. **Info/table zones built BEFORE figures** and passed as exclusion +
   column-separator inputs to figure assignment — a figure never absorbs a
   cell that lies inside an info box / table.
5. **Two-tier caption assignment** (`_assign_regions_to_captions`) — a cell
   prefers a caption it horizontally overlaps (splits side-by-side figures,
   page 85); only if none overlaps does it fall back to a centred caption
   within tolerance and not separated by an other-column anchor.
6. **Text-region filter** (`_filter_text_visual_regions`) — OWL-ViT
   false-positives on headings/paragraphs (no colour + ≥2 alphabetic words)
   are dropped so they can't be assigned to a caption and balloon a figure.
   Wide line-drawings (rulers, aspect ≥ 3) are kept.
7. **Visual-ceiling + narrow-label top-growth** (`_grow_figure_top`,
   `_snap_figure_to_column`) — a tall list-figure (page 70 Hình 12.6) whose
   interior cells OWL-ViT missed is rescued by growing UP through its own
   narrow cell-labels, bounded by the topmost visual cell and the gutter.
   Full-width body paragraphs are never absorbed.

Smoke status after v8: pages 6, 70, 85, 30, 40, 55 correct. Thin
line-drawing rulers (page 22) remain limited by OWL-ViT recall.

# ETL — Textbook image extraction (v7 anchor-first, baseline rules)

The pipeline turns a scanned SGK page into a normalised set of image crops:

```
PDF page  →  OCR text lines  →  anchor classification  →  region builders  →  final crops
                                       │                       │
                                       ▼                       ▼
                                  Hình X.Y caption       composite_figure
                                  Bảng X.Y caption       (rejected)
                                  Em có biết / ...       textbook_info_box
                                  a) b) c) labels        sub_figure splits
                                  Dụng cụ ... labels     tool_group
                                  Hãy quan sát ...       (rejected)
```

Every emitted region is whitelisted by an anchor — the detector never
returns a free-standing illustration that lacks a caption or panel
title, except labelled dashed-frame tool groups (page-13 case).

---

## 1. Rule book

### 1.1 KEEP
| Category | Anchor | Bbox derivation |
|---|---|---|
| `composite_figure` | `^Hình X.Y` caption **AND** ≥2 visual cells **AND** at least one `a)`/`b)`/… label inside | bbox grows UP from caption to cover assigned visual regions; ceiling is the nearest blocker (other caption, info-title, question prompt) above |
| `single_figure` | `^Hình X.Y` caption with no sub-labels (or only one visual cell) | same as composite, but no sub-figure splitting |
| `sub_figure` | `a)…h)` label below a visual cell inside a `composite_figure` | one crop per cell, with its label slice |
| `textbook_info_box` | OCR line starting with `Em có biết` | grows DOWN from title through continuous text lines; widens to absorb adjacent visual cell (e.g. portrait inside the panel) |
| `activity_box` | `Tìm hiểu thêm` / `Mở rộng` / `Kiến thức mới` / `Thực hành` / `Vận dụng` / `Luyện tập` | same as info-box |
| `tool_group` | `Dụng cụ đo …` / `Một số dụng cụ` / `Hộp dụng cụ` label sitting above a row of ≥2 visual cells | bbox = label ∪ row cells ∪ per-tool name labels just below; dashed-border outer box widens bbox when present |

### 1.2 REJECT
- `^Bảng X.Y` captions and the rows continuously below them (table exclusion zone).
- Lines matched by `QUESTION_PROMPT_PATTERNS` (Hãy quan sát, Em đã thấy, …) — used only as composite-ceiling blockers.
- Free-standing illustrations with no caption, info title or tool label.
- "Học xong bài học này, em có thể:" / "Em có thể:" learning-objective headers (explicitly excluded from info-box matcher).

### 1.3 OCR edge cases the rules already handle
- Tesseract often glues two side-by-side captions into one line; the
  classifier splits any line that carries multiple `Hình X.Y` markers
  into one anchor per marker (see `_split_merged_figure_caption`).
- "Trong phòng thực hành" body text used to false-match the info-box
  regex; `_match_info_box_title` now requires `Em có biết` / `Tìm hiểu
  thêm` / … to *start* the line and be followed by ≤30 chars.

---

## 2. File map

```
src/etl/image_processor.py                 ← the detector (v9)
  Per-variant tuning attributes (override on subclass — see v9 table)
  ├─ _FIG_ASSIGN_MAX_VGAP / _FIG_TOP_GROW_*  figure-growth limits
  ├─ _INFO_REQUIRE_VISUAL / _INFO_MIN_VIS    drop text-only panels (CTST)
  ├─ _RECOVER_CAPTIONS_BELOW_PHOTOS          photo→caption re-OCR
  ├─ _FIG_CAPTION_ABOVE_OK                    caption above figure (KNTT)
  └─ _SPLIT_SUBFIGURES                        a/b/c/d splitting
  Anchors / regex
  ├─ FIG_CAPTION_STRICT_REGEX               Hình X.Y anchor (CD base)
  ├─ _CTST_FIG_CAPTION_REGEX                 ▲ Hình X.Y (CtsstImageProcessor)
  ├─ _KNTT_FIG_* + _detect_pill_figure_captions  pill caption (KnttImageProcessor)
  ├─ TABLE_CAPTION_STRICT_REGEX             Bảng X.Y reject
  ├─ _INFO_BOX_TITLE_KEYS                    per-variant info-box titles
  ├─ TOOL_GROUP_LABEL_REGEX                 dụng cụ … / một số …
  OCR + anchors
  ├─ _collect_page_text_lines               OCR, with 2-column GUTTER SPLIT
  ├─ _classify_text_anchors                 lines → buckets (overridden per variant)
  ├─ _match_info_box_title                  strict, start-of-line
  ├─ _detect_colored_info_headers           pink/blue header + re-OCR
  Visual detection
  ├─ _detect_regions_with_owlvit            zero-shot regions
  ├─ _detect_framed_regions / _detect_dashed_frame_regions
  ├─ _detect_object_blobs                   CC fallback for object photos
  ├─ _filter_text_visual_regions            drop OWL text false-positives
  Region builders (order: tables+info BEFORE figures)
  ├─ _build_table_zones                     exclusion zone (column-aware)
  ├─ _build_info_panels                     panel (column-aware + gutter clamp v9)
  ├─ _assign_regions_to_captions            two-tier hov / centred
  ├─ _build_figure_composites
  │    ├─ _grow_figure_top                  narrow-label top-growth (knobbed v9)
  │    └─ _snap_figure_to_column            gutter clip + label widen
  ├─ _build_dashed_tool_groups
  Post-processing (v9, inside detect_regions_anchor_first)
  ├─ _build_uncovered_caption_regions       recover unassigned captions
  ├─ _is_text_only_panel                    drop bare-text info boxes
  ├─ _recover_captions_below_photos         re-OCR strip below uncaptioned photo
  │    └─ _reocr_caption_below / _match_recovered_caption
  ├─ _suppress_overlapping_regions          dedupe final crops
  ├─ _split_region_sub_figures              per-cell sub-figures (general)
  └─ detect_regions_anchor_first            top-level entrypoint
src/test/test_random_pages_etl.py          ← canonical QA: random pages → HTML report
src/config.py                               ← IMAGE_EXTRACTION_VERSION
```

### QA sampling

```powershell
# all pdfs in datasources/, 5 random pages each, deterministic by --seed
python -m src.test.test_random_pages_etl --seed 42
start src/test/_out/index.html

# fast dev loop while tuning: one book, a few pages
python -m src.test.test_random_pages_etl --pdf "datasources/SGK KHTN 7 CD.pdf" --num-pages 3
```

OWL-ViT runs on CPU here (~2-8 s/page after model load, and the model reloads
per book), so iterate on a single `--pdf` first, then confirm with the full
random sample. Read the per-page Regions overlay + crops in `index.html`; a
crop that is mostly text (over-grown / "dính text") is the smell to watch for.

## 3. Publisher routing (v8 multi-publisher)

All entry-points now use `make_image_processor(pdf_filename)` to get the
right processor class.  Add the book name to the dispatching table if a new
publisher needs customisation.

```python
from src.etl.image_processor import make_image_processor

proc = make_image_processor("SGK KHTN 6 CTST.pdf")   # → CtsstImageProcessor
proc = make_image_processor("SGK KHTN 7 CD.pdf")     # → ImageProcessor (CD)
```

| Keyword in filename | Variant key | Class |
|---|---|---|
| `_CTST` (or `<space>CTST`) | `ctst` | `CtsstImageProcessor` |
| `_KNTT` / `<space>KNTT` | `kntt` | `KnttImageProcessor` |
| anything else | `cd` | `ImageProcessor` (base) |

### CD — base `ImageProcessor` (Cánh Diều)

Owns all shared geometry/OCR. Caption `^Hình X.Y` OR `^Hình X.` (the
sub-number is optional — `FIG_CAPTION_STRICT_REGEX` already allows "Hình 2.").
Captions always sit BELOW the figure. Photos with a missed caption are
recovered by `_recover_captions_below_photos`.

### CTST differences (`CtsstImageProcessor`)

| Aspect | CD behaviour | CTST behaviour |
|---|---|---|
| Figure caption | `^Hình X.Y` | `^[ÀÁ▲.]? Hình X.Y` — prefix `▲` OCR'd as `À`/`Á`; a fully-mangled or dropped caption is recovered by re-OCRing the strip below the photo (v11 does this BEFORE building via `_inject_recovered_captions`, deduped by figure number) |
| Figure building | CD `_build_figure_composites` (centred caption claims cells near its centre) | **own** `_build_figure_composites` (v11): left-aligned caption claims the full-width band above it; each cell → nearest caption by x-centre. CD's builder is untouched. |
| Info-box titles | Em có biết, Tìm hiểu thêm, … | Bài tập, Khám phá, Vận dụng, Tổng kết, Ôn tập, Thực hành, Thí nghiệm — but **`_INFO_REQUIRE_VISUAL=True`** so bare section headers on white ("Tìm hiểu về …", "Thí nghiệm 1:") are dropped, not emitted. `tim hieu` is NOT a key. |
| Sub-figure labels | `a)`, `b)` … | identical |
| Sub-figure split | titles, multi-row | titles enabled but **single-row only** (`_SUBFIG_TITLE_MAX_ROWS=1`) so flow diagrams stay whole |

### KNTT differences (`KnttImageProcessor`)

| Aspect | CD behaviour | KNTT behaviour |
|---|---|---|
| Figure caption | plain OCR text | rendered in a **colour-varying pill** — `_detect_pill_figure_captions` masks ANY saturated hue, upscales + re-OCRs white-on-colour text, `_normalise_kntt_caption` repairs the dropped space/dot ("Hình193"→"Hình 19.3"), injects synthetic lines (v12) |
| Caption match | strict `^Hình X.Y` | `_KNTT_FIG_START_REGEX` — must START with the marker (≤4 junk chars) so inline body refs "(Hình X.Y)" aren't captions; then `_dedupe_kntt_captions` collapses a pill + its OCR'd line by number (v12) |
| Caption position | below figure | may sit ABOVE **or** below; v13 `_build_figure_composites` picks the side by the smaller adjacent cell gap (page 109 above, page 131 below) |
| Figure building | CD centred-caption | **own** builder (v13): rows of captions → columns at caption x-midpoints (splits a wide merged cell, page 95), lone caption → full-width / multi-photo union (page 131 sheep+bees); band capped by `_KNTT_FIG_MAX_BAND_FRAC` |
| Sub-figures | a/b/c/d | **disabled** (v14, `_SPLIT_SUBFIGURES=False`) — KNTT per-cell crops came out partial / title-less; keep the composite whole |
| Body refs | n/a | (v14) a line whose `Hình X.Y` is followed by a function word or ")" is a reference, dropped whole (page 152) |

### Adding a new publisher

1. Identify the figure caption format, info-box titles, and caption position.
2. Subclass `ImageProcessor`.
3. Override the SEAMS, not the builders: `_classify_text_anchors`,
   `_INFO_BOX_TITLE_KEYS`, the v9 tuning attributes (table above), and
   `_match_recovered_caption` if the caption vocabulary differs.
4. Add a keyword check in `get_pdf_variant` and return the class from
   `make_image_processor`.
5. Add a smoke-test page set to the runbook and export the class in
   `src/etl/__init__.py`.

`IMAGE_EXTRACTION_VERSION` gates the per-page status DB — bump it whenever
the detector logic changes so pages are reprocessed on next run.

---

## 3. Workflows

### 3.1 Adding a new textbook PDF
1. Drop the file in `datasources/`.
2. Run a quick visual QA:
   ```powershell
   python -m src.test_etl.test_image_extraction_full `
     --pdf "datasources/<new-book>.pdf" `
     --pages 1-20 `
     --out-dir scripts/_out_test_etl_full
   start scripts/_out_test_etl_full/_index.html
   ```
3. Scan the HTML index. For each page:
   - Confirm every `Hình X.Y` is captured as `composite_figure`/`single_figure`.
   - Confirm `Em có biết` / `Tìm hiểu thêm` panels are intact (incl. embedded portraits).
   - Confirm `Bảng X.Y` rows are NOT extracted.
   - Spot-check that any dashed-bordered tool grid is captured as `tool_group`.
4. If everything passes, run the full pipeline (writes to DB):
   ```powershell
   python main.py            # or whichever entry the project uses
   ```

### 3.2 Tuning when extraction breaks on a new layout
- Re-run `test_image_extraction_full` on the offending page(s).
- Open `01_anchors.png` first. **If the right anchor is missing, the rule fails before any geometry.** Most regressions are anchor-detection issues:
  - tighten / loosen one of the regexes at the top of `image_processor.py`,
  - or add a new helper to `_classify_text_anchors`.
- Open `02_visual_regions.png` next. If OWL-ViT missed a cell, options:
  - lower `OWL_VIT_CONFIDENCE_THRESHOLD` in `.env` (default 0.1),
  - add a more specific query to `OWL_VIT_TEXT_QUERIES`,
  - or rely on the dashed/framed detectors which work on HSV strokes.
- Open `03_final_regions.png` last for geometry / merging issues.

### 3.3 Adding a new info-box title
Append the keyword (normalised, no accents) to `_INFO_BOX_TITLE_KEYS`:
```python
_INFO_BOX_TITLE_KEYS = [
    ("em co biet", "textbook_info_box"),
    ("tim hieu them", "activity_box"),
    ...,
    ("new key here", "activity_box"),
]
```
Then bump `IMAGE_EXTRACTION_VERSION` so processed pages are re-evaluated.

### 3.4 Adding a new tool-group label pattern
Extend `TOOL_GROUP_LABEL_REGEX`. The regex anchors at line start and
matches noun phrases that always head a row of instruments.

---

## 4. Calibration knobs

| Knob | File | Default | Effect |
|---|---|---|---|
| `OWL_VIT_CONFIDENCE_THRESHOLD` | `.env` / `config.py` | 0.1 | lower → more cells but more noise |
| `OWL_VIT_TEXT_QUERIES` | `image_processor.py` | 20 phrases | controls what OWL-ViT looks for |
| `IMAGE_EXTRACTION_VERSION` | `config.py` | `v7_anchor_first` | bump to force re-processing |
| margin in `_build_figure_composites` (`page_height * 0.42` vertical, `1.0 + 1.4` weights) | `image_processor.py` | tuned for A4@150dpi | adjust if a new book uses a different paper size |
| `max_gap = page_height * 0.055` in `_build_info_panels` | `image_processor.py` | 5.5% | raise if a panel has unusually large internal padding |
| `_dedupe_visual_regions` `iou_threshold` | `image_processor.py` | 0.55 (visual), 0.65 (tool cells) | lower → more merging |

---

## 5. Known limitations
- Tesseract sometimes garbles the small "Em có biết" header (e.g. page 22).
  When OCR fails on the anchor, the panel is not detected. Mitigations:
  upgrade the OCR (paddle-OCR / VietOCR), or maintain a per-PDF override map.
- Line-drawing figures (rulers, microscopes in pencil sketch) are below
  OWL-ViT's `0.1` confidence for some queries. If a new book has many of
  these, add `"a black and white line drawing"` to `OWL_VIT_TEXT_QUERIES`.
- `_split_merged_figure_caption` partitions the merged line linearly by
  character count. Works for normal Vietnamese spacing; if a book uses
  exotic kerning, re-OCR the page with `--psm 4` (single column) and feed
  results back.

---

## 6. Smoke-test pages (per book)

When validating a new run, always inspect these layouts at minimum:

| Page type | Example in SGK KHTN 6 CD |
|---|---|
| Composite figure with a–g sub-labels | page 6 (Hình 1.1) |
| Dashed-border tool grid w/o caption | page 13 |
| Two captions stacked (single + composite) | page 22 (Hình 3.3 + 3.4) |
| Side-by-side captions on same OCR line | page 85 (Hình 14.1 + 14.2) |
| Bảng X.Y to be rejected | page 55 (Bảng 9.1), page 70 (Bảng 12.1) |
| Em có biết panel with embedded portrait | page 6 (Marie Curie) |

For each new book, identify the analogous pages and rebuild the
smoke-test list before running the full pipeline.
