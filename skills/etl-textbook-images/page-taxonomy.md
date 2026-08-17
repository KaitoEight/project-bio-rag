# Page taxonomy — Vietnamese SGK image ETL

Reference card listing every layout the v9 detector currently understands,
the anchor signal it relies on, and the expected output regions.

Use this list to sanity-check QA overlays when validating a new textbook.
Per-variant differences (CD / CTST / KNTT) are in `SKILL.md` §v9.

---

## A. Hình composite (caption + sub-figures with a/b/c labels)

**Anchor:** OCR line starting with `Hình X.Y` (strict regex
`FIG_CAPTION_STRICT_REGEX`).

**Required structure:**
- Caption sits BELOW the figure(s).
- ≥2 OWL-ViT cells assigned to the same caption.
- ≥1 sub-figure label `a)`/`b)`/… inside the composite bbox.

**Output:**
- 1 × `composite_figure` (bbox = cells union ∪ caption row).
- N × `sub_figure` (one per cell + its label slice).

**Example:** page 6 (Hình 1.1, six activity photos a–g).

## B. Hình single (caption + one illustration)

**Anchor:** `Hình X.Y` caption with no sub-labels in its area.

**Output:** 1 × `single_figure`.

**Example:** page 30 (Hình 4.4 thermometers), page 85 (Hình 14.1 and 14.2 side-by-side).

## C. Hình side-by-side (two captions on the same OCR line)

**Anchor:** A single OCR line carrying two `Hình X.Y` markers — split
linearly by character index into two anchors.

**Output:** 1 × `single_figure` per marker. Each caption claims the
visual cell whose horizontal centre is closest.

**Example:** page 85. *Limitation:* if OWL-ViT returns ONE wide cell covering
both photos, they merge into a single crop (CD KHTN7 p9 Hình 2+3).

## C-bis. Caption ABOVE the figure (KNTT pill labels)

**Anchor:** a `Hình X.Y` pill caption that sits ABOVE its figure / sub-figure
row instead of below it. Enabled per-variant with `_FIG_CAPTION_ABOVE_OK=True`.

**Output:** `composite_figure` spanning caption + the cell row below it, then
split into `sub_figure` crops.

**Example:** KNTT KHTN6 p109 (Hình 32.1 above the a/b/c/d mushroom row).

## C-ter. Photo whose caption OCR was lost

**Anchor:** none survives page-level OCR (e.g. CTST `▲ Hình 40.3` mangled to
`Aình403` and merged with the other column). Recovered by re-OCRing the strip
directly below each detected photo (`_recover_captions_below_photos`).

**Output:** 1 × `single_figure`. **Example:** CTST KHTN6 p174 (Hình 40.3).

## D. Info-box panel (Em có biết / Tìm hiểu thêm / …)

**Anchor:** OCR line starting with one of:
- `Em có biết` → `textbook_info_box`
- `Tìm hiểu thêm`, `Mở rộng`, `Kiến thức mới`,
  `Thực hành`, `Vận dụng`, `Luyện tập` → `activity_box`

Must be at line start AND followed by ≤30 chars (header, not body).

**v9 drop rule:** when the variant sets `_INFO_REQUIRE_VISUAL=True` (CTST), a
panel is kept ONLY if it has a coloured background or an embedded picture
(`visual_content_score ≥ _INFO_MIN_VIS`). Bare section headers on white
("Tìm hiểu về …", "Thí nghiệm 1:") are body text and are dropped.

**Bbox:** grows DOWN through continuous text lines (max gap 5.5% page H)
and widens horizontally to pull in any OWL-ViT visual cell that sits in
the same y-band (this is how the Marie Curie portrait gets included on
page 6 even though it sits to the side of the panel text).

**Stops at:** another anchor (figure / table / info title) below, or a
text gap > max_gap.

## E. Table (REJECTED)

**Anchor:** OCR line starting with `Bảng X.Y`.

Caption + the row block below it becomes a **table_zone** which is used
ONLY to filter out:
- visual cells that fall inside it (otherwise they'd be wrongly assigned
  to a nearby figure caption).
- info-box panel scans that would otherwise leak into the table area.

**Output:** none. Tables are never emitted.

## F. Tool group (dashed border + label-only fallback)

**Anchor:** OCR line matching `TOOL_GROUP_LABEL_REGEX`:
- `Dụng cụ đo …` / `Dụng cụ trong …`
- `Một số dụng cụ`
- `Hộp dụng cụ`
- `Chuẩn bị:` / `Tiến hành:` (lab activity sections — see notes below)

**Required structure:**
- ≥2 OWL-ViT cells in the row directly below the label (≤10% page H gap).
- Label must NOT sit inside any other anchor zone.

**Bbox:** label ∪ row cells ∪ per-instrument text labels below ∪ outer
dashed-border bbox (when present).

**Output:** 1 × `tool_group` per label.

**Example:** page 13 ("Dụng cụ đo chiều dài" with 3 rulers, "Dụng cụ đo
khối lượng" with 4 scales — neither has a sub-figure a/b/c label).

> Note: `Chuẩn bị:` / `Tiến hành:` labels can introduce instrument rows OR
> step-by-step procedure photos that aren't really "tools". The current
> heuristic accepts both; if a new book over-emits these, restrict the
> regex to the noun-phrase variants only.

## G. Question prompt (REJECTED)

**Anchor:** lines matching `QUESTION_PROMPT_PATTERNS` ("Hãy quan sát",
"Hãy tìm", "Em đã thấy", "Thảo luận", "Em hãy …").

Used as a **ceiling** when growing composite bboxes UPWARD so the prompt
is not absorbed into the figure crop, and as a textual stop signal when
growing info-boxes / tables.

**Output:** none.

## H. Out-of-scope (currently silently dropped)

These appear in textbooks but the v7 rules intentionally skip them:
- Free-standing illustrations with no caption (e.g. decorative icons,
  lightbulb-icon next to a question).
- Step-by-step procedure photos without "Chuẩn bị:" / "Tiến hành:" labels.
- Tables of plant photos (the "table" exclusion masks the whole grid).

If any of these turn out to be valuable for downstream RAG, add a new
rule following the same anchor-first pattern.

---

## Anchor cheatsheet

| Line text (Vietnamese) | Anchor bucket | Action |
|---|---|---|
| `Hình 1.1. Một số …` | `figure_captions` | build composite/single |
| `Bảng 9.1. Tên …` | `table_captions` | build exclusion zone |
| `Em có biết` | `info_titles` (label=`textbook_info_box`) | build info panel |
| `Tìm hiểu thêm` | `info_titles` (label=`activity_box`) | build info panel |
| `Em có thể:` | (ignored) | learning objectives — NOT a panel |
| `a) Tìm hiểu vi khuẩn …` | `sub_labels` | drive sub-figure splitting |
| `Hãy quan sát hình 1.1` | `question_prompts` | composite ceiling, then drop |
| `Dụng cụ đo chiều dài` | `tool_group_labels` | build tool group |
| `Trong phòng thực hành` (in body) | (ignored) | NOT an info-title |
