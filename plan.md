# Metadata-aware image captioning plan

## Goal

Improve image captioning so `IMAGE_CAPTION_MODEL` sees both the crop image and useful page metadata/OCR context. This should help cases where the visual crop shows fish/coral and nearby text says "Dai duong", so queries such as "hinh anh ca o dai duong" can retrieve the right image.

## Checklist

- [x] Verify current flow: captioner only receives the crop image and image hash.
- [x] Add a context-aware caption prompt that includes page title, figure caption, local OCR, nearby page text, and page/file identifiers.
- [x] Include context in the caption cache key so old image-only captions are not reused for context-aware captions.
- [x] Reorder PDF image ETL so context is built before captioning.
- [x] Keep local image import compatible when no PDF context exists.
- [x] Bump `IMAGE_EXTRACTION_VERSION` default so processed pages can be recaptioned with context-aware metadata.
- [x] Validate with a small syntax/prompt check.
- [ ] Run a focused ETL smoke test when local model/runtime dependencies are available.

## Design Notes

- Visual captioning should still avoid hallucination: use context to name textbook concepts and scene terms, but keep the visible-object description grounded in the image.
- Search quality comes from both generated caption fields and the existing metadata fields in `search_text`.
- The cache key must include a context version and context fingerprint; otherwise an old image-only caption can hide the improved flow.

---

# OWL-ViT image-region detection plan

## Goal

Replace manual OpenCV contour bounding-box discovery and the extra CLIP filtering phase with open-vocabulary object detection using OWL-ViT. This should produce more precise crops for complex textbook assets such as formulas, diagrams, tables, charts, framed illustrations, and scientific photos.

## Checklist

- [x] Add OWL-ViT config defaults for model name and confidence threshold.
- [x] Replace `CLIPModel` / `CLIPProcessor` initialization in `ImageProcessor` with `OwlViTForObjectDetection` / `OwlViTProcessor`.
- [x] Add `_detect_regions_with_owlvit(image, text_queries)` to run zero-shot object detection and return `(x0, y0, x1, y1)` boxes.
- [x] Replace `_detect_contour_regions` behavior so it no longer uses OpenCV contours for bbox discovery.
- [x] Update `extract_images_from_pdf()` to use OWL-ViT detections and skip the old Phase 3 CLIP filter.
- [x] Preserve `image_path`, `page_snapshot_path`, metadata schema, OCR/context, and captioning flow.
- [x] Bump `IMAGE_EXTRACTION_VERSION` default so pages are reprocessed with OWL-ViT crops.
- [x] Validate with compile/import checks.
- [x] Run focused OWL-ViT detection smoke against `datasources/SGK_KHTN_6_CD_sample_pages_34-36.pdf`.
- [ ] Run full image ETL against the sample PDF with captioning enabled.

## Detection Queries

Default OWL-ViT prompts target textbook visual regions:

- `a scientific formula`
- `a biology diagram`
- `a textbook illustration`
- `a data table`
- `a chart or graph`
- `a science experiment setup`
- `a microscope image`
- `a photo in a textbook`
- `a framed textbook picture`
- `a material sample photo`
- `an object photo`
- `a framed object photo`
- `a product-style object photo`
