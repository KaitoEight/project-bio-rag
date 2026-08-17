"""Import local images bypassing PDF extraction."""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm
from langchain_core.documents import Document

from src.config import IMAGE_REVIEW_MANIFEST_PATH, IMAGE_EXTRACTION_VERSION
from src.etl.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)

class LocalImageImporter:
    def __init__(self):
        self.captioner = ImageCaptioner()
        self.review_manifest_path = IMAGE_REVIEW_MANIFEST_PATH
        self.image_extraction_version = IMAGE_EXTRACTION_VERSION

    def _compute_image_hash(self, image_data: bytes) -> str:
        return hashlib.md5(image_data).hexdigest()

    def _build_image_id(self, pdf_hash: str, page_num: int, filename: str, image_hash: str) -> str:
        payload = f"{pdf_hash}:{page_num}:{filename}:{image_hash}"
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

    def import_directory(self, source_dir: str) -> List[Document]:
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            logger.error(f"Directory not found: {source_dir}")
            return []

        pdf_filename = f"{source_path.name}.pdf"
        pdf_hash = hashlib.md5(pdf_filename.encode("utf-8")).hexdigest()
        
        extracted_docs = []
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            image_files.extend(source_path.glob(ext))
            
        logger.info(f"Found {len(image_files)} images in {source_dir}")

        for filepath in tqdm(image_files, desc=f"[{source_path.name}] Processing local images"):
            filename = filepath.name
            
            page_match = re.search(r"page_(\d+)", filename, re.IGNORECASE)
            page_number = int(page_match.group(1)) if page_match else 0
            
            try:
                with Image.open(filepath) as img:
                    crop = img.convert("RGB")
                    img_bytes = filepath.read_bytes()
                    img_hash = self._compute_image_hash(img_bytes)
            except Exception as e:
                logger.warning(f"Could not read image {filepath}: {e}")
                continue

            image_id = self._build_image_id(pdf_hash, page_number, filename, img_hash)
            
            caption_context = {
                "pdf_filename": pdf_filename,
                "page_number": page_number,
                "image_type": "image_region",
            }
            visual_metadata = self.captioner.caption(
                crop, img_hash, context=caption_context)
            visual_caption = visual_metadata.get("visual_caption_vi", "")
            
            metadata = {
                "image_id": image_id,
                "image_path": str(filepath.absolute()),
                "page_snapshot_path": str((source_path / "pages" / f"page_{page_number}_snapshot.png").absolute()),
                "image_hash": img_hash,
                "page_number": page_number,
                "pdf_hash": pdf_hash,
                "pdf_filename": pdf_filename,
                "lesson_title": "",
                "section_title": "",
                "figure_label": "",
                "figure_caption": "",
                **visual_metadata,
                "image_type": "image_region",
                "keywords_vi": "",
                "caption": visual_caption,
                "caption_vi": visual_caption,
                "caption_vi_manual": "",
                "context_text": "",
                "crop_text": "",
                "nearby_text": "",
                "review_status": "pending",
                "is_active": True,
                "review_notes": "",
                "reviewed_by": "",
                "reviewed_at": "",
                "keywords_vi_manual": "",
                "final_caption_vi": visual_caption,
                "final_keywords_vi": "",
                "extraction_version": self.image_extraction_version,
                "bbox": "",
                "image_width": crop.width,
                "image_height": crop.height,
                "visual_content_score": 1.0,
                "clip_positive_score": 1.0,
                "clip_negative_score": 0.0,
            }
            
            caption_text = metadata.get("caption", "")
            parts = [
                metadata.get("image_type", ""),
                caption_text,
                f"Trang {metadata.get('page_number', '')}",
                metadata.get("pdf_filename", ""),
            ]
            search_text = "\n".join(part.strip() for part in parts if part and part.strip())
            metadata["search_text"] = search_text
            
            self._append_review_manifest(metadata)
            doc = Document(page_content=search_text, metadata=metadata)
            extracted_docs.append(doc)
            
            logger.debug(f"Imported: {filepath.name}")
            
        logger.info(f"Processed {len(extracted_docs)} local images from {source_dir}")
        return extracted_docs
