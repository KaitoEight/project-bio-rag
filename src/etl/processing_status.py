"""Processing status tracking for PDF pages with fine-grained resume support."""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import PERSIST_DIR, STATUS_COLLECTION_NAME, HF_TOKEN, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file for deduplication."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute MD5 hash of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class ProcessingStatus:
    """Track processing status per PDF page for resume-capable ETL."""

    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {},
        )
        self.db = Chroma(
            collection_name=STATUS_COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=str(PERSIST_DIR),
        )
        self._status_cache: Dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load all status records into memory for fast lookup."""
        try:
            results = self.db.get()
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"]):
                    doc = results["documents"][i]
                    self._status_cache[doc_id] = json.loads(doc)
        except Exception:
            self._status_cache = {}

    def _make_doc_id(self, pdf_hash: str, page_number: int) -> str:
        return f"{pdf_hash}_page_{page_number}"

    def get_status(self, pdf_hash: str, page_number: int) -> Optional[dict]:
        """Get processing status for a specific PDF page."""
        doc_id = self._make_doc_id(pdf_hash, page_number)
        return self._status_cache.get(doc_id)

    def needs_text_processing(self, pdf_hash: str, page_number: int) -> bool:
        """Check if a page needs text indexing."""
        status = self.get_status(pdf_hash, page_number)
        return status is None or not status.get("text_indexed", False)

    def needs_image_processing(self, pdf_hash: str, page_number: int) -> bool:
        """Check if a page needs image extraction."""
        return self.needs_image_processing_versioned(pdf_hash, page_number, required_version=None)

    def needs_image_processing_versioned(
        self,
        pdf_hash: str,
        page_number: int,
        required_version: Optional[str] = None,
    ) -> bool:
        """Check if a page needs image extraction for the current extraction version."""
        status = self.get_status(pdf_hash, page_number)
        if status is None or not status.get("image_extracted", False):
            return True

        if not required_version:
            return False

        current_version = str(status.get("image_extraction_version") or "").strip()
        return current_version != required_version

    def update_status(
        self,
        pdf_hash: str,
        page_number: int,
        text_indexed: Optional[bool] = None,
        image_extracted: Optional[bool] = None,
        image_extraction_version: Optional[str] = None,
        pdf_filename: Optional[str] = None,
    ):
        """Update processing status for a PDF page."""
        doc_id = self._make_doc_id(pdf_hash, page_number)
        existing = self.get_status(pdf_hash, page_number) or {}

        updated = {
            "pdf_hash": pdf_hash,
            "page_number": page_number,
            "pdf_filename": pdf_filename or existing.get("pdf_filename"),
            "text_indexed": text_indexed if text_indexed is not None else existing.get("text_indexed", False),
            "image_extracted": image_extracted if image_extracted is not None else existing.get("image_extracted", False),
            "image_extraction_version": image_extraction_version
            if image_extraction_version is not None
            else existing.get("image_extraction_version", ""),
            "last_updated": datetime.now().isoformat(),
        }

        self._status_cache[doc_id] = updated

        from langchain_core.documents import Document

        doc = Document(page_content=json.dumps(updated), metadata={"pdf_hash": pdf_hash, "page": page_number})
        self.db.add_documents([doc], ids=[doc_id])

    def mark_text_indexed(self, pdf_hash: str, page_number: int, pdf_filename: str):
        """Mark text as indexed for a specific page."""
        self.update_status(pdf_hash, page_number, text_indexed=True, pdf_filename=pdf_filename)
        logger.debug(f"[{pdf_filename}] Page {page_number}: text indexed")

    def mark_image_extracted(
        self,
        pdf_hash: str,
        page_number: int,
        pdf_filename: str,
        image_extraction_version: Optional[str] = None,
    ):
        """Mark images as extracted for a specific page."""
        self.update_status(
            pdf_hash,
            page_number,
            image_extracted=True,
            image_extraction_version=image_extraction_version,
            pdf_filename=pdf_filename,
        )
        logger.debug(f"[{pdf_filename}] Page {page_number}: images extracted")

    def get_pages_needing_text(self, pdf_hash: str, total_pages: int) -> List[int]:
        """Get list of page numbers that need text processing."""
        pages = []
        for page_num in range(1, total_pages + 1):
            if self.needs_text_processing(pdf_hash, page_num):
                pages.append(page_num)
        return pages

    def get_pages_needing_images(self, pdf_hash: str, total_pages: int) -> List[int]:
        """Get list of page numbers that need image extraction."""
        pages = []
        for page_num in range(1, total_pages + 1):
            if self.needs_image_processing(pdf_hash, page_num):
                pages.append(page_num)
        return pages

    def get_all_status_for_pdf(self, pdf_hash: str) -> List[dict]:
        """Get all page statuses for a given PDF."""
        return [
            status
            for doc_id, status in self._status_cache.items()
            if doc_id.startswith(f"{pdf_hash}_page_")
        ]
