"""Manual review utilities for image metadata quality control."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from ..config import IMAGE_REVIEW_MANIFEST_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ImageReviewManager:
    """Export/apply/upsert human review updates for extracted images."""

    UPSERTABLE_FIELDS = {
        "pdf_filename",
        "page_number",
        "image_path",
        "page_snapshot_path",
        "bbox",
        "figure_label",
        "figure_caption",
        "caption_vi",
        "keywords_vi",
        "caption",
        "section_title",
        "image_type",
        "context_text",
        "lesson_title",
        "crop_text",
        "nearby_text",
    }

    def __init__(self, manifest_path: Path = IMAGE_REVIEW_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path)

    def _read_manifest_records(self) -> Dict[str, dict]:
        if not self.manifest_path.exists():
            return {}

        records: Dict[str, dict] = {}
        try:
            for line in self.manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                image_id = str(payload.get("image_id") or "").strip()
                if not image_id:
                    continue
                records[image_id] = payload
        except Exception as e:
            logger.warning(f"Could not read image review manifest: {e}")
        return records

    def _write_manifest_records(self, records: Dict[str, dict]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_records = sorted(
            records.values(),
            key=lambda row: (
                str(row.get("pdf_filename") or ""),
                int(row.get("page_number") or 0),
                str(row.get("image_path") or ""),
            ),
        )
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for record in sorted_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_search_text(self, metadata: dict) -> str:
        caption_text = metadata.get("caption_vi_manual", "") or metadata.get("caption_vi", "") or metadata.get("caption", "")
        keywords_text = metadata.get("keywords_vi_manual", "") or metadata.get("keywords_vi", "")
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
        return "\n".join(str(part).strip() for part in parts if str(part).strip())

    def _normalize_filesystem_paths(self, record: dict) -> None:
        for field in ("image_path", "page_snapshot_path"):
            raw_path = str(record.get(field) or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            record[field] = str(path.resolve())

    def _to_bool(self, value, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "active"}:
                return True
            if lowered in {"0", "false", "no", "n", "inactive"}:
                return False
        return default

    def _build_review_row(self, record: dict) -> dict:
        review_status = str(record.get("review_status") or "pending")
        return {
            "image_id": record.get("image_id", ""),
            "pdf_filename": record.get("pdf_filename", ""),
            "page_number": record.get("page_number", ""),
            "image_path": record.get("image_path", ""),
            "page_snapshot_path": record.get("page_snapshot_path", ""),
            "bbox": record.get("bbox", ""),
            "figure_label": record.get("figure_label", ""),
            "figure_caption": record.get("figure_caption", ""),
            "caption_vi": record.get("caption_vi", ""),
            "keywords_vi": record.get("keywords_vi", ""),
            "caption_vi_manual": record.get("caption_vi_manual", ""),
            "keywords_vi_manual": record.get("keywords_vi_manual", ""),
            "review_status": review_status or "pending",
            "is_active": self._to_bool(record.get("is_active", True), default=True),
            "review_notes": record.get("review_notes", ""),
        }

    def _apply_update_to_record(self, record: dict, update: dict, reviewed_by: str, now: str) -> dict:
        for field in self.UPSERTABLE_FIELDS:
            if field in update and update.get(field) is not None:
                record[field] = update.get(field)
        self._normalize_filesystem_paths(record)

        status = str(update.get("review_status") or record.get("review_status") or "pending").strip().lower()
        if update.get("delete") is True:
            status = "rejected"

        record["review_status"] = status
        record["review_notes"] = str(update.get("review_notes") or record.get("review_notes") or "").strip()
        record["reviewed_by"] = str(update.get("reviewed_by") or reviewed_by).strip()
        record["reviewed_at"] = now

        caption_manual = str(update.get("caption_vi_manual") or record.get("caption_vi_manual") or "").strip()
        keywords_manual = str(update.get("keywords_vi_manual") or record.get("keywords_vi_manual") or "").strip()
        record["caption_vi_manual"] = caption_manual
        record["keywords_vi_manual"] = keywords_manual

        default_active = self._to_bool(record.get("is_active", True), default=True)
        is_active = self._to_bool(update.get("is_active"), default=default_active)
        if status in {"rejected", "deleted"}:
            is_active = False
        if status in {"approved", "edited"} and not is_active:
            is_active = True
        record["is_active"] = is_active

        record["final_caption_vi"] = caption_manual or str(record.get("caption_vi") or record.get("caption") or "")
        record["final_keywords_vi"] = keywords_manual or str(record.get("keywords_vi") or "")
        record["search_text"] = self._build_search_text(record)
        record["updated_at"] = now
        return record

    def _sync_changed_records(self, changed: Dict[str, dict], deleted_ids: List[str]) -> int:
        if not changed and not deleted_ids:
            return 0

        from ..rag.image_vectorstore import ImageVectorDB

        vdb = ImageVectorDB()
        if deleted_ids:
            vdb.delete_documents(deleted_ids)

        docs_to_upsert = [
            Document(page_content=str(row.get("search_text") or ""), metadata=row)
            for image_id, row in changed.items()
            if image_id not in set(deleted_ids)
        ]
        if docs_to_upsert:
            vdb.add_documents(docs_to_upsert)
        return len(docs_to_upsert)

    def _is_indexable_record(self, record: dict) -> bool:
        if not self._to_bool(record.get("is_active"), default=True):
            return False
        status = str(record.get("review_status") or "").strip().lower()
        return status not in {"rejected", "deleted"}

    def _sync_replaced_records(self, previous_ids: set[str], records: Dict[str, dict]) -> int:
        from ..rag.image_vectorstore import ImageVectorDB

        vdb = ImageVectorDB()
        ids_to_clear = sorted(previous_ids | set(records.keys()))
        if ids_to_clear:
            vdb.delete_documents(ids_to_clear)

        docs_to_upsert = [
            Document(page_content=str(row.get("search_text") or ""), metadata=row)
            for row in records.values()
            if self._is_indexable_record(row)
        ]
        if docs_to_upsert:
            vdb.add_documents(docs_to_upsert)
        return len(docs_to_upsert)

    def _normalize_replacement_record(self, item: dict, reviewed_by: str, now: str) -> dict:
        record = dict(item)
        self._normalize_filesystem_paths(record)
        status = str(record.get("review_status") or "pending").strip().lower()
        if self._to_bool(record.pop("delete", False), default=False):
            status = "rejected"

        record["review_status"] = status
        record["review_notes"] = str(record.get("review_notes") or "").strip()
        record["reviewed_by"] = str(record.get("reviewed_by") or reviewed_by).strip()
        record["reviewed_at"] = now

        caption_manual = str(record.get("caption_vi_manual") or "").strip()
        keywords_manual = str(record.get("keywords_vi_manual") or "").strip()
        record["caption_vi_manual"] = caption_manual
        record["keywords_vi_manual"] = keywords_manual

        is_active = self._to_bool(record.get("is_active"), default=True)
        if status in {"rejected", "deleted"}:
            is_active = False
        record["is_active"] = is_active

        record["final_caption_vi"] = caption_manual or str(record.get("caption_vi") or record.get("caption") or "")
        record["final_keywords_vi"] = keywords_manual or str(record.get("keywords_vi") or "")
        record["search_text"] = self._build_search_text(record)
        record["updated_at"] = now
        return record

    def export_for_review(
        self,
        output_path: str,
        pdf_filename: Optional[str] = None,
        only_pending: bool = True,
    ) -> int:
        records = self._read_manifest_records()
        rows = []
        for record in records.values():
            if pdf_filename and str(record.get("pdf_filename") or "") != pdf_filename:
                continue

            review_status = str(record.get("review_status") or "pending")
            if only_pending and review_status not in {"pending", ""}:
                continue

            rows.append(self._build_review_row(record))

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Exported {len(rows)} review rows to {output_file}")
        return len(rows)

    def export_db_snapshot(self, output_path: str, pdf_filename: Optional[str] = None) -> int:
        """Export full image metadata records currently stored in manifest DB."""
        records = self._read_manifest_records()
        rows = []
        for row in records.values():
            if pdf_filename and str(row.get("pdf_filename") or "") != pdf_filename:
                continue
            rows.append(dict(row))

        rows.sort(
            key=lambda row: (
                str(row.get("pdf_filename") or ""),
                int(row.get("page_number") or 0),
                str(row.get("image_path") or ""),
            )
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Exported {len(rows)} DB rows to {output_file}")
        return len(rows)

    def get_db_snapshot(self, pdf_filename: Optional[str] = None) -> list[dict]:
        """Get full image metadata records currently stored in manifest DB as a list."""
        records = self._read_manifest_records()
        rows = []
        for row in records.values():
            if pdf_filename and str(row.get("pdf_filename") or "") != pdf_filename:
                continue
            rows.append(dict(row))

        rows.sort(
            key=lambda row: (
                str(row.get("pdf_filename") or ""),
                int(row.get("page_number") or 0),
                str(row.get("image_path") or ""),
            )
        )

        return rows

    def replace_image_db(self, snapshot_path: str, reviewed_by: str = "human") -> dict:
        """Replace manifest records and image vector index from a JSON array snapshot."""
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Image DB snapshot file not found: {snapshot_file}")

        payload = json.loads(snapshot_file.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, list):
            raise ValueError("Image DB replacement file must be a JSON array")

        previous_records = self._read_manifest_records()
        previous_ids = set(previous_records.keys())
        records: Dict[str, dict] = {}
        skipped_count = 0
        now = datetime.now().isoformat()

        for item in payload:
            if not isinstance(item, dict):
                skipped_count += 1
                continue

            image_id = str(item.get("image_id") or "").strip()
            if not image_id:
                skipped_count += 1
                continue
            if image_id in records:
                raise ValueError(f"Duplicate image_id in replacement file: {image_id}")

            record = self._normalize_replacement_record(item, reviewed_by=reviewed_by, now=now)
            record["image_id"] = image_id
            records[image_id] = record

        self._write_manifest_records(records)
        upserted = self._sync_replaced_records(previous_ids, records)

        removed_ids = previous_ids - set(records.keys())
        inactive_count = sum(1 for record in records.values() if not self._is_indexable_record(record))
        summary = {
            "replaced": len(records),
            "removed": len(removed_ids),
            "inactive": inactive_count,
            "upserted": upserted,
            "skipped": skipped_count,
        }
        logger.info(f"Replaced image DB from snapshot: {summary}")
        return summary

    def replace_image_db_from_payload(self, payload: list, reviewed_by: str = "human") -> dict:
        """Replace manifest records and image vector index from a JSON array payload."""
        if not isinstance(payload, list):
            raise ValueError("Image DB replacement payload must be a JSON array")

        previous_records = self._read_manifest_records()
        previous_ids = set(previous_records.keys())
        records: Dict[str, dict] = {}
        skipped_count = 0
        now = datetime.now().isoformat()

        for item in payload:
            if not isinstance(item, dict):
                skipped_count += 1
                continue

            image_id = str(item.get("image_id") or "").strip()
            if not image_id:
                skipped_count += 1
                continue
            if image_id in records:
                raise ValueError(f"Duplicate image_id in replacement payload: {image_id}")

            record = self._normalize_replacement_record(item, reviewed_by=reviewed_by, now=now)
            record["image_id"] = image_id
            records[image_id] = record

        self._write_manifest_records(records)
        upserted = self._sync_replaced_records(previous_ids, records)

        removed_ids = previous_ids - set(records.keys())
        inactive_count = sum(1 for record in records.values() if not self._is_indexable_record(record))
        summary = {
            "replaced": len(records),
            "removed": len(removed_ids),
            "inactive": inactive_count,
            "upserted": upserted,
            "skipped": skipped_count,
        }
        logger.info(f"Replaced image DB from payload: {summary}")
        return summary

    def upsert_review_item(self, item: dict, reviewed_by: str = "human") -> dict:
        """Upsert a single review record by image_id and sync vector DB."""
        if not isinstance(item, dict):
            raise ValueError("Upsert payload must be a JSON object")

        image_id = str(item.get("image_id") or "").strip()
        if not image_id:
            raise ValueError("Upsert payload must contain image_id")

        records = self._read_manifest_records()
        existed = image_id in records
        record = dict(records.get(image_id) or {})

        if not record:
            record = {
                "image_id": image_id,
                "review_status": "pending",
                "is_active": True,
            }

        now = datetime.now().isoformat()
        record = self._apply_update_to_record(record, item, reviewed_by=reviewed_by, now=now)
        records[image_id] = record

        self._write_manifest_records(records)

        deleted_ids = [image_id] if not self._to_bool(record.get("is_active"), default=True) else []
        upserted = self._sync_changed_records({image_id: record}, deleted_ids)

        summary = {
            "updated": 1,
            "created": 0 if existed else 1,
            "deleted": len(deleted_ids),
            "upserted": upserted,
            "image_id": image_id,
        }
        logger.info(f"Upserted image review item: {summary}")
        return summary

    def apply_review_updates(
        self,
        review_path: str,
        reviewed_by: str = "human",
        pdf_filename: Optional[str] = None,
        allow_create: bool = True,
    ) -> dict:
        review_file = Path(review_path)
        if not review_file.exists():
            raise FileNotFoundError(f"Review file not found: {review_file}")

        payload = json.loads(review_file.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, list):
            raise ValueError("Review file must be a JSON array")

        records = self._read_manifest_records()
        changed: Dict[str, dict] = {}
        deleted_ids = set()
        now = datetime.now().isoformat()

        created_count = 0
        skipped_count = 0

        for update in payload:
            if not isinstance(update, dict):
                skipped_count += 1
                continue

            image_id = str(update.get("image_id") or "").strip()
            if not image_id:
                skipped_count += 1
                continue

            existing = records.get(image_id)
            if existing is None and not allow_create:
                skipped_count += 1
                continue

            target_pdf = str((update.get("pdf_filename") or (existing or {}).get("pdf_filename") or "")).strip()
            if pdf_filename and target_pdf != pdf_filename:
                skipped_count += 1
                continue

            if existing is None:
                created_count += 1
                record = {
                    "image_id": image_id,
                    "review_status": "pending",
                    "is_active": True,
                }
                if pdf_filename and not update.get("pdf_filename"):
                    update = {**update, "pdf_filename": pdf_filename}
            else:
                record = dict(existing)

            record = self._apply_update_to_record(record, update, reviewed_by=reviewed_by, now=now)
            records[image_id] = record
            changed[image_id] = record
            if not self._to_bool(record.get("is_active"), default=True):
                deleted_ids.add(image_id)

        if not changed:
            return {
                "updated": 0,
                "created": 0,
                "deleted": 0,
                "upserted": 0,
                "skipped": skipped_count,
            }

        self._write_manifest_records(records)

        deleted_list = sorted(deleted_ids)
        upserted = self._sync_changed_records(changed, deleted_list)

        summary = {
            "updated": len(changed),
            "created": created_count,
            "deleted": len(deleted_list),
            "upserted": upserted,
            "skipped": skipped_count,
        }
        if pdf_filename:
            summary["pdf_filename"] = pdf_filename

        logger.info(f"Applied image review updates: {summary}")
        return summary
