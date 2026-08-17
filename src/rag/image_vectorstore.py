"""Image vector store using CLIP embeddings via sentence-transformers."""

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPModel, CLIPProcessor

from ..config import (
    CLIP_MODEL,
    EMBEDDING_MODEL,
    IMAGE_COLLECTION_NAME,
    IMAGE_METADATA_COLLECTION_NAME,
    IMAGE_METADATA_FETCH_K,
    PERSIST_DIR,
    HF_TOKEN,
    IMAGE_RETRIEVER_K,
    IMAGE_RETRIEVER_FETCH_K,
    IMAGE_RELEVANCE_THRESHOLD,
)
from .query_intent import (
    has_image_intent,
    normalize_query_text,
    normalize_accented_text,
    strip_accents,
)

logger = logging.getLogger(__name__)

VIETNAMESE_TO_ENGLISH_VISUAL_HINTS = {
    "ca": "fish",
    "ca xiem": "betta fish siamese fighting fish",
    "mau": "color colorful",
    "doi mau": "changing color",
    "te bao": "cell",
    "gan": "liver",
    "muoi": "mosquito",
    "sot ret": "malaria",
    "thuc vat": "plant",
    "dong vat": "animal",
    "trau": "water buffalo buffalo cattle",
    "hoa": "flower",
    "hat": "seed",
    "re": "root",
}

# Words that carry no object meaning in an image request ("cho tôi hình ...").
# Stripped when extracting the query's content phrase, but classifier words
# like "con"/"cây" are kept so the phrase "con trâu" stays intact.
PHRASE_FILLER_WORDS = {
    "a", "an", "and", "anh", "are", "as", "can", "chi", "cho", "cua", "co",
    "duoc", "em", "gi", "giup", "hay", "hinh", "image", "la", "lay", "minh",
    "mot", "muon", "nhung", "photo", "picture", "please", "sach", "show",
    "the", "tim", "toi", "trong", "tui", "ve", "what", "which", "with", "xem",
}
# Classifiers kept inside phrases but ignored as standalone match tokens
# (they are too generic on their own: "con", "cây", ...).
CLASSIFIER_WORDS = {"con", "cay", "cai", "cac", "loai", "nhom"}
# Figure-identity fields used for the high-precision phrase boost: the OCR of
# the cropped image region plus the visual model's description of that crop.
# Caption/keyword fields are deliberately excluded here because composite-figure
# captions list several subjects and page-level keyword dumps repeat a term
# across every figure on the page, both of which cause false phrase matches.
PHRASE_IDENTITY_FIELDS = (
    "crop_text", "visual_objects_vi", "visual_caption_vi", "visual_keywords_vi",
)
# Broader set used only to pull candidates into the pool (recall, not ranking).
LEXICAL_IDENTITY_FIELDS = PHRASE_IDENTITY_FIELDS + (
    "figure_label", "figure_caption", "final_caption_vi", "caption",
    "keywords_vi", "caption_vi",
)

class ImageVectorDB:
    """ChromaDB-backed image store with CLIP embeddings for semantic image search."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        collection_name: str = IMAGE_COLLECTION_NAME,
        persist_dir: str = str(PERSIST_DIR),
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.metadata_collection_name = IMAGE_METADATA_COLLECTION_NAME
        self._clip_model: Optional[CLIPModel] = None
        self._clip_processor: Optional[CLIPProcessor] = None
        self._metadata_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {},
        )

        self._init_clip()
        self._init_chroma(documents)

        logger.info(f"ImageVectorDB initialized, collection: {self.collection_name}")

    def _init_clip(self):
        """Initialize CLIP model and processor via sentence-transformers."""
        logger.info(f"Loading CLIP model: {CLIP_MODEL}")
        self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL, token=HF_TOKEN)
        self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL, token=HF_TOKEN)
        self._clip_model.eval()
        logger.info("CLIP model loaded")

    def _init_chroma(self, documents: Optional[List[Document]]):
        """Initialize ChromaDB for image storage."""
        from langchain_chroma import Chroma

        self._chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=_DummyEmbeddingFunction(),
            persist_directory=self.persist_dir,
        )
        self._metadata_chroma = Chroma(
            collection_name=self.metadata_collection_name,
            embedding_function=_DummyMetadataEmbeddingFunction(self._metadata_embedding),
            persist_directory=self.persist_dir,
        )

        if documents:
            self.add_documents(documents)

    def _to_projected_embedding(self, output: Any, projection_name: str) -> torch.Tensor:
        """Extract a tensor embedding across transformers versions."""
        if isinstance(output, torch.Tensor):
            return output

        for attr_name in ("image_embeds", "text_embeds"):
            embedding = getattr(output, attr_name, None)
            if isinstance(embedding, torch.Tensor):
                return embedding

        pooled = getattr(output, "pooler_output", None)
        if isinstance(pooled, torch.Tensor):
            return self._apply_projection_if_needed(pooled, projection_name)

        if isinstance(output, (tuple, list)) and output:
            return self._to_projected_embedding(output[0], projection_name)

        raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")

    def _apply_projection_if_needed(self, embedding: torch.Tensor, projection_name: str) -> torch.Tensor:
        """Project raw CLIP hidden states only when their shape matches the projection layer."""
        projection = getattr(self._clip_model, projection_name, None)
        if projection is None:
            return embedding

        embedding_dim = embedding.shape[-1]
        in_features = getattr(projection, "in_features", None)
        out_features = getattr(projection, "out_features", None)

        if in_features is not None and embedding_dim == in_features:
            return projection(embedding)

        if out_features is not None and embedding_dim == out_features:
            return embedding

        logger.debug(
            "Skipping %s for CLIP embedding with dim=%s (expected input=%s, output=%s)",
            projection_name,
            embedding_dim,
            in_features,
            out_features,
        )
        return embedding

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize CLIP embeddings so vector distance maps more closely to cosine similarity."""
        embedding = embedding.squeeze(0)
        norm = embedding.norm(p=2).clamp(min=1e-12)
        return embedding / norm

    def _encode_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Encode an image to CLIP embedding using PIL."""
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                output = self._clip_model.get_image_features(**inputs)
                embedding = self._to_projected_embedding(output, "visual_projection")
            return self._normalize_embedding(embedding)
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return None

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text query to CLIP embedding.

        CLIP's text encoder caps at 77 positions, so truncate long queries
        (the eval's generated questions can exceed it) instead of erroring.
        """
        inputs = self._clip_processor(
            text=[text], return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        )
        with torch.no_grad():
            output = self._clip_model.get_text_features(**inputs)
            embedding = self._to_projected_embedding(output, "text_projection")
        return self._normalize_embedding(embedding)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Keep metadata Chroma-compatible and predictable."""
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, Path):
                sanitized[key] = str(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = ",".join(str(item) for item in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    def _build_search_text(self, doc: Document) -> str:
        """Combine visual caption and local OCR text for bilingual image retrieval."""
        metadata = doc.metadata or {}
        manual_caption = metadata.get("caption_vi_manual", "")
        manual_keywords = metadata.get("keywords_vi_manual", "")
        final_caption = metadata.get("final_caption_vi", "")
        final_keywords = metadata.get("final_keywords_vi", "")
        parts = [
            doc.page_content,
            metadata.get("figure_label", ""),
            metadata.get("figure_caption", ""),
            manual_caption,
            manual_keywords,
            final_caption,
            final_keywords,
            metadata.get("visual_caption_vi", ""),
            metadata.get("visual_keywords_vi", ""),
            metadata.get("visual_objects_vi", ""),
            metadata.get("visual_scene_vi", ""),
            metadata.get("lesson_title", ""),
            metadata.get("section_title", ""),
            metadata.get("image_type", ""),
            metadata.get("keywords_vi", ""),
            metadata.get("caption_vi", ""),
            metadata.get("caption", ""),
            metadata.get("caption_en", ""),
            metadata.get("context_text", ""),
            metadata.get("crop_text", ""),
            metadata.get("ocr_text", ""),
            metadata.get("pdf_filename", ""),
            f"trang {metadata.get('page_number', '')}",
        ]

        seen = set()
        cleaned_parts = []
        for part in parts:
            text = str(part or "").strip()
            if text and text not in seen:
                cleaned_parts.append(text)
                seen.add(text)

        return "\n".join(cleaned_parts)

    def _doc_id_from_metadata(self, metadata: Dict[str, Any], image_path: str) -> str:
        image_id = str(metadata.get("image_id") or "").strip()
        if image_id:
            return image_id
        return f"{Path(image_path).stem}_{Path(image_path).parent.name}"

    def add_documents(self, documents: List[Document]):
        """Add image documents to the vector store."""
        if not documents:
            return

        metadata_ids = []
        metadata_embeddings = []
        metadata_metadatas = []
        metadata_page_contents = []
        visual_ids = []
        visual_embeddings = []
        visual_metadatas = []
        visual_page_contents = []

        for doc in documents:
            image_path = doc.metadata.get("image_path")
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue

            doc_id = self._doc_id_from_metadata(doc.metadata or {}, image_path)
            search_text = self._build_search_text(doc)
            metadata = self._sanitize_metadata({**doc.metadata, "search_text": search_text})

            metadata_ids.append(doc_id)
            metadata_metadatas.append(metadata)
            metadata_page_contents.append(search_text)

            visual_embedding = self._encode_image(image_path)
            if visual_embedding is None:
                continue

            visual_ids.append(doc_id)
            visual_embeddings.append(visual_embedding.cpu().numpy().tolist())
            visual_metadatas.append(metadata)
            visual_page_contents.append(search_text)

        if metadata_ids:
            metadata_embeddings = self._metadata_embedding.embed_documents(metadata_page_contents)
            self._metadata_chroma._collection.upsert(
                ids=metadata_ids,
                embeddings=metadata_embeddings,
                metadatas=metadata_metadatas,
                documents=metadata_page_contents,
            )
            logger.info(f"Added {len(metadata_ids)} image metadata docs to ImageVectorDB")

        if visual_ids:
            self._chroma._collection.upsert(
                ids=visual_ids,
                embeddings=visual_embeddings,
                metadatas=visual_metadatas,
                documents=visual_page_contents,
            )
            logger.info(f"Added {len(visual_ids)} visual image docs to ImageVectorDB")

    def delete_documents(self, image_ids: List[str]) -> int:
        """Delete image records from both metadata and visual collections by image_id/doc_id."""
        ids = [str(image_id).strip() for image_id in image_ids if str(image_id).strip()]
        if not ids:
            return 0

        try:
            self._metadata_chroma._collection.delete(ids=ids)
        except Exception as e:
            logger.warning(f"Failed deleting image metadata docs: {e}")

        try:
            self._chroma._collection.delete(ids=ids)
        except Exception as e:
            logger.warning(f"Failed deleting visual image docs: {e}")

        return len(ids)

    # Accent-free stopwords; matched against the accent-stripped form of each
    # token so accented inputs ("hình", "tôi") are still recognised.
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "anh", "cho", "cua", "co", "con",
        "duoc", "gi", "hinh", "la", "mot", "nhung", "the", "tim", "toi",
        "trong", "ve", "xem", "what", "which", "with",
    }

    def _normalize_text(self, text: str) -> str:
        return normalize_query_text(text)

    def _normalize_accented(self, text: str) -> str:
        """Diacritic-preserving normalization for exact Vietnamese matching."""
        return normalize_accented_text(text)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize keeping diacritics, but drop stopwords by their bare form."""
        return [
            token
            for token in self._normalize_accented(text).split()
            if len(token) > 1 and strip_accents(token) not in self._STOPWORDS
        ]

    def _expand_query_for_clip(self, query: str) -> str:
        """Add lightweight English visual hints for CLIP's English-heavy text encoder."""
        normalized_query = self._normalize_text(query)
        hints = []
        for vietnamese_term, english_hint in VIETNAMESE_TO_ENGLISH_VISUAL_HINTS.items():
            if vietnamese_term in normalized_query:
                hints.append(english_hint)

        if not hints:
            return query

        return f"{query}. {' '.join(dict.fromkeys(hints))}"

    def _has_image_intent(self, query: str) -> bool:
        return has_image_intent(query)

    def _field_overlap_score(self, query: str, *texts: str) -> float:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return 0.0

        doc_tokens = set(self._tokenize(" ".join(str(text or "") for text in texts)))
        if not doc_tokens:
            return 0.0

        return len(query_tokens & doc_tokens) / len(query_tokens)

    def _direct_evidence_score(self, query: str, doc: Document) -> float:
        metadata = doc.metadata or {}
        return self._field_overlap_score(
            query,
            metadata.get("figure_label"),
            metadata.get("figure_caption"),
            metadata.get("visual_caption_vi"),
            metadata.get("visual_keywords_vi"),
            metadata.get("visual_objects_vi"),
            metadata.get("visual_scene_vi"),
            metadata.get("lesson_title"),
            metadata.get("section_title"),
            metadata.get("caption_vi_manual"),
            metadata.get("keywords_vi_manual"),
            metadata.get("final_caption_vi"),
            metadata.get("final_keywords_vi"),
            metadata.get("keywords_vi"),
            metadata.get("caption_vi"),
            metadata.get("caption"),
            metadata.get("caption_en"),
            metadata.get("context_text"),
        )

    def _lexical_score(self, query: str, doc: Document) -> float:
        metadata = doc.metadata or {}
        direct_score = self._direct_evidence_score(query, doc)
        weak_context_score = self._field_overlap_score(
            query,
            doc.page_content[:700] if doc.page_content else "",
            str(metadata.get("search_text") or "")[:700],
            str(metadata.get("nearby_text") or "")[:350],
        )

        normalized_query = self._normalize_accented(query).strip()
        direct_text = self._normalize_accented(
            " ".join(
                str(metadata.get(field) or "")
                for field in (
                    "figure_label",
                    "figure_caption",
                    "visual_caption_vi",
                    "visual_keywords_vi",
                    "visual_objects_vi",
                    "visual_scene_vi",
                    "lesson_title",
                    "section_title",
                    "caption_vi_manual",
                    "keywords_vi_manual",
                    "final_caption_vi",
                    "final_keywords_vi",
                    "keywords_vi",
                    "caption_vi",
                    "caption",
                    "context_text",
                )
            )
        )
        exact_bonus = 0.15 if normalized_query and normalized_query in direct_text else 0.0
        return min(1.0, (direct_score * 0.8) + (weak_context_score * 0.2) + exact_bonus)

    def _image_quality_adjustment(self, doc: Document) -> float:
        metadata = doc.metadata or {}
        adjustment = 0.0

        try:
            width = float(metadata.get("image_width") or 0)
            height = float(metadata.get("image_height") or 0)
            if width > 0 and height > 0:
                aspect_ratio = width / height
                if width < 80 or height < 80:
                    adjustment -= 0.2
                if aspect_ratio > 5.5 or aspect_ratio < 0.18:
                    adjustment -= 0.18
        except (TypeError, ValueError):
            pass

        try:
            visual_content_score = float(metadata.get("visual_content_score") or 0)
            if 0 < visual_content_score < 0.015:
                adjustment -= 0.12
            elif visual_content_score >= 0.08:
                adjustment += 0.04
        except (TypeError, ValueError):
            pass

        image_type = str(metadata.get("image_type") or "")
        if image_type == "figure":
            adjustment += 0.04
        elif image_type == "table":
            adjustment -= 0.08
        elif image_type in {"activity_box", "textbook_info_box"}:
            adjustment -= 0.04
        elif image_type == "text_crop":
            adjustment -= 0.35

        crop_text = self._normalize_text(str(metadata.get("crop_text") or ""))
        crop_tokens = [token for token in crop_text.split() if len(token) > 1]
        if len(crop_tokens) >= 2 and len(crop_tokens) <= 12:
            adjustment -= 0.18

        try:
            clip_positive = float(metadata.get("clip_positive_score") or 0)
            clip_negative = float(metadata.get("clip_negative_score") or 0)
            if clip_negative > clip_positive and clip_negative >= 0.58:
                adjustment -= 0.12
        except (TypeError, ValueError):
            pass

        return adjustment

    def _page_key(self, doc: Document) -> Optional[int]:
        page = (doc.metadata or {}).get("page_number")
        try:
            return int(page)
        except (TypeError, ValueError):
            return None

    def _extract_requested_page(self, query: str) -> Optional[int]:
        normalized_query = self._normalize_text(query)
        match = re.search(r"\btrang\s+(\d{1,4})\b", normalized_query)
        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _distance_to_similarity(self, distance: Optional[float]) -> float:
        if distance is None or not math.isfinite(distance):
            return 0.0
        if distance <= 2:
            return max(0.0, 1.0 - (distance / 2.0))
        return max(0.0, 1.0 / (1.0 + distance))

    def _extract_related_pages(self, related_text_docs: Optional[List[Document]]) -> Dict[int, float]:
        page_boosts: Dict[int, float] = {}
        for rank, doc in enumerate(related_text_docs or []):
            metadata = doc.metadata or {}
            page = metadata.get("page") or metadata.get("page_number")
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue

            boost = max(0.12, 0.3 - (rank * 0.05))
            page_boosts[page_number] = max(page_boosts.get(page_number, 0.0), boost)
            page_boosts[page_number - 1] = max(page_boosts.get(page_number - 1, 0.0), boost * 0.45)
            page_boosts[page_number + 1] = max(page_boosts.get(page_number + 1, 0.0), boost * 0.45)
        return page_boosts

    def _docs_from_query_results(
        self,
        results: dict,
        score_field: str,
        distance_field: str,
        source: str,
    ) -> List[Document]:
        docs = []
        ids = results.get("ids", [[]])[0] if results else []
        documents = results.get("documents", [[]])[0] if results else []
        metadatas = results.get("metadatas", [[]])[0] if results else []
        distances = results.get("distances", [[]])[0] if results else []

        for i, doc_id in enumerate(ids):
            metadata = dict(metadatas[i] or {})
            distance = distances[i] if i < len(distances) else None
            metadata["image_doc_id"] = doc_id
            metadata[distance_field] = distance
            metadata[score_field] = self._distance_to_similarity(distance)
            metadata["image_retrieval_source"] = source
            docs.append(Document(page_content=documents[i] or "", metadata=metadata))
        return docs

    def _get_page_candidates(self, page_boosts: Dict[int, float], source: str = "page_context") -> List[Document]:
        if not page_boosts:
            return []

        docs = []
        for page_number in sorted(page_boosts):
            if page_number <= 0:
                continue
            try:
                results = self._metadata_chroma._collection.get(
                    where={"page_number": page_number},
                    include=["documents", "metadatas"],
                )
            except Exception as e:
                logger.debug(f"Could not fetch image candidates for page {page_number}: {e}")
                continue

            for i, doc_id in enumerate(results.get("ids", [])):
                metadata = dict((results.get("metadatas") or [])[i] or {})
                metadata["image_doc_id"] = doc_id
                metadata["image_metadata_distance"] = None
                metadata["image_metadata_score"] = 0.0
                metadata["image_visual_distance"] = None
                metadata["image_visual_score"] = 0.0
                metadata["image_retrieval_source"] = source
                docs.append(Document(page_content=(results.get("documents") or [""])[i] or "", metadata=metadata))
        return docs

    def _page_boost(self, doc: Document, page_boosts: Dict[int, float]) -> float:
        page = (doc.metadata or {}).get("page_number")
        try:
            return page_boosts.get(int(page), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _rerank(
        self,
        query: str,
        candidates: List[Document],
        page_boosts: Dict[int, float],
        k: int,
        min_score: float,
    ) -> List[Document]:
        unique: Dict[str, Document] = {}
        for doc in candidates:
            metadata = doc.metadata or {}
            dedupe_key = metadata.get("image_doc_id") or metadata.get("image_path") or doc.page_content
            if dedupe_key not in unique:
                unique[dedupe_key] = doc
                continue

            existing = unique[dedupe_key]
            merged_metadata = {**(existing.metadata or {})}
            for key, value in metadata.items():
                if key in {"image_metadata_score", "image_visual_score", "image_page_boost", "image_relevance_score"}:
                    merged_metadata[key] = max(float(merged_metadata.get(key) or 0.0), float(value or 0.0))
                elif key not in merged_metadata or merged_metadata[key] in (None, "", 0):
                    merged_metadata[key] = value

            sources = {
                source
                for source in (
                    str(merged_metadata.get("image_retrieval_source") or ""),
                    str(metadata.get("image_retrieval_source") or ""),
                )
                if source
            }
            if sources:
                merged_metadata["image_retrieval_source"] = ",".join(sorted(sources))
            page_content = existing.page_content if len(existing.page_content) >= len(doc.page_content) else doc.page_content
            unique[dedupe_key] = Document(page_content=page_content, metadata=merged_metadata)

        scored_docs = []
        has_image_intent = self._has_image_intent(query)
        for doc in unique.values():
            metadata = dict(doc.metadata or {})
            if metadata.get("is_active") is False:
                continue
            if str(metadata.get("review_status") or "").lower() in {"rejected", "deleted"}:
                continue
            metadata_score = float(metadata.get("image_metadata_score") or 0.0)
            visual_score = float(metadata.get("image_visual_score") or 0.0)
            lexical_score = self._lexical_score(query, doc)
            direct_evidence_score = self._direct_evidence_score(query, doc)
            phrase_score = self._phrase_match_score(query, doc)
            page_score = self._page_boost(doc, page_boosts)
            quality_adjustment = self._image_quality_adjustment(doc)
            if phrase_score > 0:
                # The queried object literally appears on this figure (its OCR
                # label / caption), so don't let the generic "looks like a text
                # crop" penalty bury an otherwise exact match.
                quality_adjustment = max(quality_adjustment, 0.0)
            final_score = (
                (metadata_score * 0.35)
                + (lexical_score * 0.35)
                + (visual_score * 0.12)
                + (phrase_score * 0.45)
                + page_score
                + quality_adjustment
            )

            metadata["image_phrase_score"] = round(phrase_score, 4)
            metadata["image_lexical_score"] = round(lexical_score, 4)
            metadata["image_direct_evidence_score"] = round(direct_evidence_score, 4)
            metadata["image_quality_adjustment"] = round(quality_adjustment, 4)
            metadata["image_page_boost"] = round(page_score, 4)
            metadata["image_relevance_score"] = round(final_score, 4)
            scored_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        scored_docs.sort(key=lambda doc: doc.metadata.get("image_relevance_score", 0.0), reverse=True)
        top_score = scored_docs[0].metadata.get("image_relevance_score", 0.0) if scored_docs else 0.0
        has_page_request = any(
            "requested_page" in str((doc.metadata or {}).get("image_retrieval_source") or "")
            for doc in scored_docs
        )
        score_window = 0.14 if has_image_intent else 0.08
        if has_page_request:
            score_window = 0.32
            effective_min_score = 0.18
        else:
            effective_min_score = min_score if has_image_intent else max(min_score, 0.48)
        per_page_limit = 3 if has_image_intent else 1
        if has_page_request:
            per_page_limit = max(k, 3)

        filtered = []
        page_counts: Dict[int, int] = {}
        for doc in scored_docs:
            metadata = doc.metadata or {}
            relevance_score = float(metadata.get("image_relevance_score") or 0.0)
            direct_evidence_score = float(metadata.get("image_direct_evidence_score") or 0.0)
            lexical_score = float(metadata.get("image_lexical_score") or 0.0)
            source = str(metadata.get("image_retrieval_source") or "")

            if relevance_score < effective_min_score:
                continue
            if top_score and relevance_score < top_score - score_window:
                continue
            if (
                "requested_page" not in source
                and "page_context" in source
                and direct_evidence_score < 0.25
                and (not has_image_intent or lexical_score < 0.2)
            ):
                continue
            if not has_page_request and not has_image_intent and direct_evidence_score < 0.3 and lexical_score < 0.35:
                continue

            page = self._page_key(doc)
            if page is not None:
                if page_counts.get(page, 0) >= per_page_limit:
                    continue
                page_counts[page] = page_counts.get(page, 0) + 1

            filtered.append(doc)
            if len(filtered) >= k:
                break

        logger.info(
            "Image retrieval candidates=%s, kept=%s, threshold=%.2f, effective_threshold=%.2f, image_intent=%s, top_scores=%s",
            len(scored_docs),
            len(filtered),
            min_score,
            effective_min_score,
            has_image_intent,
            [doc.metadata.get("image_relevance_score") for doc in scored_docs[: min(5, len(scored_docs))]],
        )
        return filtered

    def _query_content_phrases(self, query: str):
        """Return (phrases, match_tokens) describing the object the user wants.

        ``phrases`` are 2+ word contiguous content phrases (e.g. "con trau") used
        for high-precision matching; ``match_tokens`` are the individual content
        tokens minus generic classifiers, used for candidate recall.
        """
        # Keep diacritics in the tokens themselves (so "trâu" != "trầu"), but
        # decide filler/classifier membership via the accent-free bare form.
        tokens = [
            token
            for token in self._normalize_accented(query).split()
            if len(token) > 1 and strip_accents(token) not in PHRASE_FILLER_WORDS
        ]
        phrases = []
        if len(tokens) >= 2:
            phrases.append(" ".join(tokens))
            phrases.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
        phrases = sorted(
            {p for p in phrases if len(p.split()) >= 2}, key=lambda p: -len(p)
        )
        match_tokens = [t for t in tokens if strip_accents(t) not in CLASSIFIER_WORDS]
        return phrases, match_tokens

    def _identity_text(self, metadata: Dict[str, Any], fields) -> str:
        return self._normalize_accented(
            " ".join(str(metadata.get(field) or "") for field in fields)
        )

    def _phrase_match_score(self, query: str, doc: Document) -> float:
        """1.0 when the query's content phrase appears in the figure's own label/caption."""
        phrases, _ = self._query_content_phrases(query)
        if not phrases:
            return 0.0
        identity_text = f" {self._identity_text(doc.metadata or {}, PHRASE_IDENTITY_FIELDS)} "
        return 1.0 if any(f" {phrase} " in identity_text for phrase in phrases) else 0.0

    def _get_lexical_candidates(self, query: str, limit: int = 30) -> List[Document]:
        """Find images whose OCR label / caption literally contains the query object.

        CLIP and the multilingual metadata embedding both miss small scanned
        sub-figures (e.g. a buffalo crop labelled "Con trâu"), so this exact-text
        channel guarantees such images enter the candidate pool.
        """
        phrases, match_tokens = self._query_content_phrases(query)
        if not phrases and not match_tokens:
            return []

        try:
            snapshot = self._metadata_chroma._collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.debug(f"Lexical candidate fetch failed: {e}")
            return []

        ids = snapshot.get("ids", []) or []
        metadatas = snapshot.get("metadatas", []) or []
        documents = snapshot.get("documents", []) or []

        scored = []
        for i, doc_id in enumerate(ids):
            metadata = dict(metadatas[i] or {})
            phrase_text = f" {self._identity_text(metadata, PHRASE_IDENTITY_FIELDS)} "
            recall_text = f" {self._identity_text(metadata, LEXICAL_IDENTITY_FIELDS)} "
            phrase_hit = any(f" {phrase} " in phrase_text for phrase in phrases)
            token_hits = sum(1 for token in match_tokens if f" {token} " in recall_text)
            if not phrase_hit and token_hits == 0:
                continue

            metadata["image_doc_id"] = doc_id
            metadata["image_metadata_distance"] = None
            metadata["image_metadata_score"] = 0.0
            metadata["image_visual_distance"] = None
            metadata["image_visual_score"] = 0.0
            metadata["image_retrieval_source"] = "lexical"
            scored.append((
                (1 if phrase_hit else 0, token_hits),
                Document(page_content=documents[i] or "", metadata=metadata),
            ))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    def similarity_search(
        self,
        query: str,
        k: int = IMAGE_RETRIEVER_K,
        related_text_docs: Optional[List[Document]] = None,
        fetch_k: int = IMAGE_RETRIEVER_FETCH_K,
        min_score: float = IMAGE_RELEVANCE_THRESHOLD,
    ) -> List[Document]:
        """Find relevant images using Vietnamese metadata first, with visual CLIP as rerank support."""
        try:
            requested_page = self._extract_requested_page(query)
            if requested_page and self._has_image_intent(query):
                page_boosts = {requested_page: 0.42}
                page_candidates = self._get_page_candidates(page_boosts, source="requested_page")
                return self._rerank(query, page_candidates, page_boosts, k, 0.0)

            metadata_query_embedding = self._metadata_embedding.embed_query(query)
            metadata_results = self._metadata_chroma._collection.query(
                query_embeddings=[metadata_query_embedding],
                n_results=max(k, IMAGE_METADATA_FETCH_K),
                include=["documents", "metadatas", "distances"],
            )

            clip_query = self._expand_query_for_clip(query)
            query_embedding = self._encode_text(clip_query).cpu().numpy().tolist()

            visual_results = self._chroma._collection.query(
                query_embeddings=[query_embedding],
                n_results=max(k, fetch_k),
                include=["documents", "metadatas", "distances"],
            )

            page_boosts = self._extract_related_pages(related_text_docs)
            candidates = self._docs_from_query_results(
                metadata_results,
                score_field="image_metadata_score",
                distance_field="image_metadata_distance",
                source="metadata",
            )
            candidates.extend(
                self._docs_from_query_results(
                    visual_results,
                    score_field="image_visual_score",
                    distance_field="image_visual_distance",
                    source="visual",
                )
            )
            candidates.extend(self._get_page_candidates(page_boosts))
            candidates.extend(self._get_lexical_candidates(query))
            return self._rerank(query, candidates, page_boosts, k, min_score)
        except Exception as e:
            logger.error(f"Image similarity search failed: {e}")
            return []

    def get_retriever(self, search_kwargs: dict = None):
        """Return a retriever for image search."""
        if search_kwargs is None:
            search_kwargs = {"k": IMAGE_RETRIEVER_K}

        return ImageRetriever(self, search_kwargs)


class _DummyEmbeddingFunction:
    """Dummy embedding function since we pre-compute CLIP embeddings manually."""

    def embed_documents(self, texts):
        return [[0.0] * 512 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 512


class _DummyMetadataEmbeddingFunction:
    """Adapter used when querying the raw Chroma collection with precomputed embeddings."""

    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.embedding_model = embedding_model

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)


class ImageRetriever:
    """Retriever wrapper for ImageVectorDB with CLIP similarity."""

    def __init__(self, image_db: ImageVectorDB, search_kwargs: dict):
        self.image_db = image_db
        self.k = search_kwargs.get("k", IMAGE_RETRIEVER_K)
        self.fetch_k = search_kwargs.get("fetch_k", IMAGE_RETRIEVER_FETCH_K)
        self.min_score = search_kwargs.get("min_score", IMAGE_RELEVANCE_THRESHOLD)

    def invoke(self, query: str, related_text_docs: Optional[List[Document]] = None) -> List[Document]:
        """Retrieve relevant images for a text query."""
        return self.image_db.similarity_search(
            query,
            k=self.k,
            related_text_docs=related_text_docs,
            fetch_k=self.fetch_k,
            min_score=self.min_score,
        )
