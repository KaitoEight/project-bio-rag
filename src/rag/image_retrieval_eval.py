"""Lightweight checks for image retrieval quality.

Run after re-indexing images:
    python -m src.rag.image_retrieval_eval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from src.rag.hybrid_retriever import HybridRetriever


@dataclass
class ImageRetrievalCase:
    query: str
    expected_page: Optional[int] = None
    expected_keywords: Optional[List[str]] = None
    max_returned: Optional[int] = None


DEFAULT_CASES = [
    ImageRetrievalCase(
        query="con cá có màu gì",
        expected_page=127,
        expected_keywords=["cá", "cá xiêm", "đổi màu", "màu"],
        max_returned=1,
    ),
    ImageRetrievalCase(
        query="cho xem hình cá xiêm đổi màu",
        expected_page=127,
        expected_keywords=["cá xiêm", "đổi màu"],
        max_returned=5,
    ),
    ImageRetrievalCase(
        query="cho tôi hình trang 74",
        expected_page=74,
        max_returned=5,
    ),
    ImageRetrievalCase(
        query="ánh sáng ảnh hưởng thế nào đến cây xanh",
        max_returned=1,
    ),
]


def _contains_expected_keyword(text: str, expected_keywords: Optional[List[str]]) -> bool:
    if not expected_keywords:
        return True
    normalized = text.lower()
    return any(keyword.lower() in normalized for keyword in expected_keywords)


def evaluate(cases: Optional[List[ImageRetrievalCase]] = None) -> List[dict]:
    retriever = HybridRetriever()
    results = []

    for case in cases or DEFAULT_CASES:
        search_result = retriever.search(case.query)
        image_rows = []
        for rank, doc in enumerate(search_result.image_docs, start=1):
            metadata = doc.metadata or {}
            haystack = " ".join(
                str(metadata.get(field, "") or "")
                for field in (
                    "search_text",
                    "visual_caption_vi",
                    "visual_keywords_vi",
                    "visual_objects_vi",
                    "caption_vi",
                    "context_text",
                    "nearby_text",
                    "keywords_vi",
                    "figure_label",
                )
            )
            image_rows.append(
                {
                    "rank": rank,
                    "page": metadata.get("page_number"),
                    "score": metadata.get("image_relevance_score"),
                    "metadata_score": metadata.get("image_metadata_score"),
                    "visual_score": metadata.get("image_visual_score"),
                    "lexical_score": metadata.get("image_lexical_score"),
                    "source": metadata.get("image_retrieval_source"),
                    "image_path": metadata.get("image_path"),
                    "visual_caption_vi": metadata.get("visual_caption_vi"),
                    "visual_keywords_vi": metadata.get("visual_keywords_vi"),
                    "keyword_hit": _contains_expected_keyword(haystack, case.expected_keywords),
                }
            )

        page_hit = any(row["page"] == case.expected_page for row in image_rows) if case.expected_page else True
        keyword_hit = any(row["keyword_hit"] for row in image_rows)
        results.append(
            {
                "query": case.query,
                "expected_page": case.expected_page,
                "page_hit": page_hit,
                "keyword_hit": keyword_hit,
                "max_returned": case.max_returned,
                "max_returned_ok": len(image_rows) <= case.max_returned if case.max_returned is not None else True,
                "returned": len(image_rows),
                "images": image_rows,
            }
        )

    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
