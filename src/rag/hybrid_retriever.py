"""Hybrid retriever combining text (MiniLM) and image (CLIP) search."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from ..config import RETRIEVER_K, IMAGE_RETRIEVER_K
from .vectorstore import VectorDB
from .image_vectorstore import ImageVectorDB
from .query_intent import is_image_only_query

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Combined search result from text and image retrieval."""

    text_docs: List[Document]
    image_docs: List[Document]
    image_only_query: bool = False

    @property
    def has_images(self) -> bool:
        return len(self.image_docs) > 0

    @property
    def all_docs(self) -> List[Document]:
        return self.text_docs + self.image_docs


class HybridRetriever:
    """Unified retriever that searches both text and image collections."""

    def __init__(
        self,
        text_retriever_k: int = RETRIEVER_K,
        image_retriever_k: int = IMAGE_RETRIEVER_K,
    ):
        self.text_db = VectorDB()
        self.image_db = ImageVectorDB()

        self.text_k = text_retriever_k
        self.image_k = image_retriever_k

        self._text_retriever = self.text_db.get_retriever({"k": self.text_k})
        self._image_retriever = self.image_db.get_retriever({"k": self.image_k})

    def search(self, query: str, text_k: int = None, image_k: int = None) -> SearchResult:
        """Perform hybrid search: text + image simultaneously."""
        text_docs = []
        image_docs = []
        image_only_query = is_image_only_query(query)

        if image_only_query:
            image_docs = self.search_image_only(query)
            return SearchResult(
                text_docs=[],
                image_docs=image_docs,
                image_only_query=True,
            )

        k = text_k if text_k else self.text_k
        img_k = image_k if image_k else self.image_k

        try:
            text_retriever = self.text_db.get_retriever({"k": k})
            text_docs = text_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"Text retrieval failed: {e}")

        try:
            image_retriever = self.image_db.get_retriever({"k": img_k})
            image_docs = image_retriever.invoke(query, related_text_docs=text_docs)
        except Exception as e:
            logger.warning(f"Image retrieval failed: {e}")

        return SearchResult(
            text_docs=text_docs,
            image_docs=image_docs,
            image_only_query=image_only_query,
        )

    def search_text_only(self, query: str) -> List[Document]:
        """Search text collection only."""
        try:
            return self._text_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []

    def search_image_only(self, query: str) -> List[Document]:
        """Search image collection only."""
        try:
            return self._image_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            return []
