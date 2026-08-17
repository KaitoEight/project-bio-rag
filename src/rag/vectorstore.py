"""Vector database management using ChromaDB."""

import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..config import (
    EMBEDDING_MODEL,
    TEXT_COLLECTION_NAME,
    PERSIST_DIR,
    RETRIEVER_FETCH_K,
    RETRIEVER_MAX_K,
    RETRIEVER_DISTANCE_MARGIN,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)


class RelevanceGatedRetriever(BaseRetriever):
    """Retriever that drops chunks far from the best match.

    Plain top-k similarity always returns ``k`` chunks regardless of how
    relevant they are, so off-topic pages bleed into the LLM context and the
    answer (e.g. a fish-colour question pulling in unrelated reproduction /
    light-mixing pages). This retriever fetches a wider candidate set, then
    keeps only chunks whose distance is within ``(1 + margin)`` of the closest
    match — chunks that are clearly farther away are discarded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    fetch_k: int = RETRIEVER_FETCH_K
    max_k: int = RETRIEVER_MAX_K
    distance_margin: float = RETRIEVER_DISTANCE_MARGIN

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Chroma returns (doc, distance) with lower distance = more similar.
        scored = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        if not scored:
            return []

        scored.sort(key=lambda pair: pair[1])
        best_distance = scored[0][1]
        cutoff = best_distance * (1.0 + self.distance_margin)

        kept = []
        for doc, distance in scored:
            if distance <= cutoff:
                kept.append(doc)
            else:
                logger.debug(
                    "Gated out chunk (dist=%.3f > cutoff=%.3f): %s",
                    distance, cutoff, doc.page_content[:60].replace("\n", " "),
                )
            if len(kept) >= self.max_k:
                break

        logger.info(
            "RelevanceGatedRetriever: %d/%d candidates kept (best=%.3f, cutoff=%.3f) for query=%r",
            len(kept), len(scored), best_distance, cutoff, query,
        )
        return kept


class VectorDB:
    """ChromaDB vector store wrapper with embedding support."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = TEXT_COLLECTION_NAME,
        persist_dir: str = str(PERSIST_DIR),
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {},
        )
        self.db = self._build_db(documents)
        logger.info(f"VectorDB initialized with {self.db._collection.count()} existing chunks")

    def _build_db(self, documents):
        if documents is None or len(documents) == 0:
            logger.info("Loading existing VectorDB")
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
        else:
            logger.info(f"Creating new VectorDB with {len(documents)} documents")
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        return db

    def get_retriever(self, search_kwargs: dict = None):
        search_kwargs = search_kwargs or {}
        # `k` (if supplied by callers like HybridRetriever) caps the number of
        # chunks kept *after* relevance gating; the wider candidate sweep is
        # controlled by fetch_k.
        max_k = search_kwargs.get("k", RETRIEVER_MAX_K)
        return RelevanceGatedRetriever(
            vectorstore=self.db,
            fetch_k=max(RETRIEVER_FETCH_K, max_k),
            max_k=max_k,
        )
