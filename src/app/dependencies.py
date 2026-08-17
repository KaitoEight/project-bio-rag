"""
Singleton manager for Biology RAG components to avoid reloading heavy models on every request.
"""
import logging
from src.rag import VectorDB, get_llm, BiologyRAG, HybridRetriever

logger = logging.getLogger(__name__)

class AppServices:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppServices, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing AppServices singleton...")
        self.vdb = VectorDB()
        self.hybrid_retriever = HybridRetriever()
        self.llm = get_llm()
        self.rag = BiologyRAG(self.llm)
        self.rag_chain = self.rag.get_chain(self.vdb.get_retriever())
        
        self._initialized = True
        logger.info("AppServices initialized successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance
