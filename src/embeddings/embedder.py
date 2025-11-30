"""HuggingFace embeddings for document chunks."""
from sentence_transformers import SentenceTransformer
from typing import List
import os
from logging_config.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """HuggingFace sentence transformer embedder."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedder with a model."""
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


