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
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            # If CUDA error occurs, try to truncate text and retry
            if "cuda" in str(e).lower() or "CUDA" in str(e):
                logger.warning(f"CUDA error during embedding, attempting to truncate text: {e}")
                # Get max sequence length and truncate text
                max_seq_length = getattr(self.model, 'max_seq_length', 256)
                # Rough estimate: ~4 chars per token, use 90% of max
                max_chars = int(max_seq_length * 0.9 * 4)
                if len(text) > max_chars:
                    truncated_text = text[:max_chars]
                    logger.info(f"Truncating text from {len(text)} to {len(truncated_text)} characters")
                    return self.model.encode(truncated_text, convert_to_numpy=True).tolist()
            # Re-raise if truncation didn't help or it's a different error
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


