"""Reranking module using cross-encoder models."""
from sentence_transformers import CrossEncoder
from typing import List
import os
from logging_config.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = None):
        """Initialize reranker with a cross-encoder model."""
        if model_name is None:
            env_model = os.getenv("RERANK_MODEL")
            if env_model:
                # If env var is set but uses old format, fix it
                if env_model.startswith("sentence-transformers/"):
                    logger.warning(f"Found old model format in RERANK_MODEL: {env_model}, converting to cross-encoder format")
                    model_name = env_model.replace("sentence-transformers/", "cross-encoder/")
                else:
                    model_name = env_model
            else:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        logger.info(f"Loading reranker model: {model_name}")
        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, trying fallback model: {e}")
            # Fallback to a well-known model
            fallback_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            try:
                self.model = CrossEncoder(fallback_model)
                self.model_name = fallback_model
                logger.info(f"Successfully loaded fallback model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model {fallback_model}: {e2}")
                raise RuntimeError(f"Could not load any reranker model. Tried {model_name} and {fallback_model}")
    
    def rerank(self, query: str, texts: List[str], top_k: int = 3) -> List[int]:
        """
        Rerank texts based on query relevance.
        Returns indices of top_k most relevant texts.
        """
        if not texts:
            return []
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Get top_k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        logger.info(f"Reranked {len(texts)} documents, returning top {top_k}")
        return top_indices
    
    def rerank_with_scores(self, query: str, texts: List[str], top_k: int = 3) -> List[tuple]:
        """
        Rerank texts and return indices with scores.
        Returns list of (index, score) tuples.
        """
        if not texts:
            return []
        
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
        
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]


