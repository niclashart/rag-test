"""ChromaDB vector store integration."""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
from pathlib import Path
from logging_config.logger import get_logger
import hashlib

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(self, db_path: str = None, collection_name: str = None, embedding_model: str = None):
        """Initialize ChromaDB client.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Explicit collection name (if None, will be generated from model)
            embedding_model: Embedding model name (if None, will be read from EMBEDDING_MODEL env var)
        """
        if db_path is None:
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        # Get embedding model name from environment if not provided
        if embedding_model is None:
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Generate collection name based on model if not explicitly provided
        if collection_name is None:
            # Check if explicit collection name is set in env
            explicit_collection = os.getenv("CHROMA_COLLECTION_NAME")
            if explicit_collection:
                collection_name = explicit_collection
            else:
                # Create a safe collection name from model name
                # Replace slashes and special chars with underscores
                safe_model_name = embedding_model.replace("/", "_").replace("-", "_").replace(":", "_")
                # Use hash to keep name short but unique
                model_hash = hashlib.md5(embedding_model.encode()).hexdigest()[:8]
                collection_name = f"documents_{safe_model_name}_{model_hash}"
        
        # Create directory if it doesn't exist
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"Initialized ChromaDB at {db_path} with collection '{collection_name}' for model '{embedding_model}'")
    
    def get_collection(self):
        """Get or create the global collection."""
        collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        return collection
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """Add documents to the vector store."""
        collection = self.get_collection()
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(texts)} documents to collection")
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """Query the vector store."""
        collection = self.get_collection()
        
        # Check collection count for debugging
        try:
            count = collection.count()
            logger.info(f"Querying collection: {count} documents in collection")
            if count == 0:
                logger.warning(f"Collection is empty!")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        
        num_results = len(results.get('ids', [[]])[0])
        logger.debug(f"Query returned results", num_results=num_results, n_results_requested=n_results)
        return results
    
    def delete_documents(self, ids: List[str]):
        """Delete documents from the vector store."""
        collection = self.get_collection()
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection")
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")


