"""ChromaDB vector store integration."""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
from pathlib import Path
from logging_config.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """Initialize ChromaDB client."""
        if db_path is None:
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        if collection_name is None:
            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "documents")
        
        # Create directory if it doesn't exist
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"Initialized ChromaDB at {db_path}")
    
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


