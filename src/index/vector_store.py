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
    
    def __init__(self, db_path: str = None, collection_prefix: str = None):
        """Initialize ChromaDB client."""
        if db_path is None:
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        if collection_prefix is None:
            collection_prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "user_")
        
        # Create directory if it doesn't exist
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.collection_prefix = collection_prefix
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"Initialized ChromaDB at {db_path}")
    
    def get_collection(self, user_id: int):
        """Get or create a collection for a user."""
        collection_name = f"{self.collection_prefix}{user_id}"
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"user_id": user_id}
        )
        return collection
    
    def add_documents(
        self,
        user_id: int,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """Add documents to the vector store."""
        collection = self.get_collection(user_id)
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(texts)} documents to collection for user {user_id}")
    
    def query(
        self,
        user_id: int,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """Query the vector store."""
        collection = self.get_collection(user_id)
        
        # Check collection count for debugging
        try:
            count = collection.count()
            logger.info(f"Querying collection for user {user_id}: {count} documents in collection")
            if count == 0:
                logger.warning(f"Collection for user {user_id} is empty!")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        
        logger.debug(f"Query returned {len(results.get('ids', [[]])[0])} results")
        return results
    
    def delete_documents(self, user_id: int, ids: List[str]):
        """Delete documents from the vector store."""
        collection = self.get_collection(user_id)
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection for user {user_id}")
    
    def delete_collection(self, user_id: int):
        """Delete a user's collection."""
        collection_name = f"{self.collection_prefix}{user_id}"
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to delete collection for user {user_id}: {e}")


