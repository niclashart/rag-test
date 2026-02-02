"""Check the status of the current ChromaDB collection."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from database.database import SessionLocal
from database.crud import get_all_documents
from logging_config.logger import get_logger

logger = get_logger(__name__)


def check_collection_status():
    """Check the status of the current collection."""
    print("=" * 60)
    print("Collection Status Check")
    print("=" * 60)
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()
    
    print(f"\n📊 Current Configuration:")
    print(f"  Embedding Model: {embedder.model_name}")
    print(f"  Collection Name: {vector_store.collection_name}")
    print(f"  DB Path: {vector_store.db_path}")
    
    # Check collection
    try:
        collection = vector_store.get_collection()
        count = collection.count()
        
        print(f"\n📦 Collection Status:")
        print(f"  Documents in collection: {count}")
        
        if count == 0:
            print(f"\n⚠️  WARNING: Collection is EMPTY!")
            print(f"   You need to re-index your documents.")
        else:
            print(f"  ✓ Collection has {count} documents")
            
            # Get a sample document
            try:
                sample = collection.peek(limit=1)
                if sample.get('ids') and len(sample['ids']) > 0:
                    print(f"\n📄 Sample Document:")
                    print(f"  ID: {sample['ids'][0]}")
                    if sample.get('metadatas'):
                        print(f"  Metadata: {sample['metadatas'][0]}")
            except Exception as e:
                print(f"  Could not get sample: {e}")
        
    except Exception as e:
        print(f"\n❌ Error accessing collection: {e}")
        print(f"   Collection might not exist yet.")
    
    # Check database
    db = SessionLocal()
    try:
        all_documents = get_all_documents(db)
        indexed_docs = [doc for doc in all_documents if doc.status == "indexed"]
        
        print(f"\n💾 Database Status:")
        print(f"  Total documents: {len(all_documents)}")
        print(f"  Indexed documents: {len(indexed_docs)}")
        
        if len(indexed_docs) > 0:
            print(f"\n  Indexed documents:")
            for doc in indexed_docs[:5]:  # Show first 5
                print(f"    - {doc.id}: {doc.filename}")
            if len(indexed_docs) > 5:
                print(f"    ... and {len(indexed_docs) - 5} more")
        
        if len(indexed_docs) > 0 and count == 0:
            print(f"\n⚠️  ACTION REQUIRED:")
            print(f"   You have {len(indexed_docs)} indexed documents in the database,")
            print(f"   but the collection is empty. Run re-indexing:")
            print(f"   python scripts/reindex_with_new_model.py")
        
    finally:
        db.close()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_collection_status()




















