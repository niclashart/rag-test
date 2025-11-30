"""CLI script for RAG queries."""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from logging_config.logger import get_logger

logger = get_logger(__name__)


def query_rag(user_id: int, question: str, use_reranking: bool = True):
    """Execute a RAG query."""
    # Initialize components
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    reranker = Reranker() if use_reranking else None
    qa_chain = QAChain()
    
    # Retrieve documents
    if use_reranking and reranker:
        retrieved_docs = retriever.retrieve_with_reranking(
            user_id=user_id,
            query=question,
            reranker=reranker
        )
    else:
        retrieved_docs = retriever.retrieve(user_id=user_id, query=question)
    
    if not retrieved_docs:
        logger.warning("No relevant documents found")
        return None
    
    # Generate answer
    result = qa_chain.answer_with_retrieved_docs(
        question=question,
        retrieved_docs=retrieved_docs
    )
    
    return result


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Query RAG pipeline")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--user-id", type=int, required=True, help="User ID")
    parser.add_argument("--no-reranking", action="store_true", help="Disable reranking")
    
    args = parser.parse_args()
    
    # Execute query
    result = query_rag(args.user_id, args.question, use_reranking=not args.no_reranking)
    
    if result:
        print("\n" + "="*80)
        print("QUESTION:")
        print("="*80)
        print(result["question"])
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        print("SOURCES:")
        print("="*80)
        for i, source in enumerate(result.get("sources", []), 1):
            print(f"\n{i}. Chunk ID: {source.get('chunk_id')}")
            print(f"   Document ID: {source.get('document_id')}")
            print(f"   Page: {source.get('page_number')}")
            print(f"   Similarity: {source.get('similarity', 0):.3f}")
    else:
        logger.error("Query failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


