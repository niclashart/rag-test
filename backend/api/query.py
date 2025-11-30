"""RAG query endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import time

from database.database import get_db
from database.models import User
from database.crud import create_query_history, get_user_query_history
from backend.dependencies import get_current_active_user
from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from logging_config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])

# Initialize components (lazy loading for reranker)
vector_store = VectorStore()
embedder = Embedder()
retriever = Retriever(vector_store, embedder)
reranker = None  # Will be initialized on first use
qa_chain = QAChain()

def get_reranker():
    """Get reranker instance (lazy initialization)."""
    global reranker
    if reranker is None:
        reranker = Reranker()
    return reranker


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    use_reranking: Optional[bool] = True
    chat_history: Optional[List[ChatMessage]] = None


class SourceInfo(BaseModel):
    chunk_id: str
    document_id: Optional[int]
    page_number: Optional[int]
    similarity: Optional[float]


class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[SourceInfo]
    retrieval_time: float
    generation_time: float


class QueryHistoryItem(BaseModel):
    id: int
    query: str
    answer: Optional[str]
    sources: List[str]
    created_at: str


@router.post("", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Execute a RAG query."""
    start_time = time.time()
    
    try:
        # Retrieve relevant documents
        retrieval_start = time.time()
        
        # For specification queries, try without reranking first to see if it helps
        # The reranker might be prioritizing title chunks over technical specification chunks
        spec_keywords = ["spezifikation", "specification", "specs", "technische", "hardware", 
                        "wieviel", "wie viel", "welche", "was ist"]
        is_spec_query = any(keyword in request.query.lower() for keyword in spec_keywords)
        
        if request.use_reranking and not is_spec_query:
            # Use reranking for non-spec queries
            retrieved_docs = retriever.retrieve_with_reranking(
                user_id=current_user.id,
                query=request.query,
                reranker=get_reranker()
            )
        else:
            # For spec queries, use direct retrieval without reranking
            # This helps find technical chunks that might be ranked lower by the reranker
            retrieved_docs = retriever.retrieve(
                user_id=current_user.id,
                query=request.query,
                n_results=15  # Get more results for spec queries
            )
        
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload and index documents first."
            )
        
        # Generate answer
        generation_start = time.time()
        try:
            # Convert chat history to dict format
            chat_history = []
            if request.chat_history:
                for msg in request.chat_history:
                    chat_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            logger.info(f"Generating answer for query with {len(retrieved_docs)} retrieved documents and {len(chat_history)} previous messages")
            result = qa_chain.answer_with_retrieved_docs(
                question=request.query,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history
            )
            generation_time = time.time() - generation_start
            
            if not result.get("answer") or result["answer"].strip() == "":
                logger.warning("QA chain returned empty answer")
                result["answer"] = "Entschuldigung, ich konnte keine Antwort generieren. Bitte versuchen Sie es mit einer anderen Formulierung."
        except Exception as e:
            logger.error(f"Error in QA chain: {e}", exc_info=True)
            generation_time = time.time() - generation_start
            result = {
                "answer": f"Fehler bei der Generierung der Antwort: {str(e)}",
                "sources": []
            }
        
        # Format sources
        sources = []
        for source in result.get("sources", []):
            sources.append(SourceInfo(
                chunk_id=source.get("chunk_id", ""),
                document_id=source.get("document_id"),
                page_number=source.get("page_number"),
                similarity=source.get("similarity")
            ))
        
        # Save to query history
        source_ids = [s.chunk_id for s in sources]
        create_query_history(
            db=db,
            user_id=current_user.id,
            query=request.query,
            answer=result["answer"],
            sources=source_ids,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": time.time() - start_time,
                "num_sources": len(sources)
            }
        )
        
        logger.info(f"Query processed for user {current_user.id}: {request.query[:50]}...")
        
        return QueryResponse(
            answer=result["answer"],
            query=request.query,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/history", response_model=List[QueryHistoryItem])
def get_query_history(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get query history for the current user."""
    history = get_user_query_history(db, current_user.id, limit=limit)
    return history


