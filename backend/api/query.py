"""RAG query endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import time

from database.database import get_db
from database.crud import create_query_history, get_query_history
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
    text: Optional[str] = None  # The actual text content of the chunk


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
    query_metadata: Optional[dict] = None  # Include metadata with response times


@router.post("", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
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
        
        # Detect processor/CPU questions specifically - these often need multiple chunks
        # because processor tables can span multiple chunks
        query_lower = request.query.lower()
        is_processor_query = any(term in query_lower for term in [
            "prozessor", "prozessoren", "processor", "processors", "cpu", "cpus",
            "welche prozessor", "welche processor", "prozessor-konfiguration",
            "prozessoroptionen", "prozessor-optionen"
        ])
        
        # Processor queries are also spec queries
        is_spec_query = any(keyword in query_lower for keyword in spec_keywords) or is_processor_query
        
        logger.info(f"Query classification: is_spec_query={is_spec_query}, is_processor_query={is_processor_query}, use_reranking={request.use_reranking}")
        
        if request.use_reranking and not is_spec_query:
            # Use reranking for non-spec queries
            # Use user_id=1 as default since vector store is global
            # For processor queries, pass higher n_results to get more chunks
            n_results_for_retrieval = 50 if is_processor_query else None
            retrieved_docs = retriever.retrieve_with_reranking(
                user_id=1,
                query=request.query,
                reranker=get_reranker(),
                n_results=n_results_for_retrieval
            )
        else:
            # For spec queries (including processor queries), use direct retrieval without reranking
            # This helps find technical chunks that might be ranked lower by the reranker
            # Get even more results for general spec queries to ensure Display, Battery, Dimensions are found
            # For processor queries, get even more chunks (80-100) because tables can span multiple chunks
            n_results_for_retrieval = 100 if is_processor_query else 50
            logger.info(f"Using direct retrieval (no reranking) with n_results={n_results_for_retrieval}")
            retrieved_docs = retriever.retrieve(
                query=request.query,
                n_results=n_results_for_retrieval
            )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
        
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
        
        # Create a mapping from chunk_id to text from retrieved_docs (fallback)
        chunk_text_map = {}
        for doc in retrieved_docs:
            chunk_id = doc.get("id", "")
            chunk_text = doc.get("text", "")
            if chunk_id:
                chunk_text_map[chunk_id] = chunk_text
        
        # Format sources with text content
        # CRITICAL: Always fetch text directly from database to ensure we get the correct chunk
        # The chunk_id is the source of truth, not the text from retrieved_docs which may be reordered
        from database.crud import get_chunk_by_id
        
        sources = []
        for source in result.get("sources", []):
            chunk_id = source.get("chunk_id", "")
            
            # Always fetch chunk text from database using chunk_id to ensure correctness
            chunk_text = ""
            source_num = source.get("source_number", "?")
            try:
                db_chunk = get_chunk_by_id(db, chunk_id)
                if db_chunk:
                    chunk_text = db_chunk.text
                    text_preview = chunk_text[:150].replace('\n', ' ')
                    logger.info(f"Source {source_num}: Fetched chunk from DB - chunk_id: {chunk_id[:8]}..., page: {db_chunk.page_number}, chunk_index: {db_chunk.chunk_index}, length: {len(chunk_text)}, preview: {text_preview}...")
                else:
                    logger.warning(f"Source {source_num}: Chunk not found in database: {chunk_id}")
                    # Fallback to text from source or chunk_text_map
                    chunk_text = source.get("text", "") or chunk_text_map.get(chunk_id, "")
            except Exception as e:
                logger.error(f"Source {source_num}: Failed to fetch chunk from database: {e}", exc_info=True)
                # Fallback to text from source or chunk_text_map
                chunk_text = source.get("text", "") or chunk_text_map.get(chunk_id, "")
            
            # Don't truncate text - show full chunk content for better table/spec visibility
            # The frontend can handle scrolling for long content
            
            sources.append(SourceInfo(
                chunk_id=chunk_id,
                document_id=source.get("document_id"),
                page_number=source.get("page_number"),
                similarity=None,  # Remove similarity display
                text=chunk_text,
                source_number=source.get("source_number")  # Preserve original source number
            ))
        
        # Save to query history
        source_ids = [s.chunk_id for s in sources]
        create_query_history(
            db=db,
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
        
        logger.info(f"Query processed: {request.query[:50]}...")
        
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
def get_query_history_endpoint(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get query history."""
    history = get_query_history(db, limit=limit)
    # Convert to dict format with metadata
    result = []
    for item in history:
        result.append({
            "id": item.id,
            "query": item.query,
            "answer": item.answer,
            "sources": item.sources or [],
            "created_at": item.created_at.isoformat() if item.created_at else "",
            "query_metadata": item.query_metadata or {}
        })
    return result


