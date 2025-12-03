"""Benchmarking endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import time
from pathlib import Path
import os

from database.database import get_db
from database.models import User
from backend.dependencies import get_current_active_user
from benchmarking.evaluator import RAGASEvaluator
from benchmarking.visualizer import BenchmarkVisualizer
from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.qa.chain import QAChain
from logging_config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

evaluator = RAGASEvaluator()
visualizer = BenchmarkVisualizer()

# Initialize RAG components (lazy loading)
_rag_components = {
    "retriever": None,
    "reranker": None,
    "qa_chain": None
}

def get_rag_components():
    """Get or initialize RAG components."""
    global _rag_components
    if _rag_components["retriever"] is None:
        embedder = Embedder()
        vector_store = VectorStore()
        _rag_components["retriever"] = Retriever(vector_store=vector_store, embedder=embedder)
        _rag_components["reranker"] = Reranker()
        _rag_components["qa_chain"] = QAChain()
    return _rag_components


class BenchmarkRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    contexts: Optional[List[List[str]]] = None


class BenchmarkResponse(BaseModel):
    results: List[Dict]
    summary: Dict
    plot_path: Optional[str] = None


@router.post("/run", response_model=BenchmarkResponse)
def run_benchmark(
    request: BenchmarkRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Run RAGAS benchmark evaluation."""
    if not request.answers or not request.contexts:
        raise HTTPException(
            status_code=400,
            detail="answers and contexts are required for benchmarking"
        )
    
    if len(request.questions) != len(request.answers) or len(request.questions) != len(request.contexts):
        raise HTTPException(
            status_code=400,
            detail="questions, answers, and contexts must have the same length"
        )
    
    # Run evaluation
    results = evaluator.evaluate_rag(
        questions=request.questions,
        answers=request.answers,
        contexts=request.contexts,
        ground_truths=request.ground_truths
    )
    
    # Create visualization
    plot_filename = f"benchmark_{current_user.id}_{int(time.time())}.html"
    plot_path = visualizer.output_dir / plot_filename
    
    visualizer.create_dashboard(results, save_path=str(plot_path))
    
    logger.info(f"Benchmark completed for user {current_user.id}")
    
    return BenchmarkResponse(
        results=results["results"],
        summary=results["summary"],
        plot_path=str(plot_path)
    )


class GoldStandardRequest(BaseModel):
    file_path: str
    use_reranking: Optional[bool] = True


@router.post("/run-from-file", response_model=BenchmarkResponse)
def run_benchmark_from_file(
    request: GoldStandardRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Führt Evaluation basierend auf Goldstandard-Datei durch.
    Lädt Fragen und Ground Truths aus JSON-Datei und führt automatisch RAG-Pipeline aus.
    """
    # Validiere Dateipfad
    gold_standard_path = Path(request.file_path)
    if not gold_standard_path.is_absolute():
        # Relativer Pfad - von Projekt-Root
        project_root = Path(__file__).parent.parent.parent
        gold_standard_path = project_root / request.file_path
    
    if not gold_standard_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Gold standard file not found: {request.file_path}"
        )
    
    # Lade Goldstandard
    try:
        with open(gold_standard_path, 'r', encoding='utf-8') as f:
            gold_standard = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in gold standard file: {str(e)}"
        )
    
    questions = [q["question"] for q in gold_standard.get("questions", [])]
    ground_truths = [q.get("ground_truth") for q in gold_standard.get("questions", [])]
    
    if not questions:
        raise HTTPException(
            status_code=400,
            detail="Gold standard file contains no questions"
        )
    
    # Initialisiere RAG-Komponenten
    components = get_rag_components()
    retriever = components["retriever"]
    reranker = components["reranker"] if request.use_reranking else None
    qa_chain = components["qa_chain"]
    
    # Führe RAG-Pipeline für alle Fragen aus
    logger.info(f"Running RAG pipeline on {len(questions)} questions from gold standard")
    answers = []
    contexts = []
    
    for i, question in enumerate(questions):
        try:
            # Retrieve
            if request.use_reranking and reranker:
                retrieved_docs = retriever.retrieve_with_reranking(
                    user_id=current_user.id,
                    query=question,
                    reranker=reranker
                )
            else:
                retrieved_docs = retriever.retrieve(query=question)
            
            context_text_list = [doc["text"] for doc in retrieved_docs]
            contexts.append(context_text_list)
            
            # Generate Answer
            result = qa_chain.answer_with_retrieved_docs(
                question=question,
                retrieved_docs=retrieved_docs
            )
            answers.append(result.get("answer", ""))
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}", exc_info=True)
            contexts.append([])
            answers.append(f"Fehler bei der Verarbeitung: {str(e)}")
    
    # Evaluiere mit RAGAS
    logger.info("Running RAGAS evaluation...")
    results = evaluator.evaluate_rag(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    # Erstelle Visualisierung
    plot_filename = f"benchmark_goldstandard_{current_user.id}_{int(time.time())}.html"
    plot_path = visualizer.output_dir / plot_filename
    visualizer.create_dashboard(results, save_path=str(plot_path))
    
    logger.info(f"Benchmark from gold standard completed for user {current_user.id}")
    
    return BenchmarkResponse(
        results=results["results"],
        summary=results["summary"],
        plot_path=str(plot_path)
    )


@router.get("/results/{result_id}")
def get_benchmark_result(
    result_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific benchmark result."""
    result_path = visualizer.output_dir / f"{result_id}.json"
    
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark result not found"
        )
    
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return result

