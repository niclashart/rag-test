"""Benchmarking endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import time
from pathlib import Path

from database.database import get_db
from database.models import User
from backend.dependencies import get_current_active_user
from benchmarking.evaluator import RAGASEvaluator
from benchmarking.visualizer import BenchmarkVisualizer
from logging_config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

evaluator = RAGASEvaluator()
visualizer = BenchmarkVisualizer()


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

