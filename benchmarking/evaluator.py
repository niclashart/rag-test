"""RAGAS evaluation module."""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
)
from datasets import Dataset
from typing import List, Dict, Optional
import pandas as pd
import os
from logging_config.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """RAGAS evaluator for RAG pipeline."""
    
    def __init__(self):
        """Initialize evaluator."""
        # Basis-Metriken, die immer verwendet werden
        self.base_metrics = [
            faithfulness,
            answer_relevancy,
            # context_precision,  # Auskommentiert für schnellere Evaluation
            context_recall
        ]
        # Metriken, die Ground Truth benötigen
        self.ground_truth_metrics = [
            answer_correctness
        ]
    
    def evaluate_rag(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate RAG pipeline using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of answers from RAG
            contexts: List of context lists (each inner list contains retrieved contexts)
            ground_truths: Optional ground truth answers
        """
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        
        # Determine which metrics to use based on availability of ground truth
        has_ground_truth = ground_truths and any(gt for gt in ground_truths if gt and gt.strip())
        
        if has_ground_truth:
            data["ground_truth"] = ground_truths
            metrics_to_use = self.base_metrics + self.ground_truth_metrics
            logger.info("Using metrics with ground truth: including answer_correctness")
        else:
            metrics_to_use = self.base_metrics
            logger.info("Using base metrics only (no ground truth available)")
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        logger.info(f"Evaluating {len(questions)} examples with RAGAS")
        
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        
        llm = ChatOpenAI(model="gpt-4o-mini", request_timeout=120, max_retries=5)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        from ragas.run_config import RunConfig
        
        run_config = RunConfig(max_workers=1, timeout=180)
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config
        )
        
        # Convert to dict
        results_dict = result.to_pandas().to_dict(orient="records")
        
        logger.info("RAGAS evaluation completed")
        
        # Build summary with available metrics
        summary = {
            "faithfulness": pd.Series([r.get("faithfulness", 0) for r in results_dict]).mean(),
            "answer_relevancy": pd.Series([r.get("answer_relevancy", 0) for r in results_dict]).mean(),
            "context_recall": pd.Series([r.get("context_recall", 0) for r in results_dict]).mean(),
        }
        
        # Add answer_correctness if ground truth was available
        if has_ground_truth:
            summary["answer_correctness"] = pd.Series([r.get("answer_correctness", 0) for r in results_dict]).mean()
        
        return {
            "results": results_dict,
            "summary": summary
        }
    
    def evaluate_from_queries(
        self,
        query_results: List[Dict]
    ) -> Dict:
        """
        Evaluate from query results.
        query_results should contain: question, answer, contexts (list of strings)
        """
        questions = [q["question"] for q in query_results]
        answers = [q["answer"] for q in query_results]
        contexts = [q.get("contexts", []) for q in query_results]
        ground_truths = [q.get("ground_truth") for q in query_results] if any(q.get("ground_truth") for q in query_results) else None
        
        return self.evaluate_rag(questions, answers, contexts, ground_truths)


