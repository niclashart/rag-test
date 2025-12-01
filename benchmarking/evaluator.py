"""RAGAS evaluation module."""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
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
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
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
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        logger.info(f"Evaluating {len(questions)} examples with RAGAS")
        
        # Import here to avoid circular imports or early initialization issues
        # Import here to avoid circular imports or early initialization issues
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Use the same model as the pipeline with higher timeout and retries
        llm = ChatOpenAI(model="gpt-4o-mini", request_timeout=60, max_retries=3)
        
        # Use HuggingFace embeddings (same as pipeline)
        # Model: sentence-transformers/all-MiniLM-L6-v2
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Use RunConfig to limit concurrency
        from ragas.run_config import RunConfig
        
        # Limit to 1 worker to avoid rate limits/timeouts
        run_config = RunConfig(max_workers=1, timeout=120)
        
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config
        )
        
        # Convert to dict
        results_dict = result.to_pandas().to_dict(orient="records")
        
        logger.info("RAGAS evaluation completed")
        return {
            "results": results_dict,
            "summary": {
                "faithfulness": pd.Series([r["faithfulness"] for r in results_dict]).mean(),
                "answer_relevancy": pd.Series([r["answer_relevancy"] for r in results_dict]).mean(),
                "context_precision": pd.Series([r["context_precision"] for r in results_dict]).mean(),
                "context_recall": pd.Series([r["context_recall"] for r in results_dict]).mean(),
            }
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


