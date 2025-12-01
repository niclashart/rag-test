import sys
import os
import asyncio
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.qa.chain import QAChain
from benchmarking.evaluator import RAGASEvaluator
from logging_config.logger import get_logger

logger = get_logger(__name__)

def run_evaluation():
    """Run Ragas evaluation on a small test set."""
    
    # 1. Initialize RAG components
    logger.info("Initializing RAG components...")
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    qa_chain = QAChain()
    
    # 2. Define test dataset (Questions and Ground Truths)
    # Using a few examples relevant to the codebase (laptops)
    test_data = [
        {
            "question": "What is the maximum RAM capacity of the ThinkPad E14 Gen 6?",
            "ground_truth": "The ThinkPad E14 Gen 6 supports up to 64GB DDR5-5600."
        }
    ]
    
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    
    # 3. Run RAG Pipeline
    logger.info("Running RAG pipeline on test questions...")
    answers = []
    contexts = []
    
    for question in questions:
        logger.info(f"Processing question: {question}")
        
        # Retrieve
        retrieved_docs = retriever.retrieve(question)
        context_text_list = [doc["text"] for doc in retrieved_docs]
        
        # Log context size for debugging
        total_chars = sum(len(text) for text in context_text_list)
        logger.info(f"Retrieved {len(context_text_list)} chunks. Total context size: {total_chars} chars")
        
        contexts.append(context_text_list)
        
        # Generate Answer
        result = qa_chain.answer_with_retrieved_docs(question, retrieved_docs)
        answers.append(result["answer"])
        
    # 4. Evaluate with Ragas
    logger.info("Running Ragas evaluation...")
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate_rag(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    # 5. Print Results
    print("\n=== Ragas Evaluation Results ===")
    print(f"Summary Metrics:")
    for metric, value in results["summary"].items():
        print(f"  {metric}: {value:.4f}")
        
    print("\nDetailed Results:")
    for i, res in enumerate(results["results"]):
        print(f"\nQ{i+1}: {res.get('question', 'N/A')}")
        print(f"A: {res.get('answer', 'N/A')}")
        print(f"Faithfulness: {res.get('faithfulness', 0.0):.4f}")
        print(f"Answer Relevancy: {res.get('answer_relevancy', 0.0):.4f}")
        print(f"Context Precision: {res.get('context_precision', 0.0):.4f}")
        print(f"Context Recall: {res.get('context_recall', 0.0):.4f}")

if __name__ == "__main__":
    run_evaluation()
