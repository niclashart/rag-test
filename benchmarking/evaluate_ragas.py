import sys
import os
import asyncio
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.qa.chain import QAChain
from benchmarking.evaluator import RAGASEvaluator
from benchmarking.visualizer import BenchmarkVisualizer
from logging_config.logger import get_logger

logger = get_logger(__name__)

def load_gold_standard(file_path: str) -> Dict:
    """Lädt Goldstandard-Dataset aus JSON-Datei."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded gold standard: {data.get('dataset_name', 'Unknown')} with {len(data.get('questions', []))} questions")
    return data


def run_evaluation_from_file(
    gold_standard_path: str,
    use_reranking: bool = True,
    save_results: bool = True
) -> Dict:
    """
    Führt Evaluation basierend auf Goldstandard-Datei durch.
    
    Args:
        gold_standard_path: Pfad zur Goldstandard-JSON-Datei
        use_reranking: Ob Reranking verwendet werden soll
        save_results: Ob Ergebnisse gespeichert werden sollen
    
    Returns:
        Dictionary mit Evaluations-Ergebnissen
    """
    # 1. Lade Goldstandard
    logger.info(f"Loading gold standard from {gold_standard_path}")
    gold_standard = load_gold_standard(gold_standard_path)
    
    questions = [q["question"] for q in gold_standard["questions"]]
    ground_truths = [q.get("ground_truth") for q in gold_standard["questions"]]
    
    # 2. Initialisiere RAG-Komponenten
    logger.info("Initializing RAG components...")
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    reranker = Reranker() if use_reranking else None
    qa_chain = QAChain()
    
    # 3. Führe RAG-Pipeline für alle Fragen aus
    logger.info(f"Running RAG pipeline on {len(questions)} questions...")
    answers = []
    contexts = []
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question[:60]}...")
        
        try:
            # Retrieve - mirror logic from backend/api/query.py
            # For specification queries, try without reranking first to see if it helps
            # The reranker might be prioritizing title chunks over technical specification chunks
            spec_keywords = ["spezifikation", "specification", "specs", "technische", "hardware", 
                            "wieviel", "wie viel", "welche", "was ist"]
            is_spec_query = any(keyword in question.lower() for keyword in spec_keywords)
            
            if use_reranking and reranker and not is_spec_query:
                # Für reranking benötigen wir user_id, aber da die VectorStore global ist,
                # verwenden wir user_id=1 als Default
                retrieved_docs = retriever.retrieve_with_reranking(
                    user_id=1,
                    query=question,
                    reranker=reranker
                )
            else:
                # For spec queries, use direct retrieval without reranking
                # This helps find technical chunks that might be ranked lower by the reranker
                # Get even more results for general spec queries to ensure Display, Battery, Dimensions are found
                n_results = 50 if is_spec_query else None
                retrieved_docs = retriever.retrieve(
                    query=question,
                    n_results=n_results
                )
            
            context_text_list = [doc["text"] for doc in retrieved_docs]
            
            # Log context size for debugging
            total_chars = sum(len(text) for text in context_text_list)
            logger.info(f"Retrieved {len(context_text_list)} chunks. Total context size: {total_chars} chars")
            
            contexts.append(context_text_list)
            
            # Generate Answer
            result = qa_chain.answer_with_retrieved_docs(
                question=question,
                retrieved_docs=retrieved_docs,
                concise_mode=True
            )
            answers.append(result.get("answer", ""))
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}", exc_info=True)
            contexts.append([])
            answers.append(f"Fehler bei der Verarbeitung: {str(e)}")
    
    # 4. Evaluiere mit RAGAS
    logger.info("Running RAGAS evaluation...")
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate_rag(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    # 5. Speichere Ergebnisse wenn gewünscht
    if save_results:
        output_dir = Path("data/benchmark_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_safe = gold_standard['dataset_name'].replace(' ', '_').replace('/', '_')
        
        # Speichere JSON-Ergebnisse
        json_path = output_dir / f"evaluation_{dataset_name_safe}_{timestamp}.json"
        evaluation_data = {
            "dataset_info": {
                "name": gold_standard.get("dataset_name"),
                "description": gold_standard.get("description"),
                "version": gold_standard.get("version"),
                "created_at": gold_standard.get("created_at"),
                "evaluated_at": datetime.now().isoformat(),
                "num_questions": len(questions),
                "use_reranking": use_reranking
            },
            "questions": [
                {
                    "id": q.get("id"),
                    "question": q.get("question"),
                    "ground_truth": q.get("ground_truth"),
                    "answer": answers[i],
                    "contexts": contexts[i],
                    "category": q.get("category")
                }
                for i, q in enumerate(gold_standard["questions"])
            ],
            "evaluation_results": results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {json_path}")
        
        # Erstelle Visualisierung
        try:
            visualizer = BenchmarkVisualizer()
            plot_filename = f"dashboard_{dataset_name_safe}_{timestamp}.html"
            plot_path = output_dir / plot_filename
            visualizer.create_dashboard(results, save_path=str(plot_path))
            logger.info(f"Dashboard saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
    
    # 6. Zeige Ergebnisse
    print("\n" + "="*60)
    print("RAGAS Evaluation Results")
    print("="*60)
    print(f"Dataset: {gold_standard['dataset_name']}")
    print(f"Questions: {len(questions)}")
    print(f"Reranking: {'Yes' if use_reranking else 'No'}")
    print("\nSummary Metrics:")
    print("-" * 60)
    for metric, value in results["summary"].items():
        print(f"  {metric:25s}: {value:.4f} ({value*100:.2f}%)")
    
    print("\nDetailed Results:")
    print("-" * 60)
    for i, res in enumerate(results["results"]):
        q_data = gold_standard["questions"][i]
        print(f"\nQ{q_data.get('id', i+1)}: {res.get('question', 'N/A')[:60]}...")
        print(f"  Ground Truth: {q_data.get('ground_truth', 'N/A')[:60]}...")
        print(f"  Answer: {res.get('answer', 'N/A')[:60]}...")
        print(f"  Faithfulness:      {res.get('faithfulness', 0.0):.4f}")
        print(f"  Answer Relevancy:  {res.get('answer_relevancy', 0.0):.4f}")
        print(f"  Context Precision: {res.get('context_precision', 0.0):.4f}")
        print(f"  Context Recall:    {res.get('context_recall', 0.0):.4f}")
    
    print("\n" + "="*60)
    
    return results


def run_evaluation():
    """Run Ragas evaluation on a small test set (legacy function)."""
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--gold-standard",
        type=str,
        default="data/gold_standard.json",
        help="Path to gold standard JSON file"
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    if args.gold_standard:
        run_evaluation_from_file(
            gold_standard_path=args.gold_standard,
            use_reranking=not args.no_reranking,
            save_results=not args.no_save
        )
    else:
        run_evaluation()
