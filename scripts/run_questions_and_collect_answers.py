#!/usr/bin/env python3
"""Führt Fragen durch die RAG-Pipeline aus und sammelt Antworten."""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.qa.chain import QAChain
from benchmarking.gold_standard import load_gold_standard
from logging_config.logger import get_logger

logger = get_logger(__name__)

def run_questions_and_collect_answers(gold_standard_path: str, max_questions: int = 10):
    """Führt Fragen durch die RAG-Pipeline aus und sammelt Antworten."""
    
    # Lade Gold Standard
    logger.info(f"Lade Gold Standard von {gold_standard_path}")
    gold_standard = load_gold_standard(gold_standard_path)
    
    questions = gold_standard.get("questions", [])
    
    # Filtere nur Fragen mit nicht-leerem question-Feld
    valid_questions = [q for q in questions if q.get("question", "").strip()]
    
    # Begrenze auf max_questions
    questions_to_process = valid_questions[:max_questions]
    
    logger.info(f"Verarbeite {len(questions_to_process)} Fragen")
    
    # Initialisiere RAG-Komponenten
    logger.info("Initialisiere RAG-Komponenten...")
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    reranker = Reranker()
    qa_chain = QAChain()
    
    results = []
    
    for i, q_data in enumerate(questions_to_process, 1):
        question = q_data.get("question", "")
        q_id = q_data.get("id", i)
        existing_ground_truth = q_data.get("ground_truth", "")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Frage {i}/{len(questions_to_process)} (ID: {q_id})")
        logger.info(f"Frage: {question}")
        
        if existing_ground_truth:
            logger.info(f"Bestehender Ground Truth: {existing_ground_truth}")
        
        try:
            # Retrieve mit Reranking
            retrieved_docs = retriever.retrieve_with_reranking(
                user_id=1,
                query=question,
                reranker=reranker,
                n_results=15
            )
            
            logger.info(f"Retrieved {len(retrieved_docs)} Dokumente")
            
            # Generate Answer
            # Verwende eval_mode wenn Ground Truth vorhanden ist
            result = qa_chain.answer_with_retrieved_docs(
                question=question,
                retrieved_docs=retrieved_docs,
                concise_mode=True,
                eval_mode=bool(existing_ground_truth),
                ground_truth=existing_ground_truth if existing_ground_truth else None
            )
            
            answer = result.get("answer", "")
            
            logger.info(f"Antwort: {answer[:200]}..." if len(answer) > 200 else f"Antwort: {answer}")
            
            results.append({
                "id": q_id,
                "question": question,
                "answer": answer,
                "existing_ground_truth": existing_ground_truth
            })
            
        except Exception as e:
            logger.error(f"Fehler bei Frage {i}: {e}", exc_info=True)
            results.append({
                "id": q_id,
                "question": question,
                "answer": f"FEHLER: {str(e)}",
                "existing_ground_truth": existing_ground_truth
            })
    
    # Zeige Zusammenfassung
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    for result in results:
        print(f"\nID {result['id']}: {result['question']}")
        print(f"  Antwort: {result['answer']}")
        if result['existing_ground_truth']:
            print(f"  (Bestehender Ground Truth: {result['existing_ground_truth']})")
    
    return results

def update_gold_standard_with_answers(gold_standard_path: str, results: list):
    """Aktualisiert die gold_standard.json mit den gesammelten Antworten."""
    path = Path(gold_standard_path)
    
    # Lese alle Zeilen
    lines = []
    id_to_answer = {r["id"]: r["answer"] for r in results}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    q = json.loads(line)
                    q_id = q.get("id")
                    
                    # Aktualisiere ground_truth wenn wir eine Antwort haben
                    if q_id in id_to_answer:
                        q["ground_truth"] = id_to_answer[q_id]
                        logger.info(f"Aktualisiere ID {q_id} mit Antwort")
                    
                    lines.append(json.dumps(q, ensure_ascii=False))
                except json.JSONDecodeError:
                    # Ungültige Zeile, behalte sie wie sie ist
                    lines.append(line)
    
    # Schreibe zurück
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    logger.info(f"Gold Standard aktualisiert: {len(id_to_answer)} Antworten eingetragen")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Führt Fragen durch RAG-Pipeline und sammelt Antworten")
    parser.add_argument(
        "--gold-standard",
        type=str,
        default="data/gold_standard.json",
        help="Pfad zur gold_standard.json"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Maximale Anzahl an Fragen zu verarbeiten"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Aktualisiert die gold_standard.json mit den Antworten"
    )
    
    args = parser.parse_args()
    
    # Führe Fragen aus
    results = run_questions_and_collect_answers(
        gold_standard_path=args.gold_standard,
        max_questions=args.max_questions
    )
    
    # Aktualisiere Gold Standard wenn gewünscht
    if args.update:
        print("\n" + "="*80)
        print("AKTUALISIERE GOLD STANDARD")
        print("="*80)
        update_gold_standard_with_answers(args.gold_standard, results)
        print("✓ Gold Standard wurde aktualisiert!")
    else:
        print("\n" + "="*80)
        print("HINWEIS: Verwende --update um die gold_standard.json zu aktualisieren")
        print("="*80)



