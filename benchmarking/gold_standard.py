"""Helper functions for managing gold standard datasets."""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from logging_config.logger import get_logger

logger = get_logger(__name__)


def load_gold_standard(file_path: str) -> Dict:
    """
    Lädt Goldstandard-Dataset aus JSON-Datei.
    
    Args:
        file_path: Pfad zur JSON-Datei
    
    Returns:
        Dictionary mit Goldstandard-Daten
    
    Raises:
        FileNotFoundError: Wenn Datei nicht gefunden wird
        ValueError: Wenn JSON ungültig ist oder Format nicht stimmt
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in gold standard file: {e}")
    
    # Validiere Format
    if "questions" not in data:
        raise ValueError("Gold standard file must contain 'questions' field")
    
    if not isinstance(data["questions"], list):
        raise ValueError("'questions' must be a list")
    
    # Validiere jede Frage
    for i, q in enumerate(data["questions"]):
        if "question" not in q:
            raise ValueError(f"Question {i+1} missing 'question' field")
        if not isinstance(q["question"], str) or not q["question"].strip():
            raise ValueError(f"Question {i+1} has invalid 'question' field")
    
    logger.info(f"Loaded gold standard: {data.get('dataset_name', 'Unknown')} with {len(data['questions'])} questions")
    return data


def save_gold_standard(data: Dict, file_path: str) -> None:
    """
    Speichert Goldstandard-Dataset in JSON-Datei.
    
    Args:
        data: Dictionary mit Goldstandard-Daten
        file_path: Pfad zur JSON-Datei
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Füge Metadaten hinzu wenn nicht vorhanden
    if "created_at" not in data:
        data["created_at"] = datetime.now().isoformat()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved gold standard to {file_path}")


def create_gold_standard(
    dataset_name: str,
    description: str,
    questions: List[Dict],
    version: str = "1.0"
) -> Dict:
    """
    Erstellt ein neues Goldstandard-Dataset.
    
    Args:
        dataset_name: Name des Datasets
        description: Beschreibung
        questions: Liste von Fragen (jede mit 'question' und optional 'ground_truth')
        version: Versionsnummer
    
    Returns:
        Dictionary mit Goldstandard-Daten
    """
    # Validiere Fragen
    validated_questions = []
    for i, q in enumerate(questions):
        if "question" not in q:
            raise ValueError(f"Question {i+1} missing 'question' field")
        
        validated_q = {
            "id": q.get("id", i + 1),
            "question": q["question"].strip(),
            "ground_truth": q.get("ground_truth", ""),
            "category": q.get("category"),
            "expected_keywords": q.get("expected_keywords", [])
        }
        validated_questions.append(validated_q)
    
    return {
        "dataset_name": dataset_name,
        "description": description,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "questions": validated_questions
    }


def add_question_to_gold_standard(
    file_path: str,
    question: str,
    ground_truth: Optional[str] = None,
    category: Optional[str] = None,
    expected_keywords: Optional[List[str]] = None
) -> None:
    """
    Fügt eine Frage zu einem bestehenden Goldstandard hinzu.
    
    Args:
        file_path: Pfad zur Goldstandard-Datei
        question: Die Frage
        ground_truth: Optional: Erwartete Antwort
        category: Optional: Kategorie
        expected_keywords: Optional: Erwartete Keywords
    """
    data = load_gold_standard(file_path)
    
    # Finde nächste ID
    max_id = max([q.get("id", 0) for q in data["questions"]], default=0)
    new_id = max_id + 1
    
    new_question = {
        "id": new_id,
        "question": question.strip(),
        "ground_truth": ground_truth or "",
        "category": category,
        "expected_keywords": expected_keywords or []
    }
    
    data["questions"].append(new_question)
    save_gold_standard(data, file_path)
    
    logger.info(f"Added question {new_id} to gold standard")


def list_gold_standards(directory: str = "data") -> List[Dict]:
    """
    Listet alle Goldstandard-Dateien in einem Verzeichnis auf.
    
    Args:
        directory: Verzeichnis zum Durchsuchen
    
    Returns:
        Liste von Dictionaries mit Informationen zu jedem Goldstandard
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    gold_standards = []
    for json_file in dir_path.glob("*.json"):
        try:
            data = load_gold_standard(str(json_file))
            gold_standards.append({
                "file_path": str(json_file),
                "dataset_name": data.get("dataset_name", "Unknown"),
                "description": data.get("description", ""),
                "version": data.get("version", "1.0"),
                "num_questions": len(data.get("questions", [])),
                "created_at": data.get("created_at", "")
            })
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")
    
    return gold_standards


def validate_gold_standard(file_path: str) -> Dict[str, any]:
    """
    Validiert eine Goldstandard-Datei und gibt Statistiken zurück.
    
    Args:
        file_path: Pfad zur Goldstandard-Datei
    
    Returns:
        Dictionary mit Validierungs-Ergebnissen und Statistiken
    """
    try:
        data = load_gold_standard(file_path)
        
        questions = data.get("questions", [])
        categories = {}
        questions_with_ground_truth = 0
        
        for q in questions:
            # Kategorien zählen
            cat = q.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
            
            # Fragen mit Ground Truth zählen
            if q.get("ground_truth"):
                questions_with_ground_truth += 1
        
        return {
            "valid": True,
            "dataset_name": data.get("dataset_name"),
            "num_questions": len(questions),
            "questions_with_ground_truth": questions_with_ground_truth,
            "categories": categories,
            "version": data.get("version"),
            "created_at": data.get("created_at")
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }




