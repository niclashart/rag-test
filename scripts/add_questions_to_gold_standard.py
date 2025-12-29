#!/usr/bin/env python3
"""Fügt neue Fragen zur gold_standard.json hinzu."""
import json
from pathlib import Path

# 9 neue technische Spezifikations-Fragen für Dell Pro 16 PC16250
new_questions = [
    {
        "id": 2,
        "question": "Welche RAM-Konfigurationen hat das Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 3,
        "question": "Welche Storage-Optionen (SSD) sind für das Dell Pro 16 PC16250 verfügbar?",
        "ground_truth": ""
    },
    {
        "id": 4,
        "question": "Welche Display-Größe und Auflösung hat das Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 5,
        "question": "Wie hoch ist die Display-Helligkeit (in nits) des Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 6,
        "question": "Welche Akku-Kapazität hat das Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 7,
        "question": "Wie viel wiegt das Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 8,
        "question": "Welche Abmessungen (Breite x Tiefe x Höhe) hat das Dell Pro 16 PC16250?",
        "ground_truth": ""
    },
    {
        "id": 9,
        "question": "Welche Grafikkarte (GPU) ist im Dell Pro 16 PC16250 verbaut?",
        "ground_truth": ""
    },
    {
        "id": 10,
        "question": "Welche Anschlüsse und Ports hat das Dell Pro 16 PC16250?",
        "ground_truth": ""
    }
]

def add_questions_to_gold_standard(file_path: str, questions: list):
    """Fügt Fragen zur gold_standard.json hinzu (JSONL-Format)."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {file_path}")
    
    # Lese alle bestehenden Zeilen
    existing_lines = []
    existing_ids = set()
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    q = json.loads(line)
                    existing_lines.append(line)
                    existing_ids.add(q.get("id"))
                except json.JSONDecodeError:
                    # Ungültige Zeile, überspringen
                    continue
    
    # Füge neue Fragen hinzu (nur wenn ID noch nicht existiert)
    added_count = 0
    for question in questions:
        q_id = question.get("id")
        if q_id not in existing_ids:
            # Finde die richtige Position (nach ID sortiert)
            inserted = False
            for i, line in enumerate(existing_lines):
                try:
                    existing_q = json.loads(line)
                    if existing_q.get("id", 0) > q_id:
                        existing_lines.insert(i, json.dumps(question, ensure_ascii=False))
                        inserted = True
                        added_count += 1
                        break
                except json.JSONDecodeError:
                    continue
            
            if not inserted:
                # Am Ende hinzufügen
                existing_lines.append(json.dumps(question, ensure_ascii=False))
                added_count += 1
        else:
            # ID existiert bereits, überschreibe die Zeile wenn sie leer ist
            for i, line in enumerate(existing_lines):
                try:
                    existing_q = json.loads(line)
                    if existing_q.get("id") == q_id:
                        # Prüfe ob Frage leer ist
                        if not existing_q.get("question", "").strip():
                            existing_lines[i] = json.dumps(question, ensure_ascii=False)
                            added_count += 1
                        break
                except json.JSONDecodeError:
                    continue
    
    # Schreibe alle Zeilen zurück
    with open(path, 'w', encoding='utf-8') as f:
        for line in existing_lines:
            f.write(line + '\n')
    
    print(f"✓ {added_count} Fragen hinzugefügt/aktualisiert")
    print(f"✓ Gesamt: {len(existing_lines)} Fragen in der Datei")
    
    return added_count

if __name__ == "__main__":
    file_path = "data/gold_standard.json"
    add_questions_to_gold_standard(file_path, new_questions)
    
    # Zeige die hinzugefügten Fragen
    print("\nHinzugefügte Fragen:")
    print("-" * 80)
    for q in new_questions:
        print(f"ID {q['id']}: {q['question']}")










