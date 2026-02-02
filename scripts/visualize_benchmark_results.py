#!/usr/bin/env python3
"""Skript zum Visualisieren von Benchmark-Ergebnissen aus JSON-Dateien."""
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarking.visualizer import BenchmarkVisualizer
from logging_config.logger import get_logger

logger = get_logger(__name__)


def load_results_from_json(json_path: Path) -> dict:
    """Lädt Evaluationsergebnisse aus JSON-Datei."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Konvertiere die Struktur für den Visualizer
    # Der Visualizer erwartet: {"summary": {...}, "results": [...], "questions": [...]}
    evaluation_results = data.get("evaluation_results", {})
    
    return {
        "summary": evaluation_results.get("summary", {}),
        "results": evaluation_results.get("results", []),
        "questions": data.get("questions", [])  # Für Frage-Labels
    }


def visualize_single_file(
    json_path: Path,
    visualizer: BenchmarkVisualizer,
    create_dashboard: bool = False
):
    """Visualisiert Ergebnisse aus einer einzelnen JSON-Datei."""
    logger.info(f"Loading results from {json_path}")
    
    try:
        results = load_results_from_json(json_path)
        
        # Erstelle Output-Prefix aus Dateinamen (ohne Extension)
        output_prefix = json_path.stem
        
        # Erstelle einzelne Plots
        logger.info(f"Creating individual plots for {output_prefix}...")
        plots = visualizer.create_individual_plots(
            results=results,
            output_prefix=output_prefix
        )
        logger.info(f"Created {len(plots)} individual plots")
        
        # Optional: Erstelle auch Dashboard (wenn gewünscht)
        if create_dashboard:
            dashboard_path = visualizer.output_dir / f"{output_prefix}_dashboard.html"
            visualizer.create_dashboard(results, save_path=str(dashboard_path))
            logger.info(f"Created dashboard at {dashboard_path}")
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}", exc_info=True)
        raise


def visualize_all_results(
    results_dir: Path,
    visualizer: BenchmarkVisualizer,
    create_dashboard: bool = False
):
    """Visualisiert alle JSON-Dateien im angegebenen Verzeichnis."""
    json_files = list(results_dir.glob("eval_*.json"))
    
    if not json_files:
        logger.warning(f"No evaluation JSON files found in {results_dir}")
        return
    
    logger.info(f"Found {len(json_files)} evaluation files to visualize")
    
    for json_file in sorted(json_files):
        try:
            visualize_single_file(json_file, visualizer, create_dashboard)
        except Exception as e:
            logger.error(f"Failed to visualize {json_file}: {e}")
            continue


def main():
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(
        description="Visualisiere Benchmark-Ergebnisse aus JSON-Dateien"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Pfad zu einer spezifischen JSON-Datei (optional)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/benchmark_results",
        help="Verzeichnis mit Benchmark-Ergebnissen (Standard: data/benchmark_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Ausgabeverzeichnis für Plots (Standard: results-dir/plots)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Erstelle zusätzlich ein zusammengefasstes Dashboard"
    )
    
    args = parser.parse_args()
    
    # Initialisiere Visualizer
    output_dir = args.output_dir or args.results_dir
    visualizer = BenchmarkVisualizer(output_dir=output_dir)
    
    if args.file:
        # Visualisiere einzelne Datei
        json_path = Path(args.file)
        if not json_path.exists():
            logger.error(f"Datei nicht gefunden: {json_path}")
            return 1
        
        visualize_single_file(json_path, visualizer, create_dashboard=args.dashboard)
    else:
        # Visualisiere alle Dateien im Verzeichnis
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.error(f"Verzeichnis nicht gefunden: {results_dir}")
            return 1
        
        visualize_all_results(results_dir, visualizer, create_dashboard=args.dashboard)
    
    logger.info("Visualisierung abgeschlossen!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

