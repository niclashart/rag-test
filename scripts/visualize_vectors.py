#!/usr/bin/env python3
"""Visualisiere Vektoren direkt aus ChromaDB."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import sqlite3
import argparse
from logging_config.logger import get_logger

logger = get_logger(__name__)


def load_vectors_from_chromadb(db_path: str = None, collection_name: str = None, limit: int = None):
    """Lade Vektoren direkt aus ChromaDB.
    
    Args:
        db_path: Pfad zur ChromaDB Datenbank
        collection_name: Name der Collection (optional)
        limit: Maximale Anzahl von Vektoren (optional)
    
    Returns:
        Tuple von (embeddings, ids, metadatas, documents)
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    
    logger.info(f"Lade Vektoren aus ChromaDB: {db_path}")
    
    # Initialisiere ChromaDB Client
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Finde alle Collections
    collections = client.list_collections()
    
    if not collections:
        logger.warning("Keine Collections in ChromaDB gefunden!")
        return None, None, None, None
    
    logger.info(f"Gefundene Collections: {[c.name for c in collections]}")
    
    # Verwende erste Collection oder spezifische Collection
    if collection_name:
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Collection '{collection_name}' nicht gefunden: {e}")
            collection = collections[0]
    else:
        collection = collections[0]
    
    logger.info(f"Verwende Collection: {collection.name}")
    
    # Lade alle Vektoren
    try:
        # Get all data from collection
        results = collection.get(include=['embeddings', 'metadatas', 'documents'])
        
        embeddings = results.get('embeddings', [])
        ids = results.get('ids', [])
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])
        
        # Konvertiere zu numpy array falls noch nicht
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        elif not isinstance(embeddings, np.ndarray):
            logger.warning("Keine Embeddings in Collection gefunden!")
            return None, None, None, None
        
        if len(embeddings) == 0:
            logger.warning("Keine Embeddings in Collection gefunden!")
            return None, None, None, None
        
        logger.info(f"Geladen: {len(embeddings)} Vektoren")
        
        # Limit falls angegeben
        if limit and limit < len(embeddings):
            embeddings = embeddings[:limit]
            ids = ids[:limit]
            metadatas = metadatas[:limit] if metadatas else None
            documents = documents[:limit] if documents else None
            logger.info(f"Begrenzt auf {limit} Vektoren")
        
        return np.array(embeddings), ids, metadatas, documents
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der Vektoren: {e}", exc_info=True)
        return None, None, None, None


def visualize_vector_space_2d(embeddings: np.ndarray, ids: List[str], 
                             metadatas: List[Dict] = None, documents: List[str] = None,
                             method: str = 'tsne', output_path: str = None):
    """Visualisiere Vektoren im 2D Raum mit t-SNE oder UMAP.
    
    Args:
        embeddings: Array von Embeddings (n_samples, n_features)
        ids: Liste von Chunk-IDs
        metadatas: Liste von Metadaten-Dictionaries (optional)
        documents: Liste von Dokument-Texten (optional)
        method: 'tsne' oder 'umap'
        output_path: Pfad zum Speichern der Visualisierung
    """
    logger.info(f"Erstelle 2D Visualisierung mit {method}...")
    
    # Dimensionsreduktion
    if method.lower() == 'tsne':
        try:
            from sklearn.manifold import TSNE
            logger.info("Berechne t-SNE...")
            perplexity = min(30, max(5, len(embeddings) // 4))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            logger.error("scikit-learn nicht installiert. Installiere mit: pip install scikit-learn")
            return None
    elif method.lower() == 'umap':
        try:
            from umap import UMAP
            logger.info("Berechne UMAP...")
            n_neighbors = min(15, max(5, len(embeddings) // 10))
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            logger.error("UMAP nicht installiert. Installiere mit: pip install umap-learn")
            return None
    else:
        logger.error(f"Unbekannte Methode: {method}. Verwende 'tsne' oder 'umap'")
        return None
    
    # Erstelle DataFrame für Visualisierung
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'id': ids
    })
    
    # Füge Metadaten hinzu falls verfügbar
    if metadatas:
        for i, meta in enumerate(metadatas):
            if meta:
                df.loc[i, 'document_id'] = meta.get('document_id', 'unknown')
                df.loc[i, 'page_number'] = meta.get('page_number', 'unknown')
                df.loc[i, 'chunk_index'] = meta.get('chunk_index', 'unknown')
    
    # Füge Dokument-Text hinzu falls verfügbar
    if documents:
        df['text_preview'] = [doc[:100] + '...' if len(doc) > 100 else doc for doc in documents]
    
    # Erstelle Visualisierung
    if metadatas and any(m.get('document_id') for m in metadatas if m):
        # Gruppiere nach Dokument
        color_col = 'document_id'
        hover_data = {'text': df.get('text_preview', '')}
    else:
        color_col = None
        hover_data = {}
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=color_col,
        hover_data=hover_data,
        hover_name='id',
        title=f'Vektor-Raum Visualisierung ({method.upper()}) - {len(embeddings)} Vektoren',
        labels={'x': f'{method.upper()} Dimension 1', 'y': f'{method.upper()} Dimension 2'}
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True
    )
    
    if output_path:
        try:
            # Versuche als PNG zu speichern
            if output_path.endswith('.html'):
                output_path = output_path.replace('.html', '.png')
            fig.write_image(output_path, width=1200, height=800, scale=2)
            logger.info(f"Visualisierung gespeichert: {output_path}")
        except Exception as e:
            logger.warning(f"PNG-Export fehlgeschlagen: {e}")
            logger.info("Versuche HTML-Export...")
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            logger.info(f"HTML-Version gespeichert: {html_path}")
    
    return fig


def visualize_vector_statistics(embeddings: np.ndarray, output_path: str = None):
    """Visualisiere Statistiken über die Vektoren.
    
    Args:
        embeddings: Array von Embeddings
        output_path: Pfad zum Speichern
    """
    logger.info("Erstelle Vektor-Statistiken...")
    
    # Berechne Statistiken
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    min_embedding = np.min(embeddings, axis=0)
    max_embedding = np.max(embeddings, axis=0)
    
    # Berechne Dimension-Statistiken
    dim_stats = pd.DataFrame({
        'mean': mean_embedding,
        'std': std_embedding,
        'min': min_embedding,
        'max': max_embedding,
        'dimension': range(len(mean_embedding))
    })
    
    # Erstelle Subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mittelwert pro Dimension', 'Standardabweichung pro Dimension',
                       'Min/Max pro Dimension', 'Vektor-Normen Verteilung'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Mittelwert
    fig.add_trace(
        go.Scatter(x=dim_stats['dimension'], y=dim_stats['mean'],
                  mode='lines+markers', name='Mittelwert',
                  marker=dict(color='blue')),
        row=1, col=1
    )
    
    # Standardabweichung
    fig.add_trace(
        go.Scatter(x=dim_stats['dimension'], y=dim_stats['std'],
                  mode='lines+markers', name='Std',
                  marker=dict(color='red')),
        row=1, col=2
    )
    
    # Min/Max
    fig.add_trace(
        go.Scatter(x=dim_stats['dimension'], y=dim_stats['min'],
                  mode='lines+markers', name='Min',
                  marker=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dim_stats['dimension'], y=dim_stats['max'],
                  mode='lines+markers', name='Max',
                  marker=dict(color='orange')),
        row=2, col=1
    )
    
    # Vektor-Normen
    norms = np.linalg.norm(embeddings, axis=1)
    fig.add_trace(
        go.Histogram(x=norms, nbinsx=50, name='Normen',
                    marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='Vektor-Statistiken',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Dimension", row=1, col=1)
    fig.update_xaxes(title_text="Dimension", row=1, col=2)
    fig.update_xaxes(title_text="Dimension", row=2, col=1)
    fig.update_xaxes(title_text="Norm", row=2, col=2)
    fig.update_yaxes(title_text="Wert", row=1, col=1)
    fig.update_yaxes(title_text="Std", row=1, col=2)
    fig.update_yaxes(title_text="Wert", row=2, col=1)
    fig.update_yaxes(title_text="Häufigkeit", row=2, col=2)
    
    if output_path:
        try:
            # Versuche als PNG zu speichern
            if output_path.endswith('.html'):
                output_path = output_path.replace('.html', '.png')
            fig.write_image(output_path, width=1200, height=800, scale=2)
            logger.info(f"Statistiken gespeichert: {output_path}")
        except Exception as e:
            logger.warning(f"PNG-Export fehlgeschlagen: {e}")
            logger.info("Versuche HTML-Export...")
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            logger.info(f"HTML-Version gespeichert: {html_path}")
    
    return fig


def visualize_vector_similarity_matrix(embeddings: np.ndarray, ids: List[str],
                                      max_vectors: int = 100, output_path: str = None):
    """Visualisiere Similarity-Matrix zwischen Vektoren.
    
    Args:
        embeddings: Array von Embeddings
        ids: Liste von IDs
        max_vectors: Maximale Anzahl für Matrix (wegen Speicher)
        output_path: Pfad zum Speichern
    """
    logger.info(f"Erstelle Similarity-Matrix (max {max_vectors} Vektoren)...")
    
    # Limit für Performance
    if len(embeddings) > max_vectors:
        logger.info(f"Begrenze auf {max_vectors} Vektoren für Matrix")
        embeddings = embeddings[:max_vectors]
        ids = ids[:max_vectors]
    
    # Berechne Cosine Similarity
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.error("scikit-learn nicht installiert")
        return None
    
    similarity_matrix = cosine_similarity(embeddings)
    
    # Visualisiere als Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"V{i+1}" for i in range(len(ids))],
        y=[f"V{i+1}" for i in range(len(ids))],
        colorscale='RdYlBu',
        zmid=0.5,
        colorbar=dict(title="Cosine Similarity"),
        text=similarity_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=f'Vektor-Similarity-Matrix ({len(embeddings)} Vektoren)',
        xaxis_title='Vektor',
        yaxis_title='Vektor',
        height=800,
        width=1000
    )
    
    if output_path:
        try:
            # Versuche als PNG zu speichern
            if output_path.endswith('.html'):
                output_path = output_path.replace('.html', '.png')
            fig.write_image(output_path, width=1000, height=800, scale=2)
            logger.info(f"Similarity-Matrix gespeichert: {output_path}")
        except Exception as e:
            logger.warning(f"PNG-Export fehlgeschlagen: {e}")
            logger.info("Versuche HTML-Export...")
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            logger.info(f"HTML-Version gespeichert: {html_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualisiere Vektoren aus ChromaDB')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Pfad zur ChromaDB (Standard: data/chroma_db)')
    parser.add_argument('--collection', type=str, default=None,
                       help='Collection Name (Standard: erste gefundene)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximale Anzahl von Vektoren')
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'], default='tsne',
                       help='Dimensionsreduktion Methode (Standard: tsne)')
    parser.add_argument('--output-dir', type=str, default='plots/generated_plots',
                       help='Output-Verzeichnis')
    parser.add_argument('--all', action='store_true',
                       help='Erstelle alle Visualisierungen')
    
    args = parser.parse_args()
    
    # Lade Vektoren
    embeddings, ids, metadatas, documents = load_vectors_from_chromadb(
        db_path=args.db_path,
        collection_name=args.collection,
        limit=args.limit
    )
    
    if embeddings is None:
        logger.error("Konnte keine Vektoren laden!")
        return 1
    
    # Erstelle Output-Verzeichnis
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Geladen: {len(embeddings)} Vektoren mit {embeddings.shape[1]} Dimensionen")
    
    # Erstelle Visualisierungen
    if args.all:
        # Alle Visualisierungen
        logger.info("Erstelle alle Visualisierungen...")
        
        # 1. 2D Visualisierung (t-SNE)
        visualize_vector_space_2d(
            embeddings, ids, metadatas, documents,
            method='tsne',
            output_path=str(output_dir / 'vector_space_tsne.png')
        )
        
        # 2. 2D Visualisierung (UMAP)
        try:
            visualize_vector_space_2d(
                embeddings, ids, metadatas, documents,
                method='umap',
                output_path=str(output_dir / 'vector_space_umap.png')
            )
        except Exception as e:
            logger.warning(f"UMAP Visualisierung fehlgeschlagen: {e}")
        
        # 3. Statistiken
        visualize_vector_statistics(
            embeddings,
            output_path=str(output_dir / 'vector_statistics.png')
        )
        
        # 4. Similarity Matrix
        visualize_vector_similarity_matrix(
            embeddings, ids,
            max_vectors=100,
            output_path=str(output_dir / 'vector_similarity_matrix.png')
        )
        
        logger.info(f"\nAlle Visualisierungen gespeichert in: {output_dir}")
    else:
        # Nur 2D Visualisierung
        visualize_vector_space_2d(
            embeddings, ids, metadatas, documents,
            method=args.method,
            output_path=str(output_dir / f'vector_space_{args.method}.png')
        )
        logger.info(f"\nVisualisierung gespeichert in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

