"""Visualization for benchmarking results."""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import os
from pathlib import Path
from logging_config.logger import get_logger

logger = get_logger(__name__)


class BenchmarkVisualizer:
    """Visualize benchmarking results."""
    
    def __init__(self, output_dir: str = None):
        """Initialize visualizer."""
        if output_dir is None:
            output_dir = os.getenv("BENCHMARK_OUTPUT_DIR", "./data/benchmark_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_ragas_metrics(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create a bar chart of RAGAS metrics."""
        summary = results.get("summary", {})
        
        metrics = list(summary.keys())
        values = list(summary.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color='lightblue',
                text=[f"{v:.3f}" for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="RAGAS Evaluation Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_metric_distribution(
        self,
        results: Dict,
        metric_name: str,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot distribution of a specific metric."""
        results_list = results.get("results", [])
        values = [r.get(metric_name, 0) for r in results_list]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=values,
                nbinsx=20,
                marker_color='skyblue'
            )
        ])
        
        fig.update_layout(
            title=f"Distribution of {metric_name}",
            xaxis_title=metric_name,
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_comparison(
        self,
        results_list: List[Dict],
        labels: List[str],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Compare multiple benchmark runs."""
        # Base metrics that are always available
        base_metrics = ["faithfulness", "answer_relevancy", "context_recall"]
        # Check if answer_correctness is available in any result
        has_answer_correctness = any(
            "answer_correctness" in result.get("summary", {}) 
            for result in results_list
        )
        
        metrics = base_metrics.copy()
        if has_answer_correctness:
            metrics.append("answer_correctness")
        
        fig = go.Figure()
        
        for i, (result, label) in enumerate(zip(results_list, labels)):
            summary = result.get("summary", {})
            values = [summary.get(m, 0) for m in metrics]
            fig.add_trace(go.Bar(
                name=label,
                x=metrics,
                y=values
            ))
        
        fig.update_layout(
            title="RAGAS Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            barmode='group',
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create a comprehensive dashboard."""
        summary = results.get("summary", {})
        results_list = results.get("results", [])
        
        # Check if answer_correctness is available
        has_answer_correctness = "answer_correctness" in summary
        
        # Determine layout based on available metrics
        if has_answer_correctness:
            # Create subplots with answer_correctness
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=("Metrics Overview", "Faithfulness Distribution", 
                              "Answer Relevancy Distribution", 
                              "Answer Correctness Distribution", "Context Recall", "Context Metrics"),
                specs=[[{"type": "bar"}, {"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}]]
            )
        else:
            # Original layout without answer_correctness
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Metrics Overview", "Faithfulness Distribution", 
                              "Answer Relevancy Distribution", "Context Metrics"),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "bar"}]]
            )
        
        # Metrics overview
        metrics = list(summary.keys())
        values = list(summary.values())
        fig.add_trace(
            go.Bar(x=metrics, y=values, name="Scores", text=[f"{v:.3f}" for v in values], textposition='outside'),
            row=1, col=1
        )
        
        # Faithfulness distribution
        faithfulness_values = [r.get("faithfulness", 0) for r in results_list]
        fig.add_trace(
            go.Histogram(x=faithfulness_values, name="Faithfulness", nbinsx=20),
            row=1, col=2
        )
        
        # Answer relevancy distribution
        relevancy_values = [r.get("answer_relevancy", 0) for r in results_list]
        if has_answer_correctness:
            fig.add_trace(
                go.Histogram(x=relevancy_values, name="Answer Relevancy", nbinsx=20),
                row=1, col=3
            )
        else:
            fig.add_trace(
                go.Histogram(x=relevancy_values, name="Answer Relevancy", nbinsx=20),
                row=2, col=1
            )
        
        # Answer correctness distribution (if available)
        if has_answer_correctness:
            correctness_values = [r.get("answer_correctness", 0) for r in results_list]
            fig.add_trace(
                go.Histogram(x=correctness_values, name="Answer Correctness", nbinsx=20),
                row=2, col=1
            )
            
            # Context recall
            context_recall_value = summary.get("context_recall", 0)
            fig.add_trace(
                go.Bar(x=["Context Recall"], y=[context_recall_value], name="Context Recall", 
                      text=[f"{context_recall_value:.3f}"], textposition='outside'),
                row=2, col=2
            )
            
            # Other metrics bar
            other_metrics = [m for m in metrics if m not in ["context_recall", "answer_correctness"]]
            other_values = [summary.get(m, 0) for m in other_metrics]
            fig.add_trace(
                go.Bar(x=other_metrics, y=other_values, name="Other Metrics",
                      text=[f"{v:.3f}" for v in other_values], textposition='outside'),
                row=2, col=3
            )
        else:
            # Context metrics (original layout)
            context_metrics = ["context_recall"]
            context_values = [summary.get(m, 0) for m in context_metrics]
            fig.add_trace(
                go.Bar(x=context_metrics, y=context_values, name="Context",
                      text=[f"{v:.3f}" for v in context_values], textposition='outside'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="RAGAS Benchmarking Dashboard",
            template="plotly_white",
            height=800 if not has_answer_correctness else 900
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved dashboard to {save_path}")
        
        return fig


