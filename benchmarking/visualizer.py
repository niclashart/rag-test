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
    
    def create_individual_plots(
        self,
        results: Dict,
        output_prefix: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """
        Erstellt einzelne Plots für jede Metrik statt eines zusammengefassten Dashboards.
        
        Args:
            results: Dictionary mit Evaluationsergebnissen (inkl. "summary", "results", "questions")
            output_prefix: Optional prefix für Ausgabedateien (ohne Extension)
        
        Returns:
            Dictionary mit Plot-Namen als Keys und Figure-Objekten als Values
        """
        summary = results.get("summary", {})
        results_list = results.get("results", [])
        questions = results.get("questions", [])
        
        plots = {}
        
        # 1. Metrics Overview (Bar Chart)
        metrics = list(summary.keys())
        values = [v * 100 for v in list(summary.values())]  # Convert to percentage
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(metrics)]
        fig_metrics = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            )
        ])
        fig_metrics.update_layout(
            title="Summary Metrics",
            xaxis_title="Metric",
            yaxis_title="Score (%)",
            yaxis_range=[0, 100],
            template="plotly_white",
            height=500
        )
        plots["metrics_overview"] = fig_metrics
        
        # 2. Individual metric distributions
        metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
        if "answer_correctness" in summary:
            metric_names.append("answer_correctness")
        
        for metric_name in metric_names:
            if metric_name in summary:
                values = [r.get(metric_name, 0) * 100 for r in results_list]  # Convert to percentage
                fig_dist = go.Figure(data=[
                    go.Histogram(
                        x=values,
                        nbinsx=20,
                        marker_color='skyblue',
                        name=metric_name
                    )
                ])
                fig_dist.update_layout(
                    title=f"{metric_name.replace('_', ' ').title()} Distribution",
                    xaxis_title="Score (%)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=500
                )
                plots[f"{metric_name}_distribution"] = fig_dist
        
        # 3. Metrics Across Questions (Line Chart)
        if results_list:
            # Create question labels
            question_labels = []
            for i, q in enumerate(questions):
                q_id = q.get("id", i+1) if isinstance(q, dict) else i+1
                question_labels.append(f"Q{q_id}")
            
            if not question_labels:
                question_labels = [f"Q{i+1}" for i in range(len(results_list))]
            
            # Determine available metrics
            base_metrics = ["faithfulness", "answer_relevancy", "context_recall"]
            has_answer_correctness = "answer_correctness" in summary
            available_metrics = base_metrics.copy()
            if has_answer_correctness:
                available_metrics.append("answer_correctness")
            
            fig_line = go.Figure()
            colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            markers = ['circle', 'square', 'triangle-up', 'diamond']
            linestyles = ['solid', 'dash', 'dot', 'dashdot']
            
            x_pos = list(range(len(question_labels)))
            
            for i, metric in enumerate(available_metrics):
                if metric in summary:
                    values = [r.get(metric, 0) * 100 for r in results_list]  # Convert to percentage
                    fig_line.add_trace(go.Scatter(
                        x=x_pos,
                        y=values,
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors_line[i % len(colors_line)], dash=linestyles[i % len(linestyles)], width=2),
                        marker=dict(symbol=markers[i % len(markers)], size=8)
                    ))
            
            fig_line.update_layout(
                title="Metrics Across Questions",
                xaxis_title="Question",
                yaxis_title="Score (%)",
                yaxis_range=[0, 100],
                template="plotly_white",
                height=600,
                xaxis=dict(tickmode='array', tickvals=x_pos, ticktext=question_labels, tickangle=45)
            )
            plots["metrics_across_questions"] = fig_line
        
        # 4. Faithfulness vs Answer Relevancy (Scatter Plot)
        if results_list:
            faithfulness = [r.get("faithfulness", 0) * 100 for r in results_list]
            answer_relevancy = [r.get("answer_relevancy", 0) * 100 for r in results_list]
            
            fig_scatter1 = go.Figure()
            fig_scatter1.add_trace(go.Scatter(
                x=faithfulness,
                y=answer_relevancy,
                mode='markers',
                marker=dict(size=10, color='#1f77b4', opacity=0.6, line=dict(width=1.5, color='black')),
                name='Data Points'
            ))
            # Add perfect correlation line
            fig_scatter1.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                name='Perfect correlation'
            ))
            fig_scatter1.update_layout(
                title="Faithfulness vs Answer Relevancy",
                xaxis_title="Faithfulness (%)",
                yaxis_title="Answer Relevancy (%)",
                xaxis_range=[0, 100],
                yaxis_range=[0, 100],
                template="plotly_white",
                height=600
            )
            plots["faithfulness_vs_answer_relevancy"] = fig_scatter1
        
        # 5. Faithfulness vs Context Recall (Scatter Plot)
        if results_list:
            faithfulness = [r.get("faithfulness", 0) * 100 for r in results_list]
            context_recall = [r.get("context_recall", 0) * 100 for r in results_list]
            
            fig_scatter2 = go.Figure()
            fig_scatter2.add_trace(go.Scatter(
                x=faithfulness,
                y=context_recall,
                mode='markers',
                marker=dict(size=10, color='#ff7f0e', opacity=0.6, line=dict(width=1.5, color='black')),
                name='Data Points'
            ))
            # Add perfect correlation line
            fig_scatter2.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                name='Perfect correlation'
            ))
            fig_scatter2.update_layout(
                title="Faithfulness vs Context Recall",
                xaxis_title="Faithfulness (%)",
                yaxis_title="Context Recall (%)",
                xaxis_range=[0, 100],
                yaxis_range=[0, 100],
                template="plotly_white",
                height=600
            )
            plots["faithfulness_vs_context_recall"] = fig_scatter2
        
        # 6. Speichere alle Plots einzeln als PNG
        if output_prefix:
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            for plot_name, fig in plots.items():
                plot_path = plots_dir / f"{output_prefix}_{plot_name}.png"
                try:
                    fig.write_image(str(plot_path), width=1200, height=800, scale=2)
                    logger.info(f"Saved individual plot to {plot_path}")
                except Exception as e:
                    logger.warning(f"Could not save PNG, falling back to HTML: {e}")
                    # Fallback zu HTML falls PNG nicht funktioniert
                    html_path = plots_dir / f"{output_prefix}_{plot_name}.html"
                    fig.write_html(str(html_path))
                    logger.info(f"Saved individual plot as HTML to {html_path}")
        
        return plots


