#!/usr/bin/env python3
"""Plot evaluation scores from RAGAS evaluation JSON files."""
import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 10

def load_evaluation_results(json_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_summary_metrics(data: dict, output_path: str = None):
    """Plot summary metrics as bar chart."""
    summary = data.get("evaluation_results", {}).get("summary", {})
    
    if not summary:
        print("No summary metrics found in data")
        return None
    
    metrics = list(summary.keys())
    values = [summary[m] * 100 for m in metrics]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(metrics)]
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(f"RAGAS Evaluation Summary Metrics\n{data.get('dataset_info', {}).get('name', 'Unknown Dataset')}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary metrics plot saved to {output_path}")
        plt.close(fig)
    else:
        return fig

def plot_metric_distribution(data: dict, output_path: str = None):
    """Plot distribution of metrics across all questions."""
    results = data.get("evaluation_results", {}).get("results", [])
    
    if not results:
        print("No results found in data")
        return None
    
    # Extract metrics
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    available_metrics = [m for m in metrics if any(m in r for r in results)]
    
    if not available_metrics:
        print("No metrics found in results")
        return None
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(available_metrics):
        values = [r.get(metric, 0) * 100 for r in results]  # Convert to percentage
        
        axes[i].hist(values, bins=10, color=colors[i % len(colors)], edgecolor='black', alpha=0.7)
        axes[i].set_xlabel('Score (%)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i].set_xlim(0, 100)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle("Distribution of Metrics Across Questions", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
        plt.close(fig)
    else:
        return fig

def plot_metrics_over_questions(data: dict, output_path: str = None):
    """Plot metrics for each question as line chart."""
    results = data.get("evaluation_results", {}).get("results", [])
    questions = data.get("questions", [])
    
    if not results:
        print("No results found in data")
        return None
    
    # Extract metrics
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    available_metrics = [m for m in metrics if any(m in r for r in results)]
    
    if not available_metrics:
        print("No metrics found in results")
        return None
    
    # Create question labels
    question_labels = []
    for i, q in enumerate(questions):
        q_id = q.get("id", i+1)
        question_labels.append(f"Q{q_id}")
    
    if not question_labels:
        question_labels = [f"Q{i+1}" for i in range(len(results))]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    x_pos = np.arange(len(question_labels))
    
    for i, metric in enumerate(available_metrics):
        values = [r.get(metric, 0) * 100 for r in results]  # Convert to percentage
        
        ax.plot(x_pos, values, 
                marker=markers[i % len(markers)], 
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=8,
                label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title("Metrics Across Questions", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(question_labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics over questions plot saved to {output_path}")
        plt.close(fig)
    else:
        return fig

def plot_metric_comparison(data: dict, output_path: str = None):
    """Create a scatter plot comparing different metrics."""
    results = data.get("evaluation_results", {}).get("results", [])
    
    if not results:
        print("No results found in data")
        return None
    
    # Extract metrics
    faithfulness = [r.get("faithfulness", 0) * 100 for r in results]
    answer_relevancy = [r.get("answer_relevancy", 0) * 100 for r in results]
    context_recall = [r.get("context_recall", 0) * 100 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Faithfulness vs Answer Relevancy
    ax1.scatter(faithfulness, answer_relevancy, s=100, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=1.5)
    ax1.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1, label='Perfect correlation')
    ax1.set_xlabel('Faithfulness (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Answer Relevancy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Faithfulness vs Answer Relevancy', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Faithfulness vs Context Recall
    ax2.scatter(faithfulness, context_recall, s=100, alpha=0.6, color='#ff7f0e', edgecolors='black', linewidth=1.5)
    ax2.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1, label='Perfect correlation')
    ax2.set_xlabel('Faithfulness (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Context Recall (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Faithfulness vs Context Recall', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    fig.suptitle("Metric Comparison Scatter Plots", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
        plt.close(fig)
    else:
        return fig

def create_comprehensive_dashboard(data: dict, output_path: str = None):
    """Create a comprehensive dashboard with all plots."""
    summary = data.get("evaluation_results", {}).get("summary", {})
    results = data.get("evaluation_results", {}).get("results", [])
    dataset_info = data.get("dataset_info", {})
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Summary Metrics (Bar Chart) - Top left
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = list(summary.keys())
    values = [summary[m] * 100 for m in metrics]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(metrics)]
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Summary Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Distribution (Histogram) - Top right
    ax2 = fig.add_subplot(gs[0, 1])
    if results:
        faithfulness_values = [r.get("faithfulness", 0) * 100 for r in results]
        ax2.hist(faithfulness_values, bins=10, color='#1f77b4', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Score (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Faithfulness Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Metrics Over Questions (Line Chart) - Bottom left (span 2 columns)
    ax3 = fig.add_subplot(gs[1, :])
    if results:
        questions = data.get("questions", [])
        question_labels = [f"Q{q.get('id', i+1)}" for i, q in enumerate(questions)] if questions else [f"Q{i+1}" for i in range(len(results))]
        
        available_metrics = ["faithfulness", "answer_relevancy", "context_recall"]
        plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        linestyles = ['-', '--', '-.']
        
        x_pos = np.arange(len(question_labels))
        
        for i, metric in enumerate(available_metrics):
            if any(metric in r for r in results):
                values = [r.get(metric, 0) * 100 for r in results]
                ax3.plot(x_pos, values, 
                         marker=markers[i], 
                         linestyle=linestyles[i],
                         color=plot_colors[i],
                         linewidth=2,
                         markersize=8,
                         label=metric.replace('_', ' ').title())
        
        ax3.set_xlabel('Question', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Metrics Across Questions', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(question_labels, rotation=45, ha='right')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Comparison Scatter (Faithfulness vs Answer Relevancy) - Bottom left
    ax4 = fig.add_subplot(gs[2, 0])
    if results:
        faithfulness = [r.get("faithfulness", 0) * 100 for r in results]
        answer_relevancy = [r.get("answer_relevancy", 0) * 100 for r in results]
        
        ax4.scatter(faithfulness, answer_relevancy, s=100, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=1.5)
        ax4.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1)
        ax4.set_xlabel('Faithfulness (%)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Answer Relevancy (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Faithfulness vs Answer Relevancy', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Comparison Scatter (Faithfulness vs Context Recall) - Bottom right
    ax5 = fig.add_subplot(gs[2, 1])
    if results:
        faithfulness = [r.get("faithfulness", 0) * 100 for r in results]
        context_recall = [r.get("context_recall", 0) * 100 for r in results]
        
        ax5.scatter(faithfulness, context_recall, s=100, alpha=0.6, color='#ff7f0e', edgecolors='black', linewidth=1.5)
        ax5.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1)
        ax5.set_xlabel('Faithfulness (%)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Context Recall (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Faithfulness vs Context Recall', fontsize=12, fontweight='bold')
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Main title
    dataset_name = dataset_info.get("name", "Unknown Dataset")
    num_questions = dataset_info.get("num_questions", len(results))
    fig.suptitle(f"RAGAS Evaluation Dashboard\n{dataset_name} - {num_questions} Questions", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive dashboard saved to {output_path}")
        plt.close(fig)
    else:
        return fig

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation scores from RAGAS evaluation JSON files")
    parser.add_argument("json_file", type=str, help="Path to evaluation JSON file")
    parser.add_argument("--output-dir", type=str, default="data/benchmark_results/plots", help="Output directory for plots")
    parser.add_argument("--format", type=str, choices=['png', 'pdf', 'both'], default='png', help="Output format (default: png)")
    parser.add_argument("--dashboard", action="store_true", help="Create comprehensive dashboard")
    parser.add_argument("--all", action="store_true", help="Create all individual plots")
    
    args = parser.parse_args()
    
    # Load data
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading evaluation results from {json_path}")
    data = load_evaluation_results(str(json_path))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base filename
    base_name = json_path.stem
    
    # Determine file extension
    ext = args.format if args.format != 'both' else 'png'
    
    if args.dashboard or (not args.all and not args.dashboard):
        # Create comprehensive dashboard (default)
        if args.format == 'both':
            dashboard_path_png = output_dir / f"{base_name}_dashboard.png"
            dashboard_path_pdf = output_dir / f"{base_name}_dashboard.pdf"
            create_comprehensive_dashboard(data, str(dashboard_path_png))
            create_comprehensive_dashboard(data, str(dashboard_path_pdf))
        else:
            dashboard_path = output_dir / f"{base_name}_dashboard.{ext}"
            create_comprehensive_dashboard(data, str(dashboard_path))
    
    if args.all:
        # Create all individual plots
        if args.format == 'both':
            plot_summary_metrics(data, str(output_dir / f"{base_name}_summary.png"))
            plot_summary_metrics(data, str(output_dir / f"{base_name}_summary.pdf"))
            plot_metric_distribution(data, str(output_dir / f"{base_name}_distribution.png"))
            plot_metric_distribution(data, str(output_dir / f"{base_name}_distribution.pdf"))
            plot_metrics_over_questions(data, str(output_dir / f"{base_name}_questions.png"))
            plot_metrics_over_questions(data, str(output_dir / f"{base_name}_questions.pdf"))
            plot_metric_comparison(data, str(output_dir / f"{base_name}_comparison.png"))
            plot_metric_comparison(data, str(output_dir / f"{base_name}_comparison.pdf"))
        else:
            plot_summary_metrics(data, str(output_dir / f"{base_name}_summary.{ext}"))
            plot_metric_distribution(data, str(output_dir / f"{base_name}_distribution.{ext}"))
            plot_metrics_over_questions(data, str(output_dir / f"{base_name}_questions.{ext}"))
            plot_metric_comparison(data, str(output_dir / f"{base_name}_comparison.{ext}"))
    
    print("\nDone!")

if __name__ == "__main__":
    main()
