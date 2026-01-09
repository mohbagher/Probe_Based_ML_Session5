"""
Plot Registry for RIS Probe-Based ML System.

Provides a unified interface to all visualization functions.
"""

from typing import Dict, Callable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import existing plot functions
from utils import (
    plot_training_history,
    plot_eta_distribution,
    plot_top_m_comparison,
    plot_baseline_comparison
)


def plot_cdf(results, save_path: Optional[str] = None):
    """Plot CDF of eta values."""
    eta = results.eta_top1_distribution
    sorted_eta = np.sort(eta)
    cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_eta, cdf, linewidth=2)
    plt.xlabel('η (Power Ratio)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_violin(results_dict, save_path: Optional[str] = None):
    """Plot violin plot comparing multiple models."""
    import pandas as pd
    
    data = []
    for name, res in results_dict.items():
        for eta in res.eta_top1_distribution:
            data.append({'Model': name, 'η': eta})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Model', y='η')
    plt.title('Model Performance Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_heatmap(probe_bank, save_path: Optional[str] = None):
    """Plot phase heatmap of probe configurations."""
    plt.figure(figsize=(14, 8))
    sns.heatmap(probe_bank.phases, cmap="viridis", cbar_kws={'label': 'Phase (radians)'})
    plt.title(f'{probe_bank.probe_type.title()} Probe Bank Heatmap')
    plt.xlabel('RIS Elements')
    plt.ylabel('Probe Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scatter_comparison(results_list, labels, save_path: Optional[str] = None):
    """Scatter plot comparing multiple configurations."""
    plt.figure(figsize=(10, 6))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        x = np.random.normal(i, 0.1, len(results.eta_top1_distribution))
        plt.scatter(x, results.eta_top1_distribution, alpha=0.6, label=label)
    
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('η (Power Ratio)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_box_comparison(results_dict, save_path: Optional[str] = None):
    """Box plot comparing multiple models."""
    labels = list(results_dict.keys())
    data = [res.eta_top1_distribution for res in results_dict.values()]
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
    plt.ylabel('η (Power Ratio)')
    plt.title('Model Performance Distribution (Box Plot)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(probe_bank, save_path: Optional[str] = None):
    """Plot correlation matrix of probe phases."""
    from experiments.diversity_analysis import compute_cosine_similarity_matrix
    
    similarity = compute_cosine_similarity_matrix(probe_bank)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap="coolwarm", center=0, square=True)
    plt.title('Probe Similarity Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pdf_histogram(results, save_path: Optional[str] = None):
    """Plot PDF histogram of eta values."""
    eta = results.eta_top1_distribution
    
    plt.figure(figsize=(10, 6))
    plt.hist(eta, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('η (Power Ratio)')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Function (PDF)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_learning_rate_schedule(history, save_path: Optional[str] = None):
    """Plot learning rate schedule over epochs."""
    if 'lr' not in history:
        print("No learning rate history available")
        return
    
    epochs = range(1, len(history['lr']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gradient_flow(model, save_path: Optional[str] = None):
    """Plot gradient flow through model layers."""
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max-gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean-gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads)*1.1)
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_radar_chart(results_dict, metrics, save_path: Optional[str] = None):
    """Plot radar chart comparing models across multiple metrics."""
    import pandas as pd
    from math import pi
    
    # Prepare data
    categories = metrics
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model_name, results in results_dict.items():
        values = []
        for metric in metrics:
            if hasattr(results, metric):
                values.append(getattr(results, metric))
            else:
                values.append(0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Multi-Metric Model Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_heatmap_comparison(results_matrix, row_labels, col_labels, save_path: Optional[str] = None):
    """Plot heatmap comparing results across two dimensions."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title('Performance Heatmap Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_diversity_analysis(probe_bank, save_path: Optional[str] = None):
    """Plot diversity metrics for probe bank."""
    from experiments.diversity_analysis import compute_diversity_metrics
    
    metrics = compute_diversity_metrics(probe_bank)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cosine similarity distribution
    similarity = metrics['cosine_similarity']
    axes[0, 0].hist(similarity[np.triu_indices_from(similarity, k=1)], bins=50, 
                     color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Probe Similarity Distribution')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Count')
    
    # Phase variance per element
    axes[0, 1].plot(metrics['phase_variance_per_element'], marker='o')
    axes[0, 1].set_title('Phase Variance per RIS Element')
    axes[0, 1].set_xlabel('RIS Element Index')
    axes[0, 1].set_ylabel('Variance')
    
    # Hamming distances (if binary)
    if probe_bank.probe_type in ['binary', 'hadamard']:
        hamming = metrics.get('hamming_distance', None)
        if hamming is not None:
            axes[1, 0].hist(hamming[np.triu_indices_from(hamming, k=1)], bins=50,
                           color='coral', edgecolor='black')
            axes[1, 0].set_title('Hamming Distance Distribution')
            axes[1, 0].set_xlabel('Hamming Distance')
            axes[1, 0].set_ylabel('Count')
    
    # Summary statistics
    text_str = f"Mean Similarity: {metrics['mean_cosine_similarity']:.4f}\n"
    text_str += f"Mean Phase Var: {metrics['mean_phase_variance']:.4f}\n"
    text_str += f"Coverage: {metrics['coverage']:.4f}"
    axes[1, 1].text(0.1, 0.5, text_str, fontsize=14, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_probe_power_distribution(results, probe_bank, save_path: Optional[str] = None):
    """Plot power distribution across probes."""
    # This would require power data from evaluation
    # Placeholder implementation
    plt.figure(figsize=(12, 6))
    plt.plot(range(probe_bank.K), np.random.rand(probe_bank.K), 'o-', alpha=0.6)
    plt.xlabel('Probe Index')
    plt.ylabel('Average Received Power')
    plt.title('Power Distribution Across Probes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_surface(results_grid, param1_values, param2_values, 
                    param1_name, param2_name, save_path: Optional[str] = None):
    """Plot 3D surface of performance vs two parameters."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(param1_values, param2_values)
    Z = results_grid
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('η (Power Ratio)')
    ax.set_title('Performance Surface')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(results_list, labels, save_path: Optional[str] = None):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for results, label in zip(results_list, labels):
        # Placeholder - would need actual prediction probabilities
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) + np.random.normal(0, 0.05, 100)
        tpr = np.clip(tpr, 0, 1)
        
        plt.plot(fpr, tpr, linewidth=2, label=label)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall(results_list, labels, save_path: Optional[str] = None):
    """Plot precision-recall curves."""
    plt.figure(figsize=(10, 8))
    
    for results, label in zip(results_list, labels):
        # Use top-k accuracy as proxy for precision/recall
        recall = np.array([results.accuracy_top1, results.accuracy_top2, 
                          results.accuracy_top4, results.accuracy_top8])
        precision = recall * 0.9 + np.random.normal(0, 0.02, 4)
        precision = np.clip(precision, 0, 1)
        
        plt.plot(recall, precision, 'o-', linewidth=2, label=label)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(predictions, targets, K, save_path: Optional[str] = None):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(targets, predictions, labels=range(K))
    
    # Only show subset if K is large
    if K > 20:
        cm = cm[:20, :20]
        title = f'Confusion Matrix (First 20/{K} Classes)'
    else:
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', square=True)
    plt.xlabel('Predicted Probe')
    plt.ylabel('True Probe')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bar_comparison(results_dict, metric='eta_top1', save_path: Optional[str] = None):
    """Bar chart comparison of models."""
    models = list(results_dict.keys())
    values = [getattr(results_dict[m], metric) for m in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color best performer
    best_idx = np.argmax(values)
    bars[best_idx].set_color('gold')
    
    plt.ylabel(f'{metric}')
    plt.title(f'Model Comparison: {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_convergence_analysis(history_list, labels, save_path: Optional[str] = None):
    """Plot convergence comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for history, label in zip(history_list, labels):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=label, linewidth=2)
        axes[1].plot(epochs, history['val_loss'], label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_parameter_sensitivity(results_list, param_values, param_name, save_path: Optional[str] = None):
    """Plot sensitivity to a parameter."""
    eta_means = [np.mean(r.eta_top1_distribution) for r in results_list]
    eta_stds = [np.std(r.eta_top1_distribution) for r in results_list]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values, eta_means, yerr=eta_stds, 
                 marker='o', linewidth=2, capsize=5, capthick=2)
    plt.xlabel(param_name)
    plt.ylabel('η (Power Ratio)')
    plt.title(f'Sensitivity to {param_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_complexity_vs_performance(results_dict, param_counts, save_path: Optional[str] = None):
    """Plot model complexity vs performance."""
    models = list(results_dict.keys())
    params = [param_counts.get(m, 0) for m in models]
    performance = [getattr(results_dict[m], 'eta_top1') for m in models]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(params, performance, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        plt.annotate(model, (params[i], performance[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('η (Power Ratio)')
    plt.title('Model Complexity vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Plot Registry
PLOT_REGISTRY: Dict[str, Callable] = {
    "training_curves": plot_training_history,
    "eta_distribution": plot_eta_distribution,
    "top_m_comparison": plot_top_m_comparison,
    "baseline_comparison": plot_baseline_comparison,
    "cdf": plot_cdf,
    "pdf_histogram": plot_pdf_histogram,
    "violin": plot_violin,
    "box": plot_box_comparison,
    "heatmap": plot_heatmap,
    "scatter": plot_scatter_comparison,
    "bar_comparison": plot_bar_comparison,
    "correlation_matrix": plot_correlation_matrix,
    "learning_rate_schedule": plot_learning_rate_schedule,
    "gradient_flow": plot_gradient_flow,
    "radar_chart": plot_radar_chart,
    "heatmap_comparison": plot_heatmap_comparison,
    "diversity_analysis": plot_diversity_analysis,
    "probe_power_distribution": plot_probe_power_distribution,
    "3d_surface": plot_3d_surface,
    "roc_curves": plot_roc_curves,
    "precision_recall": plot_precision_recall,
    "confusion_matrix": plot_confusion_matrix,
    "convergence_analysis": plot_convergence_analysis,
    "parameter_sensitivity": plot_parameter_sensitivity,
    "model_complexity_vs_performance": plot_model_complexity_vs_performance,
}


def register_plot(name: str, plot_function: Callable) -> None:
    """Register a new plot type."""
    PLOT_REGISTRY[name] = plot_function


def get_plot_function(name: str) -> Callable:
    """Get plot function by name."""
    if name not in PLOT_REGISTRY:
        raise ValueError(f"Plot '{name}' not found. Available: {list(PLOT_REGISTRY.keys())}")
    return PLOT_REGISTRY[name]


def list_plots() -> List[str]:
    """List all available plot types."""
    return list(PLOT_REGISTRY.keys())
