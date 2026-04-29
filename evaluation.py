"""
Comprehensive Evaluation Module for Anomaly Detection Models
Provides metrics, visualizations, and model comparison tools
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None,
                     model_name: str = "Model") -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for anomaly detection
    
    Parameters:
    -----------
    y_true : array-like
        True labels (0: normal, 1: anomaly)
    y_pred : array-like
        Predicted labels (0: normal, 1: anomaly)
    y_prob : array-like, optional
        Prediction probabilities or decision scores
    model_name : str
        Name of the model being evaluated
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    
    # Basic classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # False Positive Rate and False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'True Positives': int(tp),
        'True Negatives': int(tn),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'FPR': fpr,
        'FNR': fnr
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            metrics['ROC-AUC'] = roc_auc
            metrics['Average Precision'] = avg_precision
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC for {model_name}: {e}")
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model"):
    """
    Print detailed classification report
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    
    print(f"\n{'='*70}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*70}")
    
    # Get class names
    target_names = ['Normal', 'Anomaly']
    
    # Print sklearn classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=target_names,
                                   zero_division=0)
    print(report)
    
    # Additional statistics
    metrics = calculate_metrics(y_true, y_pred, model_name=model_name)
    
    print(f"\n{'Additional Metrics':^70}")
    print(f"{'-'*70}")
    print(f"Specificity (True Negative Rate): {metrics['Specificity']:.4f}")
    print(f"False Positive Rate:              {metrics['FPR']:.4f}")
    print(f"False Negative Rate:              {metrics['FNR']:.4f}")
    print(f"{'='*70}\n")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model",
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix with annotations
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', color='gray', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  model_name: str = "Model",
                  save_path: Optional[str] = None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Prediction probabilities or decision scores
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                model_name: str = "Model",
                                save_path: Optional[str] = None):
    """
    Plot Precision-Recall curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Prediction probabilities or decision scores
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot PR curve
    plt.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    
    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Random Classifier (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {model_name}', 
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    
    plt.show()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_prob: Optional[np.ndarray] = None,
                  model_name: str = "Model",
                  plot_curves: bool = True,
                  save_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation with all metrics and visualizations
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Prediction probabilities or decision scores
    model_name : str
        Name of the model
    plot_curves : bool
        Whether to plot curves
    save_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    
    # Print classification report
    print_classification_report(y_true, y_pred, model_name)
    
    # Calculate all metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob, model_name)
    
    if plot_curves:
        # Plot confusion matrix
        cm_path = f"{save_dir}/cm_{model_name.replace(' ', '_')}.png" if save_dir else None
        plot_confusion_matrix(y_true, y_pred, model_name, cm_path)
        
        # Plot ROC and PR curves if probabilities are provided
        if y_prob is not None:
            roc_path = f"{save_dir}/roc_{model_name.replace(' ', '_')}.png" if save_dir else None
            pr_path = f"{save_dir}/pr_{model_name.replace(' ', '_')}.png" if save_dir else None
            
            plot_roc_curve(y_true, y_prob, model_name, roc_path)
            plot_precision_recall_curve(y_true, y_prob, model_name, pr_path)
    
    return metrics


def compare_models(results_dict: Dict[str, Dict],
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple models and create comparison table
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results as values
        Each result should contain: y_true, y_pred, y_prob (optional)
    save_path : str, optional
        Path to save the comparison figure
        
    Returns:
    --------
    comparison_df : DataFrame
        DataFrame containing all models' metrics
    """
    
    all_metrics = []
    
    # Collect metrics from all models
    for model_name, results in results_dict.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_prob = results.get('y_prob', None)
        
        metrics = calculate_metrics(y_true, y_pred, y_prob, model_name)
        all_metrics.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)
    
    # Set Model as index
    comparison_df = comparison_df.set_index('Model')
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'MODEL COMPARISON':^100}")
    print(f"{'='*100}\n")
    
    # Select key metrics for display
    key_metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'Accuracy']
    if 'ROC-AUC' in comparison_df.columns:
        key_metrics.append('ROC-AUC')
    if 'Average Precision' in comparison_df.columns:
        key_metrics.append('Average Precision')
    
    display_df = comparison_df[key_metrics].round(4)
    print(display_df.to_string())
    print(f"\n{'='*100}\n")
    
    # Find best model for each metric
    print("Best Model for Each Metric:")
    print("-" * 100)
    for metric in key_metrics:
        best_model = display_df[metric].idxmax()
        best_value = display_df[metric].max()
        print(f"{metric:25s}: {best_model:20s} ({best_value:.4f})")
    print(f"{'='*100}\n")
    
    # Visualize comparison
    plot_model_comparison(comparison_df, key_metrics, save_path)
    
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metrics: List[str],
                         save_path: Optional[str] = None):
    """
    Create bar plots comparing models across different metrics
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame containing model metrics
    metrics : list
        List of metrics to plot
    save_path : str, optional
        Path to save the figure
    """
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(16, 10))
    axes = axes.ravel()
    
    colors = sns.color_palette("husl", len(comparison_df))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create bar plot
        comparison_df[metric].plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(comparison_df[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_multiple_roc_curves(results_dict: Dict[str, Dict],
                            save_path: Optional[str] = None):
    """
    Plot ROC curves for multiple models on the same graph
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results as values
    save_path : str, optional
        Path to save the figure
    """
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        y_true = results['y_true']
        y_prob = results.get('y_prob', None)
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            
            plt.plot(fpr, tpr, lw=2, color=colors[idx],
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--',
            label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-ROC curve saved to: {save_path}")
    
    plt.show()


def analyze_confusion_patterns(comparison_df: pd.DataFrame):
    """
    Analyze confusion matrix patterns across models
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame containing model metrics including TP, TN, FP, FN
    """
    
    print(f"\n{'='*100}")
    print(f"{'CONFUSION MATRIX ANALYSIS':^100}")
    print(f"{'='*100}\n")
    
    cm_cols = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    cm_df = comparison_df[cm_cols]
    
    print(cm_df.to_string())
    print(f"\n{'-'*100}\n")
    
    # Analyze patterns
    print("Pattern Analysis:")
    print("-" * 100)
    
    # Model with least false positives
    least_fp = cm_df['False Positives'].idxmin()
    print(f"Least False Positives:  {least_fp:20s} "
          f"(FP = {cm_df.loc[least_fp, 'False Positives']:.0f})")
    
    # Model with least false negatives
    least_fn = cm_df['False Negatives'].idxmin()
    print(f"Least False Negatives:  {least_fn:20s} "
          f"(FN = {cm_df.loc[least_fn, 'False Negatives']:.0f})")
    
    # Most balanced model (closest FP and FN)
    cm_df['Balance'] = abs(cm_df['False Positives'] - cm_df['False Negatives'])
    most_balanced = cm_df['Balance'].idxmin()
    print(f"Most Balanced Model:    {most_balanced:20s} "
          f"(|FP-FN| = {cm_df.loc[most_balanced, 'Balance']:.0f})")
    
    print(f"{'='*100}\n")


def create_evaluation_summary(comparison_df: pd.DataFrame,
                             save_path: Optional[str] = None) -> str:
    """
    Create a text summary of the evaluation results
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame containing model metrics
    save_path : str, optional
        Path to save the summary text file
        
    Returns:
    --------
    summary : str
        Text summary of evaluation
    """
    
    summary = []
    summary.append("="*100)
    summary.append("ANOMALY DETECTION MODEL EVALUATION SUMMARY")
    summary.append("="*100)
    summary.append("")
    
    # Overall best model
    key_metrics = ['F1-Score', 'ROC-AUC', 'Precision', 'Recall']
    available_metrics = [m for m in key_metrics if m in comparison_df.columns]
    
    if available_metrics:
        # Calculate average rank across metrics
        ranks = pd.DataFrame()
        for metric in available_metrics:
            ranks[metric] = comparison_df[metric].rank(ascending=False)
        
        avg_rank = ranks.mean(axis=1)
        overall_best = avg_rank.idxmin()
        
        summary.append(f"OVERALL BEST MODEL: {overall_best}")
        summary.append(f"(Average rank across {len(available_metrics)} key metrics)")
        summary.append("")
    
    # Detailed metrics
    summary.append("DETAILED METRICS:")
    summary.append("-"*100)
    summary.append(comparison_df.round(4).to_string())
    summary.append("")
    
    # Recommendations
    summary.append("RECOMMENDATIONS:")
    summary.append("-"*100)
    
    if 'Precision' in comparison_df.columns and 'Recall' in comparison_df.columns:
        high_precision = comparison_df['Precision'].idxmax()
        high_recall = comparison_df['Recall'].idxmax()
        
        summary.append(f"* For minimizing false alarms: Use {high_precision} "
                      f"(Precision = {comparison_df.loc[high_precision, 'Precision']:.4f})")
        summary.append(f"* For catching all anomalies:  Use {high_recall} "
                      f"(Recall = {comparison_df.loc[high_recall, 'Recall']:.4f})")
    
    if 'F1-Score' in comparison_df.columns:
        high_f1 = comparison_df['F1-Score'].idxmax()
        summary.append(f"* For balanced performance:    Use {high_f1} "
                      f"(F1-Score = {comparison_df.loc[high_f1, 'F1-Score']:.4f})")
    
    summary.append("")
    summary.append("="*100)
    
    summary_text = "\n".join(summary)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_text)
        print(f"Summary saved to: {save_path}")
    
    print(summary_text)
    return summary_text


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the evaluation module
    """
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (5% anomalies)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Simulated predictions for 3 different models
    models_data = {}
    
    # Model 1: High precision
    y_pred_1 = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    y_pred_1[noise_idx] = 1 - y_pred_1[noise_idx]
    y_prob_1 = np.random.beta(2, 5, n_samples) * (1 - y_pred_1) + \
               np.random.beta(5, 2, n_samples) * y_pred_1
    
    models_data['Statistical Z-Score'] = {
        'y_true': y_true,
        'y_pred': y_pred_1,
        'y_prob': y_prob_1
    }
    
    # Model 2: High recall
    y_pred_2 = y_true.copy()
    anomaly_boost = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    y_pred_2[anomaly_boost] = 1
    y_prob_2 = np.random.beta(2, 4, n_samples) * (1 - y_pred_2) + \
               np.random.beta(4, 2, n_samples) * y_pred_2
    
    models_data['Supervised LSTM'] = {
        'y_true': y_true,
        'y_pred': y_pred_2,
        'y_prob': y_prob_2
    }
    
    # Model 3: Balanced
    y_pred_3 = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    y_pred_3[noise_idx] = 1 - y_pred_3[noise_idx]
    y_prob_3 = np.random.beta(3, 4, n_samples) * (1 - y_pred_3) + \
               np.random.beta(4, 3, n_samples) * y_pred_3
    
    models_data['Unsupervised LSTM'] = {
        'y_true': y_true,
        'y_pred': y_pred_3,
        'y_prob': y_prob_3
    }
    
    print("\n" + "="*100)
    print("ANOMALY DETECTION EVALUATION - EXAMPLE DEMONSTRATION")
    print("="*100 + "\n")
    
    # Evaluate each model individually
    all_metrics = []
    for model_name, data in models_data.items():
        metrics = evaluate_model(
            data['y_true'],
            data['y_pred'],
            data['y_prob'],
            model_name=model_name,
            plot_curves=True
        )
        all_metrics.append(metrics)
    
    # Compare all models
    comparison_df = compare_models(models_data)
    
    # Plot all ROC curves together
    plot_multiple_roc_curves(models_data)
    
    # Analyze confusion patterns
    analyze_confusion_patterns(comparison_df)
    
    # Create summary
    create_evaluation_summary(comparison_df)
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100 + "\n")