import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)

import os

def compute_class_weights(y):
    """Compute dynamic class weights based on class imbalance."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}

def evaluate_metrics(y_true, y_pred, y_proba):
    """Return a dictionary of common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba)
    }

def plot_roc_curves(y_true, y_probs_dict, target_name, output_path):
    """
    Plot ROC curves for multiple models.
    
    y_probs_dict: {'model_name': y_proba, ...}
    """
    plt.figure(figsize=(8,6))
    for name, y_proba in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.2f})')
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {target_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_vs_threshold(y_true, y_proba, model_name, target_name, output_dir):
    """Plot Accuracy, Precision, Recall, F1 vs threshold for a single model."""
    thresholds = np.linspace(0, 1, 101)
    metrics_dict = {'threshold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        metrics_dict['threshold'].append(t)
        metrics_dict['accuracy'].append(evaluate_metrics(y_true, y_pred, y_proba)['accuracy'])
        metrics_dict['precision'].append(evaluate_metrics(y_true, y_pred, y_proba)['precision'])
        metrics_dict['recall'].append(evaluate_metrics(y_true, y_pred, y_proba)['recall'])
        metrics_dict['f1'].append(evaluate_metrics(y_true, y_pred, y_proba)['f1'])

    # Plot metrics
    plt.figure(figsize=(8,6))
    plt.plot(metrics_dict['threshold'], metrics_dict['accuracy'], label='Accuracy')
    plt.plot(metrics_dict['threshold'], metrics_dict['precision'], label='Precision')
    plt.plot(metrics_dict['threshold'], metrics_dict['recall'], label='Recall')
    plt.plot(metrics_dict['threshold'], metrics_dict['f1'], label='F1 Score')
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title(f"{model_name} Metrics vs Threshold - {target_name}")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_{model_name}_threshold_plot.png"), dpi=300)
    plt.close()