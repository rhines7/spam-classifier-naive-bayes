import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

def plot_class_and_content_distribution(emails, labels):
    """
    Create a figure showing class distribution (ham vs spam) and content-type distribution.
    
    Helps communicate dataset balance and MIME type variety before modeling.
    
    Args:
        emails: list of email message objects
        labels: numpy array of labels (0=ham, 1=spam)
    
    Returns:
        matplotlib figure (caller may save or show)
    """
    # Reuse same logic as explore_dataset for consistency
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    ham_count = class_dist.get(0, 0)
    spam_count = class_dist.get(1, 0)
    
    content_types = Counter()
    for msg in emails:
        content_types[msg.get_content_type()] += 1
    total = len(emails)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dataset: Class Distribution and Content Types', fontsize=16, fontweight='bold')
    
    # Left: class distribution (bar chart; green/red to match project convention)
    ax1 = axes[0]
    classes = ['Ham', 'Spam']
    values = [ham_count, spam_count]
    colors = ['green', 'red']
    bars = ax1.bar(classes, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of emails', fontsize=12)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        pct = 100 * val / total
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: content types (horizontal bar, top 10)
    ax2 = axes[1]
    top_n = min(10, len(content_types))
    ct_items = content_types.most_common(top_n)
    ct_labels = [item[0] for item in ct_items]
    ct_counts = [item[1] for item in ct_items]
    y_pos = np.arange(len(ct_labels))
    ax2.barh(y_pos, ct_counts, color='steelblue', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ct_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of emails', fontsize=12)
    ax2.set_title('Content Types (MIME)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    """
    Create large, labeled confusion matrix visualization.
    
    Displays both counts and percentages with clear axis labels.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
    
    Returns:
        fig: matplotlib figure
        cm: confusion matrix array
    """
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"\nConfusion Matrix Values:")
    print(f"  True Negatives (TN):  {tn:5d} (Ham correctly classified as Ham)")
    print(f"  False Positives (FP): {fp:5d} (Ham incorrectly classified as Spam)")
    print(f"  False Negatives (FN): {fn:5d} (Spam incorrectly classified as Ham)")
    print(f"  True Positives (TP):  {tp:5d} (Spam correctly classified as Spam)")
    print(f"  Total:                {total:5d}")
    
    # Calculate percentages
    print(f"\nPercentages:")
    print(f"  True Negatives:  {tn/total*100:6.2f}%")
    print(f"  False Positives: {fp/total*100:6.2f}%")
    print(f"  False Negatives: {fn/total*100:6.2f}%")
    print(f"  True Positives:  {tp/total*100:6.2f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap only; cell annotations carry count/percentage (no color scale legend)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False, ax=ax,
                linewidths=2, linecolor='black')
    
    # Add custom annotations with both count and percentage
    # Using data coordinates for proper positioning within heatmap cells
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = (count / total) * 100
            
            # Main count and percentage (centered in cell)
            text = f'{count:,}\n({percentage:.2f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=18, fontweight='bold',
                   color='white' if count > total/4 else 'black')
            
            # Add small TN/FP/FN/TP labels in corner of each cell
            # Using data coordinates for correct positioning
            label_map = {(0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN', (1, 1): 'TP'}
            cell_label = label_map.get((i, j), '')
            ax.text(j + 0.15, i + 0.15, cell_label,
                   ha='left', va='top',
                   fontsize=11, style='italic', fontweight='bold',
                   color='lightgray' if count > total/4 else 'gray')
    
    # Set title and labels (standard matplotlib approach)
    ax.set_title('Confusion Matrix: Spam Classification Results\n' + 
                 f'Total Samples: {total:,} | Accuracy: {(tn+tp)/total*100:.2f}%',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)
    
    # Set tick labels with proper formatting
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Ham (0)', 'Spam (1)'], fontsize=14, fontweight='bold')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Ham (0)', 'Spam (1)'], fontsize=14, fontweight='bold', rotation=0)
    
    # Adjust layout with extra padding for labels
    plt.tight_layout(pad=2.0)
    
    return fig, cm

def threshold_analysis(y_true, y_pred_proba):
    """
    Analyze effect of changing decision threshold.
    
    By default, classification uses 0.5 threshold. This analyzes how
    changing the threshold affects precision, recall, and other metrics.
    Creates ROC curve and calculates AUC.
    
    Args:
        y_true: true labels
        y_pred_proba: prediction probabilities (n_samples, 2)
    
    Returns:
        fig: matplotlib figure with threshold analysis plots
        metrics_by_threshold: dictionary of metrics at different thresholds
    """
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS")
    print("="*80)
    
    y_scores = y_pred_proba[:, 1]
    # Test different thresholds
    thresholds = np.arange(0.0, 1.01, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_scores >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred_thresh)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_thresh, average='binary', zero_division=0
        )
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Max-F1 threshold: {optimal_threshold:.2f} "
          f"(P={precisions[optimal_idx]:.4f}, R={recalls[optimal_idx]:.4f}, "
          f"F1={f1_scores[optimal_idx]:.4f})")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Decision Threshold Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Precision and Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds, precisions, 'b-', linewidth=2, marker='o', label='Precision')
    ax1.plot(thresholds, recalls, 'r-', linewidth=2, marker='s', label='Recall')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
    ax1.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
               label=f'Optimal ({optimal_threshold:.2f})')
    ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Precision and Recall vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: F1-Score and Accuracy vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds, f1_scores, 'g-', linewidth=2, marker='^', label='F1-Score')
    ax2.plot(thresholds, accuracies, 'm-', linewidth=2, marker='d', label='Accuracy')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
    ax2.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
               label=f'Optimal ({optimal_threshold:.2f})')
    ax2.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score and Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: ROC Curve
    ax3 = axes[1, 0]
    ax3.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    
    # Plot 4: Precision-Recall Curve
    ax4 = axes[1, 1]
    # Calculate precision-recall curve
    from sklearn.metrics import precision_recall_curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    ax4.plot(pr_recall, pr_precision, 'g-', linewidth=3, label='Precision-Recall Curve')
    ax4.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    metrics_by_threshold = {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'accuracies': accuracies,
        'optimal_threshold': optimal_threshold,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return fig, metrics_by_threshold

def create_additional_visualizations(fold_scores, metrics_dict):
    """
    Create any additional helpful visualizations.
    
    Creates summary visualizations showing cross-validation performance
    and overall model quality.
    
    Args:
        fold_scores: list of accuracy scores from each CV fold
        metrics_dict: dictionary of classification metrics
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Summary', fontsize=20, fontweight='bold')
    
    # Plot 1: Cross-validation fold scores
    ax1 = axes[0, 0]
    folds = range(1, len(fold_scores) + 1)
    ax1.bar(folds, fold_scores, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=np.mean(fold_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(fold_scores):.4f}')
    ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(f'Cross-Validation Fold Accuracies (Mean±Std: {np.mean(fold_scores):.4f}±{np.std(fold_scores):.4f})',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.8, 1.0)
    ax1.set_xticks(folds)
    
    # Plot 2: Per-class metrics comparison
    ax2 = axes[0, 1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    ham_scores = [metrics_dict['ham_precision'], metrics_dict['ham_recall'], metrics_dict['ham_f1']]
    spam_scores = [metrics_dict['spam_precision'], metrics_dict['spam_recall'], metrics_dict['spam_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, ham_scores, width, label='Ham', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, spam_scores, width, label='Spam', color='red', alpha=0.7)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0.8, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Confusion matrix percentages (pie chart style)
    ax3 = axes[1, 0]
    sizes = [metrics_dict['tn'], metrics_dict['fp'], metrics_dict['fn'], metrics_dict['tp']]
    labels = [f"TN\n{metrics_dict['tn']}", f"FP\n{metrics_dict['fp']}", 
              f"FN\n{metrics_dict['fn']}", f"TP\n{metrics_dict['tp']}"]
    colors = ['lightgreen', 'orange', 'red', 'darkgreen']
    explode = (0.05, 0.05, 0.05, 0.05)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
           shadow=True, startangle=45, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Confusion Matrix Distribution', fontsize=14, fontweight='bold')
    
    # Plot 4: Key metrics summary bar chart
    ax4 = axes[1, 1]
    summary_metrics = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
    summary_values = [
        metrics_dict['accuracy'],
        metrics_dict['macro_precision'],
        metrics_dict['macro_recall'],
        metrics_dict['macro_f1']
    ]
    
    bars = ax4.barh(summary_metrics, summary_values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Model Performance Summary', fontsize=14, fontweight='bold')
    ax4.set_xlim(0.8, 1.0)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, summary_values)):
        ax4.text(value, i, f' {value:.4f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    return fig

