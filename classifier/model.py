import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def train_with_cross_validation(X, y, classifier, n_folds=10):
    """
    Train classifier using stratified cross-validation.
    
    Uses StratifiedKFold to maintain class proportions in each fold,
    which is important for imbalanced datasets. Collects out-of-fold
    predictions for all samples to enable complete evaluation.
    
    Why StratifiedKFold:
    - Maintains class distribution (74% ham, 26% spam) in each fold
    - Ensures each fold is representative of the full dataset
    - Critical for imbalanced data to avoid folds with too few spam examples
    
    Why cross_val_predict:
    - Generates predictions for every sample using out-of-fold predictions
    - Each sample is predicted when it's in the test set (never trained on)
    - Allows us to build confusion matrix on 100% of data without data leakage
    
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        classifier: initialized Naive Bayes classifier
        n_folds: number of cross-validation folds (default=10)
    
    Returns:
        y_pred: predicted labels for all samples (out-of-fold)
        y_pred_proba: prediction probabilities for all samples
        fold_scores: accuracy score for each fold
        cv_object: the StratifiedKFold object used
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION TRAINING")
    print("="*80)
    
    print(f"Classifier: {classifier.__class__.__name__}, folds: {n_folds}")
    print(f"Samples: {len(y)}, features: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Ham" if label == 0 else "Spam"
        print(f"  {label_name}: {count} ({count/len(y)*100:.2f}%)")
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    start_time = time.time()
    y_pred = cross_val_predict(classifier, X, y, cv=cv, method='predict')
    y_pred_proba = cross_val_predict(classifier, X, y, cv=cv, method='predict_proba')
    
    elapsed = time.time() - start_time
    print(f"Cross-validation finished in {elapsed:.2f}s")
    
    fold_scores = []
    
    fold_num = 1
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train on this fold
        classifier.fit(X_train, y_train)
        
        # Predict on test set
        y_fold_pred = classifier.predict(X_test)
        
        # Calculate accuracy
        fold_acc = accuracy_score(y_test, y_fold_pred)
        fold_scores.append(fold_acc)
        
        print(f"  Fold {fold_num:2d}: Accuracy = {fold_acc:.4f} "
              f"(Train: {len(train_idx)}, Test: {len(test_idx)})")
        fold_num += 1
    
    # Calculate overall statistics
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    
    overall_acc = accuracy_score(y, y_pred)
    print(f"Mean accuracy: {mean_acc:.4f} (+/-{std_acc:.4f}), "
          f"range [{np.min(fold_scores):.4f}, {np.max(fold_scores):.4f}]")
    print(f"Overall accuracy (out-of-fold): {overall_acc:.4f}")
    
    return y_pred, y_pred_proba, fold_scores, cv

def generate_classification_metrics(y_true, y_pred):
    """
    Calculate and display all classification metrics.
    
    Generates comprehensive classification report with precision, recall,
    F1-score, and support for both classes.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
    
    Returns:
        metrics_dict: dictionary of all metrics
    """
    print("\n" + "="*80)
    print("CLASSIFICATION METRICS")
    print("="*80)
    
    # Generate classification report
    print("\nClassification Report:")
    print("-" * 80)
    report = classification_report(y_true, y_pred, 
                                   target_names=['Ham (0)', 'Spam (1)'],
                                   digits=4)
    print(report)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Overall metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall for Spam)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSpecificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
    print(f"False positive rate: {fp/(fp+tn):.4f}, False negative rate: {fn/(fn+tp):.4f}")
    
    metrics_dict = {
        'accuracy': accuracy,
        'ham_precision': precision[0],
        'ham_recall': recall[0],
        'ham_f1': f1[0],
        'spam_precision': precision[1],
        'spam_recall': recall[1],
        'spam_f1': f1[1],
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return metrics_dict

def analyze_misclassifications(y_true, y_pred, cleaned_emails, labels, num_examples=10):
    """
    Examine misclassified emails: false positives and false negatives with sample previews.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        cleaned_emails: list of cleaned email texts
        labels: original labels array
        num_examples: number of examples to show for each type
    """
    print("\n" + "="*80)
    print("MISCLASSIFICATIONS")
    print("="*80)
    
    misclassified = y_true != y_pred
    
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    
    print(f"Total errors: {misclassified.sum()} / {len(y_true)} ({misclassified.sum()/len(y_true)*100:.2f}%)")
    print(f"  False positives (ham as spam): {len(fp_indices)}")
    print(f"  False negatives (spam as ham): {len(fn_indices)}")
    
    if len(fp_indices) > 0:
        print(f"\nFalse positive examples ({min(num_examples, len(fp_indices))} of {len(fp_indices)}):")
        for i, idx in enumerate(fp_indices[:num_examples]):
            print(f"  [{idx}] {cleaned_emails[idx][:200]}...")
    else:
        print("\nNo false positives.")
    
    if len(fn_indices) > 0:
        print(f"\nFalse negative examples ({min(num_examples, len(fn_indices))} of {len(fn_indices)}):")
        for i, idx in enumerate(fn_indices[:num_examples]):
            print(f"  [{idx}] {cleaned_emails[idx][:200]}...")
    else:
        print("\nNo false negatives.")
    
    return fp_indices, fn_indices

