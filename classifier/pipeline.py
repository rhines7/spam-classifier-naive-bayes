"""End-to-end pipeline orchestration."""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import ComplementNB

from classifier.config import FIGURES_DIR
from classifier.features import (
    add_cluster_features,
    analyze_bow_features,
    analyze_clusters,
    create_bow_features,
    determine_optimal_clusters,
    fit_clustering_model,
)
from classifier.io import explore_dataset, load_emails, remove_empty_emails
from classifier.model import (
    analyze_misclassifications,
    generate_classification_metrics,
    train_with_cross_validation,
)
from classifier.preprocess import display_preprocessing_examples, preprocess_all_emails
from classifier.viz import (
    create_additional_visualizations,
    create_confusion_matrix_plot,
    plot_class_and_content_distribution,
    threshold_analysis,
)


def main():
    """Run the full pipeline, save figures to FIGURES_DIR, and print evaluation metrics."""
    start_time = time.time()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures: {FIGURES_DIR.resolve()}\n")

    # Load and explore
    print("--- Load data ---")
    emails, labels, raw_texts = load_emails()
    explore_dataset(emails, labels)

    class_content_fig = plot_class_and_content_distribution(emails, labels)
    class_content_fig.savefig(
        FIGURES_DIR / "00_class_and_content_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(class_content_fig)

    # Preprocess
    print("\n--- Preprocess ---")
    cleaned_emails, empty_indices = preprocess_all_emails(emails, include_subject=True)
    display_preprocessing_examples(emails, labels, cleaned_emails, num_examples=2)

    if empty_indices:
        emails, cleaned_emails, labels, raw_texts = remove_empty_emails(
            emails, cleaned_emails, labels, raw_texts, empty_indices
        )

    # TF-IDF features
    print("\n--- TF-IDF features ---")
    bow_matrix, vectorizer = create_bow_features(
        cleaned_emails,
        max_features=3000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    feature_analysis_fig = analyze_bow_features(bow_matrix, vectorizer, labels, top_n=20)
    feature_analysis_fig.savefig(
        FIGURES_DIR / "01_bow_feature_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close(feature_analysis_fig)

    # Clustering
    print("\n--- Clustering ---")
    cluster_metrics, cluster_eval_fig = determine_optimal_clusters(
        bow_matrix, k_range=range(2, 31), sample_size=3000
    )
    cluster_eval_fig.savefig(
        FIGURES_DIR / "02_cluster_evaluation.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cluster_eval_fig)

    optimal_k = cluster_metrics["best_silhouette_k"]
    sil_score = cluster_metrics["silhouette_scores"][
        cluster_metrics["k_values"].index(optimal_k)
    ]
    print(f"Selected k={optimal_k} (silhouette={sil_score:.4f})")

    clustering_model, cluster_labels = fit_clustering_model(bow_matrix, n_clusters=optimal_k)

    cluster_info, cluster_analysis_fig = analyze_clusters(
        cluster_labels, labels, cleaned_emails, vectorizer, bow_matrix
    )
    cluster_analysis_fig.savefig(
        FIGURES_DIR / "03_cluster_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cluster_analysis_fig)

    combined_matrix = add_cluster_features(bow_matrix, cluster_labels)

    # Classifier
    print("\n--- Cross-validation ---")
    classifier = ComplementNB(alpha=1.0)
    y_pred, y_pred_proba, fold_scores, cv_object = train_with_cross_validation(
        X=combined_matrix,
        y=labels,
        classifier=classifier,
        n_folds=10,
    )

    # Evaluation
    print("\n--- Evaluation ---")
    confusion_fig, cm = create_confusion_matrix_plot(labels, y_pred)
    confusion_fig.savefig(
        FIGURES_DIR / "04_confusion_matrix.png", dpi=150, bbox_inches="tight"
    )
    plt.close(confusion_fig)

    metrics_dict = generate_classification_metrics(labels, y_pred)

    fp_indices, fn_indices = analyze_misclassifications(
        labels, y_pred, cleaned_emails, labels, num_examples=5
    )

    threshold_fig, threshold_metrics = threshold_analysis(labels, y_pred_proba)
    threshold_fig.savefig(
        FIGURES_DIR / "05_threshold_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close(threshold_fig)

    summary_fig = create_additional_visualizations(fold_scores, metrics_dict)
    summary_fig.savefig(
        FIGURES_DIR / "06_performance_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close(summary_fig)

    print("\n--- Summary ---")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Mean CV accuracy: {np.mean(fold_scores):.4f} (+/-{np.std(fold_scores):.4f})")
    print(f"Ham  - P={metrics_dict['ham_precision']:.4f}, R={metrics_dict['ham_recall']:.4f}")
    print(f"Spam - P={metrics_dict['spam_precision']:.4f}, R={metrics_dict['spam_recall']:.4f}")
    print(f"ROC AUC: {threshold_metrics['roc_auc']:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\nDone in {elapsed_time:.1f}s")
