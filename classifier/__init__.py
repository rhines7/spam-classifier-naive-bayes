"""Spam classifier package: load, preprocess, feature engineering, model, and visualization."""

from classifier.config import DATA_DIR, FIGURES_DIR, PROJECT_ROOT
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
from classifier.pipeline import main
from classifier.preprocess import (
    display_preprocessing_examples,
    extract_email_body,
    preprocess_all_emails,
    preprocess_email,
)
from classifier.viz import (
    create_additional_visualizations,
    create_confusion_matrix_plot,
    plot_class_and_content_distribution,
    threshold_analysis,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "load_emails",
    "explore_dataset",
    "remove_empty_emails",
    "extract_email_body",
    "preprocess_email",
    "preprocess_all_emails",
    "display_preprocessing_examples",
    "create_bow_features",
    "analyze_bow_features",
    "determine_optimal_clusters",
    "fit_clustering_model",
    "analyze_clusters",
    "add_cluster_features",
    "train_with_cross_validation",
    "generate_classification_metrics",
    "analyze_misclassifications",
    "plot_class_and_content_distribution",
    "create_confusion_matrix_plot",
    "threshold_analysis",
    "create_additional_visualizations",
    "main",
]
