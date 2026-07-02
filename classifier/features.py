import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import matplotlib.pyplot as plt

def create_bow_features(cleaned_emails, max_features=3000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
    """
    Create Bag-of-Words feature matrix using TF-IDF.
    
    TF-IDF weights terms by importance across documents, which is effective for
    spam classification: it downweights common words and emphasizes distinctive terms.
    
    Args:
        cleaned_emails: list of preprocessed email text strings
        max_features: maximum number of features to extract
        min_df: minimum document frequency (ignore rare words)
        max_df: maximum document frequency (ignore very common words)
        ngram_range: (min_n, max_n) for n-grams. (1,2) includes unigrams and bigrams
    
    Returns:
        bow_matrix: sparse feature matrix (n_samples, n_features)
        vectorizer: fitted TfidfVectorizer
    """
    print("\n" + "="*80)
    print("TF-IDF FEATURES")
    print("="*80)
    print(f"max_features={max_features}, min_df={min_df}, max_df={max_df}, ngram_range={ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        lowercase=False,  # Already lowercased in preprocessing
        strip_accents=None,  # Already handled
        stop_words=None  # Already removed stopwords
    )
    
    print("\nFitting vectorizer...")
    bow_matrix = vectorizer.fit_transform(cleaned_emails)
    
    print(f"Matrix shape: {bow_matrix.shape}, sparsity: "
          f"{(1.0 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.2f}%")
    
    return bow_matrix, vectorizer

def analyze_bow_features(bow_matrix, vectorizer, labels, top_n=20):
    """
    Analyze BOW features to understand the data.
    
    Identifies distinctive words for spam vs ham using TF-IDF sums and
    per-class averages within each class.
    
    Args:
        bow_matrix: sparse feature matrix
        vectorizer: fitted TfidfVectorizer
        labels: email labels (0=ham, 1=spam)
        top_n: number of top features to display
    """
    print("\n" + "="*80)
    print("ANALYZING BOW FEATURES")
    print("="*80)
    
    # Pipeline uses TF-IDF only; labels and axes consistently use "TF-IDF"
    freq_label = "TF-IDF"
    print("\n(Values shown are TF-IDF sums/averages, not raw token counts.)")
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrix to array for easier manipulation
    # Note: This can be memory intensive for large matrices
    # For very large datasets, work with sparse matrices directly
    print("\nCalculating feature statistics...")
    
    # Calculate sum of each feature across all documents
    overall_freq = np.asarray(bow_matrix.sum(axis=0)).flatten()
    
    # Calculate sum for each class
    ham_mask = labels == 0
    spam_mask = labels == 1
    
    ham_freq = np.asarray(bow_matrix[ham_mask].sum(axis=0)).flatten()
    spam_freq = np.asarray(bow_matrix[spam_mask].sum(axis=0)).flatten()
    
    # Calculate average frequency per document for each class
    ham_avg = ham_freq / ham_mask.sum()
    spam_avg = spam_freq / spam_mask.sum()
    
    # Sort features by frequency
    overall_top_idx = overall_freq.argsort()[-top_n:][::-1]
    ham_top_idx = ham_avg.argsort()[-top_n:][::-1]
    spam_top_idx = spam_avg.argsort()[-top_n:][::-1]
    
    # Display overall most common features
    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURES OVERALL (by total {freq_label.lower()})")
    print(f"{'='*80}")
    print(f"{'Feature':<30} {'Total':>15} {'Ham':>15} {'Spam':>15}")
    print("-" * 80)
    for idx in overall_top_idx:
        print(f"{feature_names[idx]:<30} {overall_freq[idx]:>15.2f} "
              f"{ham_freq[idx]:>15.2f} {spam_freq[idx]:>15.2f}")
    
    # Display most common HAM features
    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURES IN HAM EMAILS (by average {freq_label.lower()})")
    print(f"{'='*80}")
    print(f"{'Feature':<30} {'Ham Avg':>15} {'Spam Avg':>15} {'Ratio (H/S)':>15}")
    print("-" * 80)
    for idx in ham_top_idx:
        ratio = ham_avg[idx] / (spam_avg[idx] + 1e-10)  # Add small value to avoid division by zero
        print(f"{feature_names[idx]:<30} {ham_avg[idx]:>15.4f} "
              f"{spam_avg[idx]:>15.4f} {ratio:>15.2f}")
    
    # Display most common SPAM features
    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURES IN SPAM EMAILS (by average {freq_label.lower()})")
    print(f"{'='*80}")
    print(f"{'Feature':<30} {'Spam Avg':>15} {'Ham Avg':>15} {'Ratio (S/H)':>15}")
    print("-" * 80)
    for idx in spam_top_idx:
        ratio = spam_avg[idx] / (ham_avg[idx] + 1e-10)
        print(f"{feature_names[idx]:<30} {spam_avg[idx]:>15.4f} "
              f"{ham_avg[idx]:>15.4f} {ratio:>15.2f}")
    
    # Create visualization of top features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bag-of-Words Feature Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Overall top features
    ax1 = axes[0, 0]
    top_features = feature_names[overall_top_idx]
    top_freqs = overall_freq[overall_top_idx]
    ax1.barh(range(len(top_features)), top_freqs, color='steelblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel(f'Total {freq_label}', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Features Overall', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Ham-specific features
    ax2 = axes[0, 1]
    ham_features = feature_names[ham_top_idx]
    ham_freqs = ham_avg[ham_top_idx]
    ax2.barh(range(len(ham_features)), ham_freqs, color='green', alpha=0.7)
    ax2.set_yticks(range(len(ham_features)))
    ax2.set_yticklabels(ham_features, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel(f'Average {freq_label} in Ham', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Ham Features', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Spam-specific features
    ax3 = axes[1, 0]
    spam_features = feature_names[spam_top_idx]
    spam_freqs = spam_avg[spam_top_idx]
    ax3.barh(range(len(spam_features)), spam_freqs, color='red', alpha=0.7)
    ax3.set_yticks(range(len(spam_features)))
    ax3.set_yticklabels(spam_features, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel(f'Average {freq_label} in Spam', fontsize=12, fontweight='bold')
    ax3.set_title(f'Top {top_n} Spam Features', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Comparison of Ham vs Spam for distinctive features
    ax4 = axes[1, 1]
    # Get features that are distinctive (high in one class, low in other)
    spam_ratio = spam_avg / (ham_avg + 1e-10)
    spam_distinctive_idx = spam_ratio.argsort()[-top_n:][::-1]
    
    x_pos = np.arange(top_n)
    width = 0.35
    
    distinctive_features = feature_names[spam_distinctive_idx]
    distinctive_ham = ham_avg[spam_distinctive_idx]
    distinctive_spam = spam_avg[spam_distinctive_idx]
    
    ax4.bar(x_pos - width/2, distinctive_ham, width, label='Ham', color='green', alpha=0.7)
    ax4.bar(x_pos + width/2, distinctive_spam, width, label='Spam', color='red', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(distinctive_features, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel(f'Average {freq_label}', fontsize=12, fontweight='bold')
    ax4.set_title('Most Distinctive Spam Features (Ham vs Spam)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def determine_optimal_clusters(bow_matrix, k_range=range(2, 31), sample_size=None):
    """
    Evaluate different numbers of clusters using multiple metrics.
    
    Uses three complementary methods to determine optimal k:
    1. Elbow Method: Looks for "elbow" in inertia curve (within-cluster sum of squares)
    2. Silhouette Score: Measures how similar points are to their own cluster vs other clusters
       - Range: [-1, 1], higher is better
       - >0.5 = good, 0.3-0.5 = reasonable, <0.3 = poor
    3. Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion
       - Higher values indicate better-defined clusters
    
    Args:
        bow_matrix: sparse feature matrix
        k_range: range of k values to test
        sample_size: if provided, sample this many points for faster computation
    
    Returns:
        metrics_dict: dictionary containing inertia, silhouette, and CH scores
    """
    print("\n" + "="*80)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("="*80)
    
    print(f"\nTesting k values from {min(k_range)} to {max(k_range)}")
    print(f"Original matrix shape: {bow_matrix.shape}")
    
    # For large datasets, sampling speeds up computation significantly
    # Clustering metrics are stable with representative samples
    if sample_size and bow_matrix.shape[0] > sample_size:
        print(f"\nSampling {sample_size} points for faster computation...")
        np.random.seed(42)
        sample_indices = np.random.choice(bow_matrix.shape[0], sample_size, replace=False)
        X_sample = bow_matrix[sample_indices]
        print(f"Sample matrix shape: {X_sample.shape}")
    else:
        X_sample = bow_matrix
        print(f"Using full dataset: {X_sample.shape}")
    
    # Initialize storage for metrics
    inertias = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    k_values = list(k_range)
    
    print("\nEvaluating clusters...")
    print(f"{'K':>5} {'Inertia':>15} {'Silhouette':>15} {'Calinski-H':>15} {'Time (s)':>10}")
    print("-" * 70)
    
    for k in k_values:
        start_time = time.time()
        
        # Fit K-Means
        # Using mini-batch for speed on large datasets
        if X_sample.shape[0] > 5000:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        cluster_labels = kmeans.fit_predict(X_sample)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_sample, cluster_labels, sample_size=min(5000, X_sample.shape[0]))
        calinski = calinski_harabasz_score(X_sample.toarray() if hasattr(X_sample, 'toarray') else X_sample, 
                                          cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski)
        
        elapsed = time.time() - start_time
        print(f"{k:>5} {inertia:>15.2f} {silhouette:>15.4f} {calinski:>15.2f} {elapsed:>10.2f}")
    
    # Create visualization of all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Number Optimization Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Elbow Method (Inertia)
    ax1 = axes[0, 0]
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values[::2])  # Show every other k value
    
    # Add annotation for potential elbow
    # Calculate rate of change to identify elbow
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs) + 1
        if elbow_idx < len(k_values):
            ax1.axvline(x=k_values[elbow_idx], color='red', linestyle='--', alpha=0.7, 
                       label=f'Potential elbow at k={k_values[elbow_idx]}')
            ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values[::2])
    
    # Add horizontal lines for interpretation
    ax2.axhline(y=0.5, color='darkgreen', linestyle=':', alpha=0.5, label='Good (>0.5)')
    ax2.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='Reasonable (>0.3)')
    
    # Mark best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    ax2.plot(k_values[best_sil_idx], silhouette_scores[best_sil_idx], 'r*', 
            markersize=20, label=f'Best: k={k_values[best_sil_idx]}')
    ax2.legend()
    
    # Plot 3: Calinski-Harabasz Index
    ax3 = axes[1, 0]
    ax3.plot(k_values, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
    ax3.set_title('Calinski-Harabasz Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_values[::2])
    
    # Mark best CH score
    best_ch_idx = np.argmax(calinski_harabasz_scores)
    ax3.plot(k_values[best_ch_idx], calinski_harabasz_scores[best_ch_idx], 'r*', 
            markersize=20, label=f'Best: k={k_values[best_ch_idx]}')
    ax3.legend()
    
    # Plot 4: Combined normalized comparison
    ax4 = axes[1, 1]
    
    # Normalize all metrics to 0-1 scale for comparison
    norm_inertia = 1 - (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))  # Invert: lower is better
    norm_silhouette = (np.array(silhouette_scores) - min(silhouette_scores)) / \
                     (max(silhouette_scores) - min(silhouette_scores))
    norm_ch = (np.array(calinski_harabasz_scores) - min(calinski_harabasz_scores)) / \
             (max(calinski_harabasz_scores) - min(calinski_harabasz_scores))
    
    ax4.plot(k_values, norm_inertia, 'b-', linewidth=2, marker='o', label='Elbow (inverted)', alpha=0.7)
    ax4.plot(k_values, norm_silhouette, 'g-', linewidth=2, marker='s', label='Silhouette', alpha=0.7)
    ax4.plot(k_values, norm_ch, 'm-', linewidth=2, marker='^', label='Calinski-H', alpha=0.7)
    
    ax4.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax4.set_title('Normalized Comparison of All Metrics', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)
    ax4.set_xticks(k_values[::2])
    
    plt.tight_layout()
    
    best_sil_k = k_values[best_sil_idx]
    best_ch_k = k_values[best_ch_idx]
    print(f"\nBest silhouette: k={best_sil_k} ({silhouette_scores[best_sil_idx]:.4f})")
    print(f"Best Calinski-Harabasz: k={best_ch_k} ({calinski_harabasz_scores[best_ch_idx]:.2f})")
    
    metrics_dict = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'best_silhouette_k': best_sil_k,
        'best_ch_k': best_ch_k
    }
    
    return metrics_dict, fig

def fit_clustering_model(bow_matrix, n_clusters):
    """
    Fit clustering model with chosen number of clusters.
    
    Uses Mini-Batch K-Means, which partitions data into k clusters by minimizing
    within-cluster variance using small random batches. For text data, this
    groups emails with similar word patterns together.
    
    Args:
        bow_matrix: sparse feature matrix
        n_clusters: number of clusters to create
    
    Returns:
        fitted model: trained MiniBatchKMeans
        cluster_labels: array of cluster assignments for each sample
    """
    print("\n" + "="*80)
    print("FITTING CLUSTERING MODEL")
    print("="*80)
    
    print(f"\nNumber of clusters: {n_clusters}, shape: {bow_matrix.shape}")
    
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1000,
        n_init=10,
        max_iter=100,
        verbose=0
    )
    
    print("Fitting Mini-Batch K-Means...")
    start_time = time.time()
    cluster_labels = model.fit_predict(bow_matrix)
    elapsed = time.time() - start_time
    
    print(f"Clustering complete in {elapsed:.2f} seconds")
    print(f"\nCluster Statistics:")
    print(f"  Inertia: {model.inertia_:.2f}")
    print(f"  Number of iterations: {model.n_iter_}")
    
    # Calculate cluster sizes
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster size distribution:")
    print(f"  {'Cluster':>10} {'Size':>10} {'Percentage':>12}")
    print("-" * 35)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  {cluster_id:>10} {count:>10} {percentage:>11.2f}%")
    
    print(f"\nCluster size statistics:")
    print(f"  Mean: {np.mean(cluster_counts):.1f}")
    print(f"  Std:  {np.std(cluster_counts):.1f}")
    print(f"  Min:  {np.min(cluster_counts)}")
    print(f"  Max:  {np.max(cluster_counts)}")
    
    return model, cluster_labels

def analyze_clusters(cluster_labels, labels, cleaned_emails, vectorizer, bow_matrix):
    """
    Analyze what each cluster represents.
    
    Examines the spam/ham distribution within each cluster and identifies
    characteristic terms for each cluster to understand what patterns were learned.
    
    Args:
        cluster_labels: array of cluster assignments
        labels: email labels (0=ham, 1=spam)
        cleaned_emails: list of cleaned email texts
        vectorizer: fitted BOW vectorizer
        bow_matrix: BOW feature matrix
    """
    print("\n" + "="*80)
    print("ANALYZING CLUSTERS")
    print("="*80)
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Calculate spam/ham distribution per cluster
    print(f"\nSpam/Ham Distribution by Cluster:")
    print(f"{'Cluster':>10} {'Total':>8} {'Ham':>8} {'Spam':>8} {'% Spam':>10} {'Classification':>15}")
    print("-" * 75)
    
    cluster_info = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_size = mask.sum()
        
        ham_count = ((labels == 0) & mask).sum()
        spam_count = ((labels == 1) & mask).sum()
        spam_pct = (spam_count / cluster_size) * 100 if cluster_size > 0 else 0
        
        # Classify cluster as primarily spam or ham
        classification = "SPAM-HEAVY" if spam_pct > 50 else "HAM-HEAVY"
        if spam_pct > 75:
            classification = "VERY SPAM"
        elif spam_pct < 25:
            classification = "VERY HAM"
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'total': cluster_size,
            'ham': ham_count,
            'spam': spam_count,
            'spam_pct': spam_pct,
            'classification': classification
        })
        
        print(f"{cluster_id:>10} {cluster_size:>8} {ham_count:>8} {spam_count:>8} "
              f"{spam_pct:>9.1f}% {classification:>15}")
    
    # Identify top terms for each cluster
    print(f"\n{'='*80}")
    print("TOP TERMS PER CLUSTER")
    print(f"{'='*80}")
    
    feature_names = vectorizer.get_feature_names_out()
    
    for cluster_id in range(min(10, n_clusters)):  # Show first 10 clusters
        mask = cluster_labels == cluster_id
        cluster_bow = bow_matrix[mask]
        
        # Calculate mean TF-IDF for each term in this cluster
        mean_tfidf = np.asarray(cluster_bow.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        info = cluster_info[cluster_id]
        print(f"\nCluster {cluster_id} ({info['classification']}, {info['spam_pct']:.1f}% spam):")
        print(f"  Top terms: {', '.join(top_terms)}")
    
    if n_clusters > 10:
        print(f"\n... (showing first 10 of {n_clusters} clusters)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Cluster sizes
    ax1 = axes[0, 0]
    cluster_ids = [info['cluster_id'] for info in cluster_info]
    cluster_sizes = [info['total'] for info in cluster_info]
    ax1.bar(cluster_ids, cluster_sizes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Spam percentage by cluster
    ax2 = axes[0, 1]
    spam_pcts = [info['spam_pct'] for info in cluster_info]
    colors = ['red' if pct > 50 else 'green' for pct in spam_pcts]
    ax2.bar(cluster_ids, spam_pcts, color=colors, alpha=0.7)
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spam Percentage', fontsize=12, fontweight='bold')
    ax2.set_title('Spam Percentage by Cluster', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Stacked bar chart of ham vs spam
    ax3 = axes[1, 0]
    ham_counts = [info['ham'] for info in cluster_info]
    spam_counts = [info['spam'] for info in cluster_info]
    ax3.bar(cluster_ids, ham_counts, label='Ham', color='green', alpha=0.7)
    ax3.bar(cluster_ids, spam_counts, bottom=ham_counts, label='Spam', color='red', alpha=0.7)
    ax3.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
    ax3.set_title('Ham vs Spam Distribution by Cluster', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Cluster classification summary
    # Use fixed order so pie colors match semantics (spam=red, ham=green)
    ax4 = axes[1, 1]
    classifications = [info['classification'] for info in cluster_info]
    class_counts = Counter(classifications)
    order = ['VERY SPAM', 'SPAM-HEAVY', 'HAM-HEAVY', 'VERY HAM']
    pie_labels = [k for k in order if k in class_counts]
    pie_sizes = [class_counts[k] for k in pie_labels]
    pie_colors = ['darkred', 'red', 'lightgreen', 'darkgreen'][:len(pie_labels)]
    ax4.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
           startangle=90)
    ax4.set_title('Cluster Classification Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return cluster_info, fig

def add_cluster_features(bow_matrix, cluster_labels):
    """
    Combine BOW features with cluster features.
    
    Adds cluster assignments as additional features to the BOW matrix.
    Using one-hot encoding so each cluster becomes a binary feature,
    allowing the Naive Bayes classifier to learn cluster-specific patterns.
    
    Args:
        bow_matrix: sparse BOW feature matrix (n_samples, n_features)
        cluster_labels: array of cluster assignments (n_samples,)
    
    Returns:
        combined_matrix: sparse matrix with BOW + cluster features
    """
    print("\n" + "="*80)
    print("ADDING CLUSTER FEATURES")
    print("="*80)
    
    print(f"\nOriginal BOW matrix shape: {bow_matrix.shape}")
    print(f"Number of clusters: {len(np.unique(cluster_labels))}")
    
    # One-hot encode cluster labels
    # This creates binary features: one column per cluster
    # Each email has a 1 in its assigned cluster column, 0 elsewhere
    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)
    
    # Create one-hot encoded matrix
    # Create cluster feature matrix
    cluster_features = np.zeros((n_samples, n_clusters))
    cluster_features[np.arange(n_samples), cluster_labels] = 1
    cluster_features_sparse = csr_matrix(cluster_features)
    
    print(f"Cluster features shape: {cluster_features_sparse.shape}")
    
    # Concatenate BOW features with cluster features
    # This preserves sparsity for memory efficiency
    combined_matrix = hstack([bow_matrix, cluster_features_sparse])
    
    print(f"\nCombined matrix shape: {combined_matrix.shape} "
          f"(BOW: {bow_matrix.shape[1]}, clusters: {n_clusters})")
    
    return combined_matrix

