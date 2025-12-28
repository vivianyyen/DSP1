import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Load the merged adversary data produced by ttpCategory.py (preferred)
merge_file = 'Categorized_Adversary_TTPs_hc.json'
fallback_file = 'enterprise-attack-18.1.json'
try:
    with open(merge_file, 'r', encoding='utf-8') as f:
        merge_list = json.load(f)
    if not isinstance(merge_list, list):
        raise ValueError(f"{merge_file} does not contain a list of merged adversaries")
    print(f"Loaded {len(merge_list)} merged adversaries from {merge_file}")
except FileNotFoundError:
    # Helpful message and hint to the user
    print(f"Error: {merge_file} not found. Please run ttpCategory.py first to extract and process adversary data.")
    print("\nSteps to fix:")
    print("1. Ensure you have 'enterprise-attack-18.1.json' (MITRE ATT&CK STIX bundle)")
    print("2. Run: python ttpCategory.py")
    print("3. Then run: python hierarchical_clustering.py")
    # If the ATT&CK bundle is present, remind user what to do
    try:
        with open(fallback_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'objects' in data:
            print(f"\n✓ Found {fallback_file} (ATT&CK STIX bundle). Run ttpCategory.py to process it.")
    except FileNotFoundError:
        print(f"\n✗ {fallback_file} not found. Download from: https://github.com/mitre/cti")
    exit(1)
except ValueError as e:
    print("Error: unexpected format in merged adversary file:", e)
    exit(1)

# Hierarchical Clustering on TTPs
def hierarchical_cluster_adversaries(merge_list, n_clusters=None, linkage_method='ward', distance_metric='euclidean'):
    """
    Cluster adversaries by TTP similarity using Hierarchical Clustering.
    
    Parameters:
    - merge_list: List of adversary dictionaries with TTPs
    - n_clusters: Number of clusters (None for auto-detection)
    - linkage_method: 'ward', 'complete', 'average', 'single'
    - distance_metric: 'euclidean', 'cosine', 'manhattan', etc.
    """
    
    if not merge_list:
        print("No merged adversaries to cluster.")
        return None, None, None, None
    
    # Convert TTP lists to strings for vectorization
    ttp_strings = [' '.join(entry.get('mitre_attack_ttps', [])) for entry in merge_list]
    
    # Handle case where all entries have no TTPs
    if all(not s.strip() for s in ttp_strings):
        print("Warning: No TTPs found. Clustering skipped.")
        return None, None, None, None
    
    # Vectorize TTPs using TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    ttp_matrix = vectorizer.fit_transform(ttp_strings)
    
    # Convert to dense array for hierarchical clustering
    ttp_dense = ttp_matrix.toarray()
    
    # Compute linkage matrix for dendrogram
    if linkage_method == 'ward':
        # Ward requires euclidean distance
        Z = linkage(ttp_dense, method='ward')
    else:
        # For other methods, compute distance matrix first
        if distance_metric == 'cosine':
            # Compute cosine distance
            distances = pdist(ttp_dense, metric='cosine')
        else:
            distances = pdist(ttp_dense, metric=distance_metric)
        Z = linkage(distances, method=linkage_method)
    
    # Auto-detect optimal k if not provided
    if n_clusters is None:
        if len(merge_list) < 3:
            n_clusters = len(merge_list)
        else:
            scores = []
            k_range = range(2, min(11, len(merge_list) + 1))
            
            # Try different numbers of clusters and evaluate
            for k in k_range:
                clusterer = AgglomerativeClustering(
                    n_clusters=k, 
                    linkage=linkage_method,
                    metric=distance_metric if linkage_method != 'ward' else 'euclidean'
                )
                labels = clusterer.fit_predict(ttp_dense)
                
                # Use silhouette score for evaluation
                score = silhouette_score(ttp_dense, labels, metric=distance_metric if linkage_method != 'ward' else 'euclidean')
                scores.append(score)
            
            n_clusters = list(k_range)[np.argmax(scores)]
            print(f"Auto-detected optimal clusters: {n_clusters} (using silhouette score)")
            print(f"Silhouette scores across k: {dict(zip(k_range, [f'{s:.3f}' for s in scores]))}")
    
    # Perform Hierarchical Clustering with optimal k
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=distance_metric if linkage_method != 'ward' else 'euclidean'
    )
    labels = clusterer.fit_predict(ttp_dense)
    
    # Assign cluster labels to adversaries
    for i, entry in enumerate(merge_list):
        entry['cluster'] = int(labels[i])
    
    # Print cluster summary
    print(f"\n{'='*60}")
    print(f"Hierarchical Clustering Results ({n_clusters} clusters)")
    print(f"Linkage Method: {linkage_method.upper()}, Distance Metric: {distance_metric.upper()}")
    print(f"{'='*60}")
    
    for cluster_id in range(n_clusters):
        members = [e['mitre_attack_name'] for e in merge_list if e.get('cluster') == cluster_id]
        print(f"\n  Cluster {cluster_id}: {len(members)} members")
        print(f"  └─ {', '.join(members[:5])}")
        if len(members) > 5:
            print(f"     ... and {len(members) - 5} more")
    
    # Compute multiple evaluation metrics
    if n_clusters >= 2 and len(merge_list) >= 2:
        silhouette_avg = silhouette_score(ttp_dense, labels, metric=distance_metric if linkage_method != 'ward' else 'euclidean')
        db_score = davies_bouldin_score(ttp_dense, labels)
        ch_score = calinski_harabasz_score(ttp_dense, labels)
        
        print(f"\n{'='*60}")
        print(f"CLUSTERING EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"  Silhouette Score:        {silhouette_avg:.4f}  (higher is better, range: -1 to 1)")
        print(f"  Davies-Bouldin Index:    {db_score:.4f}  (lower is better)")
        print(f"  Calinski-Harabasz Score: {ch_score:.2f}  (higher is better)")
        print(f"{'='*60}\n")
    else:
        print("Evaluation Metrics: N/A (need at least 2 clusters)")

    return merge_list, ttp_dense, labels, Z

def plot_dendrogram(Z, merge_list, n_clusters=None, max_display=30, show_full=False):
    """Plot hierarchical clustering dendrogram and optionally draw cut for n_clusters."""
    
    plt.figure(figsize=(16, 8))
    
    # Prepare labels (truncated to keep plot tidy)
    labels = [e['mitre_attack_name'][:15] for e in merge_list]
    
    # Decide whether to truncate the dendrogram for large datasets
    use_truncate = (len(merge_list) > max_display) and (not show_full)
    
    # Compute color threshold for cluster cut if requested
    color_threshold = None
    if n_clusters is not None and n_clusters >= 2 and Z is not None:
        try:
            # Height at which to cut to form n_clusters clusters
            color_threshold = Z[-(n_clusters - 1), 2]
            plt.axhline(y=color_threshold, color='r', linestyle='--', linewidth=1, label=f'Cut for k={n_clusters}')
        except Exception:
            color_threshold = None
    
    if use_truncate:
        # Show truncated dendrogram for large datasets, preserving color threshold
        dendrogram(Z, 
                   truncate_mode='lastp',
                   p=max_display,
                   leaf_rotation=90,
                   leaf_font_size=10,
                   show_contracted=True,
                   color_threshold=color_threshold)
        title = f'Hierarchical Clustering Dendrogram (Truncated - showing last {max_display} merges)'
    else:
        # Show full dendrogram for smaller datasets (or when requested)
        dendrogram(Z, 
                   labels=labels,
                   leaf_rotation=90,
                   leaf_font_size=8,
                   color_threshold=color_threshold)
        title = 'Hierarchical Clustering Dendrogram (Full Tree)'
    
    plt.title(title, fontsize=14, fontweight='bold')
    if color_threshold is not None:
        plt.legend(loc='upper right')
    
    plt.xlabel('Adversary / Cluster Size', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
    print("Dendrogram saved: dendrogram.png")
    plt.show()

def visualize_hierarchical_clusters(merge_list, ttp_matrix, labels, n_clusters):
    """Create comprehensive visualizations of the hierarchical clusters."""
    
    if ttp_matrix is None or labels is None:
        print("No data to visualize.")
        return
    
    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    ttp_2d = pca.fit_transform(ttp_matrix)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Scatter plot of clusters with PCA
    ax1 = plt.subplot(1, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax1.scatter(ttp_2d[mask, 0], ttp_2d[mask, 1], 
                   c=[colors[cluster_id]], 
                   label=f'Cluster {cluster_id}', 
                   s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add adversary names as annotations (for first 20 to avoid clutter)
    for i, entry in enumerate(merge_list[:20]):
        ax1.annotate(entry['mitre_attack_name'][:10], 
                   (ttp_2d[i, 0], ttp_2d[i, 1]), 
                   fontsize=7, alpha=0.6, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=10)
    ax1.set_title('Adversary Clusters\n(Hierarchical, PCA 2D projection)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster size distribution
    ax2 = plt.subplot(1, 3, 2)
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    bars = ax2.bar(range(n_clusters), cluster_sizes, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Cluster ID', fontsize=10)
    ax2.set_ylabel('Number of Adversaries', fontsize=10)
    ax2.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(n_clusters))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cluster cohesion (average within-cluster distance)
    ax3 = plt.subplot(1, 3, 3)
    cohesion_scores = []
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_points = ttp_matrix[mask]
        
        if cluster_points.shape[0] > 1:
            # Calculate average pairwise distance within cluster
            distances = pdist(cluster_points, metric='euclidean')
            avg_distance = np.mean(distances)
            cohesion_scores.append(avg_distance)
        else:
            cohesion_scores.append(0)
    
    bars = ax3.bar(range(n_clusters), cohesion_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Cluster ID', fontsize=10)
    ax3.set_ylabel('Avg. Within-Cluster Distance', fontsize=10)
    ax3.set_title('Cluster Cohesion\n(lower = more cohesive)', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(n_clusters))
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('hierarchical_clusters_analysis.png', dpi=150, bbox_inches='tight')
    print("Cluster analysis visualization saved: hierarchical_clusters_analysis.png")
    plt.show()

def plot_dendrogram_comparison(Z, merge_list, k_values=[2, 3, 4]):
    """Plot dendrograms side-by-side for different k values to visualize cut thresholds."""
    
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(6*n_k, 8))
    
    if n_k == 1:
        axes = [axes]  # Handle single subplot case
    
    labels = [e['mitre_attack_name'][:15] for e in merge_list]
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        plt.sca(ax)
        
        # Compute color threshold for this k
        try:
            color_threshold = Z[-(k - 1), 2]
        except IndexError:
            color_threshold = None
        
        # Determine if we should truncate
        use_truncate = len(merge_list) > 30
        
        if use_truncate:
            dendrogram(Z,
                       ax=ax,
                       truncate_mode='lastp',
                       p=30,
                       leaf_rotation=90,
                       leaf_font_size=8,
                       show_contracted=True,
                       color_threshold=color_threshold)
            ax.set_title(f'k={k} (Truncated)', fontsize=12, fontweight='bold')
        else:
            dendrogram(Z,
                       ax=ax,
                       labels=labels,
                       leaf_rotation=90,
                       leaf_font_size=7,
                       color_threshold=color_threshold)
            ax.set_title(f'k={k}', fontsize=12, fontweight='bold')
        
        if color_threshold is not None:
            ax.axhline(y=color_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Cut for k={k}')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.set_ylabel('Distance', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Hierarchical Clustering Dendrogram Comparison (k={k_values})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('dendrogram_comparison.png', dpi=150, bbox_inches='tight')
    print("Dendrogram comparison saved: dendrogram_comparison.png")
    plt.show()

def compare_linkage_methods(merge_list, ttp_matrix):
    """Compare different linkage methods for hierarchical clustering."""
    
    print("\n" + "="*60)
    print("COMPARING LINKAGE METHODS")
    print("="*60 + "\n")
    
    linkage_methods = ['ward', 'complete', 'average', 'single']
    results = []
    
    for method in linkage_methods:
        print(f"Testing {method.upper()} linkage...")
        
        # Perform clustering with auto-detected k
        _, ttp_dense, labels, _ = hierarchical_cluster_adversaries(
            merge_list.copy(), 
            n_clusters=None, 
            linkage_method=method,
            distance_metric='euclidean'
        )
        
        if labels is not None:
            n_clusters = len(set(labels))
            silhouette = silhouette_score(ttp_dense, labels)
            db_score = davies_bouldin_score(ttp_dense, labels)
            ch_score = calinski_harabasz_score(ttp_dense, labels)
            
            results.append({
                'method': method,
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score
            })
    
    # Display comparison table
    print("\n" + "="*80)
    print(f"{'Method':<12} {'Clusters':<10} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Calinski-Harabasz':<20}")
    print("="*80)
    for r in results:
        print(f"{r['method']:<12} {r['n_clusters']:<10} {r['silhouette']:<15.4f} {r['davies_bouldin']:<20.4f} {r['calinski_harabasz']:<20.2f}")
    print("="*80 + "\n")
    
    # Recommend best method
    best_silhouette = max(results, key=lambda x: x['silhouette'])
    print(f"✓ RECOMMENDATION: {best_silhouette['method'].upper()} linkage has the highest Silhouette Score ({best_silhouette['silhouette']:.4f})")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Hierarchical Clustering Analysis')
parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (auto-detect if not specified)')
parser.add_argument('--linkage', default='ward', choices=['ward', 'complete', 'average', 'single'], help='Linkage method')
parser.add_argument('--compare', action='store_true', help='Show dendrogram comparison for k=2,3,4')
args = parser.parse_args()

# Interactive menu if no command-line args provided and not in compare mode
if args.n_clusters is None and not args.compare:
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING - INTERACTIVE MODE")
    print("="*60)
    print("\nOptions:")
    print("  1. Auto-detect optimal k (using silhouette score)")
    print("  2. Manually specify k")
    print("  3. Compare k=2, k=3, k=4 side-by-side")
    print("  4. Exit")
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '2':
        try:
            args.n_clusters = int(input("Enter number of clusters: "))
        except ValueError:
            print("Invalid input. Using auto-detect.")
    elif choice == '3':
        args.compare = True
    elif choice == '4':
        print("Exiting.")
        exit(0)

# Run hierarchical clustering
print("\n" + "="*60)
print("STARTING HIERARCHICAL CLUSTERING ANALYSIS")
print("="*60 + "\n")

# Determine which k to use
if args.compare:
    # First, run with auto-detect to get Z
    merge_list_clustered, ttp_dense, labels, Z = hierarchical_cluster_adversaries(
        merge_list, 
        n_clusters=None,
        linkage_method=args.linkage,
        distance_metric='euclidean'
    )
    
    # Show comparison
    if Z is not None:
        print("\nGenerating dendrogram comparison for k=2, 3, 4...")
        plot_dendrogram_comparison(Z, merge_list, k_values=[2, 3, 4])
        
        # Also run final clustering with k=2 (optimal) for output
        merge_list_clustered, ttp_dense, labels, Z = hierarchical_cluster_adversaries(
            merge_list,
            n_clusters=2,  # Use optimal k
            linkage_method=args.linkage,
            distance_metric='euclidean'
        )
else:
    # Option 1: Run with specified or auto-detected k
    merge_list_clustered, ttp_dense, labels, Z = hierarchical_cluster_adversaries(
        merge_list, 
        n_clusters=args.n_clusters,  # Auto-detect if None
        linkage_method=args.linkage,
        distance_metric='euclidean'
    )

# Visualize dendrogram
if Z is not None:
    n_clusters = len(set(labels)) if labels is not None else None
    plot_dendrogram(Z, merge_list, n_clusters=n_clusters)

# Visualize clusters
if merge_list_clustered and ttp_dense is not None:
    n_clusters = len(set(labels))
    visualize_hierarchical_clusters(merge_list_clustered, ttp_dense, labels, n_clusters)
    
    # Write clustered output
    output_file = 'Hierarchical_Clustered_Adversaries.json'
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merge_list_clustered, outfile, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote clustered output: {output_file}")

# Optional: Compare different linkage methods
print("\n" + "="*60)
user_input = input("Would you like to compare different linkage methods? (y/n): ").strip().lower()
if user_input == 'y':
    compare_linkage_methods(merge_list, ttp_dense)