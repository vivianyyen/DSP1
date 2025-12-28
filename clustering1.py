import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the merged adversary data produced by ttpCategory.py (preferred)
merge_file = 'Categorized_Adversary_TTPs.json'
fallback_file = 'enterprise-attack-18.1.json'
try:
    with open(merge_file, 'r', encoding='utf-8') as f:
        merge_list = json.load(f)
    if not isinstance(merge_list, list):
        raise ValueError(f"{merge_file} does not contain a list of merged adversaries")
    print(f"Loaded {len(merge_list)} merged adversaries from {merge_file}")
except FileNotFoundError:
    # Helpful message and hint to the user
    print(f"Error: {merge_file} not found. Please run ttpCategory.py to generate it from the ATT&CK bundle (or place the correct file in this folder).")
    # If the ATT&CK bundle is present, remind user what to do
    try:
        with open(fallback_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'objects' in data:
            print(f"Note: found {fallback_file} (ATT&CK STIX bundle). Run ttpCategory.py to produce {merge_file} first.")
    except FileNotFoundError:
        pass
    exit(1)
except ValueError as e:
    print("Error: unexpected format in merged adversary file:", e)
    exit(1)

# Clustering on TTPs
def cluster_adversaries(merge_list, n_clusters=None):
    """Cluster adversaries by TTP similarity using K-means."""
    
    if not merge_list:
        print("No merged adversaries to cluster.")
        return None, None, None
    
    # Convert TTP lists to strings for vectorization
    ttp_strings = [' '.join(entry.get('mitre_attack_ttps', [])) for entry in merge_list]
    
    # Handle case where all entries have no TTPs
    if all(not s.strip() for s in ttp_strings):
        print("Warning: No TTPs found. Clustering skipped.")
        return None, None, None
    
    # Vectorize TTPs using TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    ttp_matrix = vectorizer.fit_transform(ttp_strings)
    
    # Auto-detect optimal k if not provided
    if n_clusters is None:
        if len(merge_list) < 3:
            n_clusters = len(merge_list)
        else:
            scores = []
            k_range = range(2, min(11, len(merge_list) + 1))
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(ttp_matrix)
                score = silhouette_score(ttp_matrix, labels)
                scores.append(score)
            n_clusters = list(k_range)[np.argmax(scores)]
            print(f"Auto-detected optimal clusters: {n_clusters}")
    
    # Perform K-means clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(ttp_matrix)
    
    # Assign cluster labels to adversaries
    for i, entry in enumerate(merge_list):
        entry['cluster'] = int(labels[i])
    
    # Print cluster summary
    print(f"\nClustering Results ({n_clusters} clusters):")
    for cluster_id in range(n_clusters):
        members = [e['mitre_attack_name'] for e in merge_list if e.get('cluster') == cluster_id]
        print(f"  Cluster {cluster_id}: {len(members)} members - {', '.join(members[:5])}")
        if len(members) > 5:
            print(f"             ... and {len(members) - 5} more")
    
    # Compute silhouette score only when there are at least 2 clusters
    if n_clusters >= 2 and len(merge_list) >= 2:
        silhouette_avg = silhouette_score(ttp_matrix, labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
    else:
        print("Silhouette Score: N/A (need at least 2 clusters)")

    return merge_list, ttp_matrix, labels

def visualize_clusters(merge_list, ttp_matrix, labels, n_clusters):
    """Create visualizations of the clusters."""
    
    if ttp_matrix is None or labels is None:
        print("No data to visualize.")
        return
    
    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    ttp_2d = pca.fit_transform(ttp_matrix.toarray())
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter plot of clusters
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(ttp_2d[mask, 0], ttp_2d[mask, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}', 
                   s=100, alpha=0.7, edgecolors='black')
    
    # Add adversary names as annotations (for first 20 to avoid clutter)
    for i, entry in enumerate(merge_list[:20]):
        ax.annotate(entry['mitre_attack_name'][:10], 
                   (ttp_2d[i, 0], ttp_2d[i, 1]), 
                   fontsize=7, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Adversary Clusters (TTP-based, PCA 2D projection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cluster size distribution
    ax = axes[1]
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    ax.bar(range(n_clusters), cluster_sizes, color=colors, edgecolor='black')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Adversaries')
    ax.set_title('Cluster Size Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('adversary_clusters.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved: adversary_clusters.png")
    plt.show()

# Run clustering
print("\n--- Starting TTP-based clustering ---")
merge_list, ttp_matrix, labels = cluster_adversaries(merge_list, n_clusters=None)

# Visualize
if merge_list and ttp_matrix is not None:
    n_clusters = len(set(labels))
    visualize_clusters(merge_list, ttp_matrix, labels, n_clusters)
    
    # Write clustered output
    with open('Categorized_Adversary_TTPs_Clustered.json', 'w', encoding='utf-8') as outfile:
        json.dump(merge_list, outfile, indent=2, ensure_ascii=False)
    print("Wrote clustered output: Categorized_Adversary_TTPs_Clustered.json")