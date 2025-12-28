"""
Objective 2: Supervised Learning Model for Threat Actor Classification

This script develops a predictive cyber-intelligence model that automatically classifies 
threat actors based on similarities in their TTPs, enabling identification of threat 
actors' behavioral signatures when operating under different aliases.

This builds on Objective 1 (clustering) by:
1. Using cluster labels as training data
2. Training multiple supervised learning models
3. Implementing alias detection mechanism
4. Evaluating cross-framework generalization

Input: Hierarchical_Clustered_Adversaries.json (from Objective 1)
Output: Trained models, predictions, alias detection results
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_fscore_support, accuracy_score)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OBJECTIVE 2: SUPERVISED THREAT ACTOR CLASSIFICATION MODEL")
print("="*80)

# ============================================================================
# STEP 1: LOAD CLUSTERED DATA FROM OBJECTIVE 1
# ============================================================================

def load_clustered_data(filepath='Hierarchical_Clustered_Adversaries.json'):
    """Load adversary data with cluster labels from Objective 1."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n✓ Loaded {len(data)} adversaries from {filepath}")
        return data
    except FileNotFoundError:
        print(f"\n✗ Error: {filepath} not found.")
        print("Please run hierarchical_clustering.py (Objective 1) first.")
        exit(1)

# ============================================================================
# STEP 2: PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(adversaries, min_ttp_count=5):
    """
    Prepare features (X) and labels (y) for supervised learning.
    
    Features: TTP vectors (TF-IDF)
    Labels: Cluster assignments from Objective 1
    """
    print("\n" + "="*80)
    print("STEP 1: PREPARING TRAINING DATA")
    print("="*80)
    
    # Filter adversaries with sufficient TTPs
    filtered = [adv for adv in adversaries if len(adv.get('mitre_attack_ttps', [])) >= min_ttp_count]
    print(f"Filtered: {len(filtered)}/{len(adversaries)} adversaries with >= {min_ttp_count} TTPs")
    
    if len(filtered) < 10:
        print("⚠ WARNING: Very few adversaries for training. Results may not be reliable.")
    
    # Extract features and labels
    # If an adversary has no TTPs, replace with a placeholder token so vectorizer can process it
    ttp_strings = []
    zero_ttp_count = 0
    for adv in filtered:
        ttps = adv.get('mitre_attack_ttps', [])
        if not ttps:
            zero_ttp_count += 1
            ttp_strings.append('NO_TTP')
        else:
            ttp_strings.append(' '.join(ttps))
    cluster_labels = [adv.get('cluster', -1) for adv in filtered]
    adversary_names = [adv.get('mitre_attack_name', 'Unknown') for adv in filtered]
    adversary_ids = [adv.get('mitre_attack_id', 'Unknown') for adv in filtered]
    aliases = [adv.get('aliases', []) for adv in filtered]
    if zero_ttp_count:
        print(f"Note: {zero_ttp_count} adversaries have zero TTPs and were represented with placeholder 'NO_TTP'.")    
    # Check for missing cluster labels
    if -1 in cluster_labels:
        print("⚠ WARNING: Some adversaries missing cluster labels. Run Objective 1 first.")
        filtered = [adv for adv, label in zip(filtered, cluster_labels) if label != -1]
        ttp_strings = [' '.join(adv.get('mitre_attack_ttps', [])) for adv in filtered]
        cluster_labels = [adv.get('cluster', -1) for adv in filtered]
        adversary_names = [adv.get('mitre_attack_name', 'Unknown') for adv in filtered]
        adversary_ids = [adv.get('mitre_attack_id', 'Unknown') for adv in filtered]
        aliases = [adv.get('aliases', []) for adv in filtered]
    
    # Vectorize TTPs
    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'T\d+\.\d+|T\d+|\b\w+\b',
        max_features=500,
        min_df=1,
        max_df=0.95
    )
    X = vectorizer.fit_transform(ttp_strings).toarray()
    y = np.array(cluster_labels)
    
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Number of Classes: {len(set(y))}")
    print(f"Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Create metadata dictionary
    metadata = {
        'names': adversary_names,
        'ids': adversary_ids,
        'aliases': aliases,
        'ttp_strings': ttp_strings
    }
    
    return X, y, vectorizer, metadata, filtered

# ============================================================================
# STEP 3: TRAIN MULTIPLE MODELS
# ============================================================================

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple classification models."""
    print("\n" + "="*80)
    print("STEP 2: TRAINING MULTIPLE CLASSIFICATION MODELS")
    print("="*80)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training: {name}")
        print(f"{'='*40}")
        
        # Train (guard against failures such as single-class y)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"✗ Skipping {name}: training failed: {e}")
            continue
        
        # Predict
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        except Exception as e:
            print(f"⚠ Warning: {name} prediction failed: {e}")
            y_pred = np.array([-1] * len(X_test))
            y_pred_proba = None
        
        # Evaluate
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        except Exception as e:
            print(f"⚠ Warning: evaluation failed for {name}: {e}")
            accuracy = precision = recall = f1 = 0.0
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Cross-validation (guarded)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            print(f"CV F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        except Exception as e:
            print(f"⚠ Warning: cross-validation failed for {name}: {e}")
            cv_scores = np.array([0.0])
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        trained_models[name] = model
    
    return results, trained_models

# ============================================================================
# STEP 4: SELECT BEST MODEL
# ============================================================================

def select_best_model(results, trained_models):
    """Select the best performing model based on F1-score."""
    print("\n" + "="*80)
    print("STEP 3: MODEL COMPARISON & SELECTION")
    print("="*80)
    
    # Create comparison table
    comparison = []
    for name, metrics in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'CV Mean': f"{metrics['cv_scores'].mean():.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))
    
    # Select best model by F1-score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = trained_models[best_model_name]
    
    print(f"\n✓ BEST MODEL: {best_model_name}")
    print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
    
    return best_model, best_model_name, results[best_model_name]

# ============================================================================
# STEP 5: ALIAS DETECTION MECHANISM
# ============================================================================

def detect_aliases(X, metadata, best_model, similarity_threshold=0.75):
    """
    Detect potential aliases by finding adversaries with similar TTP profiles.
    
    Method: Calculate cosine similarity between TTP vectors and flag pairs
    above threshold as potential aliases.
    """
    print("\n" + "="*80)
    print("STEP 4: ALIAS DETECTION MECHANISM")
    print("="*80)
    print(f"Similarity Threshold: {similarity_threshold}")
    
    n_adversaries = X.shape[0]
    potential_aliases = []
    
    # Calculate pairwise cosine similarities
    for i in range(n_adversaries):
        for j in range(i + 1, n_adversaries):
            # Cosine similarity
            similarity = 1 - cosine(X[i], X[j])
            
            if similarity >= similarity_threshold:
                potential_aliases.append({
                    'adversary_1': metadata['names'][i],
                    'adversary_2': metadata['names'][j],
                    'id_1': metadata['ids'][i],
                    'id_2': metadata['ids'][j],
                    'similarity': similarity,
                    'cluster_1': best_model.predict([X[i]])[0],
                    'cluster_2': best_model.predict([X[j]])[0]
                })
    
    print(f"\nFound {len(potential_aliases)} potential alias pairs (similarity >= {similarity_threshold})")
    
    if potential_aliases:
        # Sort by similarity
        potential_aliases.sort(key=lambda x: x['similarity'], reverse=True)
        
        print("\nTop 10 Potential Alias Pairs:")
        print("-"*80)
        for i, pair in enumerate(potential_aliases[:10], 1):
            print(f"{i}. {pair['adversary_1']} ({pair['id_1']}) ↔ {pair['adversary_2']} ({pair['id_2']})")
            print(f"   Similarity: {pair['similarity']:.4f} | Clusters: {pair['cluster_1']} ↔ {pair['cluster_2']}")
    else:
        print("No potential aliases detected at this threshold.")
        print("Consider lowering the threshold (e.g., 0.65) or reviewing data quality.")
    
    # Validate with known aliases
    print("\n" + "-"*80)
    print("VALIDATING WITH KNOWN ALIASES")
    print("-"*80)
    
    true_positives = 0
    false_positives = 0
    
    for pair in potential_aliases:
        # Check if adversary_1 is in adversary_2's alias list or vice versa
        idx_1 = metadata['names'].index(pair['adversary_1'])
        idx_2 = metadata['names'].index(pair['adversary_2'])
        
        aliases_1 = metadata['aliases'][idx_1]
        aliases_2 = metadata['aliases'][idx_2]
        
        # Check for alias match
        is_true_alias = (
            pair['adversary_2'] in aliases_1 or
            pair['adversary_1'] in aliases_2 or
            any(alias in aliases_2 for alias in aliases_1) or
            any(alias in aliases_1 for alias in aliases_2)
        )
        
        if is_true_alias:
            true_positives += 1
        else:
            false_positives += 1
    
    if len(potential_aliases) > 0:
        alias_precision = true_positives / len(potential_aliases)
        print(f"True Positives (Confirmed Aliases): {true_positives}")
        print(f"False Positives (Unrelated but Similar): {false_positives}")
        print(f"Alias Detection Precision: {alias_precision:.4f}")
    
    return potential_aliases

# ============================================================================
# STEP 6: PREDICTION FOR NEW/UNKNOWN ADVERSARIES
# ============================================================================

def classify_new_adversary(ttp_list, vectorizer, best_model, metadata):
    """
    Classify a new adversary based on their TTPs.
    Returns predicted cluster and confidence scores.
    """
    print("\n" + "="*80)
    print("STEP 5: CLASSIFYING NEW/UNKNOWN ADVERSARIES")
    print("="*80)
    
    # Vectorize TTPs
    ttp_string = ' '.join(ttp_list)
    X_new = vectorizer.transform([ttp_string]).toarray()
    
    # Predict
    predicted_cluster = best_model.predict(X_new)[0]
    confidence = best_model.predict_proba(X_new)[0] if hasattr(best_model, 'predict_proba') else None
    
    print(f"Predicted Cluster: {predicted_cluster}")
    if confidence is not None:
        print(f"Confidence Scores: {dict(enumerate(confidence))}")
        print(f"Max Confidence: {max(confidence):.4f}")
    
    # Find similar known adversaries
    print("\nMost Similar Known Adversaries:")
    similarities = []
    for i, known_ttp in enumerate(metadata['ttp_strings']):
        X_known = vectorizer.transform([known_ttp]).toarray()
        similarity = 1 - cosine(X_new[0], X_known[0])
        similarities.append((metadata['names'][i], metadata['ids'][i], similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    for i, (name, adv_id, sim) in enumerate(similarities[:5], 1):
        print(f"  {i}. {name} ({adv_id}) - Similarity: {sim:.4f}")
    
    return predicted_cluster, confidence, similarities

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

def create_visualizations(y_test, results, best_model_name, metadata):
    """Create comprehensive visualizations of model performance."""
    print("\n" + "="*80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Model Comparison (F1-Scores)
    ax1 = plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in model_names]
    colors = ['green' if name == best_model_name else 'skyblue' for name in model_names]
    
    bars = ax1.bar(range(len(model_names)), f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('F1-Score', fontsize=11)
    ax1.set_title('Model Comparison (F1-Score)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    y_pred_best = results[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=True)
    ax2.set_xlabel('Predicted Cluster', fontsize=11)
    ax2.set_ylabel('True Cluster', fontsize=11)
    ax2.set_title(f'Confusion Matrix ({best_model_name})', fontsize=12, fontweight='bold')
    
    # Plot 3: Precision, Recall, F1 Comparison
    ax3 = plt.subplot(2, 3, 3)
    metrics_data = []
    for name in model_names:
        metrics_data.append([
            results[name]['precision'],
            results[name]['recall'],
            results[name]['f1']
        ])
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax3.bar(x - width, [m[0] for m in metrics_data], width, label='Precision', color='#FF6B6B')
    ax3.bar(x, [m[1] for m in metrics_data], width, label='Recall', color='#4ECDC4')
    ax3.bar(x + width, [m[2] for m in metrics_data], width, label='F1-Score', color='#45B7D1')
    
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Metrics Comparison Across Models', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1])
    
    # Plot 4: Cross-Validation Scores
    ax4 = plt.subplot(2, 3, 4)
    cv_data = [results[name]['cv_scores'] for name in model_names]
    bp = ax4.boxplot(cv_data, labels=model_names, patch_artist=True)
    
    for patch, name in zip(bp['boxes'], model_names):
        if name == best_model_name:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('lightblue')
    
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title('Cross-Validation Scores (5-Fold)', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Class Distribution
    ax5 = plt.subplot(2, 3, 5)
    unique, counts = np.unique(y_test, return_counts=True)
    ax5.bar(unique, counts, color='coral', edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Cluster ID', fontsize=11)
    ax5.set_ylabel('Number of Samples', fontsize=11)
    ax5.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Feature Importance (if Random Forest is best)
    ax6 = plt.subplot(2, 3, 6)
    if 'Random Forest' in best_model_name and hasattr(results, 'feature_importances_'):
        # This would require access to the actual model
        ax6.text(0.5, 0.5, 'Feature Importance\n(Available for Random Forest)',
                ha='center', va='center', fontsize=12)
    else:
        # Show accuracy comparison instead
        accuracies = [results[name]['accuracy'] for name in model_names]
        bars = ax6.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax6.set_xticks(range(len(model_names)))
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.set_ylabel('Accuracy', fontsize=11)
        ax6.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('supervised_classification_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization: supervised_classification_results.png")
    plt.show()

# ============================================================================
# STEP 8: SAVE MODEL & RESULTS
# ============================================================================

def save_model_and_results(best_model, vectorizer, results, potential_aliases, best_model_name):
    """Save trained model and results for future use."""
    print("\n" + "="*80)
    print("STEP 7: SAVING MODEL & RESULTS")
    print("="*80)
    
    # Save model
    with open('trained_classifier_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✓ Saved model: trained_classifier_model.pkl")
    
    # Save vectorizer
    with open('ttp_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Saved vectorizer: ttp_vectorizer.pkl")
    
    # Save results summary
    results_summary = {
        'best_model': best_model_name,
        'performance': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1']
        },
        'alias_detection': {
            'total_potential_aliases': len(potential_aliases),
            'top_10_pairs': potential_aliases[:10] if potential_aliases else []
        }
    }
    
    with open('classification_results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("✓ Saved results: classification_results_summary.json")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow for Objective 2."""
    
    # Load data from Objective 1
    adversaries = load_clustered_data()
    
    # Prepare training data
    X, y, vectorizer, metadata, filtered_adversaries = prepare_training_data(adversaries, min_ttp_count=5)
    
    # Check that we have at least 2 classes
    unique_classes = set(y)
    if len(unique_classes) < 2:
        print("\n⚠ WARNING: Training data contains only one class. Attempting to relax 'min_ttp_count' to include more samples.")
        # Try relaxing the TTP threshold (fall back to 4, 3, 2, 1)
        fallback_found = False
        for new_min in [4, 3, 2, 1, 0]:
            print(f"Trying min_ttp_count={new_min}...")
            X_new, y_new, vectorizer_new, metadata_new, filtered_new = prepare_training_data(adversaries, min_ttp_count=new_min)
            if len(set(y_new)) >= 2 and X_new.shape[0] >= 10:
                print(f"Success: using min_ttp_count={new_min} yields {len(set(y_new))} classes and {X_new.shape[0]} samples.")
                X, y, vectorizer, metadata, filtered_adversaries = X_new, y_new, vectorizer_new, metadata_new, filtered_new
                fallback_found = True
                break
        if not fallback_found:
            print("\n✗ ERROR: Unable to create a training set with >=2 classes after relaxing 'min_ttp_count'.")
            print("Please review your clustering output or reduce the filtering threshold manually.")
            return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Train models
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model, best_model_name, best_results = select_best_model(results, trained_models)
    
    # Alias detection
    potential_aliases = detect_aliases(X, metadata, best_model, similarity_threshold=0.75)
    
    # Example: Classify a new adversary
    print("\n" + "="*80)
    print("EXAMPLE: CLASSIFYING A NEW ADVERSARY")
    print("="*80)
    example_ttps = [
        "T1566.001: Phishing: Spearphishing Attachment",
        "T1059.001: Command and Scripting Interpreter: PowerShell",
        "T1071.001: Application Layer Protocol: Web Protocols"
    ]
    print(f"New Adversary TTPs: {example_ttps[:3]}...")
    classify_new_adversary(example_ttps, vectorizer, best_model, metadata)
    
    # Create visualizations
    create_visualizations(y_test, results, best_model_name, metadata)
    
    # Save everything
    save_model_and_results(best_model, vectorizer, best_results, potential_aliases, best_model_name)
    
    print("\n" + "="*80)
    print("✓ OBJECTIVE 2 COMPLETE")
    print("="*80)
    print("\nNext Steps (Objective 3):")
    print("- Evaluate on held-out test data")
    print("- Test cross-framework generalization (ATLAS, DEFEND)")
    print("- Measure operational utility metrics")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()