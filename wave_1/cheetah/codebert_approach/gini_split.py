import numpy as np
from tqdm import tqdm

def gini_impurity_split(feature_values, labels):
    """Efficient Gini impurity split computation (binary labels assumed)"""
    sorted_idx = np.argsort(feature_values)
    feature_values = feature_values[sorted_idx]
    labels = labels[sorted_idx]
    
    total = len(labels)
    total_pos = np.sum(labels)
    total_neg = total - total_pos

    best_gini = 1.0
    left_pos = 0
    left_neg = 0

    for i in range(1, total):
        label = labels[i - 1]
        if label == 1:
            left_pos += 1
        else:
            left_neg += 1

        if feature_values[i] == feature_values[i - 1]:
            continue

        right_pos = total_pos - left_pos
        right_neg = total_neg - left_neg

        left_total = left_pos + left_neg
        right_total = right_pos + right_neg

        def gini(pos, neg):
            total = pos + neg
            if total == 0:
                return 0
            p = pos / total
            return 1 - p**2 - (1 - p)**2

        gini_split = (left_total / total) * gini(left_pos, left_neg) + \
                     (right_total / total) * gini(right_pos, right_neg)
        best_gini = min(best_gini, gini_split)

    return best_gini  

def select_top_k_features_by_gini(X, y, k=32):
    """Select top-k features from X based on Gini impurity reduction"""
    print(f"Selecting top {k} features based on Gini score...")
    scores = [gini_impurity_split(X[:, i], y) for i in tqdm(range(X.shape[1]), desc="Scoring features")]
    sorted_indices = np.argsort(scores)[:k]  # Top-k features
    sorted_scores = np.array(scores)[sorted_indices]
    return sorted_indices, X[:, sorted_indices], sorted_scores

def save_top_features_indices(indices, output_path):
    """
    Save the indices of the top features to a text file
    
    Args:
        indices: Indices of the top features
        output_path: Path to save the file
    """
    with open(output_path, 'w') as f:
        f.write("Top Features by Gini Impurity Reduction:\n")
        f.write("Index\n")
        f.write("-----\n")
        
        for idx in indices:
            f.write(f"{idx}\n")
    
    print(f"Top feature indices saved to {output_path}")

def save_top_features_with_scores(indices, scores, output_path):
    """
    Save the indices and scores of the top features to a text file
    
    Args:
        indices: Indices of the top features
        scores: Gini scores for the features
        output_path: Path to save the file
    """
    with open(output_path, 'w') as f:
        f.write("Top Features by Gini Impurity Reduction:\n")
        f.write("Index\tScore\n")
        f.write("-----\t-----\n")
        
        for idx, score in zip(indices, scores):
            f.write(f"{idx}\t{score:.6f}\n")
    
    print(f"Top feature indices and scores saved to {output_path}")

import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import sys
from pathlib import Path
from tqdm import tqdm

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

def main():
    print("=== Starting Training Pipeline ===")
    
    # Load pre-calculated embeddings and labels
    print("\n[1/4] Loading training data...")
    X_train = np.load('../user/data/train/embeddings.npy')
    y_train = np.load('../user/data/train/labels.npy')
    print(f"Loaded {len(X_train)} training samples")
    
    print("\n[2/4] Loading validation data...")
    X_test = np.load('../user/data/validation/embeddings.npy')
    y_test = np.load('../user/data/validation/labels.npy')
    print(f"Loaded {len(X_test)} validation samples")

    # Select top features using Gini impurity
    print("\n[3/4] Selecting top features using Gini impurity...")
    top_k = 350  # You can adjust this number
    selected_indices, X_train_selected, selected_scores = select_top_k_features_by_gini(X_train, y_train, k=top_k)
    X_test_selected = X_test[:, selected_indices]
    
    # Save the selected feature indices and scores with top_k in filename
    indices_path = script_dir / f'top_{top_k}_features_indices.txt'
    scores_path = script_dir / f'top_{top_k}_features_with_scores.txt'
    save_top_features_indices(selected_indices, indices_path)
    save_top_features_with_scores(selected_indices, selected_scores, scores_path)

    # XGBoost-Classifier
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=25,
        random_state=42
    )

    # Model training with selected features
    print("\n[4/4] Training model with selected features...")
    clf.fit(X_train_selected, y_train, 
            eval_set=[(X_test_selected, y_test)],
            verbose=True)

    # Evaluation
    print("\n=== Model Evaluation ===")
    y_pred = clf.predict(X_test_selected)
    f1 = f1_score(y_test, y_pred)
    print(f"\nOverall F1 Score: {f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
