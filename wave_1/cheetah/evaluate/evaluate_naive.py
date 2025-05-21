import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import sys
from pathlib import Path

# Add the parent directory to Python path to allow imports from word2vec_approach
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from naive_approach.solution import calc_features
from tqdm import tqdm

def load_functions(filepath):
    """
    Load functions from raw JSONL file.
    
    Args:
        filepath: Path to the functions file
    """
    # Make path relative to project root
    filepath = project_root / filepath
    
    print(f"Loading raw functions from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        functions = [json.loads(line)['function'] for line in tqdm(f.readlines(), desc="Loading raw functions")]
    return functions

def main():
    print("=== Starting Training Pipeline ===")
    
    # Load data
    print("\n[1/4] Loading training and validation data...")
    functions = load_functions('data/train/functions.jsonl')
    labels = np.load(project_root / 'data/train/labels.npy')
    print(f"Loaded {len(functions)} training samples")
    
    functions_test = load_functions('data/validation/functions.jsonl')
    labels_test = np.load(project_root / 'data/validation/labels.npy')
    print(f"Loaded {len(functions_test)} validation samples")

    print("\n[2/4] Calculating training features...")
    X_train = calc_features(functions)  
    y_train = labels

    print("\n[3/4] Calculating test features...")
    X_test = calc_features(functions_test) 
    y_test = labels_test

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

    # Model training
    print("\n[4/4] Training model...")
    clf.fit(X_train, y_train, 
            eval_set=[(X_test, y_test)],
            verbose=True)

    # Evaluation
    print("\n=== Model Evaluation ===")
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"\nOverall F1 Score: {f1:.4f}")


    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
