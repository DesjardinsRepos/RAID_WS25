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

from word2vec_approach.evaluate_w2v_feature_calculator import calc_features_w2v
from tqdm import tqdm

def load_functions(filepath, use_tokenized=True):
    """
    Load functions from either raw or tokenized JSONL file.
    
    Args:
        filepath: Path to the functions file
        use_tokenized: If True, loads from tokenized_functions.jsonl, otherwise loads raw functions
    """
    # Make path relative to project root
    filepath = project_root / filepath
    
    if use_tokenized:
        # Convert raw functions path to tokenized path
        filepath = str(filepath).replace('functions.jsonl', 'tokenized_functions.jsonl')
        print(f"Loading tokenized functions from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            functions = [json.loads(line) for line in tqdm(f.readlines(), desc="Loading tokenized functions")]
    else:
        print(f"Loading raw functions from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            functions = [json.loads(line)['function'] for line in tqdm(f.readlines(), desc="Loading raw functions")]
    return functions

def main():
    print("=== Starting Training Pipeline ===")
    
    # Load data
    print("\n[1/4] Loading training and validation data...")
    functions = load_functions('data/train/functions.jsonl', use_tokenized=True)
    labels = np.load(project_root / 'data/train/labels.npy')
    print(f"Loaded {len(functions)} training samples")
    
    functions_test = load_functions('data/validation/functions.jsonl', use_tokenized=True)
    labels_test = np.load(project_root / 'data/validation/labels.npy')
    print(f"Loaded {len(functions_test)} validation samples")

    print("\n[2/4] Calculating training features...")
    X_train = calc_features_w2v(str(project_root / 'data/train/tokenized_functions.jsonl'))  
    y_train = labels

    print("\n[3/4] Calculating test features...")
    X_test = calc_features_w2v(str(project_root / 'data/validation/tokenized_functions.jsonl')) 
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
