import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import sys
from pathlib import Path
import pickle

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
    
    # Load validation data
    print("\n[1/3] Loading validation data...")
    functions_val = load_functions('data/validation/functions.jsonl')
    labels_val = np.load(project_root / 'data/validation/labels.npy')
    print(f"Loaded {len(functions_val)} validation samples")

    print("\n[2/3] Calculating validation features...")
    X_val = calc_features(functions_val) 
    y_val = labels_val

    # XGBoost-Classifier
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=6000,
        learning_rate=0.01,
        max_depth=30,
        random_state=42
    )

    # Model training on validation data
    print("\n[3/3] Training model on validation dataset...")
    clf.fit(X_val, y_val, 
            eval_set=[(X_val, y_val)],
            verbose=True)

    # Save the trained model
    model_path = project_root / 'models' / 'xgb_model_2.pkl'
    model_path.parent.mkdir(exist_ok=True)  # Create models directory if it doesn't exist
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {model_path}")

    # Evaluation on validation set
    print("\n=== Model Evaluation ===")
    y_pred = clf.predict(X_val)

    # Calculate F1 score
    f1 = f1_score(y_val, y_pred)
    print(f"\nOverall F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
