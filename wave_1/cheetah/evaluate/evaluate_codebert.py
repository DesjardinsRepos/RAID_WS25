import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))



def main():
    print("=== Starting Training Pipeline ===")
    
    # Load pre-calculated embeddings and labels
    print("\n[1/3] Loading training data...")
    X_train = np.load('../user/data/train/embeddings.npy')
    y_train = np.load('../user/data/train/labels.npy')
    print(f"Loaded {len(X_train)} training samples")
    
    print("\n[2/3] Loading validation data...")
    X_test = np.load('../user/data/validation/embeddings.npy')
    y_test = np.load('../user/data/validation/labels.npy')
    print(f"Loaded {len(X_test)} validation samples")

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
    print("\n[3/3] Training model...")
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
