import os
import sys
import json
from pathlib import Path
import pandas as pd

# Add ml-service root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.text.baseline_model import BaselineTextModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "text"

def run_baseline_training():
    print("=" * 60)
    print("TRAINING TEXT BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)

    train_file = DATA_DIR / "train.csv"
    val_file = DATA_DIR / "val.csv"
    test_file = DATA_DIR / "test.csv"

    if not train_file.exists() or not test_file.exists():
        print(f"Error: Harmonized data not found in {DATA_DIR}. Run harmonizer.py first.")
        sys.exit(1)

    print(f"Loading training data from {train_file}...")
    train_df = pd.read_csv(train_file, engine="python").fillna("")
    val_df = pd.read_csv(val_file, engine="python").fillna("")
    test_df = pd.read_csv(test_file, engine="python").fillna("")

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # Combine title and text for rich context
    train_texts = (train_df["title"] + " " + train_df["text"]).tolist()
    train_labels = train_df["label"].tolist()

    val_texts = (val_df["title"] + " " + val_df["text"]).tolist()
    val_labels = val_df["label"].tolist()

    test_texts = (test_df["title"] + " " + test_df["text"]).tolist()
    test_labels = test_df["label"].tolist()

    model = BaselineTextModel(max_features=15000)
    print("Fitting TF-IDF Vectorizer and Logistic Regression...")
    model.train(train_texts, train_labels)

    print("\nEvaluating on Validation Set:")
    val_metrics = model.evaluate(val_texts, val_labels)
    print(f"  Val Accuracy:  {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Val Macro F1:  {val_metrics['macro_f1']:.4f}")

    print("\nEvaluating on Unseen Held-out Test Set:")
    test_metrics = model.evaluate(test_texts, test_labels)
    print(f"  Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Per-class metrics:")
    for cls_name, m in test_metrics["per_class"].items():
        print(f"    {cls_name.upper():12s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, Support={m['support']}")

    # Save model and metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_tfidf.joblib"
    metrics_path = MODEL_DIR / "baseline_metrics.json"

    model.save(str(model_path))
    with open(metrics_path, "w") as f:
        json.dump({
            "validation": val_metrics,
            "test": test_metrics
        }, f, indent=2)

    print(f"\nSaved trained baseline model to: {model_path}")
    print(f"Saved baseline metrics to: {metrics_path}")

if __name__ == "__main__":
    run_baseline_training()
