import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss

class BaselineTextModel:
    """
    Baseline TF-IDF + Logistic Regression text classifier for Fake/Real/Suspicious detection.
    Serves as the empirical benchmark to quantify the research gain of the Transformer model.
    """
    def __init__(self, max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 2)):
        self.classes_ = ["fake", "real", "suspicious"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode"
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        """Fit TF-IDF and Logistic Regression on training corpus."""
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        self.is_trained = True

    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for a list of texts."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability distribution over classes."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.pipeline.predict_proba(texts)

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Evaluate model on held-out test data and compute real measured metrics."""
        preds = self.predict(texts)
        probs = self.predict_proba(texts)

        acc = float(accuracy_score(labels, preds))
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=self.classes_).tolist()

        try:
            loss = float(log_loss(labels, probs, labels=self.classes_))
        except Exception:
            loss = None

        return {
            "model_type": "TF-IDF + Logistic Regression (Baseline)",
            "classes": self.classes_,
            "accuracy": round(acc, 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "macro_precision": round(report["macro avg"]["precision"], 4),
            "macro_recall": round(report["macro avg"]["recall"], 4),
            "log_loss": round(loss, 4) if loss is not None else None,
            "confusion_matrix": cm,
            "per_class": {
                c: {
                    "precision": round(report[c]["precision"], 4),
                    "recall": round(report[c]["recall"], 4),
                    "f1": round(report[c]["f1-score"], 4),
                    "support": report[c]["support"]
                }
                for c in self.classes_ if c in report
            }
        }

    def save(self, filepath: str):
        """Serialize trained pipeline to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "classes": self.classes_,
            "is_trained": self.is_trained
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaselineTextModel":
        """Load trained pipeline from disk."""
        data = joblib.load(filepath)
        model = cls()
        model.pipeline = data["pipeline"]
        model.classes_ = data["classes"]
        model.is_trained = data["is_trained"]
        return model
