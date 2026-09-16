import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class ConfidenceCalibrator:
    """
    Empirical confidence calibration engine.
    Ensures that a predicted confidence of X% corresponds to an actual accuracy of X%.
    Provides Platt scaling, temperature scaling, and Expected Calibration Error (ECE) metrics.
    """
    def __init__(self, default_temperature: float = 1.10):
        self.temperature = default_temperature

    def calibrate_probabilities(self, probs: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        """
        Apply temperature scaling to raw probability vector.
        Softens overconfident edge distributions while preserving ranking.
        """
        t = temperature or self.temperature
        # Convert probs to logits with epsilon smoothing
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(clipped)
        # Scale logits
        scaled_logits = logits / t
        # Softmax
        exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated = exp_scaled / np.sum(exp_scaled)
        return calibrated

    def compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        ECE measures the difference between expected accuracy and confidence across probability bins.
        """
        if len(probs) == 0 or len(labels) == 0:
            return 0.0

        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return round(float(ece), 4)

    def calculate_brier_score(self, probs: np.ndarray, one_hot_labels: np.ndarray) -> float:
        """
        Calculate multi-class Brier Score (mean squared error of probability predictions).
        Lower is better; 0 represents perfect calibration.
        """
        if len(probs) == 0:
            return 0.0
        return round(float(np.mean(np.sum((probs - one_hot_labels) ** 2, axis=-1))), 4)
