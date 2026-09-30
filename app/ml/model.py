"""
Model loader and inference engine.
Loads trained XGBoost model and runs predictions.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class CreditScoringModel:
    """Wrapper around trained XGBoost model for inference."""

    MODEL_VERSION = "1.0.0"

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run `python app/ml/train.py` to train the model first."
            )
        return joblib.load(self.model_path)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return default probabilities for each row."""
        return self.model.predict_proba(features)[:, 1]

    def predict_single(self, features: pd.DataFrame) -> float:
        """Return default probability for a single applicant."""
        return float(self.predict_proba(features)[0])

    @staticmethod
    def get_risk_level(probability: float, low_threshold: float, medium_threshold: float) -> str:
        if probability < low_threshold:
            return "LOW"
        elif probability < medium_threshold:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def get_decision(risk_level: str) -> str:
        return {
            "LOW": "APPROVE",
            "MEDIUM": "REVIEW",
            "HIGH": "REJECT",
        }[risk_level]

    @staticmethod
    def get_confidence(probability: float) -> float:
        """
        Confidence = how far the probability is from the decision boundary (0.5).
        High probability or very low probability = high confidence.
        """
        return round(abs(probability - 0.5) * 2, 4)


# Singleton — loaded once at startup
_model_instance: CreditScoringModel | None = None


def get_model(model_path: str) -> CreditScoringModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = CreditScoringModel(model_path)
    return _model_instance
