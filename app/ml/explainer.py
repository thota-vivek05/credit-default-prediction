"""
SHAP Explainer
Computes feature attributions for individual predictions.
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


FEATURE_LABELS = {
    "revolving_utilization": "Credit Utilization",
    "age": "Age",
    "times_late_30_59": "Late Payments (30–59 days)",
    "debt_ratio": "Debt-to-Income Ratio",
    "monthly_income": "Monthly Income",
    "open_credit_lines": "Open Credit Lines",
    "times_late_90": "Severe Late Payments (90+ days)",
    "real_estate_loans": "Real Estate Loans",
    "times_late_60_89": "Late Payments (60–89 days)",
    "dependents": "Number of Dependents",
    "income_per_dependent": "Income per Dependent",
    "total_late_payments": "Total Late Payments",
    "log_income": "Log Income",
    "utilization_x_debt": "Utilization × Debt Ratio",
}


class CreditExplainer:
    """SHAP-based explainer for the credit scoring model."""

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model.model)

    def explain(self, features: pd.DataFrame) -> Dict:
        """
        Compute SHAP values for a single prediction.

        Returns:
            Dict with shap_values, base_value, and top factors.
        """
        shap_values = self.explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        values = shap_values[0]
        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, list):
            base_value = base_value[1]

        feature_names = features.columns.tolist()
        shap_dict = {
            FEATURE_LABELS.get(name, name): round(float(val), 6)
            for name, val in zip(feature_names, values)
        }

        feature_values = {
            FEATURE_LABELS.get(name, name): round(float(features[name].iloc[0]), 4)
            for name in feature_names
        }

        positive = sorted(
            [(k, v) for k, v in shap_dict.items() if v > 0],
            key=lambda x: x[1], reverse=True
        )
        negative = sorted(
            [(k, v) for k, v in shap_dict.items() if v < 0],
            key=lambda x: x[1]
        )

        return {
            "shap_values": shap_dict,
            "feature_values": feature_values,
            "base_value": round(base_value, 6),
            "top_positive_factors": [k for k, _ in positive[:3]],
            "top_negative_factors": [k for k, _ in negative[:3]],
        }
