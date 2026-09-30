"""
Feature Engineering Pipeline
Transforms raw applicant data into model-ready features.
"""
import numpy as np
import pandas as pd
from typing import Dict


FEATURE_NAMES = [
    "revolving_utilization",
    "age",
    "times_late_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "times_late_90",
    "real_estate_loans",
    "times_late_60_89",
    "dependents",
    # Engineered features
    "income_per_dependent",
    "total_late_payments",
    "log_income",
    "utilization_x_debt",
]


def engineer_features(data: Dict) -> pd.DataFrame:
    """
    Transform raw applicant input into engineered feature vector.

    Args:
        data: Dictionary of raw applicant fields.

    Returns:
        Single-row DataFrame with all features in correct order.
    """
    df = pd.DataFrame([data])

    # Handle missing income
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())

    # Engineered features
    df["income_per_dependent"] = df["monthly_income"] / (df["dependents"] + 1)
    df["total_late_payments"] = (
        df["times_late_30_59"] + df["times_late_60_89"] + df["times_late_90"]
    )
    df["log_income"] = np.log1p(df["monthly_income"])
    df["utilization_x_debt"] = df["revolving_utilization"] * df["debt_ratio"]

    # Cap extreme values
    df["revolving_utilization"] = df["revolving_utilization"].clip(0, 1)
    df["debt_ratio"] = df["debt_ratio"].clip(0, 10)

    return df[FEATURE_NAMES]


def engineer_features_batch(records: list) -> pd.DataFrame:
    """
    Engineer features for a list of applicant records.

    Args:
        records: List of applicant data dicts.

    Returns:
        DataFrame with all records and engineered features.
    """
    frames = [engineer_features(r) for r in records]
    return pd.concat(frames, ignore_index=True)
