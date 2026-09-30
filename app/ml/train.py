"""
Model Training Script
Downloads data, engineers features, trains XGBoost with Optuna,
saves model artifact.

Usage:
    python app/ml/train.py
"""
import os
import joblib
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from app.ml.pipeline import engineer_features_batch, FEATURE_NAMES

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "./data/cs-training.csv"
MODEL_PATH = "./models/xgboost_model.pkl"
N_TRIALS = 100
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not Path(DATA_PATH).exists():
        print("⚠️  Dataset not found.")
        print("📥 Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data")
        print(f"   Place cs-training.csv in: {DATA_PATH}")
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"✅ Loaded {len(df):,} records")
    return df


def preprocess(df: pd.DataFrame):
    df = df.dropna(subset=["SeriousDlqin2yrs"])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    records = df.rename(columns={
        "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
        "age": "age",
        "NumberOfTime30-59DaysPastDueNotWorse": "times_late_30_59",
        "DebtRatio": "debt_ratio",
        "MonthlyIncome": "monthly_income",
        "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
        "NumberOfTimes90DaysLate": "times_late_90",
        "NumberRealEstateLoansOrLines": "real_estate_loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "times_late_60_89",
        "NumberOfDependents": "dependents",
    }).to_dict("records")

    X = engineer_features_batch(records)
    y = df["SeriousDlqin2yrs"].values
    return X, y


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train():
    print("🚀 Credit Scoring Model Training")
    print("=" * 50)

    df = load_data()
    X, y = preprocess(df)
    print(f"📊 Class distribution: {np.bincount(y.astype(int))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"🔀 Train: {len(X_train):,} | Test: {len(X_test):,}")

    print(f"\n🔍 Running Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

    best_params = study.best_params
    best_params.update({"use_label_encoder": False, "eval_metric": "auc",
                         "random_state": RANDOM_STATE, "n_jobs": -1})
    print(f"✅ Best CV AUC-ROC: {study.best_value:.4f}")

    print("\n🏋️  Training final model...")
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📈 Test AUC-ROC: {auc:.4f}")

    y_pred = (y_pred_proba > 0.15).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    Path("./models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
