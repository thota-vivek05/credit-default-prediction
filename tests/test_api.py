"""
API endpoint tests.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

SAMPLE_APPLICANT = {
    "age": 45,
    "monthly_income": 6500.0,
    "debt_ratio": 0.35,
    "revolving_utilization": 0.28,
    "open_credit_lines": 4,
    "real_estate_loans": 1,
    "dependents": 2,
    "times_late_30_59": 0,
    "times_late_60_89": 0,
    "times_late_90": 0,
}


class TestSchemas:
    def test_valid_applicant_input(self):
        from app.schemas.schemas import ApplicantInput
        applicant = ApplicantInput(**SAMPLE_APPLICANT)
        assert applicant.age == 45
        assert applicant.monthly_income == 6500.0

    def test_invalid_age_too_young(self):
        from app.schemas.schemas import ApplicantInput
        from pydantic import ValidationError
        data = {**SAMPLE_APPLICANT, "age": 10}
        with pytest.raises(ValidationError):
            ApplicantInput(**data)

    def test_utilization_capped_at_1(self):
        from app.schemas.schemas import ApplicantInput
        data = {**SAMPLE_APPLICANT, "revolving_utilization": 1.5}
        applicant = ApplicantInput(**data)
        assert applicant.revolving_utilization == 1.0

    def test_negative_income_invalid(self):
        from app.schemas.schemas import ApplicantInput
        from pydantic import ValidationError
        data = {**SAMPLE_APPLICANT, "monthly_income": -100}
        with pytest.raises(ValidationError):
            ApplicantInput(**data)


class TestFeaturePipeline:
    def test_engineer_features_returns_dataframe(self):
        import pandas as pd
        from app.ml.pipeline import engineer_features
        features = engineer_features(SAMPLE_APPLICANT)
        assert isinstance(features, pd.DataFrame)

    def test_engineer_features_correct_columns(self):
        from app.ml.pipeline import engineer_features, FEATURE_NAMES
        features = engineer_features(SAMPLE_APPLICANT)
        assert list(features.columns) == FEATURE_NAMES

    def test_engineer_features_income_per_dependent(self):
        from app.ml.pipeline import engineer_features
        features = engineer_features(SAMPLE_APPLICANT)
        expected = 6500 / (2 + 1)
        assert abs(features["income_per_dependent"].iloc[0] - expected) < 0.01

    def test_engineer_features_total_late_payments(self):
        from app.ml.pipeline import engineer_features
        data = {**SAMPLE_APPLICANT, "times_late_30_59": 2, "times_late_60_89": 1, "times_late_90": 0}
        features = engineer_features(data)
        assert features["total_late_payments"].iloc[0] == 3

    def test_engineer_features_batch(self):
        import pandas as pd
        from app.ml.pipeline import engineer_features_batch
        records = [SAMPLE_APPLICANT, SAMPLE_APPLICANT]
        features = engineer_features_batch(records)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 2


class TestModelHelpers:
    def test_get_risk_level_low(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_risk_level(0.05, 0.10, 0.25) == "LOW"

    def test_get_risk_level_medium(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_risk_level(0.15, 0.10, 0.25) == "MEDIUM"

    def test_get_risk_level_high(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_risk_level(0.50, 0.10, 0.25) == "HIGH"

    def test_get_decision_approve(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_decision("LOW") == "APPROVE"

    def test_get_decision_review(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_decision("MEDIUM") == "REVIEW"

    def test_get_decision_reject(self):
        from app.ml.model import CreditScoringModel
        assert CreditScoringModel.get_decision("HIGH") == "REJECT"

    def test_get_confidence_high_probability(self):
        from app.ml.model import CreditScoringModel
        confidence = CreditScoringModel.get_confidence(0.9)
        assert confidence == 0.8

    def test_get_confidence_low_probability(self):
        from app.ml.model import CreditScoringModel
        confidence = CreditScoringModel.get_confidence(0.05)
        assert confidence == 0.9

    def test_get_confidence_boundary(self):
        from app.ml.model import CreditScoringModel
        confidence = CreditScoringModel.get_confidence(0.5)
        assert confidence == 0.0
