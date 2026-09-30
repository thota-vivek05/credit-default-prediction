from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ApplicantInput(BaseModel):
    """Input schema for a single loan applicant."""

    age: int = Field(..., ge=18, le=100, description="Age of the borrower")
    monthly_income: float = Field(..., ge=0, description="Monthly income in USD")
    debt_ratio: float = Field(..., ge=0, le=10, description="Debt-to-income ratio")
    revolving_utilization: float = Field(..., ge=0, le=1, description="Credit utilization rate [0–1]")
    open_credit_lines: int = Field(..., ge=0, description="Number of open credit lines")
    real_estate_loans: int = Field(..., ge=0, description="Number of real estate loans")
    dependents: int = Field(..., ge=0, description="Number of dependents")
    times_late_30_59: int = Field(..., ge=0, description="Times 30–59 days past due (last 2 years)")
    times_late_60_89: int = Field(..., ge=0, description="Times 60–89 days past due (last 2 years)")
    times_late_90: int = Field(..., ge=0, description="Times 90+ days past due")

    @field_validator("revolving_utilization")
    @classmethod
    def cap_utilization(cls, v: float) -> float:
        return min(v, 1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "monthly_income": 6500,
                "debt_ratio": 0.35,
                "revolving_utilization": 0.28,
                "open_credit_lines": 4,
                "real_estate_loans": 1,
                "dependents": 2,
                "times_late_30_59": 0,
                "times_late_60_89": 0,
                "times_late_90": 0,
            }
        }
    }


class ShapSummary(BaseModel):
    top_positive_factors: List[str]
    top_negative_factors: List[str]


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction_id: str
    default_probability: float
    risk_level: str          # LOW | MEDIUM | HIGH
    decision: str            # APPROVE | REVIEW | REJECT
    confidence: float
    shap_summary: ShapSummary


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    total: int
    predictions: List[PredictionResponse]
    approved: int
    rejected: int
    review: int


class ExplainResponse(BaseModel):
    """Full SHAP explanation for a prediction."""

    prediction_id: str
    default_probability: float
    risk_level: str
    shap_values: Dict[str, float]
    feature_values: Dict[str, float]
    base_value: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    total_predictions: int
    uptime_seconds: float
