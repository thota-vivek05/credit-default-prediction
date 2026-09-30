import uuid
from datetime import datetime
from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Prediction
from app.ml.pipeline import engineer_features
from app.schemas.schemas import ApplicantInput, PredictionResponse, ShapSummary
from app.config import settings

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, applicant: ApplicantInput, db: Session = Depends(get_db)):
    """
    Predict loan default probability for a single applicant.

    Returns risk level (LOW/MEDIUM/HIGH), decision (APPROVE/REVIEW/REJECT),
    confidence score, and top SHAP factors explaining the decision.
    """
    model = request.app.state.model
    explainer = request.app.state.explainer

    features = engineer_features(applicant.model_dump())
    probability = model.predict_single(features)
    risk_level = model.get_risk_level(
        probability, settings.low_risk_threshold, settings.medium_risk_threshold
    )
    decision = model.get_decision(risk_level)
    confidence = model.get_confidence(probability)

    explanation = explainer.explain(features)

    prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    record = Prediction(
        prediction_id=prediction_id,
        default_probability=round(probability, 6),
        risk_level=risk_level,
        decision=decision,
        confidence=confidence,
        input_data=applicant.model_dump(),
        shap_values=explanation["shap_values"],
    )
    db.add(record)
    db.commit()

    return PredictionResponse(
        prediction_id=prediction_id,
        default_probability=round(probability, 4),
        risk_level=risk_level,
        decision=decision,
        confidence=confidence,
        shap_summary=ShapSummary(
            top_positive_factors=explanation["top_positive_factors"],
            top_negative_factors=explanation["top_negative_factors"],
        ),
    )
