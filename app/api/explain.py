from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Prediction
from app.ml.pipeline import engineer_features
from app.schemas.schemas import ExplainResponse

router = APIRouter()


@router.get("/explain/{prediction_id}", response_model=ExplainResponse)
def explain(prediction_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Return full SHAP explanation for a previously made prediction.

    Shows which features pushed the score up (toward default)
    and which pushed it down (toward approval).
    """
    record = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found.")

    explainer = request.app.state.explainer
    features = engineer_features(record.input_data)
    explanation = explainer.explain(features)

    return ExplainResponse(
        prediction_id=prediction_id,
        default_probability=record.default_probability,
        risk_level=record.risk_level,
        shap_values=explanation["shap_values"],
        feature_values=explanation["feature_values"],
        base_value=explanation["base_value"],
    )
