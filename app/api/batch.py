from typing import List
from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.api.predict import predict as predict_single
from app.schemas.schemas import ApplicantInput, BatchPredictionResponse, PredictionResponse

router = APIRouter()


@router.post("/batch", response_model=BatchPredictionResponse)
def batch_predict(request: Request, applicants: List[ApplicantInput], db: Session = Depends(get_db)):
    """
    Score multiple applicants in a single request.
    Returns individual predictions plus summary statistics.
    """
    predictions = [predict_single(request, applicant, db) for applicant in applicants]

    approved = sum(1 for p in predictions if p.decision == "APPROVE")
    rejected = sum(1 for p in predictions if p.decision == "REJECT")
    review = sum(1 for p in predictions if p.decision == "REVIEW")

    return BatchPredictionResponse(
        total=len(predictions),
        predictions=predictions,
        approved=approved,
        rejected=rejected,
        review=review,
    )
