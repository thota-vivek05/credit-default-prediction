from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(64), unique=True, nullable=False, index=True)
    default_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    decision = Column(String(16), nullable=False)
    confidence = Column(Float)
    input_data = Column(JSON)
    shap_values = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
