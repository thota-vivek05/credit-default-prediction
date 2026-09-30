import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import engine
from app.db.models import Base
from app.ml.model import get_model
from app.ml.explainer import CreditExplainer
from app.api import predict, batch, explain

start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and create DB tables
    Base.metadata.create_all(bind=engine)
    model = get_model(settings.model_path)
    app.state.model = model
    app.state.explainer = CreditExplainer(model)
    print(f"✅ Model loaded from {settings.model_path}")
    yield


app = FastAPI(
    title="Credit Scoring System",
    description="Production-grade ML API for loan default prediction with explainable AI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["Prediction"])
app.include_router(batch.router, tags=["Batch"])
app.include_router(explain.router, tags=["Explainability"])


@app.get("/health", tags=["System"])
def health():
    from app.db.database import SessionLocal
    from app.db.models import Prediction
    db = SessionLocal()
    total = db.query(Prediction).count()
    db.close()
    return {
        "status": "ok",
        "model_version": app.state.model.MODEL_VERSION,
        "total_predictions": total,
        "uptime_seconds": round(time.time() - start_time, 1),
    }
