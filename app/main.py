import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.train import train
from app.predict import ChurnModel
from app.monitoring import PREDICTIONS, TRAIN_RUNS, INFER_LATENCY

app = FastAPI(title="ChurnGuard – SaaS Churn Prediction", version="1.1.0")

class TrainRequest(BaseModel):
    dataset_path: str = "data/sample_data/churn.csv"

class PredictRequest(BaseModel):
    logins_per_week: float = Field(..., ge=0)
    avg_session_time: float = Field(..., ge=0)
    feedback_score: float = Field(..., ge=0)
    is_premium: int = Field(..., ge=0, le=1)
    tenure_weeks: float = Field(..., ge=0)

@app.get("/")
def health():
    return {"status": "ok", "service": "ChurnGuard", "version": "1.1.0"}

@app.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        summary = train(dataset_path=req.dataset_path)
        TRAIN_RUNS.inc()
        return {"message": "Model trained", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        start = time.time()
        model = ChurnModel()
        result = model.predict_one(req.dict())
        PREDICTIONS.inc()
        INFER_LATENCY.observe(time.time() - start)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
