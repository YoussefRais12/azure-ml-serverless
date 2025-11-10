from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
from .inference import predict_texts  # relative import

app = FastAPI(title="Serverless ML API", version="0.1.0")

class Item(BaseModel):
    id: str
    text: str

class PredictRequest(BaseModel):
    items: List[Item]

@app.get("/health")
def health():
    # lightweight health check
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict")
def predict(req: PredictRequest):
    texts = [it.text for it in req.items]
    preds = predict_texts(texts)
    # attach ids back to results in the same order
    enriched = []
    for it, pr in zip(req.items, preds):
        enriched.append({
            "id": it.id,
            "text": it.text,
            "sentiment_label": pr["sentiment_label"],
            "sentiment_score": pr["sentiment_score"],
            "processed_at": pr["processed_at"]
        })
    return enriched
