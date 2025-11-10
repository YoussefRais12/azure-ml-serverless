# api/inference.py
from typing import List, Dict
from datetime import datetime
from transformers import pipeline

# Lazy-load the model once per process
_nlp = None

def _get_pipeline():
    global _nlp
    if _nlp is None:
        # Small, fast, and good for demos
        _nlp = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _nlp

def predict_texts(texts: List[str]) -> List[Dict]:
    nlp = _get_pipeline()
    outputs = nlp(texts, truncation=True)
    ts = datetime.utcnow().isoformat() + "Z"
    results = []
    for text, out in zip(texts, outputs):
        # out = {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
        results.append({
            "text": text,
            "sentiment_label": out.get("label", "NEUTRAL").lower(),
            "sentiment_score": float(out.get("score", 0.0)),
            "processed_at": ts
        })
    return results
