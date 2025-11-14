import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from emotion_config import DEFAULT_ARTIFACT_DIR, LABELS
from emotion_inference import EmotionPredictor

logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Explorer", version="0.1.0")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

_predictor: Optional[EmotionPredictor] = None


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


def get_predictor() -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        artifact_dir = os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
        logger.info("Loading model artifacts from %s", artifact_dir)
        _predictor = EmotionPredictor(artifact_dir)
    return _predictor


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "labels": LABELS})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(request_body: PredictRequest):
    texts: List[str] = []
    if request_body.text:
        texts.append(request_body.text.strip())
    if request_body.texts:
        texts.extend([t.strip() for t in request_body.texts if t and t.strip()])

    texts = [t for t in texts if t]
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided.")

    predictor = get_predictor()
    outputs = predictor.predict_with_probs(texts)
    results = []
    for text, pred, prob in zip(texts, outputs["predictions"], outputs["probabilities"]):
        ordered_prob = [{ "label": label, "score": prob[label], "predicted": bool(pred[label]) } for label in LABELS]
        ordered_prob.sort(key=lambda x: x["score"], reverse=True)
        results.append(
            {
                "text": text,
                "predictions": pred,
                "probabilities": prob,
                "ranked": ordered_prob,
            }
        )
    return {"labels": LABELS, "results": results}
