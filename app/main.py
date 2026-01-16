from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="BanglaMisinformationGuard API",
    description="API for detecting Bangla fake news",
)

# -------------------------
# Globals
# -------------------------
model = None
vectorizer = None
authentic_db: List[str] = []
authentic_vecs = None

MODELS_DIR = "models"

# Your dataset file name in practice is usually Authentic-48.csv (not Authentic-48K.csv).
# We support both, plus the labeled authentic file as a fallback.
AUTHENTIC_CANDIDATES = [
    "data/raw/Authentic-48.csv",
    "data/raw/Authentic-48K.csv",
    "data/raw/LabeledAuthentic-7K.csv",
]


# -------------------------
# Schemas
# -------------------------
class PredictRequest(BaseModel):
    # Accept either {"content": "..."} or {"text": "..."}
    content: str = Field(..., description="News text to classify", alias="text")

    class Config:
        populate_by_name = True


class SimilarRequest(BaseModel):
    content: str = Field(..., description="News text to retrieve similar authentic articles for", alias="text")
    top_k: int = Field(5, ge=1, le=20, description="Number of similar articles to return")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    risk_score: float
    label: str


class SimilarArticle(BaseModel):
    content: str
    similarity_score: float


# -------------------------
# Helpers
# -------------------------
def _pick_existing_path(candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def _probability_of_fake(estimator, X_vec) -> float:
    """Return P(fake=1). Works for calibrated SVM, logistic, LightGBM.

    If an estimator lacks predict_proba but has decision_function, we convert the margin via sigmoid.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_vec)
        return float(proba[0][1])

    if hasattr(estimator, "decision_function"):
        score = float(np.ravel(estimator.decision_function(X_vec))[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    raise TypeError("Model does not support predict_proba or decision_function")


def _risk_label(prob_fake: float, suspicious_threshold: float = 0.33, fake_threshold: float = 0.66) -> str:
    if prob_fake < suspicious_threshold:
        return "Safe"
    if prob_fake <= fake_threshold:
        return "Suspicious"
    return "Fake"


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, authentic_db, authentic_vecs

    try:
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        print("Model and Vectorizer loaded.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        model = None
        vectorizer = None
        return

    # Load retrieval DB if present
    try:
        data_path = _pick_existing_path(AUTHENTIC_CANDIDATES)
        if data_path is None:
            print("Warning: No authentic CSV found. /similar endpoint will be unavailable.")
            authentic_db = []
            authentic_vecs = None
            return

        df = pd.read_csv(data_path)

        # Detect content column robustly
        col = None
        for cand in ["content", "text", "news", "article", "headline"]:
            if cand in df.columns:
                col = cand
                break
        if col is None:
            # fallback: first object column
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            col = obj_cols[0] if obj_cols else None
        if col is None:
            raise ValueError(f"Could not detect a text column in {data_path.name}. Columns: {list(df.columns)}")

        df[col] = df[col].fillna("").astype(str)
        authentic_db = [t for t in df[col].tolist() if t.strip()]

        print(f"Vectorizing authentic database for retrieval ({len(authentic_db)} articles)...")
        authentic_vecs = vectorizer.transform(authentic_db)
        print("Retrieval database loaded.")

    except Exception as e:
        print(f"Warning: Failed to load retrieval database: {e}")
        authentic_db = []
        authentic_vecs = None


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None and vectorizer is not None,
        "retrieval_loaded": authentic_vecs is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest, suspicious_threshold: float = 0.33, fake_threshold: float = 0.66):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if suspicious_threshold > fake_threshold:
        raise HTTPException(status_code=400, detail="suspicious_threshold must be <= fake_threshold")

    vec = vectorizer.transform([payload.content])
    prob = _probability_of_fake(model, vec)
    label = _risk_label(prob, suspicious_threshold=suspicious_threshold, fake_threshold=fake_threshold)

    return {"risk_score": float(prob), "label": label}


@app.post("/similar", response_model=List[SimilarArticle])
def find_similar(payload: SimilarRequest):
    if vectorizer is None or authentic_vecs is None or not authentic_db:
        raise HTTPException(status_code=503, detail="Retrieval database not loaded")

    vec = vectorizer.transform([payload.content])
    sims = cosine_similarity(vec, authentic_vecs).flatten()

    top_indices = sims.argsort()[-payload.top_k :][::-1]

    results: List[SimilarArticle] = []
    for idx in top_indices:
        snippet = authentic_db[idx].replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        results.append(SimilarArticle(content=snippet, similarity_score=float(sims[idx])))

    return results
