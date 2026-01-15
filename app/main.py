from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="BanglaMisinformationGuard API", description="API for detecting Bangla fake news")

# Global variables to hold artifacts
model = None
vectorizer = None
authentic_db = None
authentic_vecs = None

MODELS_DIR = "models"
DATA_PATH = "data/raw/Authentic-48K.csv"

class Article(BaseModel):
    content: str
    top_k: int = 5

class PredictionResponse(BaseModel):
    risk_score: float
    label: str
    
class SimilarArticle(BaseModel):
    content: str
    similarity_score: float

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, authentic_db, authentic_vecs
    try:
        # Load Model & Vectorizer
        with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        print("Model and Vectorizer loaded.")
        
        # Load Retrieval Database (Authentic News)
        # For efficiency, we might load a pre-computed pickle, but csv is fine for 50k rows on startup
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df['content'] = df['content'].fillna('').astype(str)
            # Keep a sample to save memory if needed, or full
            authentic_db = df['content'].tolist()
            
            # Pre-compute vectors for similarity
            # This might take a moment on startup
            print("Vectorizing authentic database for retrieval...")
            authentic_vecs = vectorizer.transform(authentic_db)
            print(f"Database loaded with {len(authentic_db)} articles.")
        else:
            print(f"Warning: Authentic data not found at {DATA_PATH}. /similar endpoint will be empty.")
            authentic_db = []
            authentic_vecs = None
            
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(article: Article):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    vec = vectorizer.transform([article.content])
    
    # Get probability of Fake (class 1)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0][1]
    else:
        # Fallback for models without probability (e.g. uncalibrated SVM)
        # Use decision function and sigmoid approximation or just binary
        decision = model.decision_function(vec)[0]
        prob = 1 / (1 + np.exp(-decision))
    
    # Label Schema
    if prob < 0.33:
        label = "Safe"
    elif prob <= 0.66:
        label = "Suspicious"
    else:
        label = "Fake"
        
    return {"risk_score": float(prob), "label": label}

@app.post("/similar")
def find_similar(article: Article):
    if not vectorizer or authentic_vecs is None:
        raise HTTPException(status_code=503, detail="Retrieval database not loaded")
        
    vec = vectorizer.transform([article.content])
    
    # Compute cosine similarity
    similarities = cosine_similarity(vec, authentic_vecs).flatten()
    
    # Get top_k indices
    top_indices = similarities.argsort()[-article.top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "content": authentic_db[idx][:500] + "...", # Truncate for response
            "similarity_score": float(similarities[idx])
        })
        
    return results
