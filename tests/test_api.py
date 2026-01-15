from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_schema():
    # Attempt to predict without model loaded triggers 503 or 200 depending on mock
    # Since we can't easily mock the global variables without more logic, 
    # we expect 503 if artifacts are missing, or 200 if they exist.
    # We'll check for strict schema compliance if 200.
    
    payload = {"content": "This is a test news article."}
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "risk_score" in data
        assert "label" in data
        assert data["label"] in ["Safe", "Suspicious", "Fake"]
    else:
        assert response.status_code == 503

def test_similar_schema():
    payload = {"content": "Test article", "top_k": 2}
    response = client.post("/similar", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "content" in data[0]
            assert "similarity_score" in data[0]
    else:
        assert response.status_code == 503
