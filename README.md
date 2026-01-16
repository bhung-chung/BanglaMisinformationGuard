# BanglaMisinformationGuard

A complete end-to-end **Bangla fake news detection system**, built as a capstone project for the **Machine Learning Zoomcamp**.
The project covers the full ML lifecycle: problem definition, data analysis, model training and selection, API development, Docker containerization, and Kubernetes deployment.

---

## 1. Problem Description

Misinformation and fake news in Bangla (Bengali) digital media can spread rapidly through social networks, blogs, and messaging platforms, often causing social confusion and harm. Automatic detection of Bangla fake news is challenging due to limited curated datasets and language-specific characteristics.

**BanglaMisinformationGuard** addresses this problem by:

* Training multiple machine-learning models on Bangla news text
* Selecting the best performing model
* Serving predictions through a production-ready REST API
* Deploying the service using Docker and Kubernetes

The system outputs:

* A **fake-news risk score** (probability of being fake)
* A human-readable **risk label**: `Safe`, `Suspicious`, or `Fake`

---

## 2. Dataset

The project uses a **public Bangla fake news dataset** from Kaggle:

**Dataset:** Bangla Fake News Dataset
**Source:** [https://www.kaggle.com/datasets/cryptexcode/banfakenews](https://www.kaggle.com/datasets/cryptexcode/banfakenews)

### Files used

* `Authentic-48.csv` (authentic news)
* `Fake-1K.csv` (fake news)

The dataset is **not committed** to the repository. Instead, users download it using their own Kaggle credentials.

### Download instructions

1. Install Kaggle CLI:

```bash
pip install kaggle
```

2. Place your Kaggle API key at:

```
~/.kaggle/kaggle.json
```

3. Download the dataset:

```bash
python src/download_data.py
```

The CSV files will be placed in:

```
data/raw/
```

---

## 3. Exploratory Data Analysis (EDA)

EDA is performed in Jupyter notebooks located in the `notebooks/` directory.

Key analyses include:

* Class imbalance analysis
* Text length distribution
* Token and n-gram frequency analysis
* Missing value handling
* Basic preprocessing validation

Notebooks:

* `notebooks/notebook.ipynb`
* `notebooks/modeling.ipynb`

---

## 4. Model Training and Selection

Multiple models were trained and compared:

| Model                   | Description              |
| ----------------------- | ------------------------ |
| Logistic Regression     | Baseline linear model    |
| Linear SVM (calibrated) | Strong linear classifier |
| LightGBM                | Gradient-boosted trees   |

### Evaluation metric

* **F1-score** (due to class imbalance)

### Best model

**Calibrated Linear SVM** was selected as the final model based on the highest F1-score on the validation set.

### Training command

```bash
python src/train.py --model_type svm
```

Trained artifacts are saved to:

```
models/
```

---

## 5. Inference Script (CLI)

A standalone prediction script is provided:

```bash
python src/predict.py "সুনীলের 'কেউ কথা রাখেনি' কবিতাটি বগুড়ার ভাষায় যেমন শোনাবে"
```

Output:

```
Fake Probability: 0.xx
Risk Label: Safe | Suspicious | Fake
```

---

## 6. REST API (FastAPI)

A production-ready REST API is implemented using **FastAPI**.

### Endpoints

* `GET /health`
  Health check

* `POST /predict`
  Predict fake-news risk

Example request:

```json
{
  "content": "সুনীলের 'কেউ কথা রাখেনি' কবিতাটি বগুড়ার ভাষায় যেমন শোনাবে"
}
```

Example response:

```json
{
  "risk_score": 0.45,
  "label": "Suspicious"
}
```

The API accepts both:

* `content`
* `text`

Swagger UI available at:

```
http://localhost:8000/docs
```

---

## 7. Docker Deployment

### Build image

```bash
docker build -t bangla-guard:latest .
```

### Run container

```bash
docker run -p 8000:8000 bangla-guard:latest
```

### Test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"content":"Bangla text here"}'
```

**Evidence:**

[!!Paste picture here!!]

---

## 8. Kubernetes Deployment (Optional)

The service can be deployed to a **local Kubernetes cluster** (Docker Desktop Kubernetes).

### Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Verify

```bash
kubectl get pods
kubectl get services
```

### Access (local clusters)

On local clusters, `EXTERNAL-IP` may show `<pending>`. Use port-forwarding:

```bash
kubectl port-forward service/bangla-guard-service 8000:80
```

Then test:

```powershell
$body = @{ content = "সুনীলের 'কেউ কথা রাখেনি' কবিতাটি বগুড়ার ভাষায় যেমন শোনাবে" } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

**Evidence:**

[!!Paste picture here!!]

[!!Paste picture here!!]

---

## 9. Project Structure

```
BanglaMisinformationGuard/
│
├── app/            # FastAPI application
├── src/            # Training and inference scripts
├── notebooks/      # EDA and experimentation
├── models/         # Trained model artifacts
├── data/            
│   └── raw/        # Downloaded dataset (not committed)
├── k8s/            # Kubernetes manifests
├── tests/          # API tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 10. Reproducibility

* All dependencies are listed in `requirements.txt`
* Training and inference are reproducible via scripts
* Dataset download instructions are provided

---

## 11. Notes on Academic Integrity

* This project is **original work**
* No reuse of previous Zoomcamp projects
* No copying of external notebooks or codebases
* Public datasets used with proper attribution

---

## 12. License

This project is released for **educational and research purposes**.
