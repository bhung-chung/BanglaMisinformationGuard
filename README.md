# BanglaMisinformationGuard

## Problem Description
False information and fake news in the Bengali language (Bangla) pose a significant challenge. This project aims to build an automated system to detect fake news in Bangla using Machine Learning and Deep Learning techniques.

We will use the **BanFakeNews** dataset to train our models. The system will be deployed as a containerized API (FastAPI) on Kubernetes.

## Dataset
We are using the [BanFakeNews](https://github.com/Rowan182/BanFakeNews) dataset.
- **Authentic News**: ~50K articles
- **Fake News**: ~3K articles
- **Location**: `data/raw/`

## Label Schema
- **Training**: Binary Classification
    - `0`: Authentic
    - `1`: Fake
- **API Output**: 3-Class Risk Assessment based on prediction probability ($p$)
    - **Safe**: $p < 0.33$
    - **Suspicious**: $0.33 \le p \le 0.66$
    - **Fake**: $p > 0.66$

## Project Structure
- `src/`: Source code for training and evaluation.
- `app/`: FastAPI application code.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `data/`: Dataset storage (ignored by git).
- `k8s/`: Kubernetes deployment manifests.

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd BanglaMisinformationGuard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data**:
   ```bash
   python src/download_data.py
   ```
